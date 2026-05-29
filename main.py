import os
import json
import time
import soundfile as sf
from kokoro_onnx import Kokoro
from groq import Groq, RateLimitError, APIError
from pydantic import BaseModel, Field
from playsound3 import playsound
from audio_input import record

# ── Constants ─────────────────────────────────────────────────────────────────

MODEL_ID = "llama-3.3-70b-versatile"
WHISPER_MODEL = "whisper-large-v3"
SYSTEM_INSTRUCTION = "You are a precise linguistic assistant. Analyze words based strictly on their sentence context. Always respond in valid JSON only, no extra text."
MAX_RETRIES = 3
RATE_LIMIT_WAIT_SECONDS = 15
TTS_VOICE = "af_sarah"
MIC_INPUT_PATH = "input/recorded_input.wav"

# ── Schema ────────────────────────────────────────────────────────────────────

class WordAnalysis(BaseModel):
    word: str = Field(description="The word being analyzed")
    contextual_meaning: str = Field(description="The meaning of the word in context")
    part_of_speech: str = Field(description="Noun, Verb, Adjective, etc.")

# ── Client ────────────────────────────────────────────────────────────────────

def get_client() -> Groq:
    api_key = "api_key_here"  # paste Groq key here
    return Groq(api_key=api_key)

# ── TTS ───────────────────────────────────────────────────────────────────────

def load_tts_model() -> Kokoro:
    """Load Kokoro TTS model. Downloads ~80MB on first run."""
    print("Loading Kokoro TTS...")
    kokoro = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")
    print("[KOKORO LOADED]")
    return kokoro

def speak_to_file(text: str, output_path: str, kokoro: Kokoro) -> None:
    """Convert text to speech using Kokoro and save as .wav file."""
    print(f"[Generating speech]: {output_path}")
    samples, sample_rate = kokoro.create(text, voice=TTS_VOICE, speed=0.8, lang="en-us")
    sf.write(output_path, samples, sample_rate)
    print(f"[DONE]: Audio saved to {output_path}")

# ── Audio Input — Record or Use File ─────────────────────────────────────────

def get_input_audio() -> str | None:
    # Record from the microphone and return that path. Returns None if no usable audio could be obtained.

    # Record from mic
    os.makedirs(os.path.dirname(MIC_INPUT_PATH), exist_ok=True)
    recorded_path = record(filename=MIC_INPUT_PATH)

    if recorded_path is None:
        print("[ERROR]: Microphone recording captured no audio.")
        return None

    return recorded_path

# ── Audio Input — Transcribe ──────────────────────────────────────────────────

def transcribe_audio(client: Groq, audio_path: str) -> str | None:
    # Transcribe audio file to text using Groq Whisper.
    
    print(f"TRANSCRIBING: {audio_path}")
    try:
        with open(audio_path, "rb") as f:
            transcription = client.audio.transcriptions.create(
                model=WHISPER_MODEL,
                file=f,
                response_format="text",
            )
        print(f"TRANSCRIBED: {transcription}")
        return transcription.strip()
    except Exception as e:
        print(f"[Transcription ERROR]: {e}")
        return None

# ── Extract Word + Sentence ───────────────────────────────────────────────────

def extract_word_and_sentence(client: Groq, transcript: str) -> tuple[str, str] | None:
    # Use LLM to extract the target word and sentence from natural speech.
    
    prompt = (
        f"The user said: '{transcript}'\n\n"
        f"Extract the target word they want analyzed and the sentence it appears in.\n"
        f"Respond ONLY with JSON: {{\"word\": \"...\", \"sentence\": \"...\"}}"
    )
    try:
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {"role": "system", "content": "Extract the word and sentence from the user's request. Respond only in JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        data = json.loads(response.choices[0].message.content)
        word = data.get("word", "").strip()
        sentence = data.get("sentence", "").strip()
        if not word or not sentence:
            print("[ERROR]: Could not extract word or sentence.")
            return None
        print(f"EXTRACTED: Word: '{word}' | Sentence: '{sentence}'")
        return word, sentence
    except Exception as e:
        print(f"[Extraction ERROR]: {e}")
        return None

# ── Prompt ────────────────────────────────────────────────────────────────────

def build_prompt(sentence: str, target_word: str) -> str:
    return (
        f"Analyze the word '{target_word}' in this sentence: '{sentence}'\n\n"
        f"Respond ONLY with a JSON object with these fields:\n"
        f"- word (string)\n"
        f"- contextual_meaning (string)\n"
        f"- part_of_speech (string)\n"
    )

# ── Analyzer ──────────────────────────────────────────────────────────────────

def analyze_word_in_context(
    client: Groq,
    sentence: str,
    target_word: str,
    _retries: int = 0,
) -> WordAnalysis | None:
    if _retries >= MAX_RETRIES:
        print(f"[Max retries ({MAX_RETRIES}) reached. Giving up.]")
        return None

    try:
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                {"role": "user", "content": build_prompt(sentence, target_word)},
            ],
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        return WordAnalysis(**json.loads(response.choices[0].message.content))

    except RateLimitError:
        wait = RATE_LIMIT_WAIT_SECONDS * (_retries + 1)
        print(f"\n[Rate limit hit. Waiting {wait}s before retry {_retries + 1}/{MAX_RETRIES}...]")
        time.sleep(wait)
        return analyze_word_in_context(client, sentence, target_word, _retries + 1)

    except APIError as e:
        print(f"[API ERROR]: {e}")
        return None

    except Exception as e:
        print(f"[Unexpected ERROR]: {e}")
        return None

# ── Format Result as Speakable Text ──────────────────────────────────────────

def format_for_speech(result: WordAnalysis) -> str:
    pos_article = "an" if result.part_of_speech.lower()[0] in "aeiou" else "a"
    return (
        f"The word {result.word} is used as {pos_article} {result.part_of_speech} in this sentence. "
        f"In this context, it means: {result.contextual_meaning}."
    )

# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    client = get_client()

    # Step 1: Record from mic
    audio_path = get_input_audio()
    if not audio_path:
        print("[FAILED]: No audio input available. Exiting.")
        return

    # Step 2: Transcribe audio
    transcript = transcribe_audio(client, audio_path)
    if not transcript:
        print("[FAILED]: Could not transcribe audio.")
        return

    # Step 3: Extract word + sentence
    extracted = extract_word_and_sentence(client, transcript)
    if not extracted:
        print("[FAILED]: Could not extract word and sentence.")
        return
    word, sentence = extracted

    # Step 4: Analyze word in context
    result = analyze_word_in_context(client, sentence, word)
    if not result:
        print("[FAILED]: Could not analyze word.")
        return

    # Step 5: Print result
    print(f"\nWord Analysis  : {result.word.upper()}")
    print(f"Part of Speech : {result.part_of_speech}")
    print(f"Meaning        : {result.contextual_meaning}")

    # Step 6: Generate and play audio
    kokoro = load_tts_model()
    speech_text = format_for_speech(result)
    os.makedirs("output", exist_ok=True)
    output_path = f"output/meaning.wav"
    speak_to_file(speech_text, output_path, kokoro)
    print(f"PLAYING AUDIO: {output_path}")
    playsound(output_path)
    """if os.path.exists(output_path):
        os.remove(output_path) #Clean up output audio file after playing
    
    if os.path.exists(audio_path):
        os.remove(audio_path) #Clean up recorded audio file after processing"""

if __name__ == "__main__":
    main()