# Word Analyzer 🎙️

Ever heard a word and thought — *what does that actually mean here?* This tool does exactly that. Say the word and the sentence out loud, and it tells you what it means in that context. Out loud, back to you.

No typing. Just talk.

---

## What it does

You speak into your mic. The program listens, figures out which word you're asking about, looks it up in context, and reads the answer back to you in a natural voice.

That's it.

---

## Before you start

You'll need:
- Python 3.10 or above
- A Groq API key (free) — grab one at [console.groq.com](https://console.groq.com)

---

## Getting it running

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Download the voice model files** (one time only, goes in your project folder):
```powershell
wget https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx -OutFile kokoro-v1.0.onnx
wget https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin -OutFile voices-v1.0.bin
```

**Add your Groq key** in `main.py`:
```python
api_key = "paste_your_key_here"
```

---

## How to use it

```bash
python main.py
```

When you see the mic prompt, just speak. Say something like:

> *"What does frantic mean in: people live at a frantic pace"*

> *"What is the meaning of light in: she carried a light backpack"*

> *"Explain the word bark in: the bark of the tree was rough"*

The recording stops on its own after you go quiet for a couple seconds. Hit `Esc` if you want to cancel.

---

## What you'll see (and hear)

```
🎙️  Recording started...

[Transcribed]: What does frantic mean in people live at a frantic pace
[Extracted] Word: 'frantic' | Sentence: 'people live at a frantic pace'

Word Analysis  : FRANTIC
Part of Speech : adjective
Meaning        : in a hurried, stressed state with little control
```

And then it reads that last part out loud.
The audio gets saved to `output/meaning.wav` too, if you want to replay it.

---

## Tweaking things

**Voice too fast or robotic?**
Change the speed in `main.py`:
```python
samples, sample_rate = kokoro.create(text, voice=TTS_VOICE, speed=0.8, lang="en-us")
```
`0.8` is slower, `1.2` is faster. `af_sarah` is the default voice — American female, pretty natural.
Other options: `af_bella`, `af_nicole`, `bf_emma` (British).

**Mic not picking up your voice?**
Lower the threshold in `audio_input.py`:
```python
threshold: float = 0.005  # try 0.001 if still not working
```

---

## Project layout

```
project/
├── main.py              # the brain
├── audio_input.py       # handles mic recording
├── kokoro-v1.0.onnx     # voice model (you download this)
├── voices-v1.0.bin      # voice data (you download this)
├── requirements.txt
├── input/               # your recorded audio lands here
└── output/              # generated speech saved here
```

---

## Something broke?

| What's happening | What to do |
|-----------------|------------|
| *"No audio input"* | Lower `threshold` in `audio_input.py` |
| *"voices-v1.0.bin not found"* | Re-run the wget commands above |
| *"Rate limit hit"* | It retries on its own, just wait |
| *"Could not extract word"* | Speak clearly and follow the example format |
| *"401 Unauthorized"* | Double-check your Groq API key |