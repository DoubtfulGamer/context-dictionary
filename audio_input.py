import numpy as np
import sounddevice as sd
import soundfile as sf
import queue
import time
from pynput import keyboard


def record(
    filename: str = "my_voice.wav",
    sample_rate: int = 44100,
    channels: int = 1,
    threshold: float = 0.01,
    silence_duration: float = 2.5,
) -> str | None:
    """
    Record microphone audio to a WAV file.

    Stops automatically after `silence_duration` seconds of silence,
    or immediately when Esc is pressed.

    Args:
        filename:         Output WAV file path.
        sample_rate:      Sample rate in Hz.
        channels:         Number of audio channels.
        threshold:        RMS volume below which audio is considered silence (0.0–1.0).
        silence_duration: Seconds of consecutive silence before auto-stop.

    Returns:
        The output filename on success, or None if no audio was captured.
    """
    audio_queue: queue.Queue = queue.Queue()
    recording_active = True
    stop_reason = "Silence detected"

    def callback(indata, frames, time_info, status):
        if recording_active:
            audio_queue.put(indata.copy())

    def on_press(key):
        nonlocal recording_active, stop_reason
        if key == keyboard.Key.esc:
            stop_reason = "Manual override (Esc pressed)"
            recording_active = False
            return False

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    stream = sd.InputStream(samplerate=sample_rate, channels=channels, callback=callback)

    print("🎙️  Recording started...")
    print("-> Stop speaking to auto-save.")
    print("-> Press 'Esc' to abruptly cancel/stop at any time.\n")

    audio_chunks = []
    silence_start_time = None

    with stream:
        while recording_active:
            while not audio_queue.empty():
                chunk = audio_queue.get()
                audio_chunks.append(chunk)

                volume_norm = np.sqrt(np.mean(chunk**2))

                if volume_norm < threshold:
                    if silence_start_time is None:
                        silence_start_time = time.time()
                    elif time.time() - silence_start_time >= silence_duration:
                        recording_active = False
                        break
                else:
                    silence_start_time = None

            time.sleep(0.1)

    listener.stop()

    if audio_chunks:
        finished_recording = np.concatenate(audio_chunks, axis=0)

        # Reject if total duration is under 0.5 seconds
        duration = len(finished_recording) / sample_rate
        if duration < 0.5:
            print("\nERROR: Recording too short — no usable audio captured.")
            return None

        # Reject if entire recording is silent (flat line)
        overall_rms = np.sqrt(np.mean(finished_recording**2))
        if overall_rms < threshold:
            print("\nERROR: Recording is silent — no speech detected.")
            return None

        print(f"\nRecording stopped. REASON: {stop_reason}")
        print("Saving file...")
        sf.write(filename, finished_recording, sample_rate)
        print(f"Successfully saved to: {filename}")
        return filename

    print("\nERROR: Recording ended before any audio data was processed.")
    return None