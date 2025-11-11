import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import scipy.io.wavfile
import time
import sys
import numpy as np
import logging
import warnings

# --- HIDE WARNINGS (NEW SECTION) ---
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("transformers").setLevel(logging.ERROR)
# ---------------------------------

MODEL_REPO_ID = "facebook/musicgen-small"  # Small model for max speed
NEGATIVE_PROMPT = "Low quality, background noise, harsh, distorted, ugly"
AUDIO_LENGTH_S = 10  # Length of audio to generate in seconds

MOOD_QUALITIES = {
    "Neutral":    {"feel": "balanced, gentle, even", "tempo": "moderate"},
    "Happiness":  {"feel": "joyful, uplifting, bright", "tempo": "fast"},
    "Surprise":   {"feel": "playful, energetic, vibrant", "tempo": "medium-fast"},
    "Sadness":    {"feel": "tender, nostalgic, soft", "tempo": "slow"},
    "Anger":      {"feel": "intense, driving, powerful", "tempo": "fast"},
    "Disgust":    {"feel": "gritty, dark, uneasy", "tempo": "medium"},
    "Fear":       {"feel": "tense, suspenseful, atmospheric", "tempo": "slow"},
    "Contempt":   {"feel": "detached, cold, minimalist", "tempo": "slow"}
}

WEATHER_INSTRUMENTS = {
    "rainy":   "warm piano, chill, reflective textures",
    "sunny":   "bright guitars, upbeat synths",
    "cloudy":  "mellow, smooth pads",
    "stormy":  "dramatic bass, intensity",
    "snowy":   "gentle bells, ambient pads"
}

def get_music_prompt(mood, confidence, weather, user_profile):
    user_genres = user_profile.get('Preferred genres', [])
    user_tempo = user_profile.get('Typical musical qualities', {}).get('tempo', 'moderate')
    user_instruments = user_profile.get('Typical musical qualities', {}).get('instruments', 'synths')
    user_vibe = user_profile.get('Typical musical qualities', {}).get('vibe', 'uplifting')

    mood_quality = MOOD_QUALITIES.get(mood, MOOD_QUALITIES["Neutral"])
    weather_descr = WEATHER_INSTRUMENTS.get(weather.lower(), "")

    if confidence > 0.7:
        genre = user_genres[0] if user_genres else "pop"
        feel = mood_quality["feel"]
        tempo = mood_quality["tempo"]
        atmosphere = weather_descr
        instruments = f"{atmosphere}, {user_instruments}"
        vibe = feel
    elif confidence >= 0.4:
        genre = user_genres[0] if user_genres else "pop"
        feel = f"{mood_quality['feel']}, {user_vibe}"
        tempo = user_tempo
        atmosphere = weather_descr
        instruments = f"{atmosphere}, {user_instruments}"
        vibe = feel
    else:
        genre = user_genres[0] if user_genres else "pop"
        feel = user_vibe
        tempo = user_tempo
        atmosphere = weather_descr
        instruments = f"{atmosphere}, {user_instruments}" if atmosphere else user_instruments
        vibe = feel

    prompt = (
        f"A {genre} track with {instruments}, {tempo} tempo, creating a {atmosphere} atmosphere, "
        f"evoking {vibe}. Style inspired by modern artists."
    )
    return prompt

def load_model():
    print("Loading MusicGen model... (This may take a few minutes)")
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            print("WARNING: CUDA not found. Running on CPU. This will be VERY slow.")
        torch_dtype = torch.float16 if device == "cuda" else torch.float32
        processor = AutoProcessor.from_pretrained(MODEL_REPO_ID)
        model = MusicgenForConditionalGeneration.from_pretrained(
            MODEL_REPO_ID,
            torch_dtype=torch_dtype,
            attn_implementation="eager"
        )
        model = model.to(device)
        print("-" * 50)
        print(f"MusicGen model loaded successfully on device: {device} (using {torch_dtype})")
        print("Type 'exit' or 'quit' at any time to close the script.")
        print("-" * 50)
        return model, processor, device
    except Exception as e:
        print(f"Fatal error loading model: {e}")
        sys.exit(1)

def main_music_controller(model, processor, device):
    sample_rate = model.config.audio_encoder.sampling_rate
    frame_rate = model.config.audio_encoder.frame_rate
    max_new_tokens = int(AUDIO_LENGTH_S * frame_rate)

    while True:
        print("\nEnter inputs for MusicGen Controller.")
        mood = input("Detected mood (Neutral, Happiness, Surprise, Sadness, Anger, Disgust, Fear, Contempt): ").strip()
        confidence = input("Confidence score for the mood (0.0 to 1.0): ").strip()
        weather = input("Current weather condition (sunny, cloudy, rainy, snowy, stormy, etc.): ").strip()
        name = input("User Name: ").strip()
        genres = input("Preferred genres (comma separated): ").strip().split(",")
        tempo = input("Typical musical tempo (e.g., slow, moderate, fast): ").strip()
        instruments = input("Typical instruments (e.g., piano, guitars, synths): ").strip()
        vibe = input("Typical vibe (e.g., uplifting, chill, dramatic): ").strip()

        try:
            confidence = float(confidence)
        except Exception:
            confidence = 0.5

        user_profile = {
            "Name": name,
            "Preferred genres": [g.strip() for g in genres if g.strip()],
            "Typical musical qualities": {
                "tempo": tempo,
                "instruments": instruments,
                "vibe": vibe
            }
        }

        prompt = get_music_prompt(mood, confidence, weather, user_profile)
        print("\nPrompt:", prompt)

        inputs = processor(
            text=[prompt],
            padding=True,
            return_tensors="pt"
        ).to(device)

        try:
            print(f"Generating {AUDIO_LENGTH_S}s of audio for: '{prompt}'...")
            start_time = time.time()

            audio_values = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens
            )

            timestamp = int(time.time())
            output_filename = f"music_musicgen_{timestamp}.wav"

            audio_numpy = audio_values[0].cpu().numpy()
            audio_mono = audio_numpy.squeeze()
            audio_int16 = (audio_mono * 32767).astype(np.int16)
            scipy.io.wavfile.write(output_filename, rate=sample_rate, data=audio_int16)

            end_time = time.time()
            print(f"Music: {output_filename}")
            print(f"Generation took {end_time - start_time:.2f} seconds.")

        except Exception as e:
            print(f"An error occurred during generation: {e}")

if __name__ == "__main__":
    model, processor, device = load_model()
    if model:
        main_music_controller(model, processor, device)