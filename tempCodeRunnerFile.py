import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import scipy.io.wavfile
import time
import sys
import numpy as np

# --- HIDE WARNINGS (NEW SECTION) ---
import logging
import warnings

# 1. Suppress the harmless "torch.tensor(sourceTensor)" UserWarning
# We're ignoring all UserWarnings, as they are not critical.
warnings.filterwarnings("ignore", category=UserWarning)

# 2. Suppress the long "Config of the..." informational messages
# This tells the 'transformers' library to only log a message if it's a real ERROR.
logging.getLogger("transformers").setLevel(logging.ERROR)
# ---------------------------------

# --- Configuration ---
MODEL_REPO_ID = "facebook/musicgen-small" # Small model for max speed
NEGATIVE_PROMPT = "Low quality, background noise, harsh, distorted, ugly"
AUDIO_LENGTH_S = 10 # Length of audio to generate in seconds
# ---------------------

def load_model():
    print("Loading MusicGen model... (This may take a few minutes)")
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            print("WARNING: CUDA not found. Running on CPU. This will be VERY slow.")
            
        torch_dtype = torch.float16 if device == "cuda" else torch.float32

        # MusicGen uses a "Processor" to handle both text and audio
        processor = AutoProcessor.from_pretrained(MODEL_REPO_ID)
        model = MusicgenForConditionalGeneration.from_pretrained(
            MODEL_REPO_ID, 
            torch_dtype=torch_dtype,
            attn_implementation="eager"  # Suppresses the "empty mask" warning
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

def generation_loop(model, processor, device):
    
    # Calculate token limit for the desired audio length
    # This model has a specific sample rate and frame rate
    sample_rate = model.config.audio_encoder.sampling_rate
    frame_rate = model.config.audio_encoder.frame_rate
    max_new_tokens = int(AUDIO_LENGTH_S * frame_rate)

    while True:
        print() 
        prompt = input("Enter your music prompt: ")
        if not prompt: continue
        if prompt.lower() in ['exit', 'quit', 'q']:
            print("Exiting...")
            break
        
        try:
            print(f"Generating {AUDIO_LENGTH_S}s of audio for: '{prompt}'...")
            start_time = time.time()
            
            # Prepare the inputs
            inputs = processor(
                text=[prompt],
                padding=True,
                return_tensors="pt"
            ).to(device)
            
            # Generate the audio
            audio_values = model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens
            )
            
            end_time = time.time()
            
            # --- Save the Audio ---
            timestamp = int(time.time())
            output_filename = f"music_musicgen_{timestamp}.wav"
            
            # The model output needs to be put on the CPU and converted to numpy
            audio_numpy = audio_values[0].cpu().numpy()

            # Squeeze the audio to be 1D (mono) instead of 2D (1, num_samples)
            audio_mono = audio_numpy.squeeze()

            # Convert from float [-1.0, 1.0] to 16-bit integer [-32767, 32767]
            # This is the most compatible format for WAV files
            audio_int16 = (audio_mono * 32767).astype(np.int16)
            
            # Save the 16-bit integer data
            scipy.io.wavfile.write(output_filename, rate=sample_rate, data=audio_int16)
            
            print(f"\nSuccess! Audio saved as {output_filename}")
            print(f"Generation took {end_time - start_time:.2f} seconds.")

        except Exception as e:
            print(f"An error occurred during generation: {e}")

if __name__ == "__main__":
    model, processor, device = load_model()
    if model:
        generation_loop(model, processor, device)