import torch
from diffusers import AudioLDM2Pipeline
import scipy.io.wavfile
import time
import sys

# --- Configuration (Set these once) ---
MODEL_REPO_ID = "cvssp/audioldm2-music" # The music-specific model
NEGATIVE_PROMPT = "Low quality, background noise, harsh, distorted, ugly"
INFERENCE_STEPS = 200  # Pro Tip: Lower this to ~100 for faster generation
AUDIO_LENGTH_S = 30.0  # Length of audio to generate in seconds
SAMPLE_RATE = 16000    # AudioLDM2 default sample rate
# -------------------------------------

def load_pipeline():
    """
    Loads the diffusion pipeline from Hugging Face and moves it to the GPU.
    This is the slow part that we only want to do once.
    """
    print("Loading audio pipeline... (This may take a few minutes the first time)")
    
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Use float16 for a massive speedup and less VRAM usage on NVIDIA GPUs
        torch_dtype = torch.float16 if device == "cuda" else torch.float32

        pipe = AudioLDM2Pipeline.from_pretrained(MODEL_REPO_ID, torch_dtype=torch_dtype)
        pipe = pipe.to(device)
        
        print("-" * 50)
        print(f"Pipeline loaded successfully on device: {device} (using {torch_dtype})")
        print("Type 'exit' or 'quit' at any time to close the script.")
        print("-" * 50)
        return pipe

    except Exception as e:
        print(f"Fatal error loading pipeline: {e}")
        print("Please ensure 'diffusers', 'transformers', and 'torch' are installed.")
        print("If you have an NVIDIA GPU, make sure CUDA is set up correctly.")
        sys.exit(1) # Exit the script if the model can't be loaded

def generation_loop(pipe):
    """
    Starts an interactive loop that takes user input and generates audio.
    """
    while True:
        print() # Add a new line for readability
        prompt = input("Enter your music prompt: ")

        if not prompt: # Handle empty input
            continue
            
        if prompt.lower() in ['exit', 'quit', 'q']:
            print("Exiting...")
            break
        
        try:
            print(f"Generating audio for: '{prompt}'...")
            start_time = time.time()
            
            # Generate the audio
            audio = pipe(
                prompt=prompt,
                negative_prompt=NEGATIVE_PROMPT,
                num_inference_steps=INFERENCE_STEPS,
                audio_length_in_s=AUDIO_LENGTH_S
            ).audios[0]
            
            end_time = time.time()
            
            # --- Save the Audio ---
            # Create a unique filename using a timestamp
            timestamp = int(time.time())
            output_filename = f"music_{timestamp}.wav"
            
            scipy.io.wavfile.write(output_filename, rate=SAMPLE_RATE, data=audio)
            
            print(f"\nSuccess! Audio saved as {output_filename}")
            print(f"Generation took {end_time - start_time:.2f} seconds.")

        except Exception as e:
            print(f"An error occurred during generation: {e}")

if __name__ == "__main__":
    pipeline = load_pipeline()
    if pipeline:
        generation_loop(pipeline)