import torch
from diffusers import AudioLDM2Pipeline
import scipy.io.wavfile
import time
import sys

# --- Configuration (Set these once) ---
MODEL_REPO_ID = "cvssp/audioldm2-music" 
NEGATIVE_PROMPT = "Low quality, background noise, harsh, distorted, ugly"
INFERENCE_STEPS = 50  # <-- THIS IS THE BIG CHANGE (was 200)
AUDIO_LENGTH_S = 10.0 
SAMPLE_RATE = 16000   
# -------------------------------------

def load_pipeline():
    print("Loading audio pipeline... (This may take a few minutes)")
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            print("WARNING: CUDA not found. Running on CPU. This will be VERY slow.")
            
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
        sys.exit(1)

def generation_loop(pipe):
    while True:
        print() 
        prompt = input("Enter your music prompt: ")
        if not prompt: continue
        if prompt.lower() in ['exit', 'quit', 'q']:
            print("Exiting...")
            break
        
        try:
            print(f"Generating with {INFERENCE_STEPS} steps for: '{prompt}'...")
            start_time = time.time()
            
            audio = pipe(
                prompt=prompt,
                negative_prompt=NEGATIVE_PROMPT,
                num_inference_steps=INFERENCE_STEPS,
                audio_length_in_s=AUDIO_LENGTH_S
            ).audios[0]
            
            end_time = time.time()
            
            timestamp = int(time.time())
            output_filename = f"music_audioldm_{timestamp}.wav"
            
            scipy.io.wavfile.write(output_filename, rate=SAMPLE_RATE, data=audio)
            
            print(f"\nSuccess! Audio saved as {output_filename}")
            print(f"Generation took {end_time - start_time:.2f} seconds.")

        except Exception as e:
            print(f"An error occurred during generation: {e}")

if __name__ == "__main__":
    pipeline = load_pipeline()
    if pipeline:
        generation_loop(pipeline)