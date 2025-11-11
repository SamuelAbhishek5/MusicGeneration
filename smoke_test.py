#!/usr/bin/env python3
"""
Smoke test for the musicserver /generate endpoint.
Verifies that the server:
  - Accepts POST requests to /generate
  - Returns valid JSON with music_url and prompt
  - Serves the generated .wav file
"""

import requests
import json
import sys
import time
import os

# Change this to your server URL
SERVER_URL = os.environ.get("SERVER_URL", "http://127.0.0.1:5000")

def test_health():
    """Test the /health endpoint."""
    print("[1/4] Testing /health endpoint...")
    try:
        resp = requests.get(f"{SERVER_URL}/health", timeout=5)
        if resp.status_code == 200 and resp.json().get("status") == "ok":
            print("  ✓ Health check passed")
            return True
        else:
            print(f"  ✗ Health check failed: {resp.status_code}")
            return False
    except Exception as e:
        print(f"  ✗ Health check error: {e}")
        return False

def test_generate():
    """Test the /generate endpoint."""
    print("[2/4] Testing /generate endpoint...")
    try:
        payload = {
            "user_id": "default",
            "mood": "happiness",
            "confidence": 0.8
        }
        resp = requests.post(
            f"{SERVER_URL}/generate",
            json=payload,
            timeout=120  # Music generation can be slow
        )
        if resp.status_code != 200:
            print(f"  ✗ Generate request failed: {resp.status_code}")
            print(f"    Response: {resp.text}")
            return None
        
        data = resp.json()
        if "music_url" not in data or "prompt" not in data:
            print(f"  ✗ Invalid response: missing music_url or prompt")
            print(f"    Response: {data}")
            return None
        
        print(f"  ✓ Generation successful")
        print(f"    Prompt: {data['prompt'][:60]}...")
        print(f"    Music URL: {data['music_url']}")
        return data
    except Exception as e:
        print(f"  ✗ Generate request error: {e}")
        return None

def test_music_download(music_url):
    """Test downloading the generated music file."""
    print("[3/4] Testing music file download...")
    try:
        resp = requests.get(music_url, timeout=10)
        if resp.status_code != 200:
            print(f"  ✗ Download failed: {resp.status_code}")
            return False
        
        file_size = len(resp.content)
        if file_size < 1000:  # Sanity check: at least 1 KB
            print(f"  ✗ Downloaded file too small: {file_size} bytes")
            return False
        
        print(f"  ✓ Downloaded music file: {file_size} bytes")
        return True
    except Exception as e:
        print(f"  ✗ Download error: {e}")
        return False

def test_moods():
    """Test generation for different moods."""
    print("[4/4] Testing different moods...")
    moods = ['happiness', 'sadness', 'anger', 'neutral', 'fear']
    results = {}
    
    for mood in moods:
        try:
            payload = {"user_id": "default", "mood": mood, "confidence": 0.7}
            resp = requests.post(
                f"{SERVER_URL}/generate",
                json=payload,
                timeout=120
            )
            if resp.status_code == 200:
                results[mood] = "✓"
                print(f"  ✓ {mood}")
            else:
                results[mood] = f"✗ ({resp.status_code})"
                print(f"  ✗ {mood}: {resp.status_code}")
        except Exception as e:
            results[mood] = f"✗ ({str(e)[:20]})"
            print(f"  ✗ {mood}: {e}")
        
        # Don't hammer the server
        time.sleep(1)
    
    return results

def main():
    print(f"\n=== MusicGen Server Smoke Test ===")
    print(f"Server: {SERVER_URL}\n")
    
    # Run tests
    health_ok = test_health()
    if not health_ok:
        print("\n✗ Server is not reachable. Aborting.")
        sys.exit(1)
    
    gen_result = test_generate()
    if not gen_result:
        print("\n✗ Generation failed. Aborting.")
        sys.exit(1)
    
    download_ok = test_music_download(gen_result["music_url"])
    if not download_ok:
        print("\n✗ Music download failed.")
        sys.exit(1)
    
    # Optional: test multiple moods (comment out if you want faster tests)
    # mood_results = test_moods()
    
    print("\n=== All tests passed! ===\n")
    sys.exit(0)

if __name__ == "__main__":
    main()
