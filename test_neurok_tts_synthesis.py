#!/usr/bin/env python3
"""
Advanced test script for the neurok-tts plugin.
This script demonstrates how to synthesize speech and save it to a file.
"""

import os
import asyncio
import tempfile
from pathlib import Path

# Import the neurok-tts plugin
from livekit.plugins.neurok_tts import TTS


async def synthesize_and_save(tts, text, output_path):
    """
    Synthesize text to speech and save to a file.
    
    Args:
        tts: TTS engine instance
        text: Text to synthesize
        output_path: Path to save the audio file
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        print(f"Synthesizing text: '{text}'")
        
        # Open the output file
        with open(output_path, "wb") as f:
            # Synthesize text to speech
            stream = tts.synthesize(text)
            
            # Process the stream events
            async for event in stream:
                # The event structure depends on the livekit.agents implementation
                # We'll try different attribute names that might contain the audio data
                audio_data = None
                
                # Try different possible attribute names
                if hasattr(event, "audio"):
                    audio_data = event.audio
                elif hasattr(event, "audio_data"):
                    audio_data = event.audio_data
                elif hasattr(event, "data"):
                    audio_data = event.data
                elif hasattr(event, "chunk"):
                    audio_data = event.chunk
                
                if audio_data:
                    f.write(audio_data)
                    print(f"Wrote {len(audio_data)} bytes of audio data")
        
        print(f"Audio saved to: {output_path}")
        print(f"Audio file size: {os.path.getsize(output_path)} bytes")
        
        return True
    
    except Exception as e:
        print(f"Error during synthesis: {e}")
        return False


async def test_tts_synthesis():
    """Test TTS synthesis functionality."""
    print("Testing TTS synthesis...")
    
    # Initialize the TTS engine with default settings (fallback mode)
    tts = TTS()
    
    # Text to synthesize
    text = "Tere! See on lihtne test neurok-tts plugina jaoks."
    
    # Create a temporary file to store the audio
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_path = temp_file.name
    
    try:
        success = await synthesize_and_save(tts, text, temp_path)
        
        if success:
            print(f"Synthesis successful. Audio saved to {temp_path}")
        else:
            print("Synthesis failed.")
        
        return success, temp_path
    
    except Exception as e:
        print(f"Error during test: {e}")
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        return False, None


async def test_tts_with_different_voices():
    """Test TTS synthesis with different voices."""
    print("\nTesting TTS synthesis with different voices...")
    
    results = []
    
    # Test with different voices
    for speaker in ["mari", "albert", "vesta"]:
        # Initialize the TTS engine with specific settings
        tts = TTS(speaker=speaker)
        
        # Text to synthesize
        text = f"See on test {speaker} häälega."
        
        # Create a temporary file to store the audio
        with tempfile.NamedTemporaryFile(suffix=f"_{speaker}.wav", delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            print(f"\nTesting with speaker '{speaker}'")
            success = await synthesize_and_save(tts, text, temp_path)
            
            if success:
                print(f"Synthesis with {speaker} successful. Audio saved to {temp_path}")
            else:
                print(f"Synthesis with {speaker} failed.")
            
            results.append((speaker, success, temp_path))
        
        except Exception as e:
            print(f"Error during test with {speaker}: {e}")
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            results.append((speaker, False, None))
    
    return results


async def main():
    """Run all tests."""
    print("=== Testing Neurok TTS Synthesis ===\n")
    
    # Test basic TTS synthesis
    basic_success, basic_path = await test_tts_synthesis()
    
    # Test TTS with different voices
    voice_results = await test_tts_with_different_voices()
    
    # Summary
    print("\n=== Test Summary ===")
    print(f"Basic synthesis test: {'Success' if basic_success else 'Failed'}")
    
    print("\nVoice tests:")
    for speaker, success, path in voice_results:
        print(f"  - {speaker}: {'Success' if success else 'Failed'}")
    
    print("\nOutput files:")
    if basic_success:
        print(f"  - Default voice: {basic_path}")
    
    for speaker, success, path in voice_results:
        if success:
            print(f"  - {speaker}: {path}")
    
    print("\nTests completed.")


if __name__ == "__main__":
    asyncio.run(main())