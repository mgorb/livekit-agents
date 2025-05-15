#!/usr/bin/env python3
"""
Debug script for the neurok-tts plugin.
This script helps diagnose issues with the TTS synthesis process.
"""

import os
import asyncio
import inspect
import tempfile
from pathlib import Path

# Import the neurok-tts plugin
from livekit.plugins.neurok_tts import TTS


async def debug_tts_event_structure(tts, text):
    """
    Debug the structure of TTS events.
    
    Args:
        tts: TTS engine instance
        text: Text to synthesize
    """
    print(f"Synthesizing text: '{text}'")
    
    # Synthesize text to speech
    stream = tts.synthesize(text)
    
    # Process the stream events
    event_count = 0
    try:
        async for event in stream:
            event_count += 1
            print(f"\nEvent {event_count}:")
            print(f"  Type: {type(event)}")
            print(f"  Dir: {dir(event)}")
            
            # Try to access common attributes
            for attr in dir(event):
                if attr.startswith('_'):
                    continue
                
                try:
                    value = getattr(event, attr)
                    if isinstance(value, (bytes, bytearray)):
                        print(f"  {attr}: <bytes> (length: {len(value)})")
                    else:
                        print(f"  {attr}: {value}")
                except Exception as e:
                    print(f"  {attr}: Error accessing - {e}")
    
    except Exception as e:
        print(f"Error during stream processing: {e}")
    
    print(f"\nTotal events received: {event_count}")


async def test_direct_file_access():
    """Test if we can access the temporary file created by the TTS engine."""
    print("\nTesting direct file access...")
    
    # Initialize the TTS engine
    tts = TTS()
    
    # Text to synthesize
    text = "Test direct file access."
    
    # Create a temporary directory to monitor
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Monitoring directory: {temp_dir}")
        
        # Get initial files
        initial_files = set(os.listdir(temp_dir))
        print(f"Initial files: {initial_files}")
        
        # Start synthesis
        print(f"Starting synthesis...")
        stream = tts.synthesize(text)
        
        # Just consume the stream
        async for _ in stream:
            pass
        
        # Check for new files
        final_files = set(os.listdir(temp_dir))
        print(f"Final files: {final_files}")
        
        new_files = final_files - initial_files
        print(f"New files: {new_files}")


async def examine_tts_implementation():
    """Examine the TTS implementation details."""
    print("\nExamining TTS implementation...")
    
    # Initialize the TTS engine
    tts = TTS()
    
    # Print class hierarchy
    print(f"TTS class: {TTS}")
    print(f"TTS base classes: {TTS.__bases__}")
    
    # Print methods
    print("\nTTS methods:")
    for name, method in inspect.getmembers(TTS, predicate=inspect.isfunction):
        if not name.startswith('_'):
            print(f"  {name}: {method}")
    
    # Print instance attributes
    print("\nTTS instance attributes:")
    for attr in dir(tts):
        if not attr.startswith('_'):
            try:
                value = getattr(tts, attr)
                if not callable(value):
                    print(f"  {attr}: {value}")
            except Exception as e:
                print(f"  {attr}: Error accessing - {e}")


async def main():
    """Run all debug tests."""
    print("=== Debugging Neurok TTS Plugin ===\n")
    
    # Initialize the TTS engine
    tts = TTS()
    
    # Debug event structure
    await debug_tts_event_structure(tts, "Tere! See on debug test.")
    
    # Test direct file access
    await test_direct_file_access()
    
    # Examine TTS implementation
    await examine_tts_implementation()
    
    print("\nDebug completed.")


if __name__ == "__main__":
    asyncio.run(main())