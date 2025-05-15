#!/usr/bin/env python3
"""
Test script for the neurok-tts plugin.
This script demonstrates basic usage of the plugin and verifies its functionality.
"""

import os
import asyncio
import tempfile
from pathlib import Path

# Import the neurok-tts plugin
from livekit.plugins.neurok_tts import TTS


async def test_tts_basic():
    """Test basic TTS functionality with fallback mode."""
    print("Testing basic TTS functionality (fallback mode)...")
    
    # Initialize the TTS engine with default settings (fallback mode)
    tts = TTS()
    
    # Text to synthesize
    text = "Tere! See on lihtne test neurok-tts plugina jaoks."
    
    try:
        print(f"Initializing TTS engine with default settings")
        print(f"TTS engine initialized successfully")
        print(f"Sample rate: {tts.sample_rate}")
        print(f"Number of channels: {tts.num_channels}")
        print(f"Streaming capability: {tts.capabilities.streaming}")
        
        return True
    
    except Exception as e:
        print(f"Error during TTS initialization: {e}")
        return False


async def test_tts_with_options():
    """Test TTS with different voice options."""
    print("\nTesting TTS with different voice options...")
    
    try:
        # Initialize the TTS engine with specific settings
        tts = TTS(speaker="albert", speed=1.2)
        
        print(f"TTS engine initialized with speaker 'albert' and speed 1.2")
        
        # Update options
        tts.update_options(speaker="mari", speed=1.0)
        print(f"TTS options updated to speaker 'mari' and speed 1.0")
        
        return True
    
    except Exception as e:
        print(f"Error during TTS initialization with options: {e}")
        return False


async def list_available_voices():
    """List all available voices."""
    print("\nListing available voices...")
    
    try:
        voices = await TTS.list_voices()
        print(f"Found {len(voices)} voices:")
        
        for voice in voices:
            print(f"  - {voice['Name']} ({voice['Gender']}, {voice['Locale']})")
        
        return voices
    
    except Exception as e:
        print(f"Error listing voices: {e}")
        return None


async def main():
    """Run all tests."""
    print("=== Testing Neurok TTS Plugin ===\n")
    
    # Test basic TTS functionality
    basic_success = await test_tts_basic()
    
    # Test TTS with different options
    options_success = await test_tts_with_options()
    
    # List available voices
    voices = await list_available_voices()
    
    # Summary
    print("\n=== Test Summary ===")
    print(f"Basic TTS test: {'Success' if basic_success else 'Failed'}")
    print(f"Options TTS test: {'Success' if options_success else 'Failed'}")
    print(f"Voice listing: {'Success' if voices else 'Failed'}")
    
    print("\nTests completed.")


if __name__ == "__main__":
    asyncio.run(main())