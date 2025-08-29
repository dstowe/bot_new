#!/usr/bin/env python3
"""
View DID Tool - Display contents of did.bin file
"""

import pickle
from pathlib import Path
from datetime import datetime

def view_did():
    """Display DID file contents with detailed information"""
    print("🔍 DID FILE VIEWER")
    print("=" * 40)
    
    did_file = Path("data/did.bin")
    
    if not did_file.exists():
        print("❌ did.bin file not found!")
        print(f"   Expected location: {did_file.absolute()}")
        print("\n💡 The file should be created automatically when you run the system")
        return
    
    try:
        # Read the DID
        with open(did_file, 'rb') as f:
            did = pickle.load(f)
        
        # File information
        stat = did_file.stat()
        modified_time = datetime.fromtimestamp(stat.st_mtime)
        
        print(f"📁 File: {did_file}")
        print(f"📊 Size: {stat.st_size} bytes")
        print(f"📅 Modified: {modified_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # DID information
        print(f"🔑 DID Value: {did}")
        print(f"📏 Length: {len(did)} characters")
        print(f"🏷️  Type: {type(did).__name__}")
        
        # Validation
        if len(did) == 32:
            print("✅ Length is correct (32 characters)")
        else:
            print(f"⚠️  Length is unusual (expected 32, got {len(did)})")
        
        # Check if it looks like a valid UUID hex
        try:
            int(did, 16)  # Try to parse as hex
            print("✅ Format appears to be valid hex")
        except ValueError:
            print("⚠️  Format doesn't appear to be valid hex")
        
        print()
        print("ℹ️  This DID helps prevent image verification during login")
        
    except Exception as e:
        print(f"❌ Error reading did.bin: {e}")
        print("   The file may be corrupted")
        print("\n💡 To regenerate:")
        print("   1. Delete the file: rm data/did.bin")
        print("   2. Run main.py - it will create a new one")
        print("   3. Or get a browser DID: python tests/check_did.py")

if __name__ == "__main__":
    view_did()