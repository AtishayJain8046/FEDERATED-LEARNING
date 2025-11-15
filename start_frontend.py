#!/usr/bin/env python
"""
Quick start script for the Federated Learning Privacy Demo frontend.
"""

import sys
import subprocess
from pathlib import Path

def check_dependencies():
    """Check if required packages are installed."""
    required = ['flask', 'flask_cors', 'torch', 'numpy', 'sklearn']
    missing = []
    
    for package in required:
        try:
            if package == 'flask_cors':
                __import__('flask_cors')
            elif package == 'sklearn':
                __import__('sklearn')
            else:
                __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print("❌ Missing dependencies:")
        for pkg in missing:
            print(f"   - {pkg}")
        print("\n📦 Install with: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies installed!")
    return True

def main():
    """Start the Flask application."""
    print("=" * 60)
    print("🔒 Federated Learning Privacy Demo")
    print("=" * 60)
    print()
    
    if not check_dependencies():
        sys.exit(1)
    
    print("\n🚀 Starting web server...")
    print("📱 Open http://localhost:5000 in your browser")
    print("⏹️  Press Ctrl+C to stop\n")
    print("=" * 60)
    print()
    
    # Import and run the app
    try:
        from app import app
        app.run(debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped. Goodbye!")
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()

