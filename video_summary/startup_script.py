#!/usr/bin/env python3
"""
Startup script for Video Summary System
This script helps you start all required services in the correct order.
"""

import subprocess
import sys
import time
import os
import signal
import threading
from pathlib import Path

class VideoSummaryStarter:
    def __init__(self):
        self.processes = []
        self.running = True
    
    def check_dependencies(self):
        """Check if required files exist"""
        required_files = [
            'ocr_server.py',
            'utils_video_summary.py', 
            'websocket_backend.py',
            'best.pt',  # YOLO model
            'license_plate_detector.pt'  # License plate detector
        ]
        
        missing_files = []
        for file in required_files:
            if not Path(file).exists():
                missing_files.append(file)
        
        if missing_files:
            print("❌ Missing required files:")
            for file in missing_files:
                print(f"   - {file}")
            print("\nPlease ensure all files are in the current directory.")
            return False
        
        print("✅ All required files found!")
        return True
    
    def start_ocr_server(self):
        """Start the OCR server"""
        print("🚀 Starting OCR server...")
        try:
            process = subprocess.Popen([
                sys.executable, 'ocr_server.py'
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.processes.append(('OCR Server', process))
            
            # Wait a moment for server to start
            time.sleep(3)
            
            if process.poll() is None:
                print("✅ OCR server started successfully!")
                return True
            else:
                print("❌ OCR server failed to start")
                return False
        except Exception as e:
            print(f"❌ Error starting OCR server: {e}")
            return False
    
    def start_websocket_server(self):
        """Start the WebSocket backend server"""
        print("🚀 Starting WebSocket backend server...")
        try:
            process = subprocess.Popen([
                sys.executable, 'websocket_backend.py'
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.processes.append(('WebSocket Server', process))
            
            # Wait a moment for server to start
            time.sleep(5)
            
            if process.poll() is None:
                print("✅ WebSocket server started successfully!")
                return True
            else:
                print("❌ WebSocket server failed to start")
                return False
        except Exception as e:
            print(f"❌ Error starting WebSocket server: {e}")
            return False
    
    def monitor_processes(self):
        """Monitor running processes"""
        while self.running:
            for name, process in self.processes:
                if process.poll() is not None:
                    print(f"⚠️ {name} has stopped unexpectedly")
                    return
            time.sleep(5)
    
    def cleanup(self):
        """Clean up all processes"""
        print("\n🛑 Shutting down servers...")
        self.running = False
        
        for name, process in self.processes:
            try:
                print(f"   Stopping {name}...")
                process.terminate()
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print(f"   Force killing {name}...")
                process.kill()
            except Exception as e:
                print(f"   Error stopping {name}: {e}")
        
        print("✅ All servers stopped!")
    
    def signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully"""
        self.cleanup()
        sys.exit(0)
    
    def run(self):
        """Main execution function"""
        # Set up signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        
        print("=" * 50)
        print("🎥 Video Summary System Startup")
        print("=" * 50)
        
        # Check dependencies
        if not self.check_dependencies():
            return False
        
        # Start OCR server
        if not self.start_ocr_server():
            return False
        
        # Start WebSocket server
        if not self.start_websocket_server():
            self.cleanup()
            return False
        
        print("\n" + "=" * 50)
        print("🎉 All servers are running!")
        print("=" * 50)
        print("📱 Open 'video_summary_frontend.html' in your browser")
        print("🌐 Frontend URL: file:///path/to/video_summary_frontend.html")
        print("🔗 WebSocket Server: ws://localhost:8765")
        print("🔗 OCR Server: localhost:6000")
        print("\nPress Ctrl+C to stop all servers")
        print("=" * 50)
        
        # Monitor processes
        try:
            monitor_thread = threading.Thread(target=self.monitor_processes, daemon=True)
            monitor_thread.start()
            
            # Keep main thread alive
            while self.running:
                time.sleep(1)
                
        except KeyboardInterrupt:
            pass
        finally:
            self.cleanup()
        
        return True

def main():
    """Main entry point"""
    starter = VideoSummaryStarter()
    success = starter.run()
    
    if not success:
        print("\n❌ Failed to start Video Summary System")
        sys.exit(1)

if __name__ == "__main__":
    main()