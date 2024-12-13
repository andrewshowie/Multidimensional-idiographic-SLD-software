# tests/integration_test.py

import tkinter as tk
from datetime import datetime
import threading
import time
import sys
import os

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from linguistic_analyzer.shared_types import AnalysisMetrics
from linguistic_analyzer.chat_client import ChatClient
from linguistic_analyzer.chat_server import ChatServer
from linguistic_analyzer.integrated_analyzer import IntegratedAnalyzer
from linguistic_analyzer.visualizer import LinguisticVisualizer
from linguistic_analyzer.logger_config import logger

def test_full_integration():
    """Test full system integration"""
    try:
        logger.info("Starting integration test")
        
        # Start server
        server = ChatServer()
        server_thread = threading.Thread(target=server.start)
        server_thread.daemon = True
        server_thread.start()
        
        # Wait for server to start
        time.sleep(2)
        
        # Create root window
        root = tk.Tk()
        root.title("Integration Test")
        
        # Create chat client
        client = ChatClient()
        
        # Create test frame
        test_frame = tk.Frame(root)
        test_frame.pack(pady=10)
        
        def send_test_message():
            test_messages = [
                "This is a simple test message.",
                "The quick brown fox jumps over the lazy dog.",
                "We are conducting an empirical analysis of the methodology.",
                "This contains sum misspelled wurds and erors.",
                "When analyzing complex theoretical frameworks, researchers must consider multiple variables."
            ]
            
            for msg in test_messages:
                client.msg_entry.set(msg)
                client.send_message()
                time.sleep(2)
        
        # Add test button
        test_button = tk.Button(
            test_frame,
            text="Run Test Messages",
            command=send_test_message
        )
        test_button.pack()
        
        # Add status label
        status_label = tk.Label(test_frame, text="Status: Ready")
        status_label.pack(pady=5)
        
        def update_status(message):
            status_label.config(text=f"Status: {message}")
        
        # Add component test buttons
        def test_analyzer():
            try:
                analyzer = IntegratedAnalyzer()
                result = analyzer.analyze_text("Test message", datetime.now())
                update_status("Analyzer Test: Success")
            except Exception as e:
                logger.error(f"Analyzer test failed: {e}")
                update_status("Analyzer Test: Failed")

        def test_visualizer():
            try:
                visualizer = LinguisticVisualizer(root)
                test_data = {
                    'mattr': 0.5,
                    'academic_ratio': 0.3,
                    'burst_length': 5,
                    'error_likelihood': 0.1
                }
                visualizer.update_visualizations(test_data)
                update_status("Visualizer Test: Success")
            except Exception as e:
                logger.error(f"Visualizer test failed: {e}")
                update_status("Visualizer Test: Failed")

        # Add test buttons
        test_frame_buttons = tk.Frame(test_frame)
        test_frame_buttons.pack(pady=5)
        
        tk.Button(
            test_frame_buttons,
            text="Test Analyzer",
            command=test_analyzer
        ).pack(side=tk.LEFT, padx=5)
        
        tk.Button(
            test_frame_buttons,
            text="Test Visualizer",
            command=test_visualizer
        ).pack(side=tk.LEFT, padx=5)
        
        # Start client
        client.start()
        
        root.mainloop()
        
    except Exception as e:
        logger.error(f"Integration test failed: {e}")
        raise

def verify_components():
    """Verify all components are working"""
    try:
        # Test analyzer
        analyzer = IntegratedAnalyzer()
        test_text = "This is a test of the integrated analysis system."
        metrics = analyzer.analyze_text(test_text, datetime.now())
        
        # Verify metrics
        assert isinstance(metrics, AnalysisMetrics), "Invalid metrics type"
        assert metrics.word_count > 0, "Word count failed"
        assert metrics.mattr >= 0, "MATTR failed"
        assert metrics.academic_ratio >= 0, "Academic ratio failed"
        
        # Test visualizer
        root = tk.Tk()
        visualizer = LinguisticVisualizer(root)
        visualizer.update_visualizations({
            'mattr': 0.5,
            'academic_ratio': 0.3,
            'burst_length': 5,
            'error_likelihood': 0.1
        })
        
        # Test server connection
        server = ChatServer()
        server_thread = threading.Thread(target=server.start)
        server_thread.daemon = True
        server_thread.start()
        time.sleep(1)
        
        logger.info("Component verification successful")
        return True
        
    except Exception as e:
        logger.error(f"Component verification failed: {e}")
        return False

def run_system_test():
    """Run complete system test"""
    try:
        # Verify components first
        if not verify_components():
            logger.error("Component verification failed")
            return False
            
        # Run full integration test
        test_full_integration()
        return True
        
    except Exception as e:
        logger.error(f"System test failed: {e}")
        return False

if __name__ == "__main__":
    print("Starting system integration test...")
    success = run_system_test()
    if success:
        print("Integration test completed successfully")
    else:
        print("Integration test failed. Check logs for details.")
        sys.exit(1)