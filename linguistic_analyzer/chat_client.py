# chat_client.py

import socket
import threading
import tkinter as tk
from tkinter import scrolledtext, ttk
from datetime import datetime
import json
from pathlib import Path
import queue
import pandas as pd

from shared_types import AnalysisMetrics, SessionState
from logger_config import logger, log_exceptions
from config import config
from integrated_analyzer import IntegratedAnalyzer
from visualizer import LinguisticVisualizer

class ChatClient:
    """Enhanced chat client with integrated analysis"""
    
    def __init__(self):
        """Initialize chat client with all components"""
        try:
            # Initialize main window
            self.root = tk.Tk()
            self.root.title("Enhanced Chat Application")
            
            # Initialize components
            self.setup_gui()
            self.initialize_data_storage()
            self.setup_analysis()
            self.setup_networking()
            
            # Message queue for thread safety
            self.message_queue = queue.Queue()
            
            # Analysis state
            self.last_analysis: Optional[AnalysisMetrics] = None
            self.session_state = SessionState()
            
            logger.info("Chat client initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing chat client: {e}")
            raise

    @log_exceptions()
    def setup_gui(self):
        """Setup GUI components"""
        try:
            # Create main frames
            self.create_frames()
            
            # Setup individual components
            self.setup_chat_area()
            self.setup_input_area()
            self.setup_analysis_display()
            self.setup_controls()
            
            # Configure grid weights
            self.root.grid_rowconfigure(0, weight=1)
            self.root.grid_columnconfigure(1, weight=1)
            
            logger.debug("GUI setup completed")
            
        except Exception as e:
            logger.error(f"Error setting up GUI: {e}")
            raise

    def create_frames(self):
        """Create main application frames"""
        # Left sidebar for controls and status
        self.control_frame = ttk.Frame(self.root, padding="5")
        self.control_frame.grid(row=0, column=0, sticky="nsew")
        
        # Main content area
        self.main_frame = ttk.Frame(self.root, padding="5")
        self.main_frame.grid(row=0, column=1, sticky="nsew")
        
        # Right sidebar for analysis
        self.analysis_frame = ttk.Frame(self.root, padding="5")
        self.analysis_frame.grid(row=0, column=2, sticky="nsew")

    def setup_chat_area(self):
        """Setup chat display area"""
        # Chat display
        self.chat_area = scrolledtext.ScrolledText(
            self.main_frame,
            wrap=tk.WORD,
            height=20
        )
        self.chat_area.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.chat_area.config(state=tk.DISABLED)
        
        # Message input
        self.input_frame = ttk.Frame(self.main_frame)
        self.input_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        
        self.msg_entry = ttk.Entry(self.input_frame)
        self.msg_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.msg_entry.bind("<Return>", self.send_message)
        
        self.send_button = ttk.Button(
            self.input_frame,
            text="Send",
            command=self.send_message
        )
        self.send_button.pack(side=tk.RIGHT, padx=5)

    def setup_analysis_display(self):
        """Setup analysis metrics display"""
        # Create notebook for different metric categories
        self.analysis_notebook = ttk.Notebook(self.analysis_frame)
        self.analysis_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Lexical metrics tab
        self.lexical_frame = ttk.Frame(self.analysis_notebook)
        self.analysis_notebook.add(self.lexical_frame, text='Lexical')
        
        # Production metrics tab
        self.production_frame = ttk.Frame(self.analysis_notebook)
        self.analysis_notebook.add(self.production_frame, text='Production')
        
        # Processing metrics tab
        self.processing_frame = ttk.Frame(self.analysis_notebook)
        self.analysis_notebook.add(self.processing_frame, text='Processing')
        
        # Setup metric labels
        self.setup_metric_labels()

    def setup_metric_labels(self):
        """Setup labels for displaying metrics"""
        # Lexical metrics
        self.lexical_labels = {
            'mattr': self.create_metric_label(self.lexical_frame, "MATTR:", 0),
            'word_count': self.create_metric_label(self.lexical_frame, "Words:", 1),
            'unique_words': self.create_metric_label(self.lexical_frame, "Unique:", 2),
            'lexical_density': self.create_metric_label(self.lexical_frame, "Density:", 3),
            'academic_ratio': self.create_metric_label(self.lexical_frame, "Academic:", 4)
        }
        
        # Production metrics
        self.production_labels = {
            'burst_length': self.create_metric_label(self.production_frame, "Burst Length:", 0),
            'burst_rate': self.create_metric_label(self.production_frame, "WPM:", 1),
            'pause_duration': self.create_metric_label(self.production_frame, "Pause:", 2),
            'total_pauses': self.create_metric_label(self.production_frame, "Pauses:", 3)
        }
        
        # Processing metrics
        self.processing_labels = {
            'ndd': self.create_metric_label(self.processing_frame, "NDD:", 0),
            'memory_load': self.create_metric_label(self.processing_frame, "Memory:", 1),
            'processing_cost': self.create_metric_label(self.processing_frame, "Cost:", 2),
            'dependencies': self.create_metric_label(self.processing_frame, "Dependencies:", 3)
        }

    def create_metric_label(self, parent, text: str, row: int) -> ttk.Label:
        """Create a labeled metric display"""
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=0, sticky="ew", padx=5, pady=2)
        
        label = ttk.Label(frame, text=text)
        label.pack(side=tk.LEFT)
        
        value = ttk.Label(frame, text="0")
        value.pack(side=tk.RIGHT)
        
        return value

    def setup_controls(self):
        """Setup control panel"""
        # Session controls
        ttk.Label(self.control_frame, text="Session Controls").pack(pady=5)
        
        self.start_button = ttk.Button(
            self.control_frame,
            text="Start Session",
            command=self.start_session
        )
        self.start_button.pack(fill=tk.X, padx=5, pady=2)
        
        self.pause_button = ttk.Button(
            self.control_frame,
            text="Pause Analysis",
            command=self.toggle_analysis
        )
        self.pause_button.pack(fill=tk.X, padx=5, pady=2)
        
        self.save_button = ttk.Button(
            self.control_frame,
            text="Save Session",
            command=self.save_session
        )
        self.save_button.pack(fill=tk.X, padx=5, pady=2)
        
        # Status display
        ttk.Separator(self.control_frame).pack(fill=tk.X, pady=10)
        self.status_label = ttk.Label(self.control_frame, text="Ready")
        self.status_label.pack(pady=5)
        
        # Session timer
        self.timer_label = ttk.Label(self.control_frame, text="00:00:00")
        self.timer_label.pack(pady=5)

    @log_exceptions()
    def initialize_data_storage(self):
        """Initialize data storage components"""
        try:
            # Create output directory if it doesn't exist
            self.output_dir = Path("output")
            self.output_dir.mkdir(exist_ok=True)
            
            # Initialize DataFrames for storing analysis results
            self.analysis_df = pd.DataFrame()
            self.chat_df = pd.DataFrame()
            
            logger.debug("Data storage initialized")
            
        except Exception as e:
            logger.error(f"Error initializing data storage: {e}")
            raise

    @log_exceptions()
    def setup_analysis(self):
        """Setup analysis components"""
        try:
            # Initialize analyzer
            self.analyzer = IntegratedAnalyzer()
            
            # Initialize visualizer
            self.visualizer = LinguisticVisualizer(self.root)
            
            # Analysis state
            self.analysis_active = True
            
            logger.debug("Analysis components initialized")
            
        except Exception as e:
            logger.error(f"Error setting up analysis: {e}")
            raise

    @log_exceptions()
    def setup_networking(self):
        """Setup network connection"""
        try:
            self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client.connect((config.network.server_ip, config.network.server_port))
            
            # Start receive thread
            self.receive_thread = threading.Thread(target=self.receive_messages)
            self.receive_thread.daemon = True
            self.receive_thread.start()
            
            logger.info("Network connection established")
            
        except Exception as e:
            logger.error(f"Error setting up network: {e}")
            raise

    @log_exceptions()
    def send_message(self, event=None):
        """Send message and perform analysis"""
        try:
            message = self.msg_entry.get().strip()
            if not message:
                return
                
            # Clear input
            self.msg_entry.delete(0, tk.END)
            
            # Perform analysis
            if self.analysis_active:
                self.analyze_text(message)
            
            # Send message
            full_msg = f"{self.screen_name}: {message}"
            self.client.send(full_msg.encode('utf-8'))
            
            # Update chat display
            self.update_chat_display(full_msg)
            
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            self.show_error("Error sending message")

    @log_exceptions()
    def analyze_text(self, text: str):
        """Perform text analysis"""
        try:
            # Perform analysis
            timestamp = datetime.now()
            metrics = self.analyzer.analyze_text(text, timestamp)
            
            # Store results
            self.last_analysis = metrics
            self.store_analysis(text, metrics)
            
            # Update displays
            self.update_metric_displays(metrics)
            self.update_visualizations(metrics)
            
        except Exception as e:
            logger.error(f"Error analyzing text: {e}")
            self.show_error("Error performing analysis")

    def store_analysis(self, text: str, metrics: AnalysisMetrics):
        """Store analysis results"""
        try:
            # Convert metrics to dict for storage
            metrics_dict = metrics.to_dict()
            metrics_dict['text'] = text
            
            # Append to DataFrame
            self.analysis_df = self.analysis_df.append(
                metrics_dict,
                ignore_index=True
            )
            
        except Exception as e:
            logger.error(f"Error storing analysis: {e}")

    def update_metric_displays(self, metrics: AnalysisMetrics):
        """Update metric displays with new values"""
        try:
            # Update lexical metrics
            self.lexical_labels['mattr'].config(text=f"{metrics.mattr:.3f}")
            self.lexical_labels['word_count'].config(text=str(metrics.word_count))
            self.lexical_labels['unique_words'].config(text=str(metrics.unique_words))
            self.lexical_labels['lexical_density'].config(text=f"{metrics.lexical_density:.3f}")
            self.lexical_labels['academic_ratio'].config(text=f"{metrics.academic_ratio:.3f}")
            
            # Update production metrics
            self.production_labels['burst_length'].config(text=f"{metrics.burst_length}")
            self.production_labels['burst_rate'].config(text=f"{metrics.burst_rate:.1f}")
            self.production_labels['pause_duration'].config(text=f"{metrics.pause_duration:.1f}s")
            self.production_labels['total_pauses'].config(text=str(metrics.total_pauses))
            
            # Update processing metrics
            self.processing_labels['ndd'].config(text=f"{metrics.ndd:.3f}")
            self.processing_labels['memory_load'].config(text=f"{metrics.working_memory_load:.2f}")
            self.processing_labels['processing_cost'].config(text=f"{metrics.processing_cost:.2f}")
            self.processing_labels['dependencies'].config(text=str(metrics.incomplete_dependencies))
            
        except Exception as e:
            logger.error(f"Error updating metric displays: {e}")

    def update_visualizations(self, metrics: AnalysisMetrics):
        """Update visualizer with new metrics"""
        try:
            self.visualizer.update_visualizations({
                'lexical_complexity': metrics.mattr,
                'burst_lengths': metrics.burst_length,
                'ndd_values': metrics.ndd,
                'working_memory_load': metrics.working_memory_load,
                'processing_cost': metrics.processing_cost,
                'error_likelihood': metrics.error_likelihood
            })
            
        except Exception as e:
            logger.error(f"Error updating visualizations: {e}")

    def receive_messages(self):
        """Receive and process incoming messages"""
        try:
            while True:
                message = self.client.recv(config.network.buffer_size).decode('utf-8')
                if message:
                    self.message_queue.put(message)
                    self.root.after(0, self.process_message)
                    
        except Exception as e:
            logger.error(f"Error receiving messages: {e}")
            self.show_error("Connection lost")

    def process_message(self):
        """Process messages from queue"""
        try:
            while not self.message_queue.empty():
                message = self.message_queue.get()
                self.update_chat_display(message)
                
        except Exception as e:
            logger.error(f"Error processing message: {e}")

def update_chat_display(self, message: str):
    """Update chat display with new message"""
    try:
        self.chat_area.config(state=tk.NORMAL)
        self.chat_area.insert(tk.END, message + "\n")
        self.chat_area.see(tk.END)
        self.chat_area.config(state=tk.DISABLED)
        
        # Store message
        self.chat_df = self.chat_df.append({
            'timestamp': datetime.now(),
            'message': message
        }, ignore_index=True)
    except Exception as e:
        logger.error(f"Error updating chat display: {e}")

    def start_session(self):
        """Start new chat session"""
        try:
            # Get screen name
            self.screen_name = self.prompt_screen_name()
            if not self.screen_name:
                return
            
            # Reset session state
            self.session_state = SessionState()
            self.analysis_active = True
            
            # Clear storage
            self.analysis_df = pd.DataFrame()
            self.chat_df = pd.DataFrame()
            
            # Update status
            self.status_label.config(text="Session Active")
            
            # Start session timer
            self.session_start = datetime.now()
            self.update_timer()
            
            # Enable controls
            self.pause_button.config(state=tk.NORMAL)
            self.save_button.config(state=tk.NORMAL)
            self.msg_entry.config(state=tk.NORMAL)
            self.send_button.config(state=tk.NORMAL)
            
            logger.info(f"Session started for user: {self.screen_name}")
            
        except Exception as e:
            logger.error(f"Error starting session: {e}")
            self.show_error("Error starting session")

    def prompt_screen_name(self) -> str:
        """Prompt user for screen name"""
        try:
            dialog = tk.Toplevel(self.root)
            dialog.title("Enter Screen Name")
            dialog.transient(self.root)
            dialog.grab_set()
            
            screen_name = tk.StringVar()
            
            ttk.Label(dialog, text="Enter your screen name:").pack(padx=10, pady=5)
            entry = ttk.Entry(dialog, textvariable=screen_name)
            entry.pack(padx=10, pady=5)
            
            def submit():
                dialog.destroy()
            
            ttk.Button(dialog, text="OK", command=submit).pack(pady=10)
            
            entry.bind("<Return>", lambda e: submit())
            entry.focus_set()
            
            dialog.wait_window()
            return screen_name.get().strip()
            
        except Exception as e:
            logger.error(f"Error prompting for screen name: {e}")
            return ""

    def toggle_analysis(self):
        """Toggle analysis on/off"""
        try:
            self.analysis_active = not self.analysis_active
            status = "Active" if self.analysis_active else "Paused"
            self.pause_button.config(text=f"Analysis: {status}")
            self.status_label.config(text=f"Analysis {status}")
            
            logger.info(f"Analysis toggled: {status}")
            
        except Exception as e:
            logger.error(f"Error toggling analysis: {e}")
            self.show_error("Error toggling analysis")

    def update_timer(self):
        """Update session timer display"""
        try:
            if hasattr(self, 'session_start'):
                duration = datetime.now() - self.session_start
                hours = duration.seconds // 3600
                minutes = (duration.seconds % 3600) // 60
                seconds = duration.seconds % 60
                self.timer_label.config(text=f"{hours:02d}:{minutes:02d}:{seconds:02d}")
                self.root.after(1000, self.update_timer)
                
        except Exception as e:
            logger.error(f"Error updating timer: {e}")

    @log_exceptions()
    def save_session(self):
        """Save session data"""
        try:
            # Create timestamp for filenames
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save analysis results
            if not self.analysis_df.empty:
                analysis_file = self.output_dir / f"analysis_{timestamp}.xlsx"
                self.analysis_df.to_excel(analysis_file, index=False)
            
            # Save chat log
            if not self.chat_df.empty:
                chat_file = self.output_dir / f"chat_{timestamp}.xlsx"
                self.chat_df.to_excel(chat_file, index=False)
            
            # Get session statistics
            stats = self.analyzer.get_session_statistics()
            stats_file = self.output_dir / f"stats_{timestamp}.json"
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=4)
            
            # Save visualizations
            self.visualizer.save_plots(self.output_dir / f"plots_{timestamp}.png")
            
            self.show_status("Session saved successfully")
            logger.info(f"Session data saved with timestamp: {timestamp}")
            
        except Exception as e:
            logger.error(f"Error saving session: {e}")
            self.show_error("Error saving session data")

    def show_status(self, message: str):
        """Show status message"""
        self.status_label.config(text=message)
        # Clear after 5 seconds
        self.root.after(5000, lambda: self.status_label.config(text="Ready"))

    def show_error(self, message: str):
        """Show error message"""
        try:
            dialog = tk.Toplevel(self.root)
            dialog.title("Error")
            dialog.transient(self.root)
            dialog.grab_set()
            
            ttk.Label(dialog, text=message, foreground="red").pack(padx=20, pady=10)
            ttk.Button(dialog, text="OK", command=dialog.destroy).pack(pady=10)
            
            dialog.wait_window()
            
        except Exception as e:
            logger.error(f"Error showing error dialog: {e}")

    def cleanup(self):
        """Clean up resources before exit"""
        try:
            # Save any unsaved data
            if hasattr(self, 'session_start'):
                self.save_session()
            
            # Close network connection
            if hasattr(self, 'client'):
                self.client.close()
            
            logger.info("Chat client cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def start(self):
        """Start the chat client"""
        try:
            # Set up exit handler
            self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
            
            # Start main loop
            self.root.mainloop()
            
        except Exception as e:
            logger.error(f"Error starting chat client: {e}")
            raise

    def on_closing(self):
        """Handle window closing"""
        try:
            self.cleanup()
            self.root.destroy()
            
        except Exception as e:
            logger.error(f"Error handling window closing: {e}")
            self.root.destroy()

def main():
    """Main entry point"""
    try:
        client = ChatClient()
        client.start()
        
    except Exception as e:
        logger.error(f"Fatal error in chat client: {e}")
        raise

if __name__ == "__main__":
    main()