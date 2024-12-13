# visualizer.py

import tkinter as tk
from tkinter import ttk, messagebox
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.dates as mdates
import numpy as np
from pathlib import Path
import seaborn as sns
from collections import deque
import pandas as pd
import json

from logger_config import logger, log_exceptions
from config import config
from shared_types import AnalysisMetrics

class LinguisticVisualizer:
    """Enhanced visualization for linguistic analysis"""
    
    LAYOUTS = {
        '2x2': (2, 2),
        '2x3': (2, 3),
        '3x2': (3, 2),
        '3x3': (3, 3)
    }
    
    PLOT_CONFIGS = {
        'lexical': {
            'title': 'Lexical Complexity',
            'xlabel': 'Time',
            'ylabel': 'Score',
            'metrics': ['MATTR', 'Academic Ratio', 'Lexical Density'],
            'colors': ['#1f77b4', '#2ca02c', '#ff7f0e']
        },
        'production': {
            'title': 'Production Metrics',
            'xlabel': 'Time',
            'ylabel': 'Words/Burst',
            'ylabel2': 'Seconds',
            'metrics': ['Burst Length', 'WPM', 'Pause Duration'],
            'colors': ['#1f77b4', '#2ca02c', '#d62728']
        },
        'processing': {
            'title': 'Processing Metrics',
            'xlabel': 'Time',
            'ylabel': 'Score',
            'metrics': ['NDD', 'Memory Load', 'Processing Cost'],
            'colors': ['#1f77b4', '#9467bd', '#e377c2']
        },
        'errors': {
            'title': 'Error Analysis',
            'xlabel': 'Time',
            'ylabel': 'Likelihood',
            'ylabel2': 'Count',
            'metrics': ['Error Likelihood', 'Error Count'],
            'colors': ['#1f77b4', '#ff7f0e']
        },
        'academic': {
            'title': 'Academic Language',
            'xlabel': 'Time',
            'ylabel': 'Ratio',
            'metrics': ['Academic Ratio'],
            'colors': ['#8c564b']
        },
        'summary': {
            'title': 'Metric Correlations',
            'xlabel': 'Metrics',
            'ylabel': 'Metrics',
            'cmap': 'coolwarm'
        }
    }
    
    def __init__(self, root: tk.Tk):
        """Initialize visualization system"""
        try:
            self.root = root
            
            # Set plot style
            plt.style.use('seaborn-v0_8-darkgrid')
            
            # Data buffers
            self.buffer_size = config.visualization.max_points
            self.initialize_buffers()
            
            # Create visualization window
            self.setup_window()
            
            # Initialize tooltips
            self.tooltip_text = tk.StringVar()
            self.setup_tooltips()
            
            logger.info("LinguisticVisualizer initialized")
            
        except Exception as e:
            logger.error(f"Error initializing visualizer: {e}")
            raise

    def initialize_buffers(self):
        """Initialize data buffers with proper typing"""
        self.data_buffers = {
            'timestamps': deque(maxlen=self.buffer_size),
            'lexical': {
                'mattr': deque(maxlen=self.buffer_size),
                'academic_ratio': deque(maxlen=self.buffer_size),
                'lexical_density': deque(maxlen=self.buffer_size)
            },
            'production': {
                'burst_length': deque(maxlen=self.buffer_size),
                'burst_rate': deque(maxlen=self.buffer_size),
                'pause_duration': deque(maxlen=self.buffer_size)
            },
            'processing': {
                'ndd': deque(maxlen=self.buffer_size),
                'memory_load': deque(maxlen=self.buffer_size),
                'processing_cost': deque(maxlen=self.buffer_size)
            },
            'errors': {
                'likelihood': deque(maxlen=self.buffer_size),
                'count': deque(maxlen=self.buffer_size)
            }
        }

    def setup_window(self):
        """Setup visualization window with enhanced controls"""
        # Create window
        self.viz_window = tk.Toplevel(self.root)
        self.viz_window.title("Linguistic Analysis Visualization")
        self.viz_window.geometry(f"{config.visualization.window_width}x{config.visualization.window_height}")
        
        # Setup menu
        self.setup_menu()
        
        # Create control panel
        self.setup_controls()
        
        # Create plot area
        self.setup_plots()
        
        # Create status bar with progress
        self.setup_status_bar()
        
        # Bind keyboard shortcuts
        self.setup_shortcuts()

    def setup_menu(self):
        """Setup menu bar"""
        menubar = tk.Menu(self.viz_window)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Save Plots", command=self.save_plots)
        file_menu.add_command(label="Export Data", command=self.export_data)
        file_menu.add_separator()
        file_menu.add_command(label="Clear Data", command=self.clear_data)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.viz_window.destroy)
        menubar.add_cascade(label="File", menu=file_menu)
        
        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        view_menu.add_checkbutton(label="Show Grid", 
                                variable=tk.BooleanVar(value=True),
                                command=self.toggle_grid)
        view_menu.add_checkbutton(label="Show Tooltips",
                                variable=tk.BooleanVar(value=True),
                                command=self.toggle_tooltips)
        menubar.add_cascade(label="View", menu=view_menu)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About", command=self.show_about)
        help_menu.add_command(label="Metrics Help", command=self.show_metrics_help)
        menubar.add_cascade(label="Help", menu=help_menu)
        
        self.viz_window.config(menu=menubar)

    def setup_controls(self):
        """Setup enhanced control panel"""
        control_frame = ttk.Frame(self.viz_window)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        # Left controls
        left_frame = ttk.Frame(control_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.X)
        
        # Pause button
        self.paused = tk.BooleanVar(value=False)
        self.pause_button = ttk.Checkbutton(
            left_frame,
            text="Pause Updates",
            variable=self.paused,
            command=self.toggle_pause
        )
        self.pause_button.pack(side=tk.LEFT, padx=5)
        
        # Layout selector with labels
        ttk.Label(left_frame, text="Layout:").pack(side=tk.LEFT, padx=(10, 2))
        self.layout_var = tk.StringVar(value=config.visualization.default_layout)
        self.layout_selector = ttk.Combobox(
            left_frame,
            textvariable=self.layout_var,
            values=list(self.LAYOUTS.keys()),
            width=10,
            state='readonly'
        )
        self.layout_selector.pack(side=tk.LEFT, padx=5)
        self.layout_selector.bind('<<ComboboxSelected>>', self.change_layout)
        
        # Right controls
        right_frame = ttk.Frame(control_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.X)
        
        # Save and Clear buttons
        self.save_button = ttk.Button(
            right_frame,
            text="Save Plots",
            command=self.save_plots
        )
        self.save_button.pack(side=tk.RIGHT, padx=5)
        
        self.clear_button = ttk.Button(
            right_frame,
            text="Clear Data",
            command=self.clear_data
        )
        self.clear_button.pack(side=tk.RIGHT, padx=5)
        
        # Export button
        self.export_button = ttk.Button(
            right_frame,
            text="Export Data",
            command=self.export_data
        )
        self.export_button.pack(side=tk.RIGHT, padx=5)

    def setup_tooltips(self):
        """Setup tooltip display"""
        self.tooltip_label = ttk.Label(
            self.viz_window,
            textvariable=self.tooltip_text,
            background='#FFFFCC',
            relief='solid',
            borderwidth=1
        )
        
        def show_tooltip(event):
            x, y = event.x_root, event.y_root
            self.tooltip_label.place(x=x + 15, y=y + 10)
            
        def hide_tooltip(event):
            self.tooltip_label.place_forget()
            
        self.canvas.get_tk_widget().bind('<Motion>', self.update_tooltip)
        self.canvas.get_tk_widget().bind('<Enter>', show_tooltip)
        self.canvas.get_tk_widget().bind('<Leave>', hide_tooltip)

    def update_tooltip(self, event):
        """Update tooltip text based on mouse position"""
        try:
            # Convert screen coordinates to data coordinates
            ax = self.get_subplot_at_position(event.x, event.y)
            if ax is None:
                return
                
            # Get data point closest to mouse position
            x, y = ax.transData.inverted().transform((event.x, event.y))
            nearest_idx = int(x + 0.5)
            
            if 0 <= nearest_idx < len(self.data_buffers['timestamps']):
                timestamp = self.data_buffers['timestamps'][nearest_idx]
                tooltip_text = f"Time: {timestamp.strftime('%H:%M:%S')}\n"
                
                # Add relevant metrics based on subplot
                subplot_name = self.get_subplot_name(ax)
                if subplot_name in self.data_buffers:
                    for metric, values in self.data_buffers[subplot_name].items():
                        if len(values) > nearest_idx:
                            tooltip_text += f"{metric}: {values[nearest_idx]:.3f}\n"
                
                self.tooltip_text.set(tooltip_text)
                
        except Exception as e:
            logger.error(f"Error updating tooltip: {e}")

    def get_subplot_at_position(self, x: int, y: int) -> Optional[plt.Axes]:
        """Get subplot at given position"""
        for ax in self.axes.values():
            if ax.contains_point((x, y)):
                return ax
        return None

    def get_subplot_name(self, ax: plt.Axes) -> str:
        """Get name of subplot"""
        for name, subplot in self.axes.items():
            if subplot == ax:
                return name
        return ""

    def setup_shortcuts(self):
        """Setup keyboard shortcuts"""
        self.viz_window.bind('<Control-s>', lambda e: self.save_plots())
        self.viz_window.bind('<Control-e>', lambda e: self.export_data())
        self.viz_window.bind('<Control-p>', lambda e: self.toggle_pause())
        self.viz_window.bind('<Control-l>', lambda e: self.layout_selector.focus())
        self.viz_window.bind('<F1>', lambda e: self.show_metrics_help())

    def create_subplots(self):
        """Create enhanced subplot layout"""
        self.fig.clear()
        layout = self.layout_var.get()
        rows, cols = self.LAYOUTS[layout]
        
        # Create grid of subplots
        self.axes = {}
        plot_config = self.get_plot_configuration(layout)
        
        for i, (name, config) in enumerate(plot_config.items()):
            if i < rows * cols:
                self.axes[name] = self.fig.add_subplot(rows, cols, i + 1)
                
                # Apply configuration
                ax = self.axes[name]
                ax.set_title(config['title'])
                ax.set_xlabel(config['xlabel'])
                ax.set_ylabel(config.get('ylabel', ''))
                
                if 'ylabel2' in config:
                    ax2 = ax.twinx()
                    ax2.set_ylabel(config['ylabel2'])
                    
                # Set time formatter for x-axis if showing time series
                if config['xlabel'] == 'Time':
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
                    ax.tick_params(axis='x', rotation=45)
        
        self.setup_plot_styles()
        self.fig.tight_layout()

    def get_plot_configuration(self, layout: str) -> Dict:
        """Get plot configuration based on layout"""
        configs = {}
        if layout == '2x2':
            configs = {k: self.PLOT_CONFIGS[k] for k in ['lexical', 'production', 'processing', 'errors']}
        elif layout == '2x3':
            configs = {k: self.PLOT_CONFIGS[k] for k in ['lexical', 'production', 'processing', 'errors', 'academic', 'summary']}
        elif layout == '3x2':
            configs = {k: self.PLOT_CONFIGS[k] for k in ['lexical', 'production', 'processing', 'errors', 'academic', 'summary']}
        elif layout == '3x3':
            configs = self.PLOT_CONFIGS.copy()  # Use all plots
        return configs

def export_data(self):
    """Export visualization data"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Convert data to DataFrame
        data = {}
        for category, buffers in self.data_buffers.items():
            if isinstance(buffers, dict):
                for metric, values in buffers.items():
                    data[f"{category}_{metric}"] = list(values)
            else:
                data[category] = list(buffers)
        
        df = pd.DataFrame(data)
        
        # Save to Excel
        excel_path = Path(f"visualization_data_{timestamp}.xlsx")
        df.to_excel(excel_path, index=False)
        
        # Save to JSON
        json_path = Path(f"visualization_data_{timestamp}.json")
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=4, default=str)  # Handle datetime serialization
        
        self.status_var.set(f"Data exported to {excel_path} and {json_path}")
        
    except Exception as e:
        logger.error(f"Error exporting data: {e}")
        self.status_var.set("Error exporting data")
        messagebox.showerror("Export Error", str(e))

    def show_about(self):
        """Show about dialog"""
        messagebox.showinfo(
            "About",
            "Linguistic Analysis Visualizer\n\n"
            "Version 1.0\n"
            "A real-time visualization tool for linguistic analysis metrics."
        )

    def show_metrics_help(self):
        """Show metrics help dialog"""
        help_text = """
Metrics Description:

Lexical Metrics:
- MATTR: Moving Average Type-Token Ratio
- Academic Ratio: Proportion of academic vocabulary
- Lexical Density: Ratio of content words to total words

Production Metrics:
- Burst Length: Number of words in current burst
- WPM: Words per minute
- Pause Duration: Duration of pauses in seconds

Processing Metrics:
- NDD: Normalized Dependency Distance
- Memory Load: Working memory load estimate
- Processing Cost: Cognitive processing cost estimate

Error Analysis:
- Error Likelihood: Probability of errors
- Error Count: Number of detected errors
        """
        messagebox.showinfo("Metrics Help", help_text)

    def toggle_grid(self):
        """Toggle grid visibility"""
        for ax in self.axes.values():
            ax.grid(not ax.get_grid())
        self.canvas.draw()

    def toggle_tooltips(self):
        """Toggle tooltip visibility"""
        if hasattr(self, 'tooltip_label'):
            if self.tooltip_label.winfo_viewable():
                self.tooltip_label.place_forget()
            self.tooltip_enabled = not getattr(self, 'tooltip_enabled', True)

    def setup_status_bar(self):
        """Setup enhanced status bar"""
        status_frame = ttk.Frame(self.viz_window)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.status_var = tk.StringVar(value="Ready")
        self.status_bar = ttk.Label(
            status_frame,
            textvariable=self.status_var,
            padding=(5, 2)
        )
        self.status_bar.pack(side=tk.LEFT)
        
        # Add progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            status_frame,
            length=200,
            mode='determinate',
            variable=self.progress_var
        )
        self.progress_bar.pack(side=tk.RIGHT, padx=5)

if __name__ == "__main__":
    # Test visualization
    root = tk.Tk()
    root.title("Visualization Test")
    viz = LinguisticVisualizer(root)
    
    def test_update():
        """Generate test data for visualization"""
        metrics = {
            'mattr': np.random.random(),
            'academic_ratio': np.random.random(),
            'lexical_density': np.random.random(),
            'burst_length': np.random.randint(1, 10),
            'burst_rate': np.random.randint(10, 60),
            'pause_duration': np.random.random() * 5,
            'ndd': np.random.random(),
            'working_memory_load': np.random.random() * 10,
            'processing_cost': np.random.random() * 15,
            'error_likelihood': np.random.random(),
            'error_locations': [i for i in range(np.random.randint(0, 5))]
        }
        
        viz.update_visualizations(metrics)
        if not viz.paused.get():
            root.after(1000, test_update)
    
    # Start test updates
    test_update()
    root.mainloop()