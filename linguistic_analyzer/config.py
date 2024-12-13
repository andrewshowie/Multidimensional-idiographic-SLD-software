# config.py

import os
from pathlib import Path
from typing import Dict
import json
from dataclasses import dataclass

@dataclass
class NetworkConfig:
    """Network-related configuration"""
    server_ip: str
    server_port: int
    buffer_size: int
    max_connections: int
    timeout: float = 60.0  # seconds
    keepalive: float = 120.0  # seconds

@dataclass
class AnalysisConfig:
    """Analysis-related configuration"""
    pause_threshold: float    # seconds
    burst_threshold: float    # seconds
    min_burst_length: int     # words
    max_context_window: int   # words
    academic_word_path: str
    update_interval: int      # milliseconds
    cache_size: int = 1000

@dataclass
class VisualizationConfig:
    """Visualization-related configuration"""
    plot_update_interval: int  # milliseconds
    max_points: int
    window_width: int
    window_height: int
    default_layout: str

class Config:
    """Central configuration management"""
    
    def __init__(self, config_path: str = "config.json"):
        self.config_path = config_path
        self.base_path = Path(__file__).parent
        
        # Set up default configurations
        self.network = NetworkConfig(
            server_ip="127.0.0.1",
            server_port=5555,
            buffer_size=1024,
            max_connections=5
        )
        
        self.analysis = AnalysisConfig(
            pause_threshold=2.0,
            burst_threshold=1.0,
            min_burst_length=3,
            max_context_window=100,
            academic_word_path=str(self.base_path / "data" / "academic_words.txt"),
            update_interval=1000,
            cache_size=1000
        )
        
        self.visualization = VisualizationConfig(
            plot_update_interval=1000,
            max_points=100,
            window_width=1200,
            window_height=800,
            default_layout="3x3"
        )
        
        # Load custom configuration if exists
        self.load_config()
        
        # Ensure required directories exist
        self.setup_directories()

    def load_config(self):
        """Load configuration from JSON file"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config_data = json.load(f)
                
                # Update network config
                if 'network' in config_data:
                    self.network = NetworkConfig(**config_data['network'])
                
                # Update analysis config
                if 'analysis' in config_data:
                    self.analysis = AnalysisConfig(**config_data['analysis'])
                
                # Update visualization config
                if 'visualization' in config_data:
                    self.visualization = VisualizationConfig(**config_data['visualization'])
        
        except Exception as e:
            print(f"Error loading config: {e}")
            print("Using default configuration")

    def save_config(self):
        """Save current configuration to JSON file"""
        try:
            config_data = {
                'network': {
                    'server_ip': self.network.server_ip,
                    'server_port': self.network.server_port,
                    'buffer_size': self.network.buffer_size,
                    'max_connections': self.network.max_connections,
                    'timeout': self.network.timeout,
                    'keepalive': self.network.keepalive
                },
                'analysis': {
                    'pause_threshold': self.analysis.pause_threshold,
                    'burst_threshold': self.analysis.burst_threshold,
                    'min_burst_length': self.analysis.min_burst_length,
                    'max_context_window': self.analysis.max_context_window,
                    'academic_word_path': self.analysis.academic_word_path,
                    'update_interval': self.analysis.update_interval,
                    'cache_size': self.analysis.cache_size
                },
                'visualization': {
                    'plot_update_interval': self.visualization.plot_update_interval,
                    'max_points': self.visualization.max_points,
                    'window_width': self.visualization.window_width,
                    'window_height': self.visualization.window_height,
                    'default_layout': self.visualization.default_layout
                }
            }
            
            with open(self.config_path, 'w') as f:
                json.dump(config_data, f, indent=4)
                
        except Exception as e:
            print(f"Error saving config: {e}")

    def setup_directories(self):
        """Create required directories if they don't exist"""
        directories = [
            'data',
            'logs',
            'output'
        ]
        
        for directory in directories:
            dir_path = self.base_path / directory
            if not dir_path.exists():
                dir_path.mkdir(parents=True)

# Create global config instance
config = Config()