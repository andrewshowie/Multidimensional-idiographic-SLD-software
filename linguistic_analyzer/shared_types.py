# shared_types.py
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Set
import numpy as np

@dataclass
class AnalysisMetrics:
    """Core metrics used across all analysis components"""
    timestamp: datetime
    
    # Lexical metrics
    mattr: float  # Moving average type-token ratio
    word_count: int
    unique_words: int
    mean_word_length: float
    lexical_density: float  # Ratio of content words to total words
    academic_ratio: float   # Ratio of academic/sophisticated words
    
    # Production metrics
    burst_length: float     # Current burst length in words
    burst_rate: float      # Words per minute in current burst
    pause_duration: float  # Duration of last pause in seconds
    total_pauses: int     # Number of pauses in session
    
    # Processing metrics
    ndd: float            # Normalized dependency distance
    working_memory_load: float
    processing_cost: float
    incomplete_dependencies: int

    # Error detection
    error_likelihood: float = 0.0
    error_locations: List[int] = field(default_factory=list)
    error_types: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert metrics to dictionary for storage/transmission"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'mattr': self.mattr,
            'word_count': self.word_count,
            'unique_words': self.unique_words,
            'mean_word_length': self.mean_word_length,
            'lexical_density': self.lexical_density,
            'academic_ratio': self.academic_ratio,
            'burst_length': self.burst_length,
            'burst_rate': self.burst_rate,
            'pause_duration': self.pause_duration,
            'total_pauses': self.total_pauses,
            'ndd': self.ndd,
            'working_memory_load': self.working_memory_load,
            'processing_cost': self.processing_cost,
            'incomplete_dependencies': self.incomplete_dependencies,
            'error_likelihood': self.error_likelihood,
            'error_locations': self.error_locations,
            'error_types': self.error_types
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'AnalysisMetrics':
        """Create metrics object from dictionary"""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)

@dataclass
class CommitmentPoint:
    """Tracks points of syntactic commitment in text processing"""
    timestamp: datetime
    token: str
    token_index: int
    dependency_type: str
    dependency_distance: int
    processing_cost: float
    incomplete_dependencies: int

    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'token': self.token,
            'token_index': self.token_index,
            'dependency_type': self.dependency_type,
            'dependency_distance': self.dependency_distance,
            'processing_cost': self.processing_cost,
            'incomplete_dependencies': self.incomplete_dependencies
        }

@dataclass
class WritingBurst:
    """Represents a continuous writing burst"""
    start_time: datetime
    end_time: datetime
    text: str
    word_count: int
    mean_word_length: float
    pause_before: float  # Duration of pause before burst in seconds
    pause_after: float   # Duration of pause after burst in seconds
    wpm: float          # Words per minute during burst

    def duration(self) -> float:
        """Calculate burst duration in seconds"""
        return (self.end_time - self.start_time).total_seconds()

@dataclass
class SessionState:
    """Maintains current session state"""
    start_time: datetime = field(default_factory=datetime.now)
    total_words: int = 0
    unique_words: Set[str] = field(default_factory=set)
    current_burst: Optional[WritingBurst] = None
    bursts: List[WritingBurst] = field(default_factory=list)
    commitment_points: List[CommitmentPoint] = field(default_factory=list)
    last_input_time: datetime = field(default_factory=datetime.now)
    pause_start: datetime = field(default_factory=datetime.now)
    
    def update_timing(self, current_time: datetime):
        """Update timing-related state"""
        self.last_input_time = current_time
        
    def add_burst(self, burst: WritingBurst):
        """Add completed burst to history"""
        self.bursts.append(burst)
        
    def add_commitment_point(self, point: CommitmentPoint):
        """Add new commitment point"""
        self.commitment_points.append(point)
        
    def get_session_duration(self) -> float:
        """Get total session duration in seconds"""
        return (datetime.now() - self.start_time).total_seconds()

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

@dataclass
class AnalysisConfig:
    """Analysis-related configuration"""
    pause_threshold: float    # seconds
    burst_threshold: float    # seconds
    min_burst_length: int     # words
    max_context_window: int   # words
    academic_word_path: str
    update_interval: int      # milliseconds

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
            update_interval=1000
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
                    'max_connections': self.network.max_connections
                },
                'analysis': {
                    'pause_threshold': self.analysis.pause_threshold,
                    'burst_threshold': self.analysis.burst_threshold,
                    'min_burst_length': self.analysis.min_burst_length,
                    'max_context_window': self.analysis.max_context_window,
                    'academic_word_path': self.analysis.academic_word_path,
                    'update_interval': self.analysis.update_interval
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