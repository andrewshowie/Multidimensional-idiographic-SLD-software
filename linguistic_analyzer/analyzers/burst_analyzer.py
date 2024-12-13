# burst_analyzer.py

import numpy as np
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from collections import defaultdict, deque

from shared_types import WritingBurst
from logger_config import logger
from config import config

class BurstAnalyzer:
    """Analyzes writing bursts and pauses"""
    
    def __init__(self):
        """Initialize burst analysis components"""
        try:
            # Current burst tracking
            self.current_burst = []
            self.bursts: List[WritingBurst] = []
            
            # Timing tracking
            self.last_input_time = datetime.now()
            self.pause_start = datetime.now()
            self.total_pauses = 0
            
            # Burst statistics
            self.burst_lengths = deque(maxlen=config.analysis.max_context_window)
            self.pause_durations = deque(maxlen=config.analysis.max_context_window)
            self.wpm_values = deque(maxlen=config.analysis.max_context_window)
            
            logger.info("BurstAnalyzer initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing BurstAnalyzer: {e}")
            raise

    @property
    def pause_threshold(self) -> float:
        """Get pause threshold from config"""
        return config.analysis.pause_threshold

    @property
    def min_burst_length(self) -> int:
        """Get minimum burst length from config"""
        return config.analysis.min_burst_length

    def analyze_burst(self, text: str, current_time: datetime) -> Dict:
        """Analyze current writing burst"""
        try:
            # Calculate time since last input
            time_since_last = (current_time - self.last_input_time).total_seconds()
            
            # Check for pause
            if time_since_last > self.pause_threshold:
                if self.current_burst:
                    # Process completed burst
                    burst = self._process_burst(current_time)
                    self.bursts.append(burst)
                    self.current_burst = []
                    
                self.total_pauses += 1
                pause_duration = time_since_last
            else:
                pause_duration = 0.0
            
            # Add new text to current burst
            words = text.split()
            self.current_burst.extend(words)
            
            # Calculate metrics
            metrics = self._calculate_burst_metrics(current_time, pause_duration)
            
            # Update timing
            self.last_input_time = current_time
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error analyzing burst: {e}")
            return self._get_default_burst_metrics()

    def _process_burst(self, end_time: datetime) -> WritingBurst:
        """Process and create WritingBurst object"""
        try:
            text = ' '.join(self.current_burst)
            
            # Estimate burst start time based on typing speed and word count
            estimated_duration = len(self.current_burst) * 0.5  # 0.5 seconds per word estimate
            start_time = end_time - timedelta(seconds=estimated_duration)
            
            # Create burst object
            burst = WritingBurst(
                start_time=start_time,
                end_time=end_time,
                text=text,
                word_count=len(self.current_burst),
                mean_word_length=np.mean([len(word) for word in self.current_burst]),
                pause_before=self._calculate_pause_before(start_time),
                pause_after=0.0,  # Will be updated when next burst starts
                wpm=self._calculate_wpm(start_time, end_time, len(self.current_burst))
            )
            
            # Update statistics
            self.burst_lengths.append(burst.word_count)
            self.pause_durations.append(burst.pause_before)
            self.wpm_values.append(burst.wpm)
            
            return burst
            
        except Exception as e:
            logger.error(f"Error processing burst: {e}")
            return self._create_default_burst(end_time)

    def _calculate_burst_metrics(self, current_time: datetime, pause_duration: float) -> Dict:
        """Calculate current burst metrics"""
        try:
            if not self.current_burst:
                return self._get_default_burst_metrics()
            
            burst_duration = (
                current_time - self.last_input_time + timedelta(seconds=0.1)
            ).total_seconds()
            
            wpm = self._calculate_wpm(
                self.last_input_time,
                current_time,
                len(self.current_burst)
            )
            
            metrics = {
                'burst_length': len(self.current_burst),
                'burst_duration': burst_duration,
                'pause_duration': pause_duration,
                'total_pauses': self.total_pauses,
                'wpm': wpm,
                'mean_word_length': np.mean([len(word) for word in self.current_burst])
            }
            
            # Add running statistics
            if self.burst_lengths:
                metrics.update({
                    'mean_burst_length': np.mean(self.burst_lengths),
                    'mean_pause_duration': np.mean(self.pause_durations),
                    'mean_wpm': np.mean(self.wpm_values)
                })
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating burst metrics: {e}")
            return self._get_default_burst_metrics()

    def _calculate_wpm(self, start_time: datetime, end_time: datetime, word_count: int) -> float:
        """Calculate words per minute"""
        try:
            duration_minutes = (end_time - start_time).total_seconds() / 60
            return word_count / duration_minutes if duration_minutes > 0 else 0.0
        except Exception as e:
            logger.error(f"Error calculating WPM: {e}")
            return 0.0

    def _calculate_pause_before(self, start_time: datetime) -> float:
        """Calculate duration of pause before current burst"""
        try:
            if not self.bursts:
                return 0.0
            return (start_time - self.bursts[-1].end_time).total_seconds()
        except Exception as e:
            logger.error(f"Error calculating pause duration: {e}")
            return 0.0

    def get_burst_statistics(self) -> Dict:
        """Calculate overall burst statistics"""
        try:
            if not self.bursts:
                return self._get_default_statistics()
            
            burst_lengths = [burst.word_count for burst in self.bursts]
            pause_durations = [burst.pause_before for burst in self.bursts]
            wpms = [burst.wpm for burst in self.bursts]
            
            return {
                'mean_burst_length': np.mean(burst_lengths),
                'mean_pause_duration': np.mean(pause_durations),
                'mean_wpm': np.mean(wpms),
                'burst_length_std': np.std(burst_lengths),
                'pause_duration_std': np.std(pause_durations),
                'wpm_std': np.std(wpms),
                'total_bursts': len(self.bursts),
                'total_pauses': self.total_pauses,
                'total_words': sum(burst_lengths)
            }
            
        except Exception as e:
            logger.error(f"Error calculating burst statistics: {e}")
            return self._get_default_statistics()

    def _get_default_burst_metrics(self) -> Dict:
        """Return default burst metrics"""
        return {
            'burst_length': 0,
            'burst_duration': 0.0,
            'pause_duration': 0.0,
            'total_pauses': self.total_pauses,
            'wpm': 0.0,
            'mean_word_length': 0.0,
            'mean_burst_length': 0.0,
            'mean_pause_duration': 0.0,
            'mean_wpm': 0.0
        }

    def _get_default_statistics(self) -> Dict:
        """Return default statistics"""
        return {
            'mean_burst_length': 0.0,
            'mean_pause_duration': 0.0,
            'mean_wpm': 0.0,
            'burst_length_std': 0.0,
            'pause_duration_std': 0.0,
            'wpm_std': 0.0,
            'total_bursts': 0,
            'total_pauses': 0,
            'total_words': 0
        }

    def _create_default_burst(self, timestamp: datetime) -> WritingBurst:
        """Create a default burst object"""
        return WritingBurst(
            start_time=timestamp,
            end_time=timestamp,
            text="",
            word_count=0,
            mean_word_length=0.0,
            pause_before=0.0,
            pause_after=0.0,
            wpm=0.0
        )

    def reset_state(self):
        """Reset analyzer state"""
        try:
            self.current_burst = []
            self.bursts.clear()
            self.last_input_time = datetime.now()
            self.pause_start = datetime.now()
            self.total_pauses = 0
            self.burst_lengths.clear()
            self.pause_durations.clear()
            self.wpm_values.clear()
            logger.debug("Burst analyzer state reset")
        except Exception as e:
            logger.error(f"Error resetting analyzer state: {e}")

if __name__ == "__main__":
    # Test the analyzer
    analyzer = BurstAnalyzer()
    test_texts = [
        "This is a test sentence.",
        "Another burst of writing.",
        "After a longer pause, this should be a new burst.",
    ]
    
    for i, text in enumerate(test_texts):
        if i > 0:
            # Simulate pauses between texts
            from time import sleep
            sleep(3 if i == 2 else 1)
        
        results = analyzer.analyze_burst(text, datetime.now())
        print(f"\nAnalyzing: {text}")
        print(f"Burst Length: {results['burst_length']}")
        print(f"WPM: {results['wpm']:.1f}")
        print(f"Pause Duration: {results['pause_duration']:.1f}s")
        
    # Print final statistics
    stats = analyzer.get_burst_statistics()
    print("\nFinal Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")