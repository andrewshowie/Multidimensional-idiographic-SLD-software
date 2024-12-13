# integrated_analyzer.py

from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime
import numpy as np
import re
from dataclasses import asdict
from pathlib import Path

# Component imports
from analyzers.lexical_analyzer import LexicalAnalyzer
from analyzers.burst_analyzer import BurstAnalyzer
from analyzers.dependency_analyzer import DependencyAnalyzer
from analyzers.error_classifier import ErrorClassifier
from academic_vocabulary import AcademicVocabulary

# Shared types and utilities
from shared_types import (
    AnalysisMetrics,
    WritingBurst,
    CommitmentPoint,
    SessionState,
    AnalysisError
)
from logger_config import logger, log_exceptions
from config import config

class IntegratedAnalyzer:
    """Coordinates all analysis components and provides unified interface"""
    
    # Analysis constants
    MAX_TEXT_LENGTH = 5000
    MIN_TEXT_LENGTH = 1
    VALID_TEXT_PATTERN = re.compile(r'^[\w\s\p{P}]*$', re.UNICODE)
    
    def __init__(self):
        """Initialize all analysis components"""
        try:
            # Initialize component analyzers
            self.lexical_analyzer = LexicalAnalyzer()
            self.burst_analyzer = BurstAnalyzer()
            self.dependency_analyzer = DependencyAnalyzer()
            self.error_classifier = ErrorClassifier()
            self.academic_vocab = AcademicVocabulary()
            
            # Initialize session state
            self.session_state = SessionState()
            
            # Analysis cache for optimization
            self._cache = {}
            self._cache_size = config.analysis.cache_size
            
            # Initialize error tracking
            self.last_error: Optional[AnalysisError] = None
            
            # Initialize data storage
            self.output_dir = Path('output')
            self.output_dir.mkdir(exist_ok=True)
            
            logger.info("IntegratedAnalyzer initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing IntegratedAnalyzer: {e}")
            raise

    def validate_input(self, text: str) -> tuple[bool, Optional[str]]:
        """Validate input text before analysis"""
        try:
            # Check for empty or whitespace text
            if not text or not text.strip():
                return False, "Empty or whitespace-only text"
                
            # Check text length
            if len(text) > self.MAX_TEXT_LENGTH:
                return False, f"Text exceeds maximum length of {self.MAX_TEXT_LENGTH}"
                
            if len(text) < self.MIN_TEXT_LENGTH:
                return False, f"Text below minimum length of {self.MIN_TEXT_LENGTH}"
                
            # Check for valid characters
            if not self.VALID_TEXT_PATTERN.match(text):
                return False, "Text contains invalid characters"
                
            return True, None
            
        except Exception as e:
            logger.error(f"Error validating input: {e}")
            return False, f"Validation error: {str(e)}"

    @log_exceptions()
    def analyze_text(self, text: str, timestamp: datetime) -> AnalysisMetrics:
        """Perform comprehensive text analysis"""
        try:
            # Validate input
            is_valid, error_message = self.validate_input(text)
            if not is_valid:
                logger.warning(f"Invalid input: {error_message}")
                self.last_error = AnalysisError(
                    error_type="InputValidation",
                    message=error_message,
                    timestamp=timestamp
                )
                return self._get_default_metrics()
            
            # Check cache
            cache_key = (text, timestamp.timestamp())
            if cache_key in self._cache:
                logger.debug("Retrieved analysis from cache")
                return self._cache[cache_key]
            
            # Perform component analyses with error handling
            analysis_results = self._perform_component_analyses(text, timestamp)
            
            # Create combined metrics
            metrics = self._create_combined_metrics(analysis_results, timestamp)
            
            # Validate metrics
            if not self._validate_metrics(metrics):
                logger.warning("Invalid metrics generated")
                return self._get_default_metrics()
            
            # Update session state
            self._update_session_state(text, timestamp, metrics, analysis_results)
            
            # Update cache
            self._update_cache(cache_key, metrics)
            
            logger.debug(f"Completed integrated analysis for text segment: {text[:50]}...")
            return metrics
            
        except Exception as e:
            logger.error(f"Error in integrated analysis: {e}")
            self.last_error = AnalysisError(
                error_type="AnalysisFailure",
                message=str(e),
                timestamp=timestamp
            )
            return self._get_default_metrics()

    def _perform_component_analyses(self, text: str, timestamp: datetime) -> Dict:
        """Perform individual component analyses with error handling"""
        results = {
            'lexical': None,
            'burst': None,
            'dependency': None,
            'error': None,
            'academic': None
        }
        
        try:
            # Lexical analysis
            results['lexical'] = self.lexical_analyzer.analyze_text(text, timestamp)
        except Exception as e:
            logger.error(f"Lexical analysis failed: {e}")
            results['lexical'] = self._get_default_lexical_metrics()

        try:
            # Burst analysis
            results['burst'] = self.burst_analyzer.analyze_burst(text, timestamp)
        except Exception as e:
            logger.error(f"Burst analysis failed: {e}")
            results['burst'] = self._get_default_burst_metrics()

        try:
            # Dependency analysis
            results['dependency'] = self.dependency_analyzer.analyze_dependencies(text, timestamp)
        except Exception as e:
            logger.error(f"Dependency analysis failed: {e}")
            results['dependency'] = self._get_default_dependency_metrics()

        try:
            # Error analysis
            results['error'] = self.error_classifier.classify_errors(text)
        except Exception as e:
            logger.error(f"Error analysis failed: {e}")
            results['error'] = self._get_default_error_metrics()

        try:
            # Academic vocabulary analysis
            results['academic'] = self.academic_vocab.get_academic_profile(text)
        except Exception as e:
            logger.error(f"Academic vocabulary analysis failed: {e}")
            results['academic'] = self._get_default_academic_metrics()

        return results

    def _create_combined_metrics(self, results: Dict, timestamp: datetime) -> AnalysisMetrics:
        """Create combined metrics from component results"""
        return AnalysisMetrics(
            timestamp=timestamp,
            # Lexical metrics
            mattr=results['lexical'].get('mattr', 0.0),
            word_count=results['lexical'].get('word_count', 0),
            unique_words=results['lexical'].get('unique_words', 0),
            mean_word_length=results['lexical'].get('sophistication', {}).get('mean_word_length', 0.0),
            lexical_density=results['lexical'].get('density', 0.0),
            academic_ratio=results['academic'].get('academic_ratio', 0.0),
            
            # Burst metrics
            burst_length=results['burst'].get('burst_length', 0.0),
            burst_rate=results['burst'].get('wpm', 0.0),
            pause_duration=results['burst'].get('pause_duration', 0.0),
            total_pauses=results['burst'].get('total_pauses', 0),
            
            # Dependency metrics
            ndd=results['dependency'].get('ndd', 0.0),
            working_memory_load=results['dependency'].get('working_memory_load', 0.0),
            processing_cost=results['dependency'].get('processing_cost', 0.0),
            incomplete_dependencies=results['dependency'].get('incomplete_dependencies', 0),
            
            # Error metrics
            error_likelihood=results['error'].get('error_likelihood', 0.0),
            error_locations=results['error'].get('error_locations', []),
            error_types=self._classify_error_types(results['error'])
        )

    def _validate_metrics(self, metrics: AnalysisMetrics) -> bool:
        """Validate metrics for consistency and reasonable values"""
        try:
            # Convert to dict for easier validation
            m = asdict(metrics)
            
            # Basic range checks
            validations = [
                0 <= m['mattr'] <= 1,
                m['word_count'] >= 0,
                m['unique_words'] <= m['word_count'],
                m['mean_word_length'] >= 0,
                0 <= m['lexical_density'] <= 1,
                0 <= m['academic_ratio'] <= 1,
                m['burst_length'] >= 0,
                m['burst_rate'] >= 0,
                m['pause_duration'] >= 0,
                m['total_pauses'] >= 0,
                m['ndd'] >= 0,
                m['working_memory_load'] >= 0,
                m['processing_cost'] >= 0,
                m['incomplete_dependencies'] >= 0,
                0 <= m['error_likelihood'] <= 1
            ]
            
            return all(validations)
            
        except Exception as e:
            logger.error(f"Error validating metrics: {e}")
            return False

    def _classify_error_types(self, error_metrics: Dict) -> Dict[str, float]:
        """Classify types of potential errors"""
        try:
            error_types = {}
            if error_metrics.get('error_tokens'):
                for token, score in zip(
                    error_metrics['error_tokens'],
                    error_metrics['perplexity_scores']
                ):
                    if score > 100:
                        error_types['severe_error'] = score
                    elif score > 50:
                        error_types['moderate_error'] = score
                    else:
                        error_types['minor_error'] = score
            
            return error_types
            
        except Exception as e:
            logger.error(f"Error classifying error types: {e}")
            return {}

    def _update_session_state(self, 
                            text: str, 
                            timestamp: datetime,
                            metrics: AnalysisMetrics,
                            analysis_results: Dict):
        """Update session state with new analysis"""
        try:
            # Update timing
            self.session_state.update_timing(timestamp)
            
            # Update word counts
            self.session_state.total_words += metrics.word_count
            self.session_state.unique_words.update(text.split())
            
            # Update current burst if relevant
            if metrics.burst_length > 0:
                new_burst = WritingBurst(
                    start_time=timestamp,
                    end_time=timestamp,
                    text=text,
                    word_count=metrics.word_count,
                    mean_word_length=metrics.mean_word_length,
                    pause_before=metrics.pause_duration,
                    pause_after=0.0,
                    wpm=metrics.burst_rate
                )
                self.session_state.add_burst(new_burst)
            
            # Add any new commitment points
            for cp in analysis_results['dependency'].get('commitment_points', []):
                self.session_state.add_commitment_point(cp)
            
        except Exception as e:
            logger.error(f"Error updating session state: {e}")

    def _update_cache(self, key: tuple, metrics: AnalysisMetrics):
        """Update analysis cache"""
        try:
            self._cache[key] = metrics
            
            # Remove oldest entries if cache is too large
            if len(self._cache) > self._cache_size:
                oldest_key = min(self._cache.keys(), key=lambda k: k[1])
                del self._cache[oldest_key]
                
        except Exception as e:
            logger.error(f"Error updating cache: {e}")

    def get_session_statistics(self) -> Dict:
        """Get comprehensive session statistics"""
        try:
            burst_stats = self.burst_analyzer.get_burst_statistics()
            dependency_stats = self.dependency_analyzer.get_analysis_statistics()
            
            return {
                'session_duration': self.session_state.get_session_duration(),
                'total_words': self.session_state.total_words,
                'unique_words': len(self.session_state.unique_words),
                'vocabulary_diversity': (
                    len(self.session_state.unique_words) / 
                    self.session_state.total_words 
                    if self.session_state.total_words > 0 else 0.0
                ),
                'total_bursts': len(self.session_state.bursts),
                'mean_burst_length': burst_stats['mean_burst_length'],
                'mean_pause_duration': burst_stats['mean_pause_duration'],
                'mean_wpm': burst_stats['mean_wpm'],
                'mean_ndd': dependency_stats['mean_ndd'],
                'max_memory_load': dependency_stats['max_memory_load'],
                'total_commitment_points': dependency_stats['total_commitment_points'],
                'mean_processing_cost': dependency_stats['mean_processing_cost']
            }
            
        except Exception as e:
            logger.error(f"Error getting session statistics: {e}")
            return self._get_default_session_stats()

    def _get_default_metrics(self) -> AnalysisMetrics:
        """Return default metrics when analysis fails"""
        return AnalysisMetrics(
            timestamp=datetime.now(),
            mattr=0.0,
            word_count=0,
            unique_words=0,
            mean_word_length=0.0,
            lexical_density=0.0,
            academic_ratio=0.0,
            burst_length=0.0,
            burst_rate=0.0,
            pause_duration=0.0,
            total_pauses=0,
            ndd=0.0,
            working_memory_load=0.0,
            processing_cost=0.0,
            incomplete_dependencies=0,
            error_likelihood=0.0,
            error_locations=[],
            error_types={}
        )

    def _get_default_session_stats(self) -> Dict:
        """Return default session statistics"""
        return {
            'session_duration': 0.0,
            'total_words': 0,
            'unique_words': 0,
            'vocabulary_diversity': 0.0,
            'total_bursts': 0,
            'mean_burst_length': 0.0,
            'mean_pause_duration': 0.0,
            'mean_wpm': 0.0,
            'mean_ndd': 0.0,
            'max_memory_load': 0.0,
            'total_commitment_points': 0,
            'mean_processing_cost': 0.0
        }

    def _get_default_lexical_metrics(self) -> Dict:
        """Return default lexical metrics"""
        return {
            'mattr': 0.0,
            'word_count': 0,
            'unique_words': 0,
            'sophistication': {
                'mean_word_length': 0.0,
                'academic_ratio': 0.0
            },
            'density': 0.0
        }

    def _get_default_burst_metrics(self) -> Dict:
        """Return default burst metrics"""
        return {
            'burst_length': 0.0,
            'wpm': 0.0,
            'pause_duration': 0.0,
            'total_pauses': 0
        }

def _get_default_burst_metrics(self) -> Dict:
    """Return default burst metrics"""
    return {
        'burst_length': 0.0,
        'wpm': 0.0,
        'pause_duration': 0.0,
        'total_pauses': 0
    }

def _get_default_dependency_metrics(self) -> Dict:
    """Return default dependency metrics"""
    return {
        'ndd': 0.0,
        'working_memory_load': 0.0,
        'processing_cost': 0.0,
        'incomplete_dependencies': 0,
        'commitment_points': []
    }

def _get_default_error_metrics(self) -> Dict:
    """Return default error metrics"""
    return {
        'error_likelihood': 0.0,
        'error_locations': [],
        'error_tokens': [],
        'perplexity_scores': []
    }

def _get_default_academic_metrics(self) -> Dict:
    """Return default academic vocabulary metrics"""
    return {
        'academic_ratio': 0.0,
        'academic_types': 0,
        'sublist_distribution': {},
        'frequency_bands': {
            'high': 0,
            'medium': 0,
            'low': 0
        }
    }

def get_last_error(self) -> Optional[AnalysisError]:
    """Get the last error that occurred during analysis"""
    return self.last_error

def clear_error(self):
    """Clear the last error"""
    self.last_error = None

    def clear_error(self):
        """Clear the last error"""
        self.last_error = None

    def save_session_data(self, filepath: Optional[Path] = None) -> Path:
        """Save current session data to file"""
        try:
            if filepath is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filepath = self.output_dir / f"session_data_{timestamp}.json"

            session_data = {
                'statistics': self.get_session_statistics(),
                'bursts': [asdict(burst) for burst in self.session_state.bursts],
                'commitment_points': [asdict(cp) for cp in self.session_state.commitment_points],
                'total_words': self.session_state.total_words,
                'unique_words': list(self.session_state.unique_words),
                'session_duration': self.session_state.get_session_duration()
            }

            import json
            with open(filepath, 'w') as f:
                json.dump(session_data, f, indent=4, default=str)

            logger.info(f"Session data saved to {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"Error saving session data: {e}")
            raise

    def load_session_data(self, filepath: Path) -> bool:
        """Load session data from file"""
        try:
            import json
            with open(filepath, 'r') as f:
                session_data = json.load(f)

            # Recreate session state
            self.session_state = SessionState()
            self.session_state.total_words = session_data['total_words']
            self.session_state.unique_words = set(session_data['unique_words'])

            # Recreate bursts
            for burst_data in session_data['bursts']:
                burst_data['start_time'] = datetime.fromisoformat(burst_data['start_time'])
                burst_data['end_time'] = datetime.fromisoformat(burst_data['end_time'])
                self.session_state.add_burst(WritingBurst(**burst_data))

            # Recreate commitment points
            for cp_data in session_data['commitment_points']:
                cp_data['timestamp'] = datetime.fromisoformat(cp_data['timestamp'])
                self.session_state.add_commitment_point(CommitmentPoint(**cp_data))

            logger.info(f"Session data loaded from {filepath}")
            return True

        except Exception as e:
            logger.error(f"Error loading session data: {e}")
            return False

    def reset_session(self):
        """Reset the analyzer session state"""
        try:
            self.session_state = SessionState()
            self._cache.clear()
            self.last_error = None
            logger.info("Analyzer session reset")
        except Exception as e:
            logger.error(f"Error resetting session: {e}")

def test_integrated_analyzer():
    """Test integrated analyzer functionality"""
    try:
        analyzer = IntegratedAnalyzer()
        
        test_texts = [
            "This is a simple test sentence.",
            "The quick brown fox jumps over the lazy dog.",
            "We are conducting an empirical analysis of the methodology.",
            "When the cat saw the dog that was chasing the mouse, it quickly climbed the tree.",
            "Thiss iz ann example of text with sum errors."
        ]
        
        for text in test_texts:
            logger.info(f"\nAnalyzing: {text}")
            
            metrics = analyzer.analyze_text(text, datetime.now())
            
            logger.info(f"MATTR: {metrics.mattr:.3f}")
            logger.info(f"Word Count: {metrics.word_count}")
            logger.info(f"Burst Length: {metrics.burst_length}")
            logger.info(f"NDD: {metrics.ndd:.3f}")
            logger.info(f"Working Memory Load: {metrics.working_memory_load:.3f}")
            logger.info(f"Error Likelihood: {metrics.error_likelihood:.3f}")
            
            # Short delay between texts
            from time import sleep
            sleep(2)
        
        # Get session statistics
        stats = analyzer.get_session_statistics()
        logger.info("\nSession Statistics:")
        for key, value in stats.items():
            logger.info(f"{key}: {value}")
        
        # Save session data
        analyzer.save_session_data()
            
    except Exception as e:
        logger.error(f"Error in integrated analyzer test: {e}")
        raise

if __name__ == "__main__":
    test_integrated_analyzer()

