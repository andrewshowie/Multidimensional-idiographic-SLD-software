# academic_vocabulary.py

from typing import Dict, Set, Optional, List, Tuple
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import numpy as np
import re

from logger_config import logger
from avl_data import AVL_DATA
from config import config

class AcademicVocabulary:
    """Academic Vocabulary Analysis using Davies and Gardner (2014) AVL"""
    
    def __init__(self):
        """Initialize academic vocabulary analyzer"""
        try:
            # Load AVL data
            self.vocabulary = AVL_DATA
            
            # Create frequency band thresholds
            self.frequency_bands = {
                'high': 100.0,  # Words with frequency > 100 per million
                'medium': 50.0,  # Words with frequency > 50 per million
                'low': 0.0      # All other academic words
            }
            
            # Track vocabulary statistics
            self.word_history = defaultdict(int)
            self.sublist_usage = defaultdict(int)
            
            # Initialize caches
            self.profile_cache = {}
            self.cache_size = config.analysis.cache_size
            
            logger.info("Academic vocabulary analyzer initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing academic vocabulary: {e}")
            raise

    def is_academic(self, word: str) -> bool:
        """Check if word is in academic vocabulary"""
        return word.lower() in self.vocabulary

    def get_word_info(self, word: str) -> Optional[Dict]:
        """Get detailed information about academic word"""
        return self.vocabulary.get(word.lower())

    def get_sublist(self, sublist_num: int) -> Set[str]:
        """Get all words from a specific sublist"""
        return {
            word for word, info in self.vocabulary.items()
            if info['sublist'] == sublist_num
        }

    def calculate_academic_ratio(self, text: str) -> float:
        """Calculate ratio of academic words in text"""
        try:
            words = self._preprocess_text(text)
            if not words:
                return 0.0
            
            academic_words = sum(1 for word in words if self.is_academic(word))
            return academic_words / len(words)
            
        except Exception as e:
            logger.error(f"Error calculating academic ratio: {e}")
            return 0.0

    def get_academic_profile(self, text: str) -> Dict:
        """Get detailed academic vocabulary profile of text"""
        try:
            # Check cache first
            cache_key = text
            if cache_key in self.profile_cache:
                return self.profile_cache[cache_key]
            
            words = self._preprocess_text(text)
            total_words = len(words)
            
            if not total_words:
                return self._get_default_profile()
            
            profile = {
                'total_words': total_words,
                'academic_words': 0,
                'academic_types': set(),
                'sublist_distribution': defaultdict(int),
                'frequency_bands': defaultdict(int),
                'sophistication_score': 0.0,
                'dispersion_metrics': defaultdict(float),
                'vocabulary_depth': defaultdict(int)
            }
            
            # Analyze each word
            for word in words:
                if self.is_academic(word):
                    info = self.vocabulary[word]
                    
                    # Update counts
                    profile['academic_words'] += 1
                    profile['academic_types'].add(word)
                    profile['sublist_distribution'][info['sublist']] += 1
                    
                    # Update frequency bands
                    freq = info['frequency']
                    if freq > self.frequency_bands['high']:
                        profile['frequency_bands']['high'] += 1
                    elif freq > self.frequency_bands['medium']:
                        profile['frequency_bands']['medium'] += 1
                    else:
                        profile['frequency_bands']['low'] += 1
                    
                    # Calculate sophistication score
                    profile['sophistication_score'] += (
                        info['frequency'] * info['dispersion'] * info['range']
                    )
                    
                    # Track dispersion
                    profile['dispersion_metrics']['mean_dispersion'] += info['dispersion']
                    
                    # Track vocabulary depth
                    profile['vocabulary_depth'][info['sublist']] += 1
                    
                    # Update historical tracking
                    self.word_history[word] += 1
                    self.sublist_usage[info['sublist']] += 1
            
            # Calculate final metrics
            academic_word_count = profile['academic_words']
            if academic_word_count > 0:
                profile['sophistication_score'] /= academic_word_count
                profile['dispersion_metrics']['mean_dispersion'] /= academic_word_count
            
            # Calculate ratios
            profile['academic_ratio'] = academic_word_count / total_words
            profile['academic_types'] = len(profile['academic_types'])
            profile['academic_type_token_ratio'] = (
                profile['academic_types'] / academic_word_count if academic_word_count > 0 else 0.0
            )
            
            # Update cache
            self._update_cache(cache_key, profile)
            
            return profile
            
        except Exception as e:
            logger.error(f"Error generating academic profile: {e}")
            return self._get_default_profile()

    def _preprocess_text(self, text: str) -> List[str]:
        """Preprocess text for analysis"""
        try:
            # Convert to lowercase
            text = text.lower()
            
            # Remove punctuation and split into words
            words = re.findall(r'\b\w+\b', text)
            
            return words
            
        except Exception as e:
            logger.error(f"Error preprocessing text: {e}")
            return []

    def get_historical_statistics(self) -> Dict:
        """Get historical usage statistics"""
        try:
            if not self.word_history:
                return self._get_default_historical_stats()
            
            total_uses = sum(self.word_history.values())
            total_words = len(self.word_history)
            
            return {
                'total_academic_words_used': total_uses,
                'unique_academic_words': total_words,
                'most_frequent_words': sorted(
                    self.word_history.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10],
                'sublist_distribution': dict(self.sublist_usage),
                'vocabulary_breadth': total_words / len(self.vocabulary),
                'usage_concentration': self._calculate_usage_concentration()
            }
            
        except Exception as e:
            logger.error(f"Error getting historical statistics: {e}")
            return self._get_default_historical_stats()

    def _calculate_usage_concentration(self) -> float:
        """Calculate concentration of academic vocabulary usage"""
        try:
            if not self.word_history:
                return 0.0
            
            total_uses = sum(self.word_history.values())
            squares_sum = sum(count * count for count in self.word_history.values())
            
            # Herfindahl-Hirschman Index
            return squares_sum / (total_uses * total_uses) if total_uses > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating usage concentration: {e}")
            return 0.0

    def _update_cache(self, key: str, profile: Dict):
        """Update profile cache"""
        try:
            self.profile_cache[key] = profile
            
            # Remove oldest entries if cache is too large
            if len(self.profile_cache) > self.cache_size:
                oldest_key = next(iter(self.profile_cache))
                del self.profile_cache[oldest_key]
                
        except Exception as e:
            logger.error(f"Error updating cache: {e}")

    def _get_default_profile(self) -> Dict:
        """Return default empty profile"""
        return {
            'total_words': 0,
            'academic_words': 0,
            'academic_types': 0,
            'academic_ratio': 0.0,
            'academic_type_token_ratio': 0.0,
            'sublist_distribution': {},
            'frequency_bands': {
                'high': 0,
                'medium': 0,
                'low': 0
            },
            'sophistication_score': 0.0,
            'dispersion_metrics': {
                'mean_dispersion': 0.0
            },
            'vocabulary_depth': {}
        }

    def _get_default_historical_stats(self) -> Dict:
        """Return default historical statistics"""
        return {
            'total_academic_words_used': 0,
            'unique_academic_words': 0,
            'most_frequent_words': [],
            'sublist_distribution': {},
            'vocabulary_breadth': 0.0,
            'usage_concentration': 0.0
        }

    def reset_state(self):
        """Reset analyzer state"""
        try:
            self.word_history.clear()
            self.sublist_usage.clear()
            self.profile_cache.clear()
            logger.debug("Academic vocabulary analyzer state reset")
        except Exception as e:
            logger.error(f"Error resetting analyzer state: {e}")

if __name__ == "__main__":
    # Test the analyzer
    analyzer = AcademicVocabulary()
    
    test_texts = [
        "This is a simple test sentence.",
        "The analysis of empirical data requires sophisticated methodology.",
        "We are conducting research to evaluate the theoretical framework.",
        "The concept demonstrates significant implications for academic discourse."
    ]
    
    for text in test_texts:
        print(f"\nAnalyzing: {text}")
        profile = analyzer.get_academic_profile(text)
        print(f"Academic Ratio: {profile['academic_ratio']:.3f}")
        print(f"Academic Words: {profile['academic_words']}")
        print(f"Academic Types: {profile['academic_types']}")
        print("Frequency Bands:", dict(profile['frequency_bands']))
        print("Sublist Distribution:", dict(profile['sublist_distribution']))
    
    # Print historical statistics
    print("\nHistorical Statistics:")
    stats = analyzer.get_historical_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")