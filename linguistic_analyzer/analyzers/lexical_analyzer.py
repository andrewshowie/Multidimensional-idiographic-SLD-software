# lexical_analyzer.py

import spacy
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime
from collections import defaultdict
from logger_config import logger
from config import config

class LexicalAnalyzer:
    """Analyzes lexical characteristics of text"""
    
    def __init__(self):
        """Initialize lexical analysis components"""
        try:
            self.nlp = spacy.load("en_core_web_sm")
            
            # Initialize moving window for MATTR
            self.previous_words = []
            self.word_frequencies = defaultdict(int)
            
            # Load word lists if needed
            self.load_word_lists()
            
            logger.info("LexicalAnalyzer initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing LexicalAnalyzer: {e}")
            raise

    def load_word_lists(self):
        """Load word lists for analysis"""
        try:
            # Placeholder for actual word list loading
            # Now handled by AcademicVocabulary class
            self.frequency_bands = defaultdict(int)
            logger.debug("Word lists loaded successfully")
        except Exception as e:
            logger.error(f"Error loading word lists: {e}")
            self.frequency_bands = defaultdict(int)

    def calculate_lexical_measures(self, text: str) -> Dict:
        """Comprehensive lexical complexity analysis"""
        try:
            words = text.lower().split()
            self.previous_words.extend(words)
            self.previous_words = self.previous_words[-config.analysis.max_context_window:]

            measures = {
                'mattr': self._calculate_mattr(),
                'sophistication': self._calculate_sophistication(words),
                'density': self._calculate_density(words),
                'diversity': self._calculate_diversity(words),
                'timestamp': datetime.now().isoformat()
            }
            
            logger.debug(f"Calculated lexical measures: {measures}")
            return measures
        except Exception as e:
            logger.error(f"Error calculating lexical measures: {e}")
            return self._get_default_measures()

    def _calculate_mattr(self) -> float:
        """Moving Average Type-Token Ratio"""
        try:
            if len(self.previous_words) < config.analysis.plot_window_size:
                return len(set(self.previous_words)) / len(self.previous_words) if self.previous_words else 0
            
            ratios = []
            for i in range(len(self.previous_words) - config.analysis.plot_window_size + 1):
                window = self.previous_words[i:i + config.analysis.plot_window_size]
                ratio = len(set(window)) / len(window)
                ratios.append(ratio)
            return np.mean(ratios)
        except Exception as e:
            logger.error(f"Error calculating MATTR: {e}")
            return 0.0

    def _calculate_sophistication(self, words: List[str]) -> Dict:
        """Calculate sophistication metrics"""
        try:
            if not words:
                return {
                    'mean_word_length': 0.0,
                    'long_words_ratio': 0.0
                }
                
            word_lengths = [len(w) for w in words]
            long_words = sum(1 for w in words if len(w) > 6)
            
            return {
                'mean_word_length': np.mean(word_lengths),
                'long_words_ratio': long_words / len(words) if words else 0
            }
        except Exception as e:
            logger.error(f"Error calculating sophistication: {e}")
            return {'mean_word_length': 0.0, 'long_words_ratio': 0.0}

    def _calculate_density(self, words: List[str]) -> float:
        """Calculate lexical density using spaCy"""
        try:
            if not words:
                return 0.0
            doc = self.nlp(" ".join(words))
            content_words = len([token for token in doc if token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV']])
            return content_words / len(words)
        except Exception as e:
            logger.error(f"Error calculating density: {e}")
            return 0.0

    def _calculate_diversity(self, words: List[str]) -> Dict:
        """Calculate various diversity metrics"""
        try:
            unique_words = len(set(words))
            total_words = len(words)
            return {
                'ttr': unique_words / total_words if total_words else 0,
                'root_ttr': unique_words / np.sqrt(total_words) if total_words else 0,
                'log_ttr': unique_words / np.log(total_words) if total_words > 1 else 0
            }
        except Exception as e:
            logger.error(f"Error calculating diversity: {e}")
            return {'ttr': 0, 'root_ttr': 0, 'log_ttr': 0}

    def analyze_text(self, text: str, timestamp: datetime) -> Dict:
        """Perform comprehensive lexical analysis"""
        try:
            if not text.strip():
                return self._get_default_measures()
            
            # Process text
            doc = self.nlp(text)
            words = [token.text for token in doc if not token.is_space]
            
            # Calculate all metrics
            measures = self.calculate_lexical_measures(text)
            
            # Add basic counts
            measures.update({
                'word_count': len(words),
                'unique_words': len(set(words)),
                'sentence_count': len(list(doc.sents))
            })
            
            return measures
            
        except Exception as e:
            logger.error(f"Error in lexical analysis: {e}")
            return self._get_default_measures()

    def _get_default_measures(self) -> Dict:
        """Return default measures in case of error"""
        return {
            'mattr': 0.0,
            'sophistication': {'mean_word_length': 0.0, 'long_words_ratio': 0.0},
            'density': 0.0,
            'diversity': {'ttr': 0.0, 'root_ttr': 0.0, 'log_ttr': 0.0},
            'word_count': 0,
            'unique_words': 0,
            'sentence_count': 0,
            'timestamp': datetime.now().isoformat()
        }

    def reset_state(self):
        """Reset analyzer state"""
        try:
            self.previous_words = []
            self.word_frequencies.clear()
            logger.debug("Lexical analyzer state reset")
        except Exception as e:
            logger.error(f"Error resetting analyzer state: {e}")

if __name__ == "__main__":
    # Test the analyzer
    analyzer = LexicalAnalyzer()
    test_texts = [
        "This is a simple test sentence.",
        "The quick brown fox jumps over the lazy dog.",
        "We are conducting empirical analysis of linguistic complexity.",
        "This text contains some more sophisticated vocabulary elements."
    ]
    
    for text in test_texts:
        results = analyzer.analyze_text(text, datetime.now())
        print(f"\nAnalyzing: {text}")
        print(f"MATTR: {results['mattr']:.3f}")
        print(f"Lexical Density: {results['density']:.3f}")
        print(f"Word Count: {results['word_count']}")
        print(f"Sophistication: {results['sophistication']}")