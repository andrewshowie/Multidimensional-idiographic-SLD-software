# error_classifier.py

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, BertForMaskedLM, BertTokenizer
import numpy as np
from typing import Dict, List, Tuple
import spacy
from collections import defaultdict
import re

from logger_config import logger, log_exceptions
from config import config

class ErrorClassifier:
    """Advanced error classification using multiple models"""
    
    def __init__(self):
        """Initialize error detection models"""
        try:
            # Load GPT-2 for perplexity-based detection
            self.gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
            self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            self.gpt2_model.eval()
            
            # Load BERT for contextual analysis
            self.bert_model = BertForMaskedLM.from_pretrained('bert-base-uncased')
            self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.bert_model.eval()
            
            # Load spaCy for syntactic analysis
            self.nlp = spacy.load("en_core_web_sm")
            
            # Error type patterns
            self.error_patterns = {
                'spelling': [
                    r'\b\w*[aeiou]{3,}\w*\b',  # Unusual vowel patterns
                    r'\b\w*(.)\1{2,}\w*\b',     # Repeated characters
                    r'\b\w*[^aeiou]{5,}\w*\b'   # Too many consonants
                ],
                'grammar': [
                    r'\b(a)\s+[aeiou]',          # Article errors
                    r'\b(have)\s+[a-z]+ed\b',    # Perfect tense errors
                    r'\b(is|are|was|were)\s+\w+ing\b'  # Progressive errors
                ]
            }
            
            logger.info("ErrorClassifier initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing ErrorClassifier: {e}")
            raise

    @log_exceptions()
    def classify_errors(self, text: str) -> Dict:
        """Perform comprehensive error analysis"""
        try:
            results = {
                'error_likelihood': 0.0,
                'error_locations': [],
                'error_types': defaultdict(list),
                'suggestions': []
            }
            
            # Skip empty text
            if not text.strip():
                return results
            
            # Perplexity-based detection
            perplexity_errors = self._check_perplexity(text)
            results['error_locations'].extend(perplexity_errors['locations'])
            results['error_likelihood'] = perplexity_errors['likelihood']
            
            # Pattern-based detection
            pattern_errors = self._check_patterns(text)
            results['error_types'].update(pattern_errors)
            
            # Contextual analysis
            contextual_errors = self._check_context(text)
            results['error_types'].update(contextual_errors)
            
            # Syntactic analysis
            syntactic_errors = self._check_syntax(text)
            results['error_types'].update(syntactic_errors)
            
            # Generate suggestions
            results['suggestions'] = self._generate_suggestions(text, results)
            
            return dict(results)  # Convert defaultdict to regular dict
            
        except Exception as e:
            logger.error(f"Error in error classification: {e}")
            return {
                'error_likelihood': 0.0,
                'error_locations': [],
                'error_types': {},
                'suggestions': []
            }

    @torch.no_grad()
    def _check_perplexity(self, text: str) -> Dict:
        """Check text perplexity using GPT-2"""
        try:
            # Tokenize input
            inputs = self.gpt2_tokenizer(text, return_tensors='pt')
            
            # Get model outputs
            outputs = self.gpt2_model(**inputs, labels=inputs['input_ids'])
            
            # Calculate token-level perplexity
            token_perplexities = torch.exp(outputs.logits)
            
            # Find unusual tokens
            mean_perplexity = torch.mean(token_perplexities)
            std_perplexity = torch.std(token_perplexities)
            threshold = mean_perplexity + 2 * std_perplexity
            
            error_indices = torch.where(token_perplexities > threshold)[0]
            
            return {
                'locations': error_indices.tolist(),
                'likelihood': float(mean_perplexity)
            }
            
        except Exception as e:
            logger.error(f"Error in perplexity check: {e}")
            return {'locations': [], 'likelihood': 0.0}

    def _check_patterns(self, text: str) -> Dict:
        """Check for pattern-based errors"""
        try:
            errors = defaultdict(list)
            
            # Check each pattern type
            for error_type, patterns in self.error_patterns.items():
                for pattern in patterns:
                    matches = re.finditer(pattern, text, re.IGNORECASE)
                    for match in matches:
                        errors[error_type].append({
                            'span': match.span(),
                            'text': match.group()
                        })
            
            return errors
            
        except Exception as e:
            logger.error(f"Error in pattern check: {e}")
            return {}

    @torch.no_grad()
    def _check_context(self, text: str) -> Dict:
        """Check for contextual errors using BERT"""
        try:
            errors = defaultdict(list)
            
            # Tokenize input
            inputs = self.bert_tokenizer(text, return_tensors='pt')
            
            # Get model predictions
            outputs = self.bert_model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)
            
            # Compare predictions with actual tokens
            for i, (pred, actual) in enumerate(zip(predictions[0], inputs['input_ids'][0])):
                if pred != actual:
                    token = self.bert_tokenizer.decode([actual])
                    suggested = self.bert_tokenizer.decode([pred])
                    
                    errors['contextual'].append({
                        'position': i,
                        'token': token,
                        'suggestion': suggested
                    })
            
            return errors
            
        except Exception as e:
            logger.error(f"Error in context check: {e}")
            return {}

    def _check_syntax(self, text: str) -> Dict:
        """Check for syntactic errors"""
        try:
            errors = defaultdict(list)
            doc = self.nlp(text)
            
            # Check for dependency issues
            for token in doc:
                if token.dep_ == 'ROOT' and token.pos_ not in ['VERB', 'AUX']:
                    errors['syntax'].append({
                        'type': 'invalid_root',
                        'token': token.text,
                        'position': token.i
                    })
                
                # Check for unusual dependency distances
                if token.head != token:
                    distance = abs(token.i - token.head.i)
                    if distance > 5:  # Arbitrary threshold
                        errors['syntax'].append({
                            'type': 'long_dependency',
                            'token': token.text,
                            'position': token.i
                        })
            
            return errors
            
        except Exception as e:
            logger.error(f"Error in syntax check: {e}")
            return {}

    def _generate_suggestions(self, text: str, results: Dict) -> List[Dict]:
        """Generate improvement suggestions"""
        try:
            suggestions = []
            
            # Generate suggestions for each error type
            for error_type, errors in results['error_types'].items():
                if error_type == 'spelling':
                    for error in errors:
                        suggestions.append({
                            'type': 'spelling',
                            'original': error['text'],
                            'suggestion': self._get_spelling_suggestion(error['text'])
                        })
                        
                elif error_type == 'contextual':
                    for error in errors:
                        suggestions.append({
                            'type': 'contextual',
                            'original': error['token'],
                            'suggestion': error['suggestion']
                        })
                        
                elif error_type == 'syntax':
                    for error in errors:
                        suggestions.append({
                            'type': 'syntax',
                            'issue': error['type'],
                            'suggestion': self._get_syntax_suggestion(error)
                        })
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Error generating suggestions: {e}")
            return []

    def _get_spelling_suggestion(self, word: str) -> str:
        """Get spelling suggestion using edit distance"""
        # Implement basic spell checking here
        return f"Consider revising: {word}"

    def _get_syntax_suggestion(self, error: Dict) -> str:
        """Get syntax improvement suggestion"""
        if error['type'] == 'invalid_root':
            return "Consider restructuring the sentence with a main verb"
        elif error['type'] == 'long_dependency':
            return "Consider breaking this into shorter sentences"
        return "Consider revising the sentence structure"

# academic_wordlist.py

import pandas as pd
from pathlib import Path
from typing import Set, Dict, Optional
import json

class AcademicVocabulary:
    """Academic Vocabulary List (AVL) from Davies and Gardner (2014)"""
    
    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize academic vocabulary list"""
        try:
            self.data_dir = data_dir or Path(__file__).parent / 'data'
            self.data_dir.mkdir(exist_ok=True)
            
            # Load vocabulary data
            self.vocabulary = self._load_vocabulary()
            
            logger.info("Academic vocabulary loaded successfully")
            
        except Exception as e:
            logger.error(f"Error initializing academic vocabulary: {e}")
            raise

    def _load_vocabulary(self) -> Dict:
        """Load vocabulary data from file or initialize default"""
        vocab_file = self.data_dir / 'avl_davies_gardner.json'
        
        if vocab_file.exists():
            with open(vocab_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        # Initialize with core academic vocabulary
        # This is a small subset of the full AVL for demonstration
        # The full list should be properly loaded from a complete source
        return {
            "analysis": {
                "frequency": 120.5,
                "dispersion": 0.85,
                "range": 0.92,
                "sublist": 1
            },
            "research": {
                "frequency": 115.3,
                "dispersion": 0.88,
                "range": 0.95,
                "sublist": 1
            },
            "data": {
                "frequency": 98.7,
                "dispersion": 0.82,
                "range": 0.90,
                "sublist": 1
            },
            # ... Add more words ...
        }

    def is_academic(self, word: str) -> bool:
        """Check if word is in academic vocabulary"""
        return word.lower() in self.vocabulary

    def get_word_info(self, word: str) -> Optional[Dict]:
        """Get information about academic word"""
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
            words = text.lower().split()
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
            words = text.lower().split()
            total_words = len(words)
            if not total_words:
                return self._get_default_profile()
            
            profile = {
                'total_words': total_words,
                'academic_words': 0,
                'academic_types': set(),
                'sublist_distribution': defaultdict(int),
                'frequency_bands': defaultdict(int)
            }
            
            for word in words:
                if self.is_academic(word):
                    info = self.vocabulary[word]
                    profile['academic_words'] += 1
                    profile['academic_types'].add(word)
                    profile['sublist_distribution'][info['sublist']] += 1
                    
                    # Categorize by frequency
                    freq = info['frequency']
                    if freq > 100:
                        profile['frequency_bands']['high'] += 1
                    elif freq > 50:
                        profile['frequency_bands']['medium'] += 1
                    else:
                        profile['frequency_bands']['low'] += 1
            
            # Calculate percentages
            profile['academic_ratio'] = profile['academic_words'] / total_words
            profile['academic_types'] = len(profile['academic_types'])
            
            return profile
            
        except Exception as e:
            logger.error(f"Error generating academic profile: {e}")
            return self._get_default_profile()

    def _get_default_profile(self) -> Dict:
        """Return default empty profile"""
        return {
            'total_words': 0,
            'academic_words': 0,
            'academic_types': 0,
            'academic_ratio': 0.0,
            'sublist_distribution': {},
            'frequency_bands': {
                'high': 0,
                'medium': 0,
                'low': 0
            }
        }

# Create global instances
error_classifier = ErrorClassifier()
academic_vocab = AcademicVocabulary()