# dependency_analyzer.py

import spacy
import numpy as np
from typing import List, Dict, Set, Tuple, Optional
from datetime import datetime
import networkx as nx
from dataclasses import dataclass
from collections import deque

from shared_types import CommitmentPoint
from logger_config import logger, log_exceptions
from config import config

@dataclass
class DependencyNode:
    """Represents a node in the dependency tree"""
    token: spacy.tokens.Token
    index: int
    depth: int
    children: List['DependencyNode']
    head_distance: int
    processing_cost: float

class DependencyAnalyzer:
    """Analyzes syntactic dependencies and processing complexity"""
    
    def __init__(self):
        """Initialize dependency analyzer"""
        try:
            # Load spaCy model
            self.nlp = spacy.load("en_core_web_sm")
            
            # Initialize processing state
            self.active_dependencies: List[Tuple[int, int]] = []
            self.dependency_graph = nx.DiGraph()
            self.working_memory = deque(maxlen=config.analysis.max_context_window)
            
            # Track commitment points
            self.commitment_points: List[CommitmentPoint] = []
            
            # Initialize memory cost weights
            self.cost_weights = {
                'dependency_distance': 0.5,
                'tree_depth': 0.3,
                'branching_factor': 0.2
            }
            
            logger.info("DependencyAnalyzer initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing DependencyAnalyzer: {e}")
            raise

    @log_exceptions()
    def analyze_dependencies(self, text: str, timestamp: datetime) -> Dict:
        """Perform comprehensive dependency analysis"""
        try:
            if not text.strip():
                return self._get_default_metrics()
                
            # Process text
            doc = self.nlp(text)
            
            # Build dependency tree
            root_node = self._build_dependency_tree(doc)
            
            # Calculate core metrics
            ndd = self._calculate_ndd(doc)
            memory_load = self._calculate_memory_load(root_node)
            processing_cost = self._calculate_processing_cost(root_node)
            
            # Update working memory
            self._update_working_memory(doc)
            
            # Detect commitment points
            new_commitment_points = self._detect_commitment_points(doc, timestamp)
            self.commitment_points.extend(new_commitment_points)
            
            metrics = {
                'ndd': ndd,
                'working_memory_load': memory_load,
                'processing_cost': processing_cost,
                'tree_depth': self._get_tree_depth(root_node),
                'branching_factor': self._get_mean_branching_factor(root_node),
                'incomplete_dependencies': len(self.active_dependencies),
                'commitment_points': new_commitment_points
            }
            
            logger.debug(f"Dependency analysis completed for text: {text[:50]}...")
            return metrics
            
        except Exception as e:
            logger.error(f"Error in dependency analysis: {e}")
            return self._get_default_metrics()

    @log_exceptions()
    def _build_dependency_tree(self, doc: spacy.tokens.Doc) -> DependencyNode:
        """Build tree representation of dependencies"""
        try:
            # Create node mapping
            nodes: Dict[int, DependencyNode] = {}
            
            # First pass: create nodes
            for token in doc:
                head_distance = abs(token.i - token.head.i) if token.head != token else 0
                nodes[token.i] = DependencyNode(
                    token=token,
                    index=token.i,
                    depth=0,  # Will be updated in second pass
                    children=[],
                    head_distance=head_distance,
                    processing_cost=0.0  # Will be updated
                )
            
            # Second pass: build tree structure
            root = None
            for token in doc:
                node = nodes[token.i]
                if token.head == token:
                    root = node
                else:
                    head_node = nodes[token.head.i]
                    head_node.children.append(node)
            
            if root is None:
                # Handle case where no root is found
                root = nodes[0] if nodes else self._create_dummy_node()
            
            # Calculate depths and costs
            self._calculate_tree_metrics(root, 0)
            
            return root
            
        except Exception as e:
            logger.error(f"Error building dependency tree: {e}")
            return self._create_dummy_node()

    def _create_dummy_node(self) -> DependencyNode:
        """Create a dummy node for error cases"""
        dummy_token = self.nlp("dummy")[0]
        return DependencyNode(
            token=dummy_token,
            index=0,
            depth=0,
            children=[],
            head_distance=0,
            processing_cost=0.0
        )

    def _calculate_tree_metrics(self, node: DependencyNode, depth: int) -> None:
        """Calculate depth and processing cost for each node"""
        node.depth = depth
        node.processing_cost = self._calculate_node_cost(node)
        
        for child in node.children:
            self._calculate_tree_metrics(child, depth + 1)

    def _calculate_node_cost(self, node: DependencyNode) -> float:
        """Calculate processing cost for a single node"""
        # Base cost
        cost = 1.0
        
        # Add distance cost
        cost += node.head_distance * self.cost_weights['dependency_distance']
        
        # Add depth cost
        cost += node.depth * self.cost_weights['tree_depth']
        
        # Add branching cost
        cost += len(node.children) * self.cost_weights['branching_factor']
        
        return cost

    @log_exceptions()
    def _calculate_ndd(self, doc: spacy.tokens.Doc) -> float:
        """Calculate Normalized Dependency Distance"""
        try:
            if len(doc) <= 1:
                return 0.0
                
            total_distance = sum(
                abs(token.i - token.head.i)
                for token in doc
                if token.head != token
            )
            
            # Normalize by sentence length
            n = len(doc)
            expected_distance = (n ** 2) / (6 * (n - 1))  # Theoretical mean
            
            return total_distance / (n * expected_distance) if n > 1 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating NDD: {e}")
            return 0.0

    @log_exceptions()
    def _calculate_memory_load(self, root_node: DependencyNode) -> float:
        """Calculate current working memory load"""
        try:
            # Base memory cost from active dependencies
            base_load = len(self.active_dependencies) * 0.5
            
            # Add cost from tree complexity
            tree_load = (
                root_node.depth * self.cost_weights['tree_depth'] +
                len(root_node.children) * self.cost_weights['branching_factor']
            )
            
            # Add cost from working memory items
            memory_load = len(self.working_memory) * 0.1
            
            return base_load + tree_load + memory_load
            
        except Exception as e:
            logger.error(f"Error calculating memory load: {e}")
            return 0.0

    @log_exceptions()
    def _calculate_processing_cost(self, root_node: DependencyNode) -> float:
        """Calculate total processing cost"""
        try:
            def sum_costs(node: DependencyNode) -> float:
                return node.processing_cost + sum(
                    sum_costs(child) for child in node.children
                )
            
            return sum_costs(root_node)
            
        except Exception as e:
            logger.error(f"Error calculating processing cost: {e}")
            return 0.0

    def _get_tree_depth(self, root_node: DependencyNode) -> int:
        """Get maximum depth of dependency tree"""
        try:
            def max_depth(node: DependencyNode) -> int:
                if not node.children:
                    return node.depth
                return max(max_depth(child) for child in node.children)
            
            return max_depth(root_node)
            
        except Exception as e:
            logger.error(f"Error calculating tree depth: {e}")
            return 0

    def _get_mean_branching_factor(self, root_node: DependencyNode) -> float:
        """Calculate mean branching factor of tree"""
        try:
            node_counts = []
            
            def count_children(node: DependencyNode):
                node_counts.append(len(node.children))
                for child in node.children:
                    count_children(child)
            
            count_children(root_node)
            return np.mean(node_counts) if node_counts else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating branching factor: {e}")
            return 0.0

    @log_exceptions()
    def _detect_commitment_points(self, 
                                doc: spacy.tokens.Doc, 
                                timestamp: datetime) -> List[CommitmentPoint]:
        """Detect syntactic commitment points"""
        try:
            commitment_points = []
            
            for token in doc:
                # Check for significant dependency distance
                if token.head != token and abs(token.i - token.head.i) > 3:
                    # Calculate processing cost
                    distance_cost = abs(token.i - token.head.i) * 0.5
                    depth_cost = len(list(token.ancestors)) * 0.3
                    
                    commitment_points.append(CommitmentPoint(
                        timestamp=timestamp,
                        trigger_word=token.text,
                        token_index=token.i,
                        dependency_type=token.dep_,
                        dependency_distance=abs(token.i - token.head.i),
                        processing_cost=distance_cost + depth_cost,
                        incomplete_dependencies=len(self.active_dependencies)
                    ))
            
            return commitment_points
            
        except Exception as e:
            logger.error(f"Error detecting commitment points: {e}")
            return []

    def _update_working_memory(self, doc: spacy.tokens.Doc):
        """Update working memory with new tokens"""
        try:
            # Add new tokens to working memory
            for token in doc:
                if not token.is_space:
                    self.working_memory.append(token)
            
            # Update active dependencies
            self._update_active_dependencies(doc)
            
        except Exception as e:
            logger.error(f"Error updating working memory: {e}")

    def _update_active_dependencies(self, doc: spacy.tokens.Doc):
        """Update set of active dependencies"""
        try:
            # Remove completed dependencies
            self.active_dependencies = [
                (head, dep) for head, dep in self.active_dependencies
                if dep >= doc[0].i
            ]
            
            # Add new dependencies
            for token in doc:
                if token.head != token:
                    self.active_dependencies.append(
                        (token.head.i, token.i)
                    )
            
        except Exception as e:
            logger.error(f"Error updating active dependencies: {e}")

    def _get_default_metrics(self) -> Dict:
        """Return default metrics when analysis fails"""
        return {
            'ndd': 0.0,
            'working_memory_load': 0.0,
            'processing_cost': 0.0,
            'tree_depth': 0,
            'branching_factor': 0.0,
            'incomplete_dependencies': 0,
            'commitment_points': []
        }

    def get_analysis_statistics(self) -> Dict:
        """Get overall analysis statistics"""
        try:
            if not self.commitment_points:
                return self._get_default_metrics()
                
            return {
                'mean_ndd': np.mean([cp.dependency_distance 
                                   for cp in self.commitment_points]),
                'max_memory_load': max([cp.incomplete_dependencies 
                                      for cp in self.commitment_points]),
                'total_commitment_points': len(self.commitment_points),
                'mean_processing_cost': np.mean([cp.processing_cost 
                                               for cp in self.commitment_points])
            }
            
        except Exception as e:
            logger.error(f"Error calculating analysis statistics: {e}")
            return self._get_default_metrics()

    def reset_state(self):
        """Reset analyzer state"""
        try:
            self.active_dependencies.clear()
            self.dependency_graph.clear()
            self.working_memory.clear()
            self.commitment_points.clear()
            logger.debug("Dependency analyzer state reset")
        except Exception as e:
            logger.error(f"Error resetting analyzer state: {e}")

if __name__ == "__main__":
    # Test the analyzer
    analyzer = DependencyAnalyzer()
    test_sentences = [
        "This is a simple test.",
        "The quick brown fox jumps over the lazy dog.",
        "When the cat saw the mouse that was eating cheese, it quickly pounced.",
        "The fact that the student who studied hard passed the exam surprised nobody."
    ]
    
    for sentence in test_sentences:
        print(f"\nAnalyzing: {sentence}")
        results = analyzer.analyze_dependencies(sentence, datetime.now())
        print(f"NDD: {results['ndd']:.3f}")
        print(f"Memory Load: {results['working_memory_load']:.2f}")
        print(f"Processing Cost: {results['processing_cost']:.2f}")
        print(f"Tree Depth: {results['tree_depth']}")
        print(f"Commitment Points: {len(results['commitment_points'])}")
        
    # Get overall statistics
    stats = analyzer.get_analysis_statistics()
    print("\nOverall Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")