"""
Advanced Network Topology for 250-Agent Brain System

This module implements sophisticated network topology management for
large-scale agent coordination and communication.
"""

import time
import random
import math
import threading
from typing import Dict, List, Any, Optional, Set, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import logging
import networkx as nx
import numpy as np

# Import existing components
from massive_agent_factory import AdvancedAgent, AgentClusterManager, massive_agent_factory
from communication_system import Signal, SignalType, Connection
from agent_specializations import AgentDomain, agent_registry

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NetworkTopology(Enum):
    """Network topology types"""
    SCALE_FREE = "scale_free"
    SMALL_WORLD = "small_world"
    HIERARCHICAL = "hierarchical"
    MESH = "mesh"
    HYBRID = "hybrid"

class ConnectionType(Enum):
    """Types of connections between agents"""
    STRONG = "strong"      # High weight, low latency
    MEDIUM = "medium"      # Medium weight, medium latency
    WEAK = "weak"         # Low weight, high latency
    DYNAMIC = "dynamic"    # Adaptive weight and latency

@dataclass
class NetworkMetrics:
    """Comprehensive network metrics"""
    total_connections: int = 0
    average_path_length: float = 0.0
    clustering_coefficient: float = 0.0
    network_density: float = 0.0
    diameter: int = 0
    centrality_scores: Dict[str, float] = field(default_factory=dict)
    efficiency_score: float = 0.0
    robustness_score: float = 0.0
    bandwidth_utilization: float = 0.0
    latency_distribution: Dict[str, float] = field(default_factory=dict)

@dataclass
class TopologyConfig:
    """Configuration for network topology"""
    topology_type: NetworkTopology = NetworkTopology.HYBRID
    max_connections_per_agent: int = 15
    connection_probability: float = 0.3
    rewire_probability: float = 0.1
    cluster_coefficient: float = 0.6
    scale_free_exponent: float = 2.5
    small_world_k: int = 6
    hierarchical_levels: int = 4
    optimization_interval: int = 100
    adaptation_rate: float = 0.05

class AdvancedNetworkTopology:
    """Advanced network topology management for large-scale agent systems"""
    
    def __init__(self, agents: Dict[str, AdvancedAgent], config: TopologyConfig = None):
        self.agents = agents
        self.config = config or TopologyConfig()
        
        # Network representation
        self.network_graph = nx.Graph()
        self.connection_matrix: Dict[str, Dict[str, Connection]] = defaultdict(dict)
        self.topology_metrics = NetworkMetrics()
        
        # Topology management
        self.topology_history = deque(maxlen=1000)
        self.optimization_history = deque(maxlen=100)
        self.adaptation_events = deque(maxlen=500)
        
        # Performance tracking
        self.signal_routing_stats = defaultdict(int)
        self.bandwidth_usage = defaultdict(float)
        self.latency_measurements = deque(maxlen=10000)
        
        # Dynamic adaptation
        self.adaptation_thresholds = {
            "high_latency": 0.1,
            "low_bandwidth": 0.8,
            "connection_failure": 0.05,
            "load_imbalance": 0.3
        }
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize network
        self._initialize_network()
    
    def _initialize_network(self):
        """Initialize the network topology"""
        self.logger.info(f"Initializing {self.config.topology_type.value} network topology")
        
        # Add all agents to network graph
        for agent_id, agent in self.agents.items():
            self.network_graph.add_node(agent_id, **self._get_node_attributes(agent))
        
        # Create topology based on configuration
        if self.config.topology_type == NetworkTopology.SCALE_FREE:
            self._create_scale_free_topology()
        elif self.config.topology_type == NetworkTopology.SMALL_WORLD:
            self._create_small_world_topology()
        elif self.config.topology_type == NetworkTopology.HIERARCHICAL:
            self._create_hierarchical_topology()
        elif self.config.topology_type == NetworkTopology.MESH:
            self._create_mesh_topology()
        elif self.config.topology_type == NetworkTopology.HYBRID:
            self._create_hybrid_topology()
        
        # Calculate initial metrics
        self._calculate_topology_metrics()
        
        self.logger.info(f"Network initialized with {self.network_graph.number_of_edges()} connections")
    
    def _get_node_attributes(self, agent: AdvancedAgent) -> Dict[str, Any]:
        """Get node attributes for network graph"""
        return {
            "domain": agent.specialization.domain.value,
            "expertise_level": agent.specialization.expertise_level,
            "priority": agent.priority.value,
            "capabilities": [cap.name for cap in agent.specialization.capabilities],
            "state": agent.state.value,
            "load": agent.load.queue_size
        }
    
    def _create_scale_free_topology(self):
        """Create scale-free network topology"""
        self.logger.info("Creating scale-free topology")
        
        # Start with small connected core
        core_agents = list(self.agents.keys())[:5]
        for i in range(len(core_agents)):
            for j in range(i + 1, len(core_agents)):
                self._add_connection(core_agents[i], core_agents[j], ConnectionType.STRONG)
        
        # Add remaining agents with preferential attachment
        remaining_agents = list(self.agents.keys())[5:]
        
        for agent_id in remaining_agents:
            # Calculate attachment probabilities based on existing degrees
            degrees = dict(self.network_graph.degree())
            total_degree = sum(degrees.values())
            
            # Select existing nodes to connect to
            connections_to_make = min(self.config.scale_free_exponent, len(degrees))
            selected_nodes = []
            
            for _ in range(connections_to_make):
                # Preferential attachment
                probabilities = [degrees[node] / total_degree for node in degrees]
                selected_node = np.random.choice(list(degrees.keys()), p=probabilities)
                selected_nodes.append(selected_node)
            
            # Add connections
            for target_id in selected_nodes:
                self._add_connection(agent_id, target_id, ConnectionType.MEDIUM)
    
    def _create_small_world_topology(self):
        """Create small-world network topology"""
        self.logger.info("Creating small-world topology")
        
        agent_ids = list(self.agents.keys())
        n = len(agent_ids)
        k = min(self.config.small_world_k, n - 1)
        
        # Create regular ring lattice
        for i, agent_id in enumerate(agent_ids):
            for j in range(1, k // 2 + 1):
                target_id = agent_ids[(i + j) % n]
                self._add_connection(agent_id, target_id, ConnectionType.MEDIUM)
        
        # Rewire connections with probability p
        edges_to_rewire = list(self.network_graph.edges())
        for source, target in edges_to_rewire:
            if random.random() < self.config.rewire_probability:
                # Remove existing edge
                self.network_graph.remove_edge(source, target)
                if source in self.connection_matrix and target in self.connection_matrix[source]:
                    del self.connection_matrix[source][target]
                if target in self.connection_matrix and source in self.connection_matrix[target]:
                    del self.connection_matrix[target][source]
                
                # Add new random edge
                new_target = random.choice([aid for aid in agent_ids if aid != source])
                self._add_connection(source, new_target, ConnectionType.MEDIUM)
    
    def _create_hierarchical_topology(self):
        """Create hierarchical network topology"""
        self.logger.info("Creating hierarchical topology")
        
        # Group agents by domain
        domain_groups = defaultdict(list)
        for agent_id, agent in self.agents.items():
            domain_groups[agent.specialization.domain].append(agent_id)
        
        # Create hierarchy levels
        levels = self.config.hierarchical_levels
        agents_per_level = len(self.agents) // levels
        
        # Create hierarchical connections
        current_level_agents = []
        for i, (domain, agent_ids) in enumerate(domain_groups.items()):
            # Domain-level clustering
            for j, agent_id in enumerate(agent_ids):
                # Connect within domain
                for k, other_id in enumerate(agent_ids):
                    if j < k:
                        self._add_connection(agent_id, other_id, ConnectionType.STRONG)
                
                # Connect to hierarchy
                if i < levels - 1:
                    # Connect to next level
                    next_level_domain = list(domain_groups.keys())[(i + 1) % len(domain_groups)]
                    next_level_agents = domain_groups[next_level_domain]
                    if next_level_agents:
                        target_id = random.choice(next_level_agents)
                        self._add_connection(agent_id, target_id, ConnectionType.MEDIUM)
    
    def _create_mesh_topology(self):
        """Create mesh network topology"""
        self.logger.info("Creating mesh topology")
        
        agent_ids = list(self.agents.keys())
        
        # Connect each agent to a subset of others
        for i, agent_id in enumerate(agent_ids):
            # Calculate connections based on probability
            possible_connections = [aid for aid in agent_ids if aid != agent_id]
            num_connections = int(len(possible_connections) * self.config.connection_probability)
            
            # Select random connections
            selected_connections = random.sample(
                possible_connections, 
                min(num_connections, len(possible_connections))
            )
            
            for target_id in selected_connections:
                self._add_connection(agent_id, target_id, ConnectionType.MEDIUM)
    
    def _create_hybrid_topology(self):
        """Create hybrid network topology combining multiple approaches"""
        self.logger.info("Creating hybrid topology")
        
        # Create base scale-free structure
        self._create_scale_free_topology()
        
        # Add small-world characteristics
        agent_ids = list(self.agents.keys())
        for agent_id in agent_ids:
            # Add random long-range connections
            if random.random() < self.config.rewire_probability:
                possible_targets = [aid for aid in agent_ids if aid != agent_id]
                if possible_targets:
                    target_id = random.choice(possible_targets)
                    if not self.network_graph.has_edge(agent_id, target_id):
                        self._add_connection(agent_id, target_id, ConnectionType.WEAK)
        
        # Add hierarchical clustering by domain
        domain_groups = defaultdict(list)
        for agent_id, agent in self.agents.items():
            domain_groups[agent.specialization.domain].append(agent_id)
        
        # Strengthen intra-domain connections
        for domain, agent_ids in domain_groups.items():
            for i, agent_id in enumerate(agent_ids):
                for j in range(i + 1, min(i + 3, len(agent_ids))):  # Connect to nearby agents
                    target_id = agent_ids[j]
                    if self.network_graph.has_edge(agent_id, target_id):
                        # Strengthen existing connection
                        connection = self.connection_matrix[agent_id][target_id]
                        connection.weight = min(1.0, connection.weight * 1.2)
    
    def _add_connection(self, source_id: str, target_id: str, connection_type: ConnectionType):
        """Add a connection between two agents"""
        if source_id not in self.agents or target_id not in self.agents:
            return
        
        source_agent = self.agents[source_id]
        target_agent = self.agents[target_id]
        
        # Calculate connection parameters
        weight = self._calculate_connection_weight(source_agent, target_agent, connection_type)
        delay = self._calculate_connection_delay(connection_type)
        
        # Create connection
        connection = Connection(
            source_id=source_id,
            target_id=target_id,
            weight=weight,
            connection_type=connection_type.value,
            delay=delay,
            metadata={
                "type": connection_type.value,
                "created_at": time.time(),
                "strength": weight,
                "latency": delay
            }
        )
        
        # Add to agents
        source_agent.add_connection(target_agent, weight, connection_type.value)
        
        # Add to network graph
        self.network_graph.add_edge(source_id, target_id, 
                                   weight=weight, 
                                   type=connection_type.value,
                                   delay=delay)
        
        # Add to connection matrix
        self.connection_matrix[source_id][target_id] = connection
        self.connection_matrix[target_id][source_id] = connection
    
    def _calculate_connection_weight(self, agent1: AdvancedAgent, agent2: AdvancedAgent, 
                                   connection_type: ConnectionType) -> float:
        """Calculate connection weight based on agents and connection type"""
        base_weights = {
            ConnectionType.STRONG: 0.9,
            ConnectionType.MEDIUM: 0.6,
            ConnectionType.WEAK: 0.3,
            ConnectionType.DYNAMIC: 0.5
        }
        
        base_weight = base_weights[connection_type]
        
        # Adjust based on agent compatibility
        compatibility_score = self._calculate_agent_compatibility(agent1, agent2)
        weight_adjustment = compatibility_score * 0.3
        
        # Adjust based on domain similarity
        domain_bonus = 0.2 if agent1.specialization.domain == agent2.specialization.domain else 0.0
        
        # Adjust based on expertise complementarity
        expertise_diff = abs(agent1.specialization.expertise_level - agent2.specialization.expertise_level)
        expertise_bonus = (1.0 - expertise_diff) * 0.2
        
        final_weight = base_weight + weight_adjustment + domain_bonus + expertise_bonus
        
        return max(0.1, min(1.0, final_weight))
    
    def _calculate_connection_delay(self, connection_type: ConnectionType) -> float:
        """Calculate connection delay based on connection type"""
        base_delays = {
            ConnectionType.STRONG: 0.001,
            ConnectionType.MEDIUM: 0.01,
            ConnectionType.WEAK: 0.1,
            ConnectionType.DYNAMIC: 0.05
        }
        
        return base_delays[connection_type] * random.uniform(0.8, 1.2)
    
    def _calculate_agent_compatibility(self, agent1: AdvancedAgent, agent2: AdvancedAgent) -> float:
        """Calculate compatibility score between two agents"""
        compatibility = 0.0
        
        # Capability compatibility
        for cap1 in agent1.specialization.capabilities:
            for cap2 in agent2.specialization.capabilities:
                if cap1.output_types and cap2.input_types:
                    overlap = len(set(cap1.output_types) & set(cap2.input_types))
                    if overlap > 0:
                        compatibility += overlap / max(len(cap1.output_types), len(cap2.input_types))
        
        # Normalize
        total_capability_pairs = len(agent1.specialization.capabilities) * len(agent2.specialization.capabilities)
        if total_capability_pairs > 0:
            compatibility /= total_capability_pairs
        
        return compatibility
    
    def _calculate_topology_metrics(self):
        """Calculate comprehensive topology metrics"""
        if not self.network_graph.nodes():
            return
        
        # Basic metrics
        self.topology_metrics.total_connections = self.network_graph.number_of_edges()
        self.topology_metrics.network_density = nx.density(self.network_graph)
        
        # Path metrics
        if nx.is_connected(self.network_graph):
            self.topology_metrics.average_path_length = nx.average_shortest_path_length(self.network_graph)
            self.topology_metrics.diameter = nx.diameter(self.network_graph)
        
        # Clustering coefficient
        self.topology_metrics.clustering_coefficient = nx.average_clustering(self.network_graph)
        
        # Centrality scores
        centrality_measures = {
            "degree_centrality": nx.degree_centrality(self.network_graph),
            "betweenness_centrality": nx.betweenness_centrality(self.network_graph),
            "closeness_centrality": nx.closeness_centrality(self.network_graph),
            "eigenvector_centrality": nx.eigenvector_centrality(self.network_graph, max_iter=1000)
        }
        
        # Combine centrality scores
        for node_id in self.network_graph.nodes():
            combined_centrality = (
                centrality_measures["degree_centrality"].get(node_id, 0) * 0.3 +
                centrality_measures["betweenness_centrality"].get(node_id, 0) * 0.3 +
                centrality_measures["closeness_centrality"].get(node_id, 0) * 0.2 +
                centrality_measures["eigenvector_centrality"].get(node_id, 0) * 0.2
            )
            self.topology_metrics.centrality_scores[node_id] = combined_centrality
        
        # Efficiency and robustness
        self.topology_metrics.efficiency_score = self._calculate_network_efficiency()
        self.topology_metrics.robustness_score = self._calculate_network_robustness()
        
        # Bandwidth and latency
        self.topology_metrics.bandwidth_utilization = self._calculate_bandwidth_utilization()
        self.topology_metrics.latency_distribution = self._calculate_latency_distribution()
    
    def _calculate_network_efficiency(self) -> float:
        """Calculate network efficiency score"""
        if not self.network_graph.nodes():
            return 0.0
        
        try:
            # Global efficiency
            efficiency = nx.global_efficiency(self.network_graph)
            
            # Adjust for network size
            size_factor = len(self.network_graph.nodes()) / 100.0
            adjusted_efficiency = efficiency / (1 + math.log(size_factor))
            
            return min(1.0, adjusted_efficiency)
        except:
            return 0.5
    
    def _calculate_network_robustness(self) -> float:
        """Calculate network robustness score"""
        if not self.network_graph.nodes():
            return 0.0
        
        # Calculate connectivity after random node removal
        original_size = len(self.network_graph.nodes())
        robustness_scores = []
        
        for _ in range(10):  # Test 10 random removals
            test_graph = self.network_graph.copy()
            
            # Remove random nodes
            nodes_to_remove = random.sample(list(test_graph.nodes()), 
                                         max(1, original_size // 10))
            test_graph.remove_nodes_from(nodes_to_remove)
            
            # Calculate connectivity
            if nx.is_connected(test_graph):
                robustness_scores.append(1.0)
            else:
                # Calculate size of largest component
                largest_component = max(nx.connected_components(test_graph), key=len)
                robustness_score = len(largest_component) / original_size
                robustness_scores.append(robustness_score)
        
        return sum(robustness_scores) / len(robustness_scores)
    
    def _calculate_bandwidth_utilization(self) -> float:
        """Calculate current bandwidth utilization"""
        total_bandwidth = sum(
            connection.weight for source, connections in self.connection_matrix.items()
            for connection in connections.values()
        )
        
        max_bandwidth = len(self.connection_matrix) * 1.0  # Maximum possible bandwidth
        
        return min(1.0, total_bandwidth / max_bandwidth) if max_bandwidth > 0 else 0.0
    
    def _calculate_latency_distribution(self) -> Dict[str, float]:
        """Calculate latency distribution across the network"""
        latencies = []
        
        for source, connections in self.connection_matrix.items():
            for connection in connections.values():
                latencies.append(connection.delay)
        
        if not latencies:
            return {"min": 0.0, "max": 0.0, "avg": 0.0, "median": 0.0}
        
        return {
            "min": min(latencies),
            "max": max(latencies),
            "avg": sum(latencies) / len(latencies),
            "median": sorted(latencies)[len(latencies) // 2]
        }
    
    def optimize_topology(self):
        """Optimize network topology based on performance metrics"""
        self.logger.info("Optimizing network topology")
        
        # Analyze current performance
        self._analyze_topology_performance()
        
        # Apply optimization strategies
        self._optimize_connection_weights()
        self._optimize_network_structure()
        self._balance_network_load()
        
        # Recalculate metrics
        self._calculate_topology_metrics()
        
        # Record optimization
        optimization_event = {
            "timestamp": time.time(),
            "metrics": self.topology_metrics.__dict__.copy(),
            "improvements": self._calculate_improvements()
        }
        
        self.optimization_history.append(optimization_event)
        
        self.logger.info("Network topology optimization completed")
    
    def _analyze_topology_performance(self):
        """Analyze current topology performance"""
        # Check for high-latency connections
        high_latency_connections = []
        for source, connections in self.connection_matrix.items():
            for target, connection in connections.items():
                if connection.delay > self.adaptation_thresholds["high_latency"]:
                    high_latency_connections.append((source, target))
        
        # Check for low-bandwidth utilization
        low_bandwidth_agents = []
        for agent_id, agent in self.agents.items():
            if agent.load.queue_size > 10:  # High load
                low_bandwidth_agents.append(agent_id)
        
        # Check for connection failures
        failure_rates = self._calculate_connection_failure_rates()
        
        # Log performance issues
        if high_latency_connections:
            self.logger.warning(f"Found {len(high_latency_connections)} high-latency connections")
        
        if low_bandwidth_agents:
            self.logger.warning(f"Found {len(low_bandwidth_agents)} agents with low bandwidth")
        
        if failure_rates:
            avg_failure_rate = sum(failure_rates.values()) / len(failure_rates)
            if avg_failure_rate > self.adaptation_thresholds["connection_failure"]:
                self.logger.warning(f"High connection failure rate: {avg_failure_rate:.3f}")
    
    def _calculate_connection_failure_rates(self) -> Dict[str, float]:
        """Calculate connection failure rates"""
        failure_rates = {}
        
        for source, connections in self.connection_matrix.items():
            total_signals = self.signal_routing_stats.get(f"{source}->*", 0)
            failed_signals = self.signal_routing_stats.get(f"{source}->*->failed", 0)
            
            if total_signals > 0:
                failure_rates[source] = failed_signals / total_signals
            else:
                failure_rates[source] = 0.0
        
        return failure_rates
    
    def _optimize_connection_weights(self):
        """Optimize connection weights based on usage patterns"""
        for source, connections in self.connection_matrix.items():
            for target, connection in connections.items():
                # Calculate usage statistics
                usage_key = f"{source}->{target}"
                signal_count = self.signal_routing_stats.get(usage_key, 0)
                
                if signal_count > 0:
                    # Calculate success rate
                    failed_count = self.signal_routing_stats.get(f"{usage_key}->failed", 0)
                    success_rate = 1.0 - (failed_count / signal_count)
                    
                    # Adjust weight based on performance
                    if success_rate > 0.9:
                        # Strengthen successful connections
                        connection.weight = min(1.0, connection.weight * 1.1)
                    elif success_rate < 0.7:
                        # Weaken failing connections
                        connection.weight = max(0.1, connection.weight * 0.9)
                    
                    # Update network graph
                    if self.network_graph.has_edge(source, target):
                        self.network_graph[source][target]["weight"] = connection.weight
    
    def _optimize_network_structure(self):
        """Optimize overall network structure"""
        # Identify poorly connected agents
        connectivity_scores = {}
        for agent_id in self.agents.keys():
            degree = self.network_graph.degree(agent_id)
            max_possible = len(self.agents) - 1
            connectivity_scores[agent_id] = degree / max_possible
        
        # Add connections for poorly connected agents
        poorly_connected = [aid for aid, score in connectivity_scores.items() if score < 0.1]
        
        for agent_id in poorly_connected:
            # Find beneficial connections
            beneficial_targets = self._find_beneficial_connections(agent_id)
            
            for target_id in beneficial_targets[:3]:  # Add up to 3 connections
                if not self.network_graph.has_edge(agent_id, target_id):
                    self._add_connection(agent_id, target_id, ConnectionType.DYNAMIC)
        
        # Remove redundant connections
        self._remove_redundant_connections()
    
    def _find_beneficial_connections(self, agent_id: str) -> List[str]:
        """Find beneficial connection targets for an agent"""
        agent = self.agents[agent_id]
        current_connections = set(self.network_graph.neighbors(agent_id))
        
        # Score potential connections
        potential_targets = []
        for target_id, target_agent in self.agents.items():
            if target_id != agent_id and target_id not in current_connections:
                score = self._calculate_connection_benefit(agent, target_agent)
                potential_targets.append((target_id, score))
        
        # Sort by benefit score
        potential_targets.sort(key=lambda x: x[1], reverse=True)
        
        return [target_id for target_id, score in potential_targets]
    
    def _calculate_connection_benefit(self, agent1: AdvancedAgent, agent2: AdvancedAgent) -> float:
        """Calculate benefit score for potential connection"""
        benefit = 0.0
        
        # Capability complementarity
        capability_score = self._calculate_agent_compatibility(agent1, agent2)
        benefit += capability_score * 0.4
        
        # Load balance potential
        load_balance = 1.0 - abs(agent1.load.queue_size - agent2.load.queue_size) / max(1, agent1.load.queue_size + agent2.load.queue_size)
        benefit += load_balance * 0.3
        
        # Expertise complementarity
        expertise_diff = abs(agent1.specialization.expertise_level - agent2.specialization.expertise_level)
        expertise_complementarity = 1.0 - expertise_diff
        benefit += expertise_complementarity * 0.2
        
        # Network position benefit
        current_degree1 = self.network_graph.degree(agent1.id)
        current_degree2 = self.network_graph.degree(agent2.id)
        
        # Prefer connecting to well-connected agents
        degree_benefit = min(current_degree1, current_degree2) / len(self.agents)
        benefit += degree_benefit * 0.1
        
        return benefit
    
    def _remove_redundant_connections(self):
        """Remove redundant connections to optimize network"""
        edges_to_remove = []
        
        for source, connections in self.connection_matrix.items():
            for target, connection in connections.items():
                # Check if connection is redundant
                if self._is_connection_redundant(source, target):
                    edges_to_remove.append((source, target))
        
        # Remove redundant connections
        for source, target in edges_to_remove:
            self._remove_connection(source, target)
        
        if edges_to_remove:
            self.logger.info(f"Removed {len(edges_to_remove)} redundant connections")
    
    def _is_connection_redundant(self, source: str, target: str) -> bool:
        """Check if a connection is redundant"""
        # Check if there are alternative paths
        if self.network_graph.has_edge(source, target):
            # Temporarily remove edge
            self.network_graph.remove_edge(source, target)
            
            # Check if still connected
            still_connected = nx.has_path(self.network_graph, source, target)
            
            # Restore edge
            self.network_graph.add_edge(source, target)
            
            # Consider redundant if alternative paths exist and connection is weak
            connection = self.connection_matrix[source][target]
            return still_connected and connection.weight < 0.3
        
        return False
    
    def _remove_connection(self, source: str, target: str):
        """Remove a connection between two agents"""
        # Remove from agents
        if source in self.agents and target in self.agents:
            source_agent = self.agents[source]
            target_agent = self.agents[target]
            
            # Remove from agent connection lists
            source_agent.connections = [conn for conn in source_agent.connections 
                                       if conn.target_id != target]
            target_agent.connections = [conn for conn in target_agent.connections 
                                       if conn.target_id != source]
        
        # Remove from network graph
        if self.network_graph.has_edge(source, target):
            self.network_graph.remove_edge(source, target)
        
        # Remove from connection matrix
        if source in self.connection_matrix and target in self.connection_matrix[source]:
            del self.connection_matrix[source][target]
        if target in self.connection_matrix and source in self.connection_matrix[target]:
            del self.connection_matrix[target][source]
    
    def _balance_network_load(self):
        """Balance load across the network"""
        # Calculate load distribution
        load_scores = {}
        for agent_id, agent in self.agents.items():
            load_scores[agent_id] = agent.load.queue_size
        
        # Identify overloaded and underloaded agents
        avg_load = sum(load_scores.values()) / len(load_scores)
        overloaded = [aid for aid, load in load_scores.items() if load > avg_load * 1.5]
        underloaded = [aid for aid, load in load_scores.items() if load < avg_load * 0.5]
        
        # Redistribute connections
        for overloaded_id in overloaded:
            # Find connections to underloaded agents
            for underloaded_id in underloaded:
                if not self.network_graph.has_edge(overloaded_id, underloaded_id):
                    # Add connection to balance load
                    self._add_connection(overloaded_id, underloaded_id, ConnectionType.DYNAMIC)
    
    def _calculate_improvements(self) -> Dict[str, float]:
        """Calculate improvements from optimization"""
        if len(self.optimization_history) < 2:
            return {}
        
        current_metrics = self.optimization_history[-1]["metrics"]
        previous_metrics = self.optimization_history[-2]["metrics"]
        
        improvements = {}
        for key in current_metrics:
            if isinstance(current_metrics[key], (int, float)) and key in previous_metrics:
                if previous_metrics[key] != 0:
                    improvement = (current_metrics[key] - previous_metrics[key]) / previous_metrics[key]
                    improvements[key] = improvement
        
        return improvements
    
    def route_signal(self, signal: Signal) -> List[str]:
        """Route signal through optimal path in network"""
        if signal.target_agent_id:
            # Direct routing
            return [signal.target_agent_id]
        
        # Find optimal path based on signal characteristics
        optimal_path = self._find_optimal_path(signal)
        
        # Update routing statistics
        for i in range(len(optimal_path) - 1):
            source = optimal_path[i]
            target = optimal_path[i + 1]
            routing_key = f"{source}->{target}"
            self.signal_routing_stats[routing_key] += 1
        
        return optimal_path
    
    def _find_optimal_path(self, signal: Signal) -> List[str]:
        """Find optimal path for signal routing"""
        # Select source agent based on signal content
        source_agent = self._select_source_agent(signal)
        
        # Select target agents based on signal type and content
        target_agents = self._select_target_agents(signal)
        
        if not target_agents:
            return [source_agent]
        
        # Find optimal path to each target
        optimal_paths = []
        for target_agent in target_agents:
            try:
                # Calculate path weights based on signal requirements
                path_weights = {}
                for edge in self.network_graph.edges(data=True):
                    source, target, data = edge
                    weight = self._calculate_path_weight(data, signal)
                    path_weights[(source, target)] = weight
                    path_weights[(target, source)] = weight
                
                # Find shortest path
                path = nx.shortest_path(
                    self.network_graph, 
                    source_agent, 
                    target_agent, 
                    weight=path_weights
                )
                optimal_paths.append(path)
            except nx.NetworkXNoPath:
                # No path found, use direct broadcast
                optimal_paths.append([source_agent, target_agent])
        
        # Return the shortest path
        if optimal_paths:
            return min(optimal_paths, key=len)
        
        return [source_agent]
    
    def _select_source_agent(self, signal: Signal) -> str:
        """Select optimal source agent for signal"""
        # Use signal metadata or content to determine best source
        if hasattr(signal, 'metadata') and 'preferred_source' in signal.metadata:
            preferred_source = signal.metadata['preferred_source']
            if preferred_source in self.agents:
                return preferred_source
        
        # Select based on agent load and capability
        best_agent = None
        best_score = -1
        
        for agent_id, agent in self.agents.items():
            # Calculate suitability score
            load_score = 1.0 / (1.0 + agent.load.queue_size)
            expertise_score = agent.specialization.expertise_level
            
            # Check if agent can handle signal type
            capability_score = 0.0
            for cap in agent.specialization.capabilities:
                if signal.type.value in cap.input_types or "data" in cap.input_types:
                    capability_score = 1.0
                    break
            
            total_score = load_score * 0.3 + expertise_score * 0.4 + capability_score * 0.3
            
            if total_score > best_score:
                best_score = total_score
                best_agent = agent_id
        
        return best_agent or list(self.agents.keys())[0]
    
    def _select_target_agents(self, signal: Signal) -> List[str]:
        """Select target agents for signal"""
        targets = []
        
        # Select based on signal type and content
        for agent_id, agent in self.agents.items():
            # Check if agent can handle signal
            can_handle = False
            for cap in agent.specialization.capabilities:
                if signal.type.value in cap.input_types or "data" in cap.input_types:
                    can_handle = True
                    break
            
            if can_handle:
                # Score based on expertise and current load
                expertise_score = agent.specialization.expertise_level
                load_score = 1.0 / (1.0 + agent.load.queue_size)
                
                total_score = expertise_score * 0.7 + load_score * 0.3
                
                # Select if score is above threshold
                if total_score > 0.6:
                    targets.append(agent_id)
        
        return targets[:5]  # Limit to top 5 targets
    
    def _calculate_path_weight(self, edge_data: Dict[str, Any], signal: Signal) -> float:
        """Calculate weight for path finding"""
        base_weight = edge_data.get("weight", 1.0)
        delay = edge_data.get("delay", 0.01)
        
        # Adjust for signal priority
        priority_factor = 1.0
        if hasattr(signal, 'strength'):
            priority_factor = 2.0 - signal.strength  # Higher strength = lower weight
        
        # Adjust for latency sensitivity
        latency_factor = 1.0 + delay * 10  # Higher delay = higher weight
        
        return base_weight * priority_factor * latency_factor
    
    def adapt_to_conditions(self):
        """Adapt network topology to changing conditions"""
        self.logger.info("Adapting network topology to conditions")
        
        # Monitor network conditions
        self._monitor_network_conditions()
        
        # Apply adaptive strategies
        self._adapt_connection_weights()
        self._adapt_network_structure()
        self._adapt_routing_strategies()
        
        # Record adaptation
        adaptation_event = {
            "timestamp": time.time(),
            "trigger": "condition_monitoring",
            "changes": self._calculate_adaptation_changes()
        }
        
        self.adaptation_events.append(adaptation_event)
        
        self.logger.info("Network topology adaptation completed")
    
    def _monitor_network_conditions(self):
        """Monitor current network conditions"""
        # Monitor latency
        current_latencies = []
        for source, connections in self.connection_matrix.items():
            for target, connection in connections.items():
                current_latencies.append(connection.delay)
        
        if current_latencies:
            avg_latency = sum(current_latencies) / len(current_latencies)
            self.latency_measurements.append(avg_latency)
        
        # Monitor bandwidth usage
        current_bandwidth = self._calculate_bandwidth_utilization()
        
        # Monitor connection failures
        failure_rates = self._calculate_connection_failure_rates()
        avg_failure_rate = sum(failure_rates.values()) / len(failure_rates) if failure_rates else 0.0
        
        # Log conditions if thresholds are exceeded
        if avg_latency > self.adaptation_thresholds["high_latency"]:
            self.logger.warning(f"High average latency: {avg_latency:.4f}")
        
        if current_bandwidth > self.adaptation_thresholds["low_bandwidth"]:
            self.logger.warning(f"High bandwidth utilization: {current_bandwidth:.3f}")
        
        if avg_failure_rate > self.adaptation_thresholds["connection_failure"]:
            self.logger.warning(f"High connection failure rate: {avg_failure_rate:.3f}")
    
    def _adapt_connection_weights(self):
        """Adapt connection weights based on current conditions"""
        for source, connections in self.connection_matrix.items():
            for target, connection in connections.items():
                # Calculate adaptation factor
                adaptation_factor = self._calculate_adaptation_factor(connection)
                
                # Apply adaptation
                new_weight = connection.weight * (1.0 + adaptation_factor)
                connection.weight = max(0.1, min(1.0, new_weight))
                
                # Update network graph
                if self.network_graph.has_edge(source, target):
                    self.network_graph[source][target]["weight"] = connection.weight
    
    def _calculate_adaptation_factor(self, connection: Connection) -> float:
        """Calculate adaptation factor for connection"""
        adaptation_factor = 0.0
        
        # Adapt based on latency
        if connection.delay > self.adaptation_thresholds["high_latency"]:
            adaptation_factor -= 0.1  # Weaken high-latency connections
        
        # Adapt based on usage
        usage_key = f"{connection.source_id}->{connection.target_id}"
        usage_count = self.signal_routing_stats.get(usage_key, 0)
        
        if usage_count > 10:  # High usage
            adaptation_factor += 0.05  # Strengthen frequently used connections
        
        # Adapt based on failure rate
        failed_count = self.signal_routing_stats.get(f"{usage_key}->failed", 0)
        if usage_count > 0:
            failure_rate = failed_count / usage_count
            if failure_rate > self.adaptation_thresholds["connection_failure"]:
                adaptation_factor -= 0.15  # Weaken failing connections
        
        return adaptation_factor * self.config.adaptation_rate
    
    def _adapt_network_structure(self):
        """Adapt network structure based on conditions"""
        # Add connections for high-load agents
        high_load_agents = [
            agent_id for agent_id, agent in self.agents.items()
            if agent.load.queue_size > 15
        ]
        
        for agent_id in high_load_agents:
            # Find additional connections to distribute load
            additional_targets = self._find_load_distribution_targets(agent_id)
            
            for target_id in additional_targets[:2]:  # Add up to 2 connections
                if not self.network_graph.has_edge(agent_id, target_id):
                    self._add_connection(agent_id, target_id, ConnectionType.DYNAMIC)
        
        # Remove connections for consistently failing paths
        failing_connections = self._identify_failing_connections()
        
        for source, target in failing_connections:
            self._remove_connection(source, target)
    
    def _find_load_distribution_targets(self, agent_id: str) -> List[str]:
        """Find targets for load distribution"""
        agent = self.agents[agent_id]
        current_connections = set(self.network_graph.neighbors(agent_id))
        
        # Find underloaded agents with compatible capabilities
        potential_targets = []
        for target_id, target_agent in self.agents.items():
            if (target_id != agent_id and 
                target_id not in current_connections and
                target_agent.load.queue_size < 5):  # Underloaded
                
                # Check capability compatibility
                compatibility = self._calculate_agent_compatibility(agent, target_agent)
                if compatibility > 0.3:  # Minimum compatibility threshold
                    potential_targets.append((target_id, compatibility))
        
        # Sort by compatibility
        potential_targets.sort(key=lambda x: x[1], reverse=True)
        
        return [target_id for target_id, compatibility in potential_targets]
    
    def _identify_failing_connections(self) -> List[Tuple[str, str]]:
        """Identify consistently failing connections"""
        failing_connections = []
        
        for source, connections in self.connection_matrix.items():
            for target, connection in connections.items():
                usage_key = f"{source}->{target}"
                total_signals = self.signal_routing_stats.get(usage_key, 0)
                failed_signals = self.signal_routing_stats.get(f"{usage_key}->failed", 0)
                
                if total_signals > 5:  # Minimum usage threshold
                    failure_rate = failed_signals / total_signals
                    if failure_rate > self.adaptation_thresholds["connection_failure"] * 2:
                        failing_connections.append((source, target))
        
        return failing_connections
    
    def _adapt_routing_strategies(self):
        """Adapt routing strategies based on conditions"""
        # Update routing preferences based on current conditions
        current_avg_latency = sum(self.latency_measurements) / len(self.latency_measurements) if self.latency_measurements else 0.01
        
        # Adjust adaptation thresholds based on conditions
        if current_avg_latency > 0.05:  # High latency
            self.adaptation_thresholds["high_latency"] *= 1.1  # Increase threshold
        else:
            self.adaptation_thresholds["high_latency"] *= 0.9  # Decrease threshold
        
        # Ensure thresholds stay within reasonable bounds
        self.adaptation_thresholds["high_latency"] = max(0.01, min(0.5, self.adaptation_thresholds["high_latency"]))
    
    def _calculate_adaptation_changes(self) -> Dict[str, Any]:
        """Calculate changes made during adaptation"""
        changes = {
            "connections_added": 0,
            "connections_removed": 0,
            "weights_adjusted": 0,
            "thresholds_updated": []
        }
        
        # Count changes (this would be tracked during actual adaptation)
        for adaptation_event in list(self.adaptation_events)[-10:]:  # Last 10 events
            if "changes" in adaptation_event:
                changes["connections_added"] += adaptation_event["changes"].get("connections_added", 0)
                changes["connections_removed"] += adaptation_event["changes"].get("connections_removed", 0)
                changes["weights_adjusted"] += adaptation_event["changes"].get("weights_adjusted", 0)
        
        return changes
    
    def get_topology_status(self) -> Dict[str, Any]:
        """Get comprehensive topology status"""
        return {
            "topology_type": self.config.topology_type.value,
            "total_agents": len(self.agents),
            "total_connections": self.topology_metrics.total_connections,
            "network_density": self.topology_metrics.network_density,
            "average_path_length": self.topology_metrics.average_path_length,
            "clustering_coefficient": self.topology_metrics.clustering_coefficient,
            "efficiency_score": self.topology_metrics.efficiency_score,
            "robustness_score": self.topology_metrics.robustness_score,
            "bandwidth_utilization": self.topology_metrics.bandwidth_utilization,
            "latency_distribution": self.topology_metrics.latency_distribution,
            "optimization_count": len(self.optimization_history),
            "adaptation_count": len(self.adaptation_events)
        }
    
    def get_network_visualization_data(self) -> Dict[str, Any]:
        """Get data for network visualization"""
        nodes = []
        edges = []
        
        # Node data
        for agent_id, agent in self.agents.items():
            node_data = {
                "id": agent_id,
                "domain": agent.specialization.domain.value,
                "expertise_level": agent.specialization.expertise_level,
                "load": agent.load.queue_size,
                "state": agent.state.value,
                "centrality": self.topology_metrics.centrality_scores.get(agent_id, 0.0),
                "capabilities": [cap.name for cap in agent.specialization.capabilities[:3]]  # Top 3 capabilities
            }
            nodes.append(node_data)
        
        # Edge data
        for source, connections in self.connection_matrix.items():
            for target, connection in connections.items():
                edge_data = {
                    "source": source,
                    "target": target,
                    "weight": connection.weight,
                    "type": connection.connection_type,
                    "delay": connection.delay
                }
                edges.append(edge_data)
        
        return {
            "nodes": nodes,
            "edges": edges,
            "metrics": self.topology_metrics.__dict__
        }

# Global topology manager instance
advanced_topology_manager = None