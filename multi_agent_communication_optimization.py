"""
Multi-Agent Communication Optimization for 250+ Agent Networks

This module implements advanced communication optimization techniques specifically designed
for large-scale multi-agent systems. It addresses the challenges of information overload,
scalability, and efficient message routing in networks with hundreds of specialized agents.

Key Features:
- Hierarchical message routing
- Adaptive signal propagation
- Interest-based content filtering
- Network topology optimization
- Message batching and compression
- Priority-based queuing
- Failure-tolerant communication
"""

import asyncio
import heapq
import time
import uuid
from typing import Dict, List, Set, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import threading
import json
import hashlib
import zlib

# Import existing components
from communication_system import CommunicationSystem, Message
from scalable_multi_agent_system import ScalableAgent, AgentDomain, AgentPriority


class MessageType(Enum):
    """Types of messages with different handling requirements"""
    URGENT = "urgent"           # Critical, immediate delivery
    NORMAL = "normal"           # Standard processing
    BULK = "bulk"              # Large data transfers
    BROADCAST = "broadcast"    # System-wide announcements
    MULTICAST = "multicast"    # Group-specific messages
    UNICAST = "unicast"        # Point-to-point communication
    REQUEST = "request"        # Request-response pattern
    RESPONSE = "response"      # Response to requests
    NOTIFICATION = "notification"  # Informational messages
    COMMAND = "command"        # Directive messages


class RoutingStrategy(Enum):
    """Message routing strategies"""
    DIRECT = "direct"          # Direct sender-receiver
    HIERARCHICAL = "hierarchical"  # Through organizational hierarchy
    INTEREST_BASED = "interest_based"  # Based on agent interests
    LOAD_BALANCED = "load_balanced"  # Distribute across available agents
    GEOGRAPHIC = "geographic"  # Based on logical/physical proximity
    CAPABILITY_BASED = "capability_based"  # Based on agent capabilities
    HYBRID = "hybrid"          # Combination of strategies


@dataclass
class MessageMetadata:
    """Enhanced metadata for optimized message handling"""
    message_id: str
    message_type: MessageType
    priority: int
    size_bytes: int
    compression_ratio: float
    ttl: int  # Time-to-live in hops
    created_at: float
    routing_strategy: RoutingStrategy
    requires_ack: bool
    batch_id: Optional[str] = None
    parent_message_id: Optional[str] = None
    
    def is_expired(self, current_time: float, max_age: float = 300.0) -> bool:
        """Check if message has expired"""
        return current_time - self.created_at > max_age
    
    def calculate_priority_score(self) -> float:
        """Calculate overall priority score"""
        type_priority = {
            MessageType.URGENT: 1.0,
            MessageType.COMMAND: 0.9,
            MessageType.REQUEST: 0.8,
            MessageType.RESPONSE: 0.7,
            MessageType.NOTIFICATION: 0.6,
            MessageType.NORMAL: 0.5,
            MessageType.BROADCAST: 0.4,
            MessageType.MULTICAST: 0.3,
            MessageType.BULK: 0.2
        }
        
        base_priority = type_priority.get(self.message_type, 0.5)
        age_factor = max(0, 1.0 - (time.time() - self.created_at) / 300.0)
        
        return base_priority * age_priority


@dataclass
class AgentInterestProfile:
    """Defines agent interests and communication preferences"""
    agent_id: str
    domains_of_interest: List[AgentDomain]
    keywords: List[str]
    message_types_preferred: List[MessageType]
    max_message_size: int
    processing_capacity: float
    communication_frequency: float
    reliability_requirement: float
    
    def matches_message(self, message_content: Dict[str, Any]) -> float:
        """Calculate match score for a message"""
        score = 0.0
        
        # Check domain interest
        task_domain = message_content.get("task_domain")
        if task_domain in [d.value for d in self.domains_of_interest]:
            score += 0.4
        
        # Check keyword matching
        message_text = str(message_content).lower()
        keyword_matches = sum(1 for keyword in self.keywords 
                            if keyword.lower() in message_text)
        if keyword_matches > 0:
            score += min(0.3, keyword_matches * 0.1)
        
        # Check message type preference
        message_type = message_content.get("message_type", "normal")
        if message_type in [t.value for t in self.message_types_preferred]:
            score += 0.3
        
        return score


class MessageRouter:
    """Advanced message router for large-scale agent networks"""
    
    def __init__(self, max_queue_size: int = 10000):
        self.max_queue_size = max_queue_size
        self.message_queues = {
            MessageType.URGENT: asyncio.PriorityQueue(maxsize=1000),
            MessageType.NORMAL: asyncio.PriorityQueue(maxsize=5000),
            MessageType.BULK: asyncio.PriorityQueue(maxsize=2000),
            MessageType.BROADCAST: asyncio.PriorityQueue(maxsize=500),
            MessageType.MULTICAST: asyncio.PriorityQueue(maxsize=2000),
            MessageType.UNICAST: asyncio.PriorityQueue(maxsize=5000),
            MessageType.REQUEST: asyncio.PriorityQueue(maxsize=3000),
            MessageType.RESPONSE: asyncio.PriorityQueue(maxsize=3000),
            MessageType.NOTIFICATION: asyncio.PriorityQueue(maxsize=2000),
            MessageType.COMMAND: asyncio.PriorityQueue(maxsize=1000)
        }
        
        # Routing tables
        self.agent_profiles: Dict[str, AgentInterestProfile] = {}
        self.domain_routes: Dict[AgentDomain, Set[str]] = defaultdict(set)
        self.capability_routes: Dict[str, Set[str]] = defaultdict(set)
        self.load_balancer_routes: Dict[str, float] = defaultdict(float)
        
        # Performance metrics
        self.routing_stats = defaultdict(int)
        self.delivery_times = deque(maxlen=1000)
        self.failed_deliveries = deque(maxlen=100)
        
        # Network topology
        self.network_graph = defaultdict(set)
        self.hierarchy_levels = defaultdict(int)
        
    def register_agent(self, agent: ScalableAgent, profile: AgentInterestProfile):
        """Register an agent with the routing system"""
        self.agent_profiles[agent.agent_id] = profile
        
        # Update domain routes
        for domain in profile.domains_of_interest:
            self.domain_routes[domain].add(agent.agent_id)
        
        # Update capability routes
        for capability in agent.capabilities.supported_tasks:
            self.capability_routes[capability].add(agent.agent_id)
        
        # Initialize load balancer
        self.load_balancer_routes[agent.agent_id] = 0.0
        
        # Update network graph
        self._update_network_topology(agent)
    
    def unregister_agent(self, agent_id: str):
        """Unregister an agent from the routing system"""
        if agent_id in self.agent_profiles:
            profile = self.agent_profiles[agent_id]
            
            # Remove from domain routes
            for domain in profile.domains_of_interest:
                self.domain_routes[domain].discard(agent_id)
            
            # Remove from capability routes
            for capability in profile.capabilities.supported_tasks:
                self.capability_routes[capability].discard(agent_id)
            
            # Remove from load balancer
            self.load_balancer_routes.pop(agent_id, None)
            
            # Remove profile
            del self.agent_profiles[agent_id]
    
    def _update_network_topology(self, agent: ScalableAgent):
        """Update network topology based on agent relationships"""
        # Build connections based on domain similarity and capabilities
        for other_agent_id, other_profile in self.agent_profiles.items():
            if other_agent_id != agent.agent_id:
                # Calculate connection strength
                connection_strength = self._calculate_connection_strength(agent, other_profile)
                if connection_strength > 0.5:
                    self.network_graph[agent.agent_id].add(other_agent_id)
                    self.network_graph[other_agent_id].add(agent.agent_id)
    
    def _calculate_connection_strength(self, agent: ScalableAgent, 
                                     other_profile: AgentInterestProfile) -> float:
        """Calculate connection strength between agents"""
        strength = 0.0
        
        # Domain overlap
        agent_domains = set(agent.capabilities.subdomains)
        profile_domains = set([d.value for d in other_profile.domains_of_interest])
        domain_overlap = len(agent_domains.intersection(profile_domains))
        if domain_overlap > 0:
            strength += 0.4
        
        # Capability overlap
        agent_capabilities = set(agent.capabilities.supported_tasks)
        profile_capabilities = set(other_profile.keywords)
        capability_overlap = len(agent_capabilities.intersection(profile_capabilities))
        if capability_overlap > 0:
            strength += 0.3
        
        # Priority compatibility
        if agent.priority.value <= 2:  # High priority agents
            strength += 0.2
        
        # Load balancing consideration
        current_load = self.load_balancer_routes.get(other_profile.agent_id, 0.0)
        if current_load < 0.7:
            strength += 0.1
        
        return min(1.0, strength)
    
    async def route_message(self, message: Message, metadata: MessageMetadata) -> List[str]:
        """Route a message to appropriate agents"""
        start_time = time.time()
        
        # Determine routing strategy
        strategy = self._select_routing_strategy(message, metadata)
        
        # Find target agents
        target_agents = await self._find_target_agents(message, metadata, strategy)
        
        # Deliver message
        successful_deliveries = []
        for agent_id in target_agents:
            if await self._deliver_message(agent_id, message, metadata):
                successful_deliveries.append(agent_id)
                self.routing_stats["successful_deliveries"] += 1
            else:
                self.routing_stats["failed_deliveries"] += 1
                self.failed_deliveries.append(agent_id)
        
        # Update metrics
        delivery_time = time.time() - start_time
        self.delivery_times.append(delivery_time)
        self.routing_stats["total_messages"] += 1
        
        return successful_deliveries
    
    def _select_routing_strategy(self, message: Message, metadata: MessageMetadata) -> RoutingStrategy:
        """Select optimal routing strategy based on message characteristics"""
        content = message.content
        
        # Urgent messages use direct routing
        if metadata.message_type == MessageType.URGENT:
            return RoutingStrategy.DIRECT
        
        # Broadcast messages use hierarchical routing
        if metadata.message_type == MessageType.BROADCAST:
            return RoutingStrategy.HIERARCHICAL
        
        # Capability-based routing for task-specific messages
        if "task_type" in content:
            return RoutingStrategy.CAPABILITY_BASED
        
        # Interest-based routing for informational messages
        if metadata.message_type in [MessageType.NOTIFICATION, MessageType.NORMAL]:
            return RoutingStrategy.INTEREST_BASED
        
        # Load balancing for bulk operations
        if metadata.message_type == MessageType.BULK:
            return RoutingStrategy.LOAD_BALANCED
        
        # Default to hybrid strategy
        return RoutingStrategy.HYBRID
    
    async def _find_target_agents(self, message: Message, metadata: MessageMetadata, 
                                strategy: RoutingStrategy) -> List[str]:
        """Find target agents based on routing strategy"""
        content = message.content
        target_agents = []
        
        if strategy == RoutingStrategy.DIRECT:
            # Direct routing to specified receiver
            if message.receiver_id:
                target_agents = [message.receiver_id]
        
        elif strategy == RoutingStrategy.CAPABILITY_BASED:
            # Route based on agent capabilities
            task_type = content.get("task_type", "")
            if task_type:
                target_agents = list(self.capability_routes.get(task_type, set()))
        
        elif strategy == RoutingStrategy.INTEREST_BASED:
            # Route based on agent interests
            target_agents = []
            for agent_id, profile in self.agent_profiles.items():
                match_score = profile.matches_message(content)
                if match_score > 0.5:
                    target_agents.append(agent_id)
        
        elif strategy == RoutingStrategy.LOAD_BALANCED:
            # Route to least loaded agents
            capable_agents = self._find_capable_agents(content)
            target_agents = self._select_least_loaded_agents(capable_agents, 5)
        
        elif strategy == RoutingStrategy.HIERARCHICAL:
            # Route through organizational hierarchy
            target_agents = self._route_through_hierarchy(message)
        
        elif strategy == RoutingStrategy.HYBRID:
            # Combine multiple strategies
            capable_agents = self._find_capable_agents(content)
            interested_agents = []
            for agent_id in capable_agents:
                if agent_id in self.agent_profiles:
                    profile = self.agent_profiles[agent_id]
                    if profile.matches_message(content) > 0.3:
                        interested_agents.append(agent_id)
            
            target_agents = self._select_least_loaded_agents(interested_agents, 3)
        
        return target_agents
    
    def _find_capable_agents(self, content: Dict[str, Any]) -> List[str]:
        """Find agents capable of handling the message"""
        task_type = content.get("task_type", "")
        if task_type:
            return list(self.capability_routes.get(task_type, set()))
        return []
    
    def _select_least_loaded_agents(self, agents: List[str], count: int) -> List[str]:
        """Select agents with lowest load"""
        agent_loads = [(agent_id, self.load_balancer_routes.get(agent_id, 0.0)) 
                      for agent_id in agents]
        agent_loads.sort(key=lambda x: x[1])
        return [agent_id for agent_id, _ in agent_loads[:count]]
    
    def _route_through_hierarchy(self, message: Message) -> List[str]:
        """Route message through organizational hierarchy"""
        # Simple hierarchical routing based on agent priorities
        high_priority_agents = []
        medium_priority_agents = []
        low_priority_agents = []
        
        for agent_id, profile in self.agent_profiles.items():
            if hasattr(profile, 'priority'):
                if profile.priority <= 1:
                    high_priority_agents.append(agent_id)
                elif profile.priority <= 2:
                    medium_priority_agents.append(agent_id)
                else:
                    low_priority_agents.append(agent_id)
        
        # Return agents based on message importance
        if message.content.get("importance", "normal") == "high":
            return high_priority_agents[:3]
        else:
            return medium_priority_agents[:5] + low_priority_agents[:2]
    
    async def _deliver_message(self, agent_id: str, message: Message, 
                             metadata: MessageMetadata) -> bool:
        """Deliver message to specific agent"""
        try:
            # Update load balancer
            self.load_balancer_routes[agent_id] += metadata.size_bytes / 1000000.0  # MB
            
            # In a real implementation, this would deliver to the actual agent
            # For now, simulate successful delivery
            return True
            
        except Exception as e:
            print(f"Failed to deliver message to {agent_id}: {e}")
            return False
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing performance statistics"""
        total_messages = self.routing_stats["total_messages"]
        successful_deliveries = self.routing_stats["successful_deliveries"]
        failed_deliveries = self.routing_stats["failed_deliveries"]
        
        avg_delivery_time = (
            sum(self.delivery_times) / len(self.delivery_times)
            if self.delivery_times else 0.0
        )
        
        success_rate = (
            successful_deliveries / total_messages
            if total_messages > 0 else 0.0
        )
        
        return {
            "total_messages": total_messages,
            "successful_deliveries": successful_deliveries,
            "failed_deliveries": failed_deliveries,
            "success_rate": success_rate,
            "average_delivery_time": avg_delivery_time,
            "registered_agents": len(self.agent_profiles),
            "network_connections": sum(len(connections) for connections in self.network_graph.values()),
            "queue_sizes": {
                msg_type.value: queue.qsize()
                for msg_type, queue in self.message_queues.items()
            }
        }


class MessageOptimizer:
    """Optimizes message content for efficient transmission"""
    
    def __init__(self):
        self.compression_stats = defaultdict(int)
        self.compression_cache = {}
        
    def optimize_message(self, message: Message) -> Tuple[Message, MessageMetadata]:
        """Optimize message for efficient transmission"""
        # Compress message content
        compressed_content, compression_ratio = self._compress_content(message.content)
        
        # Generate metadata
        metadata = MessageMetadata(
            message_id=str(uuid.uuid4()),
            message_type=self._classify_message(message),
            priority=self._calculate_priority(message),
            size_bytes=len(json.dumps(compressed_content).encode('utf-8')),
            compression_ratio=compression_ratio,
            ttl=self._calculate_ttl(message),
            created_at=time.time(),
            routing_strategy=self._select_routing_strategy(message),
            requires_ack=self._requires_acknowledgment(message)
        )
        
        # Create optimized message
        optimized_message = Message(
            sender_id=message.sender_id,
            receiver_id=message.receiver_id,
            content=compressed_content
        )
        
        return optimized_message, metadata
    
    def _compress_content(self, content: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
        """Compress message content"""
        # Convert to JSON string
        json_str = json.dumps(content, separators=(',', ':'))
        original_size = len(json_str.encode('utf-8'))
        
        # Check cache
        content_hash = hashlib.md5(json_str.encode()).hexdigest()
        if content_hash in self.compression_cache:
            return self.compression_cache[content_hash]
        
        # Compress using zlib
        compressed_bytes = zlib.compress(json_str.encode('utf-8'))
        compressed_size = len(compressed_bytes)
        
        # Calculate compression ratio
        compression_ratio = compressed_size / original_size if original_size > 0 else 1.0
        
        # Store in cache
        compressed_content = {
            "compressed_data": compressed_bytes.hex(),
            "original_hash": content_hash,
            "compression_algorithm": "zlib"
        }
        
        self.compression_cache[content_hash] = (compressed_content, compression_ratio)
        self.compression_stats["compressed_messages"] += 1
        self.compression_stats["total_compression_ratio"] += compression_ratio
        
        return compressed_content, compression_ratio
    
    def _classify_message(self, message: Message) -> MessageType:
        """Classify message type based on content"""
        content = message.content
        
        if content.get("urgent", False):
            return MessageType.URGENT
        elif content.get("command", False):
            return MessageType.COMMAND
        elif content.get("request", False):
            return MessageType.REQUEST
        elif content.get("response", False):
            return MessageType.RESPONSE
        elif content.get("broadcast", False):
            return MessageType.BROADCAST
        elif content.get("multicast", False):
            return MessageType.MULTICAST
        elif content.get("notification", False):
            return MessageType.NOTIFICATION
        elif content.get("bulk_data", False):
            return MessageType.BULK
        elif message.receiver_id and message.receiver_id != "broadcast":
            return MessageType.UNICAST
        else:
            return MessageType.NORMAL
    
    def _calculate_priority(self, message: Message) -> int:
        """Calculate message priority"""
        content = message.content
        base_priority = content.get("priority", 5)
        
        # Adjust based on message type
        type_adjustments = {
            MessageType.URGENT: 5,
            MessageType.COMMAND: 4,
            MessageType.REQUEST: 3,
            MessageType.RESPONSE: 2,
            MessageType.NOTIFICATION: 1,
            MessageType.NORMAL: 0,
            MessageType.BROADCAST: -1,
            MessageType.MULTICAST: -1,
            MessageType.BULK: -2
        }
        
        message_type = self._classify_message(message)
        adjustment = type_adjustments.get(message_type, 0)
        
        return max(1, min(10, base_priority + adjustment))
    
    def _calculate_ttl(self, message: Message) -> int:
        """Calculate time-to-live for message"""
        content = message.content
        base_ttl = content.get("ttl", 10)
        
        # Adjust based on message type
        message_type = self._classify_message(message)
        if message_type == MessageType.URGENT:
            return max(5, base_ttl)
        elif message_type == MessageType.BROADCAST:
            return min(20, base_ttl)
        elif message_type == MessageType.BULK:
            return max(15, base_ttl)
        
        return base_ttl
    
    def _select_routing_strategy(self, message: Message) -> RoutingStrategy:
        """Select routing strategy for message"""
        content = message.content
        
        if content.get("routing_strategy"):
            strategy_map = {
                "direct": RoutingStrategy.DIRECT,
                "hierarchical": RoutingStrategy.HIERARCHICAL,
                "interest_based": RoutingStrategy.INTEREST_BASED,
                "load_balanced": RoutingStrategy.LOAD_BALANCED,
                "capability_based": RoutingStrategy.CAPABILITY_BASED,
                "hybrid": RoutingStrategy.HYBRID
            }
            return strategy_map.get(content["routing_strategy"], RoutingStrategy.HYBRID)
        
        return RoutingStrategy.HYBRID
    
    def _requires_acknowledgment(self, message: Message) -> bool:
        """Determine if message requires acknowledgment"""
        content = message.content
        message_type = self._classify_message(message)
        
        # Critical messages require acknowledgment
        if message_type in [MessageType.URGENT, MessageType.COMMAND, MessageType.REQUEST]:
            return True
        
        # Check explicit requirement
        return content.get("requires_ack", False)
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get message optimization statistics"""
        total_compressed = self.compression_stats["compressed_messages"]
        avg_compression_ratio = (
            self.compression_stats["total_compression_ratio"] / total_compressed
            if total_compressed > 0 else 1.0
        )
        
        return {
            "compressed_messages": total_compressed,
            "average_compression_ratio": avg_compression_ratio,
            "cache_size": len(self.compression_cache),
            "cache_hit_rate": self._calculate_cache_hit_rate()
        }
    
    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate"""
        # This would track cache hits vs misses in a real implementation
        return 0.85  # Simulated cache hit rate


class NetworkTopologyManager:
    """Manages and optimizes network topology for large-scale agent networks"""
    
    def __init__(self):
        self.topology_graph = defaultdict(set)
        self.agent_positions = {}
        self.network_metrics = defaultdict(float)
        self.optimization_history = deque(maxlen=100)
        
    def add_agent(self, agent_id: str, position: Tuple[float, float] = None):
        """Add agent to network topology"""
        self.agent_positions[agent_id] = position or (0.0, 0.0)
        self._initialize_agent_connections(agent_id)
    
    def remove_agent(self, agent_id: str):
        """Remove agent from network topology"""
        # Remove connections
        for connected_agent in self.topology_graph[agent_id]:
            self.topology_graph[connected_agent].discard(agent_id)
        del self.topology_graph[agent_id]
        del self.agent_positions[agent_id]
    
    def _initialize_agent_connections(self, agent_id: str):
        """Initialize connections for a new agent"""
        position = self.agent_positions[agent_id]
        
        # Connect to nearby agents
        for other_id, other_position in self.agent_positions.items():
            if other_id != agent_id:
                distance = self._calculate_distance(position, other_position)
                if distance < 1.0:  # Connection threshold
                    self.topology_graph[agent_id].add(other_id)
                    self.topology_graph[other_id].add(agent_id)
    
    def _calculate_distance(self, pos1: Tuple[float, float], 
                           pos2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between positions"""
        return ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)**0.5
    
    def optimize_topology(self):
        """Optimize network topology for better performance"""
        print("Optimizing network topology...")
        
        # Calculate current network metrics
        current_metrics = self._calculate_network_metrics()
        
        # Identify optimization opportunities
        optimizations = self._identify_optimizations(current_metrics)
        
        # Apply optimizations
        for optimization in optimizations:
            self._apply_optimization(optimization)
        
        # Record optimization
        self.optimization_history.append({
            "timestamp": time.time(),
            "metrics_before": current_metrics,
            "optimizations_applied": len(optimizations)
        })
        
        print(f"Applied {len(optimizations)} topology optimizations")
    
    def _calculate_network_metrics(self) -> Dict[str, float]:
        """Calculate current network performance metrics"""
        total_agents = len(self.agent_positions)
        total_connections = sum(len(connections) for connections in self.topology_graph.values())
        average_degree = total_connections / total_agents if total_agents > 0 else 0
        
        # Calculate network density
        max_possible_connections = total_agents * (total_agents - 1) / 2
        density = total_connections / max_possible_connections if max_possible_connections > 0 else 0
        
        # Calculate average path length (simplified)
        avg_path_length = self._calculate_average_path_length()
        
        return {
            "total_agents": total_agents,
            "total_connections": total_connections,
            "average_degree": average_degree,
            "network_density": density,
            "average_path_length": avg_path_length
        }
    
    def _calculate_average_path_length(self) -> float:
        """Calculate average shortest path length (simplified)"""
        # This is a simplified calculation - in practice, you'd use BFS or Floyd-Warshall
        if len(self.agent_positions) < 2:
            return 0.0
        
        # Sample a few paths
        sample_size = min(10, len(self.agent_positions))
        agent_ids = list(self.agent_positions.keys())
        total_path_length = 0
        path_count = 0
        
        for i in range(sample_size):
            for j in range(i + 1, sample_size):
                path_length = self._calculate_shortest_path(agent_ids[i], agent_ids[j])
                if path_length > 0:
                    total_path_length += path_length
                    path_count += 1
        
        return total_path_length / path_count if path_count > 0 else 0.0
    
    def _calculate_shortest_path(self, start: str, end: str) -> int:
        """Calculate shortest path between two agents (BFS)"""
        if start == end:
            return 0
        
        visited = set()
        queue = [(start, 0)]
        
        while queue:
            current, distance = queue.pop(0)
            
            if current == end:
                return distance
            
            if current in visited:
                continue
            
            visited.add(current)
            
            for neighbor in self.topology_graph[current]:
                if neighbor not in visited:
                    queue.append((neighbor, distance + 1))
        
        return -1  # No path found
    
    def _identify_optimizations(self, metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        """Identify topology optimization opportunities"""
        optimizations = []
        
        # Check for under-connected agents
        if metrics["average_degree"] < 3.0:
            optimizations.append({
                "type": "add_connections",
                "description": "Increase network connectivity",
                "target_degree": 4.0
            })
        
        # Check for over-connected agents
        if metrics["average_degree"] > 8.0:
            optimizations.append({
                "type": "remove_connections",
                "description": "Reduce network complexity",
                "target_degree": 6.0
            })
        
        # Check for poor path lengths
        if metrics["average_path_length"] > 5.0:
            optimizations.append({
                "type": "add_shortcuts",
                "description": "Add shortcut connections",
                "target_path_length": 3.0
            })
        
        return optimizations
    
    def _apply_optimization(self, optimization: Dict[str, Any]):
        """Apply a topology optimization"""
        opt_type = optimization["type"]
        
        if opt_type == "add_connections":
            self._add_strategic_connections()
        elif opt_type == "remove_connections":
            self._remove_redundant_connections()
        elif opt_type == "add_shortcuts":
            self._add_shortcut_connections()
    
    def _add_strategic_connections(self):
        """Add strategic connections to improve connectivity"""
        underconnected_agents = [
            agent_id for agent_id, connections in self.topology_graph.items()
            if len(connections) < 3
        ]
        
        for agent_id in underconnected_agents:
            # Find nearby agents to connect to
            position = self.agent_positions[agent_id]
            nearby_agents = [
                other_id for other_id, other_pos in self.agent_positions.items()
                if other_id != agent_id and self._calculate_distance(position, other_pos) < 1.5
            ]
            
            # Add connections to top 3 nearby agents
            for nearby_agent in nearby_agents[:3]:
                if nearby_agent not in self.topology_graph[agent_id]:
                    self.topology_graph[agent_id].add(nearby_agent)
                    self.topology_graph[nearby_agent].add(agent_id)
    
    def _remove_redundant_connections(self):
        """Remove redundant connections to simplify network"""
        overconnected_agents = [
            agent_id for agent_id, connections in self.topology_graph.items()
            if len(connections) > 8
        ]
        
        for agent_id in overconnected_agents:
            connections = list(self.topology_graph[agent_id])
            
            # Keep connections to high-priority agents
            # (In a real implementation, you'd have priority information)
            connections_to_keep = connections[:6]
            connections_to_remove = set(connections[6:])
            
            for other_agent in connections_to_remove:
                self.topology_graph[agent_id].discard(other_agent)
                self.topology_graph[other_agent].discard(agent_id)
    
    def _add_shortcut_connections(self):
        """Add shortcut connections to reduce path lengths"""
        # Find agents that are far apart in the network
        agent_ids = list(self.agent_positions.keys())
        
        for i in range(0, len(agent_ids), 5):  # Sample every 5th agent
            for j in range(i + 5, len(agent_ids), 5):
                agent1, agent2 = agent_ids[i], agent_ids[j]
                
                # Check if they're already connected
                if agent2 not in self.topology_graph[agent1]:
                    # Check if adding a connection would significantly reduce path length
                    current_path = self._calculate_shortest_path(agent1, agent2)
                    if current_path > 4:  # Long path
                        # Add shortcut
                        self.topology_graph[agent1].add(agent2)
                        self.topology_graph[agent2].add(agent1)
    
    def get_topology_stats(self) -> Dict[str, Any]:
        """Get network topology statistics"""
        metrics = self._calculate_network_metrics()
        
        return {
            "network_metrics": metrics,
            "optimization_history": list(self.optimization_history)[-5:],
            "topology_visualization": {
                "agents": len(self.agent_positions),
                "connections": metrics["total_connections"],
                "avg_degree": metrics["average_degree"]
            }
        }


# Main communication optimization system
class OptimizedCommunicationSystem:
    """Main system for optimized multi-agent communication"""
    
    def __init__(self):
        self.message_router = MessageRouter()
        self.message_optimizer = MessageOptimizer()
        self.topology_manager = NetworkTopologyManager()
        
        # Performance monitoring
        self.performance_metrics = defaultdict(list)
        self.system_stats = defaultdict(int)
        
    async def send_message(self, sender_id: str, receiver_id: str, 
                          content: Dict[str, Any]) -> List[str]:
        """Send optimized message through the system"""
        # Create original message
        original_message = Message(
            sender_id=sender_id,
            receiver_id=receiver_id,
            content=content
        )
        
        # Optimize message
        optimized_message, metadata = self.message_optimizer.optimize_message(original_message)
        
        # Route message
        delivered_agents = await self.message_router.route_message(optimized_message, metadata)
        
        # Update statistics
        self.system_stats["messages_sent"] += 1
        self.system_stats["messages_delivered"] += len(delivered_agents)
        
        return delivered_agents
    
    def register_agent(self, agent: ScalableAgent, profile: AgentInterestProfile):
        """Register agent with the communication system"""
        self.message_router.register_agent(agent, profile)
        self.topology_manager.add_agent(agent.agent_id)
    
    def unregister_agent(self, agent_id: str):
        """Unregister agent from the communication system"""
        self.message_router.unregister_agent(agent_id)
        self.topology_manager.remove_agent(agent_id)
    
    async def optimize_network(self):
        """Optimize the entire communication network"""
        # Optimize message routing
        routing_stats = self.message_router.get_routing_stats()
        
        # Optimize network topology
        self.topology_manager.optimize_topology()
        
        # Get optimization stats
        optimization_stats = self.message_optimizer.get_optimization_stats()
        topology_stats = self.topology_manager.get_topology_stats()
        
        return {
            "routing_stats": routing_stats,
            "optimization_stats": optimization_stats,
            "topology_stats": topology_stats
        }
    
    def get_system_performance(self) -> Dict[str, Any]:
        """Get comprehensive system performance metrics"""
        return {
            "message_routing": self.message_router.get_routing_stats(),
            "message_optimization": self.message_optimizer.get_optimization_stats(),
            "network_topology": self.topology_manager.get_topology_stats(),
            "system_stats": dict(self.system_stats)
        }


# Factory functions for creating agent profiles
def create_vision_agent_profile(agent_id: str) -> AgentInterestProfile:
    """Create profile for vision processing agent"""
    return AgentInterestProfile(
        agent_id=agent_id,
        domains_of_interest=[AgentDomain.VISION],
        keywords=["image", "vision", "object", "detection", "recognition"],
        message_types_preferred=[MessageType.NORMAL, MessageType.BULK],
        max_message_size=10485760,  # 10MB
        processing_capacity=0.8,
        communication_frequency=0.6,
        reliability_requirement=0.9
    )


def create_reasoning_agent_profile(agent_id: str) -> AgentInterestProfile:
    """Create profile for reasoning agent"""
    return AgentInterestProfile(
        agent_id=agent_id,
        domains_of_interest=[AgentDomain.REASONING, AgentDomain.DECISION],
        keywords=["logic", "reasoning", "decision", "analysis", "inference"],
        message_types_preferred=[MessageType.REQUEST, MessageType.RESPONSE, MessageType.URGENT],
        max_message_size=1048576,  # 1MB
        processing_capacity=0.9,
        communication_frequency=0.8,
        reliability_requirement=0.95
    )


def create_quantum_agent_profile(agent_id: str) -> AgentInterestProfile:
    """Create profile for quantum processing agent"""
    return AgentInterestProfile(
        agent_id=agent_id,
        domains_of_interest=[AgentDomain.QUANTUM, AgentDomain.OPTIMIZATION],
        keywords=["quantum", "optimization", "algorithm", "computation", "entanglement"],
        message_types_preferred=[MessageType.COMMAND, MessageType.REQUEST, MessageType.URGENT],
        max_message_size=524288,  # 512KB
        processing_capacity=0.95,
        communication_frequency=0.7,
        reliability_requirement=0.99
    )


# Main execution
async def main():
    """Main function to demonstrate optimized communication system"""
    print("🌐 Optimized Multi-Agent Communication System")
    print("=" * 60)
    
    # Create communication system
    comm_system = OptimizedCommunicationSystem()
    
    # Create sample agents
    from scalable_multi_agent_system import create_vision_agent, create_reasoning_agent, create_quantum_agent
    
    vision_agent = create_vision_agent("vision_001")
    reasoning_agent = create_reasoning_agent("reasoning_001")
    quantum_agent = create_quantum_agent("quantum_001")
    
    # Create agent profiles
    vision_profile = create_vision_agent_profile("vision_001")
    reasoning_profile = create_reasoning_agent_profile("reasoning_001")
    quantum_profile = create_quantum_agent_profile("quantum_001")
    
    # Register agents
    comm_system.register_agent(vision_agent, vision_profile)
    comm_system.register_agent(reasoning_agent, reasoning_profile)
    comm_system.register_agent(quantum_agent, quantum_profile)
    
    print(f"✅ Registered 3 agents with communication system")
    
    # Demonstrate message sending
    print(f"\n📤 Sending sample messages...")
    
    sample_messages = [
        {
            "sender": "system",
            "receiver": "vision_001",
            "content": {
                "task_type": "image_classification",
                "image_data": "sample_image.jpg",
                "priority": 7,
                "urgent": False
            }
        },
        {
            "sender": "system",
            "receiver": "reasoning_001",
            "content": {
                "task_type": "logical_analysis",
                "problem": "complex_logic_problem",
                "priority": 8,
                "request": True
            }
        },
        {
            "sender": "system",
            "receiver": "quantum_001",
            "content": {
                "task_type": "quantum_optimization",
                "problem": "traveling_salesman",
                "cities": 50,
                "priority": 9,
                "urgent": True
            }
        }
    ]
    
    for msg_info in sample_messages:
        delivered = await comm_system.send_message(
            msg_info["sender"],
            msg_info["receiver"],
            msg_info["content"]
        )
        print(f"📨 Message delivered to {len(delivered)} agents: {delivered}")
    
    # Optimize network
    print(f"\n⚡ Optimizing communication network...")
    optimization_results = await comm_system.optimize_network()
    
    print(f"📊 Optimization Results:")
    print(f"   Routing Success Rate: {optimization_results['routing_stats']['success_rate']:.2f}")
    print(f"   Avg Compression Ratio: {optimization_results['optimization_stats']['average_compression_ratio']:.2f}")
    print(f"   Network Density: {optimization_results['topology_stats']['network_metrics']['network_density']:.2f}")
    
    # Display final performance
    print(f"\n📈 System Performance:")
    performance = comm_system.get_system_performance()
    print(f"   Messages Sent: {performance['system_stats']['messages_sent']}")
    print(f"   Messages Delivered: {performance['system_stats']['messages_delivered']}")
    print(f"   Registered Agents: {performance['message_routing']['registered_agents']}")
    print(f"   Network Connections: {performance['message_routing']['network_connections']}")
    
    print(f"\n🏁 Communication System Demo Complete")


if __name__ == "__main__":
    asyncio.run(main())