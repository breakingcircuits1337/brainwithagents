"""
Massive Agent Factory for 250-Agent Brain System

This module provides factories and managers for creating and managing
250 specialized AI agents with sophisticated coordination and communication.
"""

import time
import random
import threading
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict, deque

# Import existing components
from hrm_system import HRMSystem, Task, TaskPriority
from communication_system import (
    CommunicationNetwork, Agent, Signal, SignalType, Connection,
    create_specialized_agent
)
from agent_specializations import (
    AgentSpecializationRegistry, AgentDomain, AgentSpecialization,
    AgentCapability, agent_registry
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentState(Enum):
    """Agent operational states"""
    IDLE = "idle"
    PROCESSING = "processing"
    LEARNING = "learning"
    COORDINATING = "coordinating"
    OPTIMIZING = "optimizing"
    ERROR = "error"
    SLEEPING = "sleeping"

class AgentPriority(Enum):
    """Agent priority levels for resource allocation"""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    BACKGROUND = 5

@dataclass
class AgentMetrics:
    """Comprehensive metrics for individual agents"""
    activation_count: int = 0
    processing_time: float = 0.0
    success_rate: float = 1.0
    error_count: int = 0
    resource_usage: Dict[str, float] = field(default_factory=dict)
    collaboration_count: int = 0
    learning_events: int = 0
    last_activity: float = field(default_factory=time.time)
    efficiency_score: float = 1.0
    contribution_score: float = 0.0

@dataclass
class AgentLoad:
    """Current load information for an agent"""
    queue_size: int = 0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    network_usage: float = 0.0
    processing_complexity: float = 0.0
    estimated_completion_time: float = 0.0

class AdvancedAgent(Agent):
    """Enhanced agent with advanced capabilities for large-scale systems"""
    
    def __init__(self, agent_id: str, specialization: AgentSpecialization):
        # Initialize base agent with specialization-specific threshold
        threshold = random.uniform(*specialization.threshold_range)
        super().__init__(agent_id, specialization.name, threshold)
        
        # Advanced properties
        self.specialization = specialization
        self.state = AgentState.IDLE
        self.priority = self._determine_agent_priority()
        self.metrics = AgentMetrics()
        self.load = AgentLoad()
        self.collaboration_history = []
        self.learning_buffer = deque(maxlen=100)
        self.resource_limits = specialization.resource_profile
        self.performance_history = deque(maxlen=1000)
        
        # Enhanced HRM system with specialization-specific setup
        self._setup_specialized_hrm()
        
        # Coordination properties
        self.coordination_partners = set()
        self.coordination_role = self._determine_coordination_role()
        self.coordination_weight = random.uniform(0.1, 1.0)
        
        # Learning properties
        self.learning_rate = 0.01 + (specialization.expertise_level * 0.04)
        self.adaptation_threshold = 0.7 + (specialization.expertise_level * 0.2)
        
        self.logger = logging.getLogger(f"{__name__}.AdvancedAgent.{agent_id}")
    
    def _determine_agent_priority(self) -> AgentPriority:
        """Determine agent priority based on specialization"""
        domain_priority_map = {
            AgentDomain.SECURITY: AgentPriority.CRITICAL,
            AgentDomain.COORDINATION: AgentPriority.HIGH,
            AgentDomain.MONITORING: AgentPriority.HIGH,
            AgentDomain.OPTIMIZATION: AgentPriority.MEDIUM,
            AgentDomain.LEARNING: AgentPriority.MEDIUM,
            AgentDomain.ANALYSIS: AgentPriority.MEDIUM,
            AgentDomain.COMMUNICATION: AgentPriority.MEDIUM,
            AgentDomain.DATA_PROCESSING: AgentPriority.LOW,
            AgentDomain.CREATIVITY: AgentPriority.LOW,
            AgentDomain.SPECIALIZED: AgentPriority.HIGH
        }
        
        return domain_priority_map.get(self.specialization.domain, AgentPriority.MEDIUM)
    
    def _determine_coordination_role(self) -> str:
        """Determine coordination role based on specialization"""
        domain_roles = {
            AgentDomain.COORDINATION: "coordinator",
            AgentDomain.OPTIMIZATION: "optimizer",
            AgentDomain.MONITORING: "monitor",
            AgentDomain.SECURITY: "security",
            AgentDomain.LEARNING: "learner",
            AgentDomain.ANALYSIS: "analyzer",
            AgentDomain.COMMUNICATION: "communicator",
            AgentDomain.DATA_PROCESSING: "processor",
            AgentDomain.CREATIVITY: "creator",
            AgentDomain.SPECIALIZED: "specialist"
        }
        
        return domain_roles.get(self.specialization.domain, "general")
    
    def _setup_specialized_hrm(self):
        """Setup HRM system with specialization-specific configuration"""
        # Set core mission based on domain
        domain_missions = {
            AgentDomain.DATA_PROCESSING: "Process and transform data efficiently",
            AgentDomain.ANALYSIS: "Extract insights and patterns from data",
            AgentDomain.COMMUNICATION: "Facilitate effective information exchange",
            AgentDomain.LEARNING: "Acquire knowledge and improve capabilities",
            AgentDomain.CREATIVITY: "Generate novel and valuable content",
            AgentDomain.OPTIMIZATION: "Optimize system performance and resource usage",
            AgentDomain.MONITORING: "Monitor system health and detect issues",
            AgentDomain.SECURITY: "Ensure system security and integrity",
            AgentDomain.COORDINATION: "Coordinate agent activities and workflows",
            AgentDomain.SPECIALIZED: "Handle specialized domain-specific tasks"
        }
        
        mission = domain_missions.get(self.specialization.domain, "Contribute to system goals")
        self.hrm_system.visionary.set_core_mission(mission)
        
        # Add domain-specific ethical guidelines
        domain_guidelines = {
            AgentDomain.SECURITY: ["Prioritize security above all else", "Maintain data integrity"],
            AgentDomain.LEARNING: ["Ensure learning benefits the system", "Avoid harmful knowledge acquisition"],
            AgentDomain.CREATIVITY: ["Create original and valuable content", "Respect intellectual property"],
            AgentDomain.COORDINATION: ["Ensure fair resource allocation", "Maintain system harmony"],
            AgentDomain.MONITORING: ["Report issues accurately and promptly", "Minimize false positives"],
            AgentDomain.OPTIMIZATION: ["Optimize for overall system benefit", "Consider trade-offs carefully"],
            AgentDomain.ANALYSIS: ["Provide accurate and unbiased analysis", "Consider multiple perspectives"],
            AgentDomain.COMMUNICATION: ["Communicate clearly and honestly", "Protect sensitive information"],
            AgentDomain.DATA_PROCESSING: ["Process data accurately and efficiently", "Respect data privacy"],
            AgentDomain.SPECIALIZED: ["Excel in specialized domain", "Collaborate with other specialists"]
        }
        
        guidelines = domain_guidelines.get(self.specialization.domain, ["Act ethically and responsibly"])
        for guideline in guidelines:
            self.hrm_system.visionary.add_ethical_guideline(guideline)
        
        # Add specialization-specific objectives
        objectives = []
        for capability in self.specialization.capabilities:
            objectives.append(f"Excel at {capability.name}")
        
        for objective in objectives:
            self.hrm_system.visionary.add_fundamental_objective(objective)
    
    def receive_signal(self, signal: Signal):
        """Enhanced signal reception with load management"""
        # Check agent capacity
        if self._is_overloaded():
            self.logger.warning(f"Agent {self.id} is overloaded, rejecting signal")
            return
        
        # Update load metrics
        self.load.queue_size += 1
        self.load.processing_complexity += signal.strength
        
        # Process signal with priority handling
        if self._should_process_signal(signal):
            super().receive_signal(signal)
            self._update_metrics_on_reception(signal)
        else:
            self.logger.debug(f"Agent {self.id} deferred signal {signal.id}")
    
    def _is_overloaded(self) -> bool:
        """Check if agent is overloaded"""
        load_score = (
            self.load.queue_size * 0.3 +
            self.load.cpu_usage * 0.3 +
            self.load.memory_usage * 0.2 +
            self.load.network_usage * 0.2
        )
        
        return load_score > 0.8
    
    def _should_process_signal(self, signal: Signal) -> bool:
        """Determine if signal should be processed now"""
        # Priority-based processing
        if signal.strength >= self.threshold * 1.5:  # High priority signals
            return True
        
        # Load-based decision
        if self.load.queue_size < 3:  # Low load
            return True
        
        # Expertise-based decision
        if self.specialization.expertise_level > 0.9:  # High expertise
            return True
        
        return False
    
    def _update_metrics_on_reception(self, signal: Signal):
        """Update metrics when receiving a signal"""
        self.metrics.last_activity = time.time()
        self.metrics.activation_count += 1
        
        # Update resource usage estimates
        for resource, requirement in self.specialization.resource_profile.items():
            if resource not in self.metrics.resource_usage:
                self.metrics.resource_usage[resource] = 0.0
            self.metrics.resource_usage[resource] += requirement * signal.strength
    
    def _activate(self, signal: Signal):
        """Enhanced activation with state management"""
        self.state = AgentState.PROCESSING
        start_time = time.time()
        
        try:
            # Process with specialized HRM
            result = self._process_with_hrm(signal)
            
            # Create output signal
            output_signal = Signal(
                id=f"output_{self.id}_{int(time.time())}",
                type=SignalType.DATA,
                content=result,
                source_agent_id=self.id,
                strength=signal.strength * self._calculate_output_strength(),
                metadata={
                    "processing_time": time.time() - start_time,
                    "agent_state": self.state.value,
                    "expertise_level": self.specialization.expertise_level
                }
            )
            
            self.output_buffer.append(output_signal)
            self._update_metrics_on_success(time.time() - start_time)
            
        except Exception as e:
            self.logger.error(f"Error in agent activation: {e}")
            self.state = AgentState.ERROR
            self._update_metrics_on_error()
            
            # Create error signal
            error_signal = Signal(
                id=f"error_{self.id}_{int(time.time())}",
                type=SignalType.ERROR,
                content={"error": str(e), "agent_id": self.id},
                source_agent_id=self.id,
                strength=signal.strength * 0.5
            )
            
            self.output_buffer.append(error_signal)
        
        finally:
            # Update load
            self.load.queue_size = max(0, self.load.queue_size - 1)
            self.state = AgentState.IDLE
    
    def _calculate_output_strength(self) -> float:
        """Calculate output signal strength based on agent performance"""
        base_strength = 0.8
        
        # Adjust based on expertise
        expertise_bonus = self.specialization.expertise_level * 0.2
        
        # Adjust based on recent performance
        performance_bonus = self.metrics.efficiency_score * 0.1
        
        # Adjust based on current load
        load_penalty = min(self.load.queue_size * 0.05, 0.3)
        
        return max(0.1, min(1.0, base_strength + expertise_bonus + performance_bonus - load_penalty))
    
    def _update_metrics_on_success(self, processing_time: float):
        """Update metrics on successful processing"""
        self.metrics.processing_time += processing_time
        self.metrics.success_rate = (
            (self.metrics.success_rate * (self.metrics.activation_count - 1) + 1.0) /
            self.metrics.activation_count
        )
        
        # Update efficiency score
        expected_time = sum(cap.processing_time for cap in self.specialization.capabilities) / len(self.specialization.capabilities)
        efficiency = min(1.0, expected_time / processing_time)
        self.metrics.efficiency_score = (
            (self.metrics.efficiency_score * 0.9) + (efficiency * 0.1)
        )
        
        # Record performance
        self.performance_history.append({
            "timestamp": time.time(),
            "processing_time": processing_time,
            "success": True,
            "efficiency": efficiency
        })
    
    def _update_metrics_on_error(self):
        """Update metrics on error"""
        self.metrics.error_count += 1
        self.metrics.success_rate = (
            (self.metrics.success_rate * (self.metrics.activation_count - 1)) /
            self.metrics.activation_count
        )
        
        # Record performance
        self.performance_history.append({
            "timestamp": time.time(),
            "processing_time": 0,
            "success": False,
            "efficiency": 0
        })
    
    def coordinate_with(self, other_agent: 'AdvancedAgent', coordination_type: str):
        """Coordinate activities with another agent"""
        self.coordination_partners.add(other_agent.id)
        other_agent.coordination_partners.add(self.id)
        
        coordination_event = {
            "timestamp": time.time(),
            "partner_id": other_agent.id,
            "type": coordination_type,
            "initiator": self.id
        }
        
        self.collaboration_history.append(coordination_event)
        other_agent.collaboration_history.append(coordination_event)
        
        self.metrics.collaboration_count += 1
        other_agent.metrics.collaboration_count += 1
        
        self.logger.debug(f"Coordination established: {self.id} <-> {other_agent.id}")
    
    def learn_from_experience(self, experience_data: Dict[str, Any]):
        """Learn from experience and adapt behavior"""
        if self.specialization.domain != AgentDomain.LEARNING:
            return
        
        # Add to learning buffer
        self.learning_buffer.append({
            "timestamp": time.time(),
            "experience": experience_data,
            "context": self.get_status()
        })
        
        # Update learning metrics
        self.metrics.learning_events += 1
        
        # Adapt threshold based on experience
        if len(self.learning_buffer) >= 10:
            self._adapt_from_learning()
    
    def _adapt_from_learning(self):
        """Adapt agent behavior based on learning"""
        recent_experiences = list(self.learning_buffer)[-10:]
        
        # Calculate average success rate from experiences
        success_count = sum(1 for exp in recent_experiences if exp.get("success", False))
        recent_success_rate = success_count / len(recent_experiences)
        
        # Adapt threshold
        if recent_success_rate > 0.9:
            # Increase threshold for more selective processing
            self.threshold = min(0.99, self.threshold * 1.05)
        elif recent_success_rate < 0.7:
            # Decrease threshold for more processing
            self.threshold = max(0.1, self.threshold * 0.95)
        
        self.logger.info(f"Agent {self.id} adapted threshold to {self.threshold:.3f}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive agent status"""
        base_status = super().get_status()
        
        return {
            **base_status,
            "state": self.state.value,
            "priority": self.priority.value,
            "specialization": {
                "domain": self.specialization.domain.value,
                "expertise_level": self.specialization.expertise_level,
                "capabilities": [cap.name for cap in self.specialization.capabilities]
            },
            "metrics": {
                "activation_count": self.metrics.activation_count,
                "success_rate": self.metrics.success_rate,
                "efficiency_score": self.metrics.efficiency_score,
                "collaboration_count": self.metrics.collaboration_count,
                "learning_events": self.metrics.learning_events
            },
            "load": {
                "queue_size": self.load.queue_size,
                "cpu_usage": self.load.cpu_usage,
                "memory_usage": self.load.memory_usage,
                "network_usage": self.load.network_usage
            },
            "coordination": {
                "partners": list(self.coordination_partners),
                "role": self.coordination_role,
                "weight": self.coordination_weight
            },
            "performance": {
                "average_processing_time": self.metrics.processing_time / max(1, self.metrics.activation_count),
                "error_rate": self.metrics.error_count / max(1, self.metrics.activation_count)
            }
        }
    
    def optimize_performance(self):
        """Optimize agent performance based on historical data"""
        if len(self.performance_history) < 20:
            return
        
        # Analyze recent performance
        recent_performance = list(self.performance_history)[-20:]
        recent_success_rate = sum(1 for p in recent_performance if p["success"]) / len(recent_performance)
        avg_efficiency = sum(p["efficiency"] for p in recent_performance) / len(recent_performance)
        
        # Optimize based on performance
        if recent_success_rate < 0.8:
            # Improve reliability
            self.threshold = max(0.1, self.threshold * 0.95)
            self.logger.info(f"Agent {self.id} optimizing for reliability")
        
        if avg_efficiency < 0.7:
            # Improve efficiency
            self.learning_rate = min(0.1, self.learning_rate * 1.1)
            self.logger.info(f"Agent {self.id} optimizing for efficiency")

class MassiveAgentFactory:
    """Factory for creating and managing 250 specialized agents"""
    
    def __init__(self):
        self.registry = agent_registry
        self.agents: Dict[str, AdvancedAgent] = {}
        self.creation_stats = {
            "total_created": 0,
            "by_domain": defaultdict(int),
            "by_complexity": defaultdict(int),
            "creation_time": 0.0
        }
        self.logger = logging.getLogger(__name__)
    
    def create_all_agents(self) -> Dict[str, AdvancedAgent]:
        """Create all 250 specialized agents"""
        start_time = time.time()
        self.logger.info("Creating 250 specialized agents...")
        
        # Create agents for each specialization
        for specialization in self.registry.get_all_specializations():
            agent = self._create_agent_from_specialization(specialization)
            self.agents[agent.id] = agent
            
            # Update statistics
            self.creation_stats["total_created"] += 1
            self.creation_stats["by_domain"][specialization.domain.value] += 1
            
            for capability in specialization.capabilities:
                self.creation_stats["by_complexity"][capability.complexity.value] += 1
        
        # Establish initial connections
        self._establish_initial_connections()
        
        # Calculate creation time
        self.creation_stats["creation_time"] = time.time() - start_time
        
        self.logger.info(f"Created {len(self.agents)} agents in {self.creation_stats['creation_time']:.2f} seconds")
        self.logger.info(f"Domain distribution: {dict(self.creation_stats['by_domain'])}")
        
        return self.agents
    
    def _create_agent_from_specialization(self, specialization: AgentSpecialization) -> AdvancedAgent:
        """Create an advanced agent from specialization"""
        agent_id = specialization.id
        agent = AdvancedAgent(agent_id, specialization)
        
        self.logger.debug(f"Created agent: {agent_id} ({specialization.name})")
        
        return agent
    
    def _establish_initial_connections(self):
        """Establish initial connections between agents"""
        self.logger.info("Establishing initial agent connections...")
        
        connection_count = 0
        
        for agent_id, agent in self.agents.items():
            # Get compatible connections
            compatible_targets = self.registry.get_compatible_connections(agent_id)
            
            # Filter to actual agents
            available_targets = [target_id for target_id in compatible_targets 
                               if target_id in self.agents and target_id != agent_id]
            
            # Establish connections based on specialization preferences
            for preferred_id in agent.specialization.preferred_connections:
                if preferred_id in self.agents:
                    # Calculate connection weight based on compatibility
                    weight = self._calculate_connection_weight(agent, self.agents[preferred_id])
                    agent.add_connection(self.agents[preferred_id], weight)
                    connection_count += 1
            
            # Add additional connections for network robustness
            additional_targets = random.sample(
                available_targets, 
                min(3, len(available_targets))
            )
            
            for target_id in additional_targets:
                if target_id not in agent.specialization.preferred_connections:
                    weight = self._calculate_connection_weight(agent, self.agents[target_id])
                    agent.add_connection(self.agents[target_id], weight * 0.7)  # Lower weight for non-preferred
                    connection_count += 1
        
        self.logger.info(f"Established {connection_count} initial connections")
    
    def _calculate_connection_weight(self, agent1: AdvancedAgent, agent2: AdvancedAgent) -> float:
        """Calculate connection weight between two agents"""
        base_weight = 0.5
        
        # Domain compatibility bonus
        if agent1.specialization.domain == agent2.specialization.domain:
            base_weight += 0.3
        
        # Expertise alignment bonus
        expertise_diff = abs(agent1.specialization.expertise_level - agent2.specialization.expertise_level)
        expertise_bonus = (1.0 - expertise_diff) * 0.2
        base_weight += expertise_bonus
        
        # Capability compatibility bonus
        compatibility_score = self._calculate_capability_compatibility(agent1, agent2)
        base_weight += compatibility_score * 0.3
        
        # Random variation
        base_weight += random.uniform(-0.1, 0.1)
        
        return max(0.1, min(1.0, base_weight))
    
    def _calculate_capability_compatibility(self, agent1: AdvancedAgent, agent2: AdvancedAgent) -> float:
        """Calculate capability compatibility between two agents"""
        compatibility_score = 0.0
        total_checks = 0
        
        for cap1 in agent1.specialization.capabilities:
            for cap2 in agent2.specialization.capabilities:
                total_checks += 1
                
                # Check if output types match input types
                if cap1.output_types and cap2.input_types:
                    overlap = len(set(cap1.output_types) & set(cap2.input_types))
                    if overlap > 0:
                        compatibility_score += overlap / max(len(cap1.output_types), len(cap2.input_types))
        
        return compatibility_score / max(1, total_checks)
    
    def get_agent_by_domain(self, domain: AgentDomain) -> List[AdvancedAgent]:
        """Get all agents in a specific domain"""
        domain_agents = []
        
        for agent in self.agents.values():
            if agent.specialization.domain == domain:
                domain_agents.append(agent)
        
        return domain_agents
    
    def get_agent_by_capability(self, capability_name: str) -> List[AdvancedAgent]:
        """Get agents with specific capability"""
        capable_agents = []
        
        for agent in self.agents.values():
            for capability in agent.specialization.capabilities:
                if capability.name == capability_name:
                    capable_agents.append(agent)
                    break
        
        return capable_agents
    
    def get_network_statistics(self) -> Dict[str, Any]:
        """Get comprehensive network statistics"""
        stats = {
            "total_agents": len(self.agents),
            "creation_stats": self.creation_stats,
            "domain_distribution": {},
            "connection_stats": {
                "total_connections": 0,
                "average_connections_per_agent": 0,
                "connection_density": 0.0
            },
            "agent_states": {},
            "performance_summary": {}
        }
        
        # Domain distribution
        for agent in self.agents.values():
            domain = agent.specialization.domain.value
            if domain not in stats["domain_distribution"]:
                stats["domain_distribution"][domain] = 0
            stats["domain_distribution"][domain] += 1
        
        # Connection statistics
        total_connections = sum(len(agent.connections) for agent in self.agents.values())
        stats["connection_stats"]["total_connections"] = total_connections
        stats["connection_stats"]["average_connections_per_agent"] = total_connections / len(self.agents)
        max_possible_connections = len(self.agents) * (len(self.agents) - 1)
        stats["connection_stats"]["connection_density"] = total_connections / max_possible_connections
        
        # Agent states
        for agent in self.agents.values():
            state = agent.state.value
            if state not in stats["agent_states"]:
                stats["agent_states"][state] = 0
            stats["agent_states"][state] += 1
        
        # Performance summary
        total_activations = sum(agent.metrics.activation_count for agent in self.agents.values())
        total_errors = sum(agent.metrics.error_count for agent in self.agents.values())
        avg_efficiency = sum(agent.metrics.efficiency_score for agent in self.agents.values()) / len(self.agents)
        
        stats["performance_summary"] = {
            "total_activations": total_activations,
            "total_errors": total_errors,
            "overall_success_rate": (total_activations - total_errors) / max(1, total_activations),
            "average_efficiency": avg_efficiency
        }
        
        return stats
    
    def optimize_agent_network(self):
        """Optimize the entire agent network"""
        self.logger.info("Optimizing agent network...")
        
        # Optimize individual agents
        for agent in self.agents.values():
            agent.optimize_performance()
        
        # Optimize connections
        self._optimize_connections()
        
        # Establish new beneficial connections
        self._establish_beneficial_connections()
        
        self.logger.info("Agent network optimization completed")
    
    def _optimize_connections(self):
        """Optimize existing connections based on usage"""
        for agent in self.agents.values():
            # Analyze connection usage
            connection_usage = {}
            for connection in agent.connections:
                usage_key = f"{agent.id}->{connection.target_id}"
                connection_usage[usage_key] = random.uniform(0, 1)  # Simulated usage
            
            # Adjust weights based on usage
            for connection in agent.connections:
                usage_key = f"{agent.id}->{connection.target_id}"
                usage = connection_usage.get(usage_key, 0.5)
                
                if usage > 0.7:
                    connection.weight = min(1.0, connection.weight * 1.1)
                elif usage < 0.3:
                    connection.weight = max(0.1, connection.weight * 0.9)
    
    def _establish_beneficial_connections(self):
        """Establish new connections that would be beneficial"""
        for agent in self.agents.values():
            current_connections = {conn.target_id for conn in agent.connections}
            
            # Find agents that would benefit from connection
            compatible_targets = self.registry.get_compatible_connections(agent.id)
            potential_targets = [target_id for target_id in compatible_targets 
                               if target_id in self.agents and target_id not in current_connections]
            
            # Select top candidates
            if potential_targets:
                # Score potential connections
                scored_targets = []
                for target_id in potential_targets:
                    target_agent = self.agents[target_id]
                    score = self._calculate_connection_score(agent, target_agent)
                    scored_targets.append((target_id, score))
                
                # Sort by score and establish top connections
                scored_targets.sort(key=lambda x: x[1], reverse=True)
                
                for target_id, score in scored_targets[:2]:  # Top 2 connections
                    if score > 0.6:  # Only establish beneficial connections
                        weight = score * 0.8
                        agent.add_connection(self.agents[target_id], weight)
    
    def _calculate_connection_score(self, agent1: AdvancedAgent, agent2: AdvancedAgent) -> float:
        """Calculate connection score between two agents"""
        score = 0.0
        
        # Capability compatibility
        capability_score = self._calculate_capability_compatibility(agent1, agent2)
        score += capability_score * 0.4
        
        # Performance complementarity
        perf_diff = abs(agent1.metrics.efficiency_score - agent2.metrics.efficiency_score)
        perf_complementarity = 1.0 - perf_diff
        score += perf_complementarity * 0.3
        
        # Load balance potential
        load_balance = 1.0 - abs(agent1.load.queue_size - agent2.load.queue_size) / max(1, agent1.load.queue_size + agent2.load.queue_size)
        score += load_balance * 0.2
        
        # Expertise complementarity
        expertise_diff = abs(agent1.specialization.expertise_level - agent2.specialization.expertise_level)
        expertise_complementarity = 1.0 - expertise_diff
        score += expertise_complementarity * 0.1
        
        return score

class AgentClusterManager:
    """Manages agent clusters for efficient coordination"""
    
    def __init__(self, agents: Dict[str, AdvancedAgent]):
        self.agents = agents
        self.clusters: Dict[str, Set[str]] = {}
        self.cluster_leaders: Dict[str, str] = {}
        self.logger = logging.getLogger(__name__)
    
    def form_clusters(self, cluster_size: int = 10):
        """Form clusters of agents for efficient coordination"""
        self.logger.info(f"Forming clusters of size {cluster_size}")
        
        # Clear existing clusters
        self.clusters.clear()
        self.cluster_leaders.clear()
        
        # Group agents by domain
        domain_groups = defaultdict(list)
        for agent_id, agent in self.agents.items():
            domain_groups[agent.specialization.domain].append(agent_id)
        
        # Create clusters within each domain
        cluster_id = 0
        for domain, agent_ids in domain_groups.items():
            # Sort by expertise level
            sorted_agents = sorted(agent_ids, 
                                 key=lambda aid: self.agents[aid].specialization.expertise_level,
                                 reverse=True)
            
            # Form clusters
            for i in range(0, len(sorted_agents), cluster_size):
                cluster_members = set(sorted_agents[i:i + cluster_size])
                cluster_name = f"cluster_{cluster_id:03d}"
                
                self.clusters[cluster_name] = cluster_members
                
                # Select cluster leader (highest expertise)
                leader_id = max(cluster_members, 
                              key=lambda aid: self.agents[aid].specialization.expertise_level)
                self.cluster_leaders[cluster_name] = leader_id
                
                # Establish cluster coordination
                self._establish_cluster_coordination(cluster_name, cluster_members, leader_id)
                
                cluster_id += 1
        
        self.logger.info(f"Formed {len(self.clusters)} clusters")
    
    def _establish_cluster_coordination(self, cluster_name: str, members: Set[str], leader_id: str):
        """Establish coordination within a cluster"""
        leader = self.agents[leader_id]
        
        for member_id in members:
            if member_id != leader_id:
                member = self.agents[member_id]
                
                # Establish coordination relationship
                leader.coordinate_with(member, "cluster_leadership")
                member.coordinate_with(leader, "cluster_membership")
                
                # Set up strong connections
                leader.add_connection(member, 0.9)
                member.add_connection(leader, 0.8)
    
    def get_cluster_for_agent(self, agent_id: str) -> Optional[str]:
        """Get the cluster for a specific agent"""
        for cluster_name, members in self.clusters.items():
            if agent_id in members:
                return cluster_name
        return None
    
    def get_cluster_members(self, cluster_name: str) -> Set[str]:
        """Get members of a specific cluster"""
        return self.clusters.get(cluster_name, set())
    
    def coordinate_cluster_activity(self, cluster_name: str, task_data: Dict[str, Any]):
        """Coordinate activity within a cluster"""
        if cluster_name not in self.clusters:
            return
        
        leader_id = self.cluster_leaders[cluster_name]
        leader = self.agents[leader_id]
        
        # Create coordination signal
        signal = Signal(
            id=f"cluster_coord_{cluster_name}_{int(time.time())}",
            type=SignalType.CONTROL,
            content={
                "type": "cluster_coordination",
                "cluster": cluster_name,
                "task": task_data,
                "coordinator": leader_id
            },
            source_agent_id="cluster_manager",
            target_agent_id=leader_id,
            strength=0.9
        )
        
        # Send to cluster leader
        leader.receive_signal(signal)
        
        self.logger.debug(f"Coordinated cluster {cluster_name} for task: {task_data.get('type', 'unknown')}")

# Global factory instance
massive_agent_factory = MassiveAgentFactory()