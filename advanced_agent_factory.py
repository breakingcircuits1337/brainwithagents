"""
Advanced Agent Factory for 250-Agent Brain System

This module provides factory classes for creating and managing
250 specialized AI agents with advanced capabilities and coordination.
"""

import random
import time
import threading
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import existing systems
from hrm_system import HRMSystem, Task, TaskPriority
from communication_system import Agent, Connection, Signal, SignalType, CommunicationNetwork
from agent_specializations import (
    AgentSpecialization, AgentSpecializationRegistry, AgentDomain, AgentComplexity
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentState(Enum):
    """Agent operational states"""
    IDLE = "idle"
    INITIALIZING = "initializing"
    ACTIVE = "active"
    PROCESSING = "processing"
    LEARNING = "learning"
    OPTIMIZING = "optimizing"
    ERROR = "error"
    SLEEPING = "sleeping"

class AgentHealth(Enum):
    """Agent health status"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"

@dataclass
class AgentMetrics:
    """Comprehensive metrics for individual agents"""
    total_signals_processed: int = 0
    successful_signals: int = 0
    failed_signals: int = 0
    average_processing_time: float = 0.0
    current_load: float = 0.0  # 0.0 to 1.0
    resource_usage: Dict[str, float] = field(default_factory=dict)
    uptime: float = 0.0
    last_activity: float = 0.0
    learning_progress: float = 0.0  # 0.0 to 1.0
    adaptation_score: float = 0.0  # 0.0 to 1.0
    error_rate: float = 0.0  # 0.0 to 1.0
    efficiency_score: float = 1.0  # 0.0 to 1.0
    collaboration_score: float = 0.5  # 0.0 to 1.0

@dataclass
class AgentProfile:
    """Complete profile for an advanced agent"""
    agent_id: str
    specialization: AgentSpecialization
    instance: Agent
    state: AgentState = AgentState.IDLE
    health: AgentHealth = AgentHealth.EXCELLENT
    metrics: AgentMetrics = field(default_factory=AgentMetrics)
    current_tasks: List[str] = field(default_factory=list)
    task_history: List[Dict[str, Any]] = field(default_factory=list)
    connections: List[str] = field(default_factory=list)
    performance_history: List[Dict[str, Any]] = field(default_factory=list)
    learning_data: List[Dict[str, Any]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)
    
    def update_metrics(self, processing_time: float, success: bool):
        """Update agent metrics after processing"""
        self.metrics.total_signals_processed += 1
        self.metrics.last_activity = time.time()
        
        if success:
            self.metrics.successful_signals += 1
        else:
            self.metrics.failed_signals += 1
        
        # Update average processing time
        total = self.metrics.total_signals_processed
        current_avg = self.metrics.average_processing_time
        self.metrics.average_processing_time = (
            (current_avg * (total - 1) + processing_time) / total
        )
        
        # Update error rate
        self.metrics.error_rate = self.metrics.failed_signals / total
        
        # Update last updated timestamp
        self.last_updated = time.time()

class AdvancedAgentFactory:
    """Factory for creating and managing advanced agents"""
    
    def __init__(self, specialization_registry: AgentSpecializationRegistry):
        self.specialization_registry = specialization_registry
        self.agent_profiles: Dict[str, AgentProfile] = {}
        self.domain_leaders: Dict[AgentDomain, str] = {}
        self.performance_thresholds = {
            "min_efficiency": 0.7,
            "max_error_rate": 0.3,
            "min_collaboration": 0.4,
            "max_load": 0.8
        }
        self.logger = logging.getLogger(f"{__name__}.AdvancedAgentFactory")
        
        # Threading for concurrent operations
        self.executor = ThreadPoolExecutor(max_workers=10)
        
    def create_agent(self, agent_id: str, specialization_id: str) -> Optional[AgentProfile]:
        """Create a single advanced agent"""
        specialization = self.specialization_registry.get_specialization(specialization_id)
        if not specialization:
            self.logger.error(f"Specialization {specialization_id} not found")
            return None
        
        try:
            # Create base agent instance
            threshold = random.uniform(*specialization.threshold_range)
            base_agent = Agent(agent_id, specialization.name, threshold)
            
            # Set up HRM system with specialization-specific configuration
            self._configure_agent_hrm(base_agent, specialization)
            
            # Create agent profile
            profile = AgentProfile(
                agent_id=agent_id,
                specialization=specialization,
                instance=base_agent,
                state=AgentState.INITIALIZING
            )
            
            # Initialize agent metrics
            profile.metrics.resource_usage = {
                "cpu": 0.1,
                "memory": 0.1,
                "network": 0.1
            }
            
            # Store profile
            self.agent_profiles[agent_id] = profile
            
            # Initialize agent
            self._initialize_agent(profile)
            
            self.logger.info(f"Created agent {agent_id} with specialization {specialization.name}")
            return profile
            
        except Exception as e:
            self.logger.error(f"Failed to create agent {agent_id}: {e}")
            return None
    
    def create_agent_batch(self, agent_configs: List[Tuple[str, str]]) -> Dict[str, AgentProfile]:
        """Create multiple agents concurrently"""
        self.logger.info(f"Creating batch of {len(agent_configs)} agents")
        
        profiles = {}
        futures = []
        
        # Submit creation tasks
        for agent_id, specialization_id in agent_configs:
            future = self.executor.submit(self.create_agent, agent_id, specialization_id)
            futures.append((agent_id, future))
        
        # Collect results
        for agent_id, future in futures:
            try:
                profile = future.result(timeout=30.0)
                if profile:
                    profiles[agent_id] = profile
            except Exception as e:
                self.logger.error(f"Failed to create agent {agent_id}: {e}")
        
        self.logger.info(f"Successfully created {len(profiles)} agents")
        return profiles
    
    def create_domain_agents(self, domain: AgentDomain, count: int) -> Dict[str, AgentProfile]:
        """Create agents for a specific domain"""
        domain_agents = self.specialization_registry.get_domain_agents(domain)
        
        if not domain_agents:
            self.logger.error(f"No agents found for domain {domain}")
            return {}
        
        # Select agents based on available specializations
        selected_agents = domain_agents[:min(count, len(domain_agents))]
        
        # Create agent configurations
        agent_configs = []
        for i, specialization_id in enumerate(selected_agents):
            agent_id = f"{domain.value}_{i+1:03d}"
            agent_configs.append((agent_id, specialization_id))
        
        return self.create_agent_batch(agent_configs)
    
    def create_all_agents(self) -> Dict[str, AgentProfile]:
        """Create all 250 agents"""
        self.logger.info("Creating all 250 specialized agents")
        
        all_profiles = {}
        
        # Create agents for each domain
        for domain in AgentDomain:
            domain_count = len(self.specialization_registry.get_domain_agents(domain))
            domain_profiles = self.create_domain_agents(domain, domain_count)
            all_profiles.update(domain_profiles)
        
        # Set up domain leaders
        self._establish_domain_leaders()
        
        self.logger.info(f"Successfully created {len(all_profiles)} agents")
        return all_profiles
    
    def _configure_agent_hrm(self, agent: Agent, specialization: AgentSpecialization):
        """Configure agent's HRM system based on specialization"""
        # Set core mission based on domain
        domain_missions = {
            AgentDomain.DATA_PROCESSING: "Process and transform data efficiently and accurately",
            AgentDomain.ANALYSIS: "Analyze data to extract meaningful insights and patterns",
            AgentDomain.COMMUNICATION: "Facilitate effective communication and information exchange",
            AgentDomain.LEARNING: "Learn and adapt from data and experiences",
            AgentDomain.CREATIVITY: "Generate creative and innovative solutions",
            AgentDomain.OPTIMIZATION: "Optimize systems and processes for maximum efficiency",
            AgentDomain.MONITORING: "Monitor systems and detect anomalies or issues",
            AgentDomain.SECURITY: "Ensure system security and protect against threats",
            AgentDomain.COORDINATION: "Coordinate agent activities and system operations",
            AgentDomain.SPECIALIZED: "Perform specialized tasks requiring expert knowledge"
        }
        
        mission = domain_missions.get(specialization.domain, "Contribute to system objectives")
        agent.hrm_system.visionary.set_core_mission(mission)
        
        # Add ethical guidelines based on domain
        if specialization.domain == AgentDomain.SECURITY:
            agent.hrm_system.visionary.add_ethical_guideline("Prioritize security and privacy")
        elif specialization.domain == AgentDomain.LEARNING:
            agent.hrm_system.visionary.add_ethical_guideline("Ensure fair and unbiased learning")
        elif specialization.domain == AgentDomain.CREATIVITY:
            agent.hrm_system.visionary.add_ethical_guideline("Create original and appropriate content")
        
        # Add domain-specific objectives
        objectives = self._get_domain_objectives(specialization.domain)
        for objective in objectives:
            agent.hrm_system.visionary.add_fundamental_objective(objective)
    
    def _get_domain_objectives(self, domain: AgentDomain) -> List[str]:
        """Get fundamental objectives for a domain"""
        objectives_map = {
            AgentDomain.DATA_PROCESSING: [
                "Ensure data quality and integrity",
                "Optimize data processing speed",
                "Maintain data consistency"
            ],
            AgentDomain.ANALYSIS: [
                "Extract accurate insights",
                "Identify meaningful patterns",
                "Provide actionable recommendations"
            ],
            AgentDomain.COMMUNICATION: [
                "Ensure message delivery",
                "Minimize communication latency",
                "Maintain message integrity"
            ],
            AgentDomain.LEARNING: [
                "Continuously improve performance",
                "Adapt to new information",
                "Share knowledge effectively"
            ],
            AgentDomain.CREATIVITY: [
                "Generate innovative solutions",
                "Maintain originality",
                "Meet creative requirements"
            ],
            AgentDomain.OPTIMIZATION: [
                "Maximize efficiency",
                "Minimize resource usage",
                "Improve performance metrics"
            ],
            AgentDomain.MONITORING: [
                "Detect anomalies early",
                "Provide accurate monitoring",
                "Generate timely alerts"
            ],
            AgentDomain.SECURITY: [
                "Protect against threats",
                "Maintain system integrity",
                "Ensure compliance"
            ],
            AgentDomain.COORDINATION: [
                "Coordinate effectively",
                "Optimize resource allocation",
                "Maintain system balance"
            ],
            AgentDomain.SPECIALIZED: [
                "Perform specialized tasks accurately",
                "Maintain domain expertise",
                "Collaborate with other agents"
            ]
        }
        
        return objectives_map.get(domain, ["Contribute to system success"])
    
    def _initialize_agent(self, profile: AgentProfile):
        """Initialize an agent after creation"""
        try:
            # Perform initialization through HRM
            init_directive = f"Initialize agent {profile.agent_id} for {profile.specialization.name}"
            result = profile.instance.hrm_system.process_directive(init_directive)
            
            if result.get("success", False):
                profile.state = AgentState.IDLE
                profile.health = AgentHealth.EXCELLENT
                self.logger.debug(f"Agent {profile.agent_id} initialized successfully")
            else:
                profile.state = AgentState.ERROR
                profile.health = AgentHealth.POOR
                self.logger.warning(f"Agent {profile.agent_id} initialization failed")
                
        except Exception as e:
            profile.state = AgentState.ERROR
            profile.health = AgentHealth.CRITICAL
            self.logger.error(f"Agent {profile.agent_id} initialization error: {e}")
    
    def _establish_domain_leaders(self):
        """Establish domain leaders for coordination"""
        for domain in AgentDomain:
            domain_agents = [
                agent_id for agent_id, profile in self.agent_profiles.items()
                if profile.specialization.domain == domain
            ]
            
            if domain_agents:
                # Select leader based on complexity and capabilities
                leader_candidates = []
                for agent_id in domain_agents:
                    profile = self.agent_profiles[agent_id]
                    if profile.specialization.complexity == AgentComplexity.EXPERT:
                        leader_candidates.append(agent_id)
                
                # If no expert agents, select advanced
                if not leader_candidates:
                    for agent_id in domain_agents:
                        profile = self.agent_profiles[agent_id]
                        if profile.specialization.complexity == AgentComplexity.ADVANCED:
                            leader_candidates.append(agent_id)
                
                # Select first candidate or random if none
                if leader_candidates:
                    leader = leader_candidates[0]
                else:
                    leader = random.choice(domain_agents)
                
                self.domain_leaders[domain] = leader
                self.logger.info(f"Established {leader} as leader for {domain} domain")
    
    def get_agent_profile(self, agent_id: str) -> Optional[AgentProfile]:
        """Get agent profile by ID"""
        return self.agent_profiles.get(agent_id)
    
    def get_domain_agents(self, domain: AgentDomain) -> List[str]:
        """Get all agent IDs in a domain"""
        return [
            agent_id for agent_id, profile in self.agent_profiles.items()
            if profile.specialization.domain == domain
        ]
    
    def get_agents_by_state(self, state: AgentState) -> List[str]:
        """Get agents in a specific state"""
        return [
            agent_id for agent_id, profile in self.agent_profiles.items()
            if profile.state == state
        ]
    
    def get_agents_by_health(self, health: AgentHealth) -> List[str]:
        """Get agents with specific health status"""
        return [
            agent_id for agent_id, profile in self.agent_profiles.items()
            if profile.health == health
        ]
    
    def update_agent_state(self, agent_id: str, new_state: AgentState):
        """Update agent state"""
        if agent_id in self.agent_profiles:
            profile = self.agent_profiles[agent_id]
            old_state = profile.state
            profile.state = new_state
            profile.last_updated = time.time()
            
            self.logger.debug(f"Agent {agent_id} state changed from {old_state} to {new_state}")
    
    def update_agent_health(self, agent_id: str, new_health: AgentHealth):
        """Update agent health status"""
        if agent_id in self.agent_profiles:
            profile = self.agent_profiles[agent_id]
            old_health = profile.health
            profile.health = new_health
            profile.last_updated = time.time()
            
            self.logger.debug(f"Agent {agent_id} health changed from {old_health} to {new_health}")
    
    def process_agent_signal(self, agent_id: str, signal: Signal) -> bool:
        """Process a signal through a specific agent"""
        profile = self.agent_profiles.get(agent_id)
        if not profile:
            self.logger.error(f"Agent {agent_id} not found")
            return False
        
        if profile.state == AgentState.ERROR:
            self.logger.warning(f"Agent {agent_id} is in error state")
            return False
        
        try:
            # Update agent state
            self.update_agent_state(agent_id, AgentState.PROCESSING)
            
            # Process signal
            start_time = time.time()
            profile.instance.receive_signal(signal)
            
            # Wait for processing to complete (simplified)
            processing_time = time.time() - start_time
            
            # Update metrics
            success = profile.state != AgentState.ERROR
            profile.update_metrics(processing_time, success)
            
            # Update load
            profile.metrics.current_load = min(1.0, len(profile.current_tasks) / profile.specialization.max_concurrent_tasks)
            
            # Return to idle state
            if profile.state != AgentState.ERROR:
                self.update_agent_state(agent_id, AgentState.IDLE)
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error processing signal in agent {agent_id}: {e}")
            self.update_agent_state(agent_id, AgentState.ERROR)
            self.update_agent_health(agent_id, AgentHealth.CRITICAL)
            return False
    
    def get_agent_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for all agents"""
        total_agents = len(self.agent_profiles)
        if total_agents == 0:
            return {"total_agents": 0}
        
        # Calculate aggregate metrics
        total_signals = sum(p.metrics.total_signals_processed for p in self.agent_profiles.values())
        successful_signals = sum(p.metrics.successful_signals for p in self.agent_profiles.values())
        total_processing_time = sum(p.metrics.average_processing_time * p.metrics.total_signals_processed 
                                  for p in self.agent_profiles.values())
        
        avg_processing_time = total_processing_time / total_signals if total_signals > 0 else 0
        overall_success_rate = successful_signals / total_signals if total_signals > 0 else 0
        
        # State distribution
        state_distribution = {}
        for state in AgentState:
            state_distribution[state.value] = len(self.get_agents_by_state(state))
        
        # Health distribution
        health_distribution = {}
        for health in AgentHealth:
            health_distribution[health.value] = len(self.get_agents_by_health(health))
        
        # Domain distribution
        domain_distribution = {}
        for domain in AgentDomain:
            domain_distribution[domain.value] = len(self.get_domain_agents(domain))
        
        return {
            "total_agents": total_agents,
            "total_signals_processed": total_signals,
            "successful_signals": successful_signals,
            "overall_success_rate": overall_success_rate,
            "average_processing_time": avg_processing_time,
            "state_distribution": state_distribution,
            "health_distribution": health_distribution,
            "domain_distribution": domain_distribution,
            "domain_leaders": {domain.value: leader for domain, leader in self.domain_leaders.items()}
        }
    
    def get_agent_recommendations(self) -> List[Dict[str, Any]]:
        """Get recommendations for agent optimization"""
        recommendations = []
        
        for agent_id, profile in self.agent_profiles.items():
            agent_recs = []
            
            # Check efficiency
            if profile.metrics.efficiency_score < self.performance_thresholds["min_efficiency"]:
                agent_recs.append({
                    "type": "efficiency",
                    "message": "Low efficiency detected, consider optimization",
                    "priority": "medium"
                })
            
            # Check error rate
            if profile.metrics.error_rate > self.performance_thresholds["max_error_rate"]:
                agent_recs.append({
                    "type": "error_rate",
                    "message": "High error rate, investigate and fix issues",
                    "priority": "high"
                })
            
            # Check collaboration
            if profile.metrics.collaboration_score < self.performance_thresholds["min_collaboration"]:
                agent_recs.append({
                    "type": "collaboration",
                    "message": "Low collaboration score, improve agent interactions",
                    "priority": "low"
                })
            
            # Check load
            if profile.metrics.current_load > self.performance_thresholds["max_load"]:
                agent_recs.append({
                    "type": "load",
                    "message": "High load detected, consider load balancing",
                    "priority": "high"
                })
            
            # Check health
            if profile.health in [AgentHealth.POOR, AgentHealth.CRITICAL]:
                agent_recs.append({
                    "type": "health",
                    "message": f"Agent health is {profile.health.value}, requires attention",
                    "priority": "high"
                })
            
            if agent_recs:
                recommendations.append({
                    "agent_id": agent_id,
                    "agent_name": profile.specialization.name,
                    "recommendations": agent_recs
                })
        
        return recommendations
    
    def optimize_agents(self) -> Dict[str, Any]:
        """Optimize agent performance based on metrics"""
        self.logger.info("Starting agent optimization")
        
        optimization_results = {
            "agents_optimized": 0,
            "improvements_made": [],
            "failed_optimizations": []
        }
        
        for agent_id, profile in self.agent_profiles.items():
            try:
                improvements = []
                
                # Optimize threshold based on performance
                if profile.metrics.error_rate > 0.3:
                    # Increase threshold to reduce false activations
                    new_threshold = min(1.0, profile.instance.threshold * 1.1)
                    profile.instance.threshold = new_threshold
                    improvements.append(f"Increased threshold to {new_threshold:.2f}")
                
                # Optimize learning rate
                if profile.metrics.learning_progress < 0.5:
                    # Increase learning rate
                    new_learning_rate = min(1.0, profile.specialization.learning_rate * 1.2)
                    profile.specialization.learning_rate = new_learning_rate
                    improvements.append(f"Increased learning rate to {new_learning_rate:.2f}")
                
                # Optimize connections
                if profile.metrics.collaboration_score < 0.4:
                    # Add more connections
                    improvements.append("Enhanced agent connections")
                
                if improvements:
                    optimization_results["agents_optimized"] += 1
                    optimization_results["improvements_made"].append({
                        "agent_id": agent_id,
                        "improvements": improvements
                    })
                    
                    self.logger.info(f"Optimized agent {agent_id}: {improvements}")
                
            except Exception as e:
                optimization_results["failed_optimizations"].append({
                    "agent_id": agent_id,
                    "error": str(e)
                })
                self.logger.error(f"Failed to optimize agent {agent_id}: {e}")
        
        self.logger.info(f"Optimization completed: {optimization_results['agents_optimized']} agents optimized")
        return optimization_results
    
    def shutdown(self):
        """Shutdown the agent factory"""
        self.logger.info("Shutting down agent factory")
        
        # Shutdown thread pool
        self.executor.shutdown(wait=True)
        
        # Update agent states
        for agent_id in self.agent_profiles:
            self.update_agent_state(agent_id, AgentState.SLEEPING)
        
        self.logger.info("Agent factory shutdown completed")

class NetworkTopologyManager:
    """Manages network topology for 250 agents"""
    
    def __init__(self, agent_factory: AdvancedAgentFactory):
        self.agent_factory = agent_factory
        self.connection_strategies = {
            "domain_centric": self._create_domain_centric_topology,
            "hierarchical": self._create_hierarchical_topology,
            "small_world": self._create_small_world_topology,
            "scale_free": self._create_scale_free_topology,
            "hybrid": self._create_hybrid_topology
        }
        self.logger = logging.getLogger(f"{__name__}.NetworkTopologyManager")
    
    def create_topology(self, strategy: str = "hybrid") -> Dict[str, List[str]]:
        """Create network topology using specified strategy"""
        if strategy not in self.connection_strategies:
            self.logger.error(f"Unknown topology strategy: {strategy}")
            return {}
        
        self.logger.info(f"Creating {strategy} network topology")
        return self.connection_strategies[strategy]()
    
    def _create_domain_centric_topology(self) -> Dict[str, List[str]]:
        """Create domain-centric topology"""
        connections = {}
        
        # Connect agents within domains
        for domain in AgentDomain:
            domain_agents = self.agent_factory.get_domain_agents(domain)
            
            # Create mesh within domain
            for i, agent_id in enumerate(domain_agents):
                connections[agent_id] = []
                
                # Connect to other agents in same domain
                for j, other_id in enumerate(domain_agents):
                    if i != j:
                        connections[agent_id].append(other_id)
                
                # Connect to domain leader
                if domain in self.agent_factory.domain_leaders:
                    leader = self.agent_factory.domain_leaders[domain]
                    if leader != agent_id:
                        connections[agent_id].append(leader)
        
        # Connect domain leaders to master coordinator
        if "master_coordinator_001" in self.agent_factory.agent_profiles:
            master_coordinator = "master_coordinator_001"
            connections[master_coordinator] = []
            
            for leader in self.agent_factory.domain_leaders.values():
                if leader != master_coordinator:
                    connections[master_coordinator].append(leader)
                    connections[leader].append(master_coordinator)
        
        self.logger.info("Created domain-centric topology")
        return connections
    
    def _create_hierarchical_topology(self) -> Dict[str, List[str]]:
        """Create hierarchical topology"""
        connections = {}
        
        # Level 1: Master coordinator
        master_coordinator = "master_coordinator_001"
        connections[master_coordinator] = []
        
        # Level 2: Domain coordinators
        domain_coordinators = []
        for domain in AgentDomain:
            domain_agents = self.agent_factory.get_domain_agents(domain)
            if domain_agents:
                # Find domain coordinator (first agent or domain leader)
                domain_coord = self.agent_factory.domain_leaders.get(domain, domain_agents[0])
                domain_coordinators.append(domain_coord)
                
                # Connect to master coordinator
                connections[master_coordinator].append(domain_coord)
                connections[domain_coord] = [master_coordinator]
        
        # Level 3: Regular agents
        for domain in AgentDomain:
            domain_agents = self.agent_factory.get_domain_agents(domain)
            domain_coord = self.agent_factory.domain_leaders.get(domain)
            
            for agent_id in domain_agents:
                if agent_id != domain_coord:
                    connections[agent_id] = [domain_coord]
                    if domain_coord:
                        connections[domain_coord].append(agent_id)
        
        self.logger.info("Created hierarchical topology")
        return connections
    
    def _create_small_world_topology(self) -> Dict[str, List[str]]:
        """Create small-world topology"""
        connections = {}
        
        all_agents = list(self.agent_factory.agent_profiles.keys())
        n = len(all_agents)
        
        # Create regular lattice
        k = 6  # Each node connects to k nearest neighbors
        
        for i, agent_id in enumerate(all_agents):
            connections[agent_id] = []
            
            # Connect to k nearest neighbors
            for j in range(1, k // 2 + 1):
                neighbor_idx = (i + j) % n
                connections[agent_id].append(all_agents[neighbor_idx])
                
                neighbor_idx = (i - j) % n
                connections[agent_id].append(all_agents[neighbor_idx])
        
        # Add random long-range connections
        num_random = n // 10  # 10% random connections
        
        for _ in range(num_random):
            agent1, agent2 = random.sample(all_agents, 2)
            if agent2 not in connections[agent1]:
                connections[agent1].append(agent2)
                connections[agent2].append(agent1)
        
        self.logger.info("Created small-world topology")
        return connections
    
    def _create_scale_free_topology(self) -> Dict[str, List[str]]:
        """Create scale-free topology"""
        connections = {}
        
        all_agents = list(self.agent_factory.agent_profiles.keys())
        n = len(all_agents)
        
        # Start with small core network
        m0 = 3
        core_agents = all_agents[:m0]
        
        # Connect core agents
        for agent_id in core_agents:
            connections[agent_id] = [a for a in core_agents if a != agent_id]
        
        # Add remaining agents with preferential attachment
        for new_agent in all_agents[m0:]:
            connections[new_agent] = []
            
            # Calculate degrees (number of connections)
            degrees = {agent: len(connections[agent]) for agent in connections}
            total_degree = sum(degrees.values())
            
            # Select m nodes to connect to (preferential attachment)
            m = 3
            for _ in range(m):
                # Select node with probability proportional to degree
                rand_val = random.uniform(0, total_degree)
                cumulative = 0
                
                for agent, degree in degrees.items():
                    cumulative += degree
                    if cumulative >= rand_val:
                        connections[new_agent].append(agent)
                        connections[agent].append(new_agent)
                        degrees[agent] += 1
                        total_degree += 1
                        break
        
        self.logger.info("Created scale-free topology")
        return connections
    
    def _create_hybrid_topology(self) -> Dict[str, List[str]]:
        """Create hybrid topology combining multiple strategies"""
        connections = {}
        
        # Start with domain-centric base
        domain_connections = self._create_domain_centric_topology()
        
        # Add small-world properties within domains
        for domain in AgentDomain:
            domain_agents = self.agent_factory.get_domain_agents(domain)
            
            # Add random connections within domain
            for _ in range(len(domain_agents) // 5):  # 20% random connections
                agent1, agent2 = random.sample(domain_agents, 2)
                if agent2 not in domain_connections[agent1]:
                    domain_connections[agent1].append(agent2)
                    domain_connections[agent2].append(agent1)
        
        # Add scale-free properties for coordination agents
        coord_agents = ["master_coordinator_001"] + list(self.agent_factory.domain_leaders.values())
        
        for coord_agent in coord_agents:
            if coord_agent in domain_connections:
                # Connect coordination agents with higher probability
                for other_coord in coord_agents:
                    if other_coord != coord_agent and random.random() < 0.7:
                        if other_coord not in domain_connections[coord_agent]:
                            domain_connections[coord_agent].append(other_coord)
                            domain_connections[other_coord].append(coord_agent)
        
        connections = domain_connections
        self.logger.info("Created hybrid topology")
        return connections
    
    def apply_topology(self, connections: Dict[str, List[str]], network: CommunicationNetwork):
        """Apply topology to communication network"""
        self.logger.info("Applying topology to communication network")
        
        connections_created = 0
        
        for source_id, target_ids in connections.items():
            source_profile = self.agent_factory.agent_profiles.get(source_id)
            if not source_profile:
                continue
            
            for target_id in target_ids:
                target_profile = self.agent_factory.agent_profiles.get(target_id)
                if not target_profile:
                    continue
                
                # Calculate connection weight based on specializations
                weight = self._calculate_connection_weight(source_profile, target_profile)
                
                # Create connection
                source_profile.instance.add_connection(target_profile.instance, weight)
                connections_created += 1
                
                # Update connection lists
                if target_id not in source_profile.connections:
                    source_profile.connections.append(target_id)
                if source_id not in target_profile.connections:
                    target_profile.connections.append(source_id)
        
        self.logger.info(f"Applied topology: {connections_created} connections created")
        return connections_created
    
    def _calculate_connection_weight(self, source_profile: AgentProfile, target_profile: AgentProfile) -> float:
        """Calculate connection weight between two agents"""
        base_weight = 0.5
        
        # Increase weight for preferred connections
        if target_profile.agent_id in source_profile.specialization.preferred_connections:
            base_weight += 0.3
        
        # Use specialization weights if available
        spec_weights = source_profile.specialization.connection_weights
        if target_profile.specialization.name in spec_weights:
            base_weight = spec_weights[target_profile.specialization.name]
        
        # Adjust based on domain relationship
        if source_profile.specialization.domain == target_profile.specialization.domain:
            base_weight += 0.2  # Same domain preference
        
        # Adjust based on complexity
        if (source_profile.specialization.complexity == AgentComplexity.EXPERT and
            target_profile.specialization.complexity == AgentComplexity.EXPERT):
            base_weight += 0.1
        
        # Normalize to [0.1, 1.0]
        return max(0.1, min(1.0, base_weight))
    
    def get_topology_metrics(self, connections: Dict[str, List[str]]) -> Dict[str, Any]:
        """Get topology metrics"""
        total_agents = len(connections)
        if total_agents == 0:
            return {}
        
        total_connections = sum(len(targets) for targets in connections.values())
        avg_connections = total_connections / total_agents
        
        # Calculate density
        max_possible = total_agents * (total_agents - 1)
        density = total_connections / max_possible if max_possible > 0 else 0
        
        # Find most connected agents
        connection_counts = {agent: len(targets) for agent, targets in connections.items()}
        max_connections = max(connection_counts.values()) if connection_counts else 0
        most_connected = [agent for agent, count in connection_counts.items() if count == max_connections]
        
        return {
            "total_agents": total_agents,
            "total_connections": total_connections,
            "average_connections": avg_connections,
            "density": density,
            "max_connections": max_connections,
            "most_connected_agents": most_connected
        }

# Factory function for easy initialization
def create_advanced_agent_factory() -> AdvancedAgentFactory:
    """Create and initialize advanced agent factory"""
    registry = AgentSpecializationRegistry()
    factory = AdvancedAgentFactory(registry)
    return factory