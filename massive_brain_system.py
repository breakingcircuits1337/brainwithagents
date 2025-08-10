"""
250-Agent Brain System - Complete Implementation

This module integrates all components to create a sophisticated brain system
with 250 specialized AI agents working together collaboratively.
"""

import time
import random
import threading
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import logging
import json

# Import all components
from agent_specializations import agent_registry, AgentDomain, AgentSpecialization
from massive_agent_factory import MassiveAgentFactory, AdvancedAgent, AgentClusterManager
from advanced_network_topology import AdvancedNetworkTopology, NetworkTopology, TopologyConfig
from sophisticated_coordination import (
    CoordinationManager, CoordinationStrategy, CoordinationTask, TaskPriority
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BrainSystemState(Enum):
    """Brain system operational states"""
    INITIALIZING = "initializing"
    READY = "ready"
    PROCESSING = "processing"
    OPTIMIZING = "optimizing"
    LEARNING = "learning"
    ERROR = "error"
    SHUTDOWN = "shutdown"

class BrainSystemMode(Enum):
    """Brain system operational modes"""
    COLLABORATIVE = "collaborative"
    COMPETITIVE = "competitive"
    ADAPTIVE = "adaptive"
    EMERGENT = "emergent"
    HYBRID = "hybrid"

@dataclass
class BrainSystemMetrics:
    """Comprehensive metrics for the 250-agent brain system"""
    total_agents: int = 250
    active_agents: int = 0
    total_connections: int = 0
    network_efficiency: float = 0.0
    coordination_efficiency: float = 0.0
    processing_throughput: float = 0.0
    learning_rate: float = 0.0
    adaptation_score: float = 0.0
    system_health: float = 1.0
    resource_utilization: Dict[str, float] = field(default_factory=dict)
    domain_performance: Dict[str, float] = field(default_factory=dict)
    collaboration_metrics: Dict[str, float] = field(default_factory=dict)
    error_rate: float = 0.0
    uptime: float = 0.0

class MassiveBrainSystem:
    """Main brain system class for 250-agent coordination"""
    
    def __init__(self, brain_id: str = "massive_brain_250"):
        self.id = brain_id
        self.state = BrainSystemState.INITIALIZING
        self.mode = BrainSystemMode.HYBRID
        
        # Core components
        self.agent_factory = MassiveAgentFactory()
        self.agents: Dict[str, AdvancedAgent] = {}
        self.cluster_manager: Optional[AgentClusterManager] = None
        self.topology_manager: Optional[AdvancedNetworkTopology] = None
        self.coordination_manager: Optional[CoordinationManager] = None
        
        # System configuration
        self.config = {
            "auto_optimization": True,
            "auto_learning": True,
            "adaptation_rate": 0.05,
            "optimization_interval": 100,
            "health_check_interval": 10,
            "max_concurrent_tasks": 100,
            "cluster_size": 10,
            "network_topology": NetworkTopology.HYBRID,
            "coordination_strategy": CoordinationStrategy.HYBRID
        }
        
        # Metrics and monitoring
        self.metrics = BrainSystemMetrics()
        self.system_history = deque(maxlen=10000)
        self.performance_history = deque(maxlen=1000)
        self.health_checks = deque(maxlen=1000)
        
        # System management
        self.start_time = time.time()
        self.last_optimization = time.time()
        self.is_running = False
        self.system_lock = threading.Lock()
        
        # Event handlers
        self.event_handlers = defaultdict(list)
        
        self.logger = logging.getLogger(f"{__name__}.MassiveBrainSystem.{brain_id}")
        
        # Initialize system
        self._initialize_system()
    
    def _initialize_system(self):
        """Initialize the 250-agent brain system"""
        self.logger.info(f"Initializing 250-agent brain system: {self.id}")
        
        try:
            # Step 1: Create all specialized agents
            self.logger.info("Creating 250 specialized agents...")
            self.agents = self.agent_factory.create_all_agents()
            self.logger.info(f"Created {len(self.agents)} specialized agents")
            
            # Step 2: Create cluster manager
            self.logger.info("Creating cluster manager...")
            self.cluster_manager = AgentClusterManager(self.agents)
            self.cluster_manager.form_clusters(self.config["cluster_size"])
            self.logger.info("Agent clusters formed")
            
            # Step 3: Create network topology manager
            self.logger.info("Creating network topology manager...")
            topology_config = TopologyConfig(
                topology_type=self.config["network_topology"],
                max_connections_per_agent=15,
                connection_probability=0.3,
                rewire_probability=0.1,
                cluster_coefficient=0.6
            )
            self.topology_manager = AdvancedNetworkTopology(self.agents, topology_config)
            self.logger.info("Network topology established")
            
            # Step 4: Create coordination manager
            self.logger.info("Creating coordination manager...")
            self.coordination_manager = CoordinationManager(self.agents, self.topology_manager)
            self.logger.info("Coordination system established")
            
            # Step 5: Initialize metrics
            self._initialize_metrics()
            
            # Step 6: Set system state to ready
            self.state = BrainSystemState.READY
            self.logger.info("250-agent brain system initialization completed")
            
            # Trigger system ready event
            self._trigger_event("system_ready", {
                "total_agents": len(self.agents),
                "clusters": len(self.cluster_manager.clusters) if self.cluster_manager else 0,
                "connections": self.topology_manager.topology_metrics.total_connections if self.topology_manager else 0
            })
            
        except Exception as e:
            self.logger.error(f"Error during system initialization: {e}")
            self.state = BrainSystemState.ERROR
            self._trigger_event("system_error", {"error": str(e)})
    
    def _initialize_metrics(self):
        """Initialize system metrics"""
        self.metrics.total_agents = len(self.agents)
        self.metrics.active_agents = len(self.agents)
        self.metrics.total_connections = (
            self.topology_manager.topology_metrics.total_connections if self.topology_manager else 0
        )
        
        # Initialize domain performance
        for domain in AgentDomain:
            self.metrics.domain_performance[domain.value] = 0.0
        
        # Initialize resource utilization
        self.metrics.resource_utilization = {
            "cpu": 0.0,
            "memory": 0.0,
            "network": 0.0,
            "coordination": 0.0
        }
        
        # Initialize collaboration metrics
        self.metrics.collaboration_metrics = {
            "total_collaborations": 0,
            "successful_collaborations": 0,
            "average_collaboration_size": 0.0,
            "collaboration_efficiency": 0.0
        }
    
    def start_system(self):
        """Start the 250-agent brain system"""
        if self.state != BrainSystemState.READY:
            self.logger.error(f"Cannot start system in state: {self.state}")
            return False
        
        self.logger.info("Starting 250-agent brain system")
        
        try:
            # Start coordination system
            if self.coordination_manager:
                self.coordination_manager.start_coordination()
            
            # Set system state
            self.state = BrainSystemState.PROCESSING
            self.is_running = True
            
            # Start system management threads
            self._start_management_threads()
            
            self.logger.info("250-agent brain system started successfully")
            
            # Trigger system started event
            self._trigger_event("system_started", {
                "start_time": time.time(),
                "mode": self.mode.value
            })
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting system: {e}")
            self.state = BrainSystemState.ERROR
            self._trigger_event("system_error", {"error": str(e)})
            return False
    
    def stop_system(self):
        """Stop the 250-agent brain system"""
        self.logger.info("Stopping 250-agent brain system")
        
        try:
            # Stop coordination system
            if self.coordination_manager:
                self.coordination_manager.stop_coordination()
            
            # Set system state
            self.is_running = False
            self.state = BrainSystemState.SHUTDOWN
            
            # Update uptime
            self.metrics.uptime = time.time() - self.start_time
            
            self.logger.info("250-agent brain system stopped successfully")
            
            # Trigger system stopped event
            self._trigger_event("system_stopped", {
                "stop_time": time.time(),
                "uptime": self.metrics.uptime
            })
            
        except Exception as e:
            self.logger.error(f"Error stopping system: {e}")
            self._trigger_event("system_error", {"error": str(e)})
    
    def _start_management_threads(self):
        """Start system management threads"""
        # Health monitoring thread
        threading.Thread(target=self._health_monitoring_loop, daemon=True).start()
        
        # Performance optimization thread
        threading.Thread(target=self._performance_optimization_loop, daemon=True).start()
        
        # System adaptation thread
        threading.Thread(target=self._system_adaptation_loop, daemon=True).start()
        
        # Metrics collection thread
        threading.Thread(target=self._metrics_collection_loop, daemon=True).start()
    
    def _health_monitoring_loop(self):
        """Health monitoring loop"""
        while self.is_running:
            try:
                time.sleep(self.config["health_check_interval"])
                
                # Perform health check
                health_status = self._perform_health_check()
                
                # Record health check
                self.health_checks.append({
                    "timestamp": time.time(),
                    "health_status": health_status,
                    "system_state": self.state.value
                })
                
                # Handle health issues
                if not health_status["healthy"]:
                    self._handle_health_issues(health_status)
                
            except Exception as e:
                self.logger.error(f"Error in health monitoring loop: {e}")
    
    def _performance_optimization_loop(self):
        """Performance optimization loop"""
        while self.is_running:
            try:
                time.sleep(self.config["optimization_interval"])
                
                if self.config["auto_optimization"]:
                    # Optimize network topology
                    if self.topology_manager:
                        self.topology_manager.optimize_topology()
                    
                    # Optimize coordination
                    if self.coordination_manager:
                        self.coordination_manager.performance_optimizer.optimize_coordination()
                    
                    # Update last optimization time
                    self.last_optimization = time.time()
                    
                    self.logger.debug("Performance optimization completed")
                
            except Exception as e:
                self.logger.error(f"Error in performance optimization loop: {e}")
    
    def _system_adaptation_loop(self):
        """System adaptation loop"""
        while self.is_running:
            try:
                time.sleep(30.0)  # Adapt every 30 seconds
                
                if self.config["auto_learning"]:
                    # Adapt network topology
                    if self.topology_manager:
                        self.topology_manager.adapt_to_conditions()
                    
                    # Adapt coordination strategy
                    self._adapt_coordination_strategy()
                    
                    # Update adaptation score
                    self.metrics.adaptation_score = min(1.0, self.metrics.adaptation_score + 0.01)
                    
                    self.logger.debug("System adaptation completed")
                
            except Exception as e:
                self.logger.error(f"Error in system adaptation loop: {e}")
    
    def _metrics_collection_loop(self):
        """Metrics collection loop"""
        while self.is_running:
            try:
                time.sleep(5.0)  # Collect metrics every 5 seconds
                
                # Update system metrics
                self._update_system_metrics()
                
                # Record performance history
                self.performance_history.append({
                    "timestamp": time.time(),
                    "metrics": self.metrics.__dict__.copy()
                })
                
            except Exception as e:
                self.logger.error(f"Error in metrics collection loop: {e}")
    
    def _perform_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        health_status = {
            "healthy": True,
            "issues": [],
            "component_status": {}
        }
        
        # Check agent health
        active_agents = 0
        for agent_id, agent in self.agents.items():
            if agent.state.value != "error":
                active_agents += 1
            else:
                health_status["issues"].append(f"Agent {agent_id} in error state")
        
        health_status["component_status"]["agents"] = {
            "total": len(self.agents),
            "active": active_agents,
            "health_ratio": active_agents / len(self.agents)
        }
        
        # Check network health
        if self.topology_manager:
            network_health = self.topology_manager.topology_metrics.robustness_score
            health_status["component_status"]["network"] = {
                "robustness": network_health,
                "connections": self.topology_manager.topology_metrics.total_connections
            }
            
            if network_health < 0.7:
                health_status["issues"].append("Low network robustness")
        
        # Check coordination health
        if self.coordination_manager:
            coord_efficiency = self.coordination_manager.metrics.coordination_efficiency
            health_status["component_status"]["coordination"] = {
                "efficiency": coord_efficiency,
                "active_tasks": len(self.coordination_manager.active_tasks)
            }
            
            if coord_efficiency < 0.7:
                health_status["issues"].append("Low coordination efficiency")
        
        # Determine overall health
        if health_status["issues"]:
            health_status["healthy"] = False
        
        return health_status
    
    def _handle_health_issues(self, health_status: Dict[str, Any]):
        """Handle health issues"""
        self.logger.warning(f"Health issues detected: {health_status['issues']}")
        
        # Try to resolve issues
        for issue in health_status["issues"]:
            if "Agent" in issue and "error state" in issue:
                # Restart error agents
                self._restart_error_agents()
            elif "Low network robustness" in issue:
                # Optimize network
                if self.topology_manager:
                    self.topology_manager.optimize_topology()
            elif "Low coordination efficiency" in issue:
                # Optimize coordination
                if self.coordination_manager:
                    self.coordination_manager.performance_optimizer.optimize_coordination()
    
    def _restart_error_agents(self):
        """Restart agents in error state"""
        error_agents = [
            agent_id for agent_id, agent in self.agents.items()
            if agent.state.value == "error"
        ]
        
        for agent_id in error_agents:
            try:
                # Reset agent state
                agent = self.agents[agent_id]
                agent.state = agent.state.IDLE
                agent.load.queue_size = 0
                agent.metrics.error_count = 0
                
                self.logger.info(f"Restarted agent: {agent_id}")
                
            except Exception as e:
                self.logger.error(f"Error restarting agent {agent_id}: {e}")
    
    def _adapt_coordination_strategy(self):
        """Adapt coordination strategy based on performance"""
        if not self.coordination_manager:
            return
        
        # Get current strategy performance
        current_strategy = self.coordination_manager.active_strategy
        strategy_performance = self.coordination_manager.strategy_performance.get(current_strategy.value, 0.5)
        
        # Try different strategies if performance is low
        if strategy_performance < 0.6:
            strategies = list(CoordinationStrategy)
            current_index = strategies.index(current_strategy)
            
            # Try next strategy
            next_strategy = strategies[(current_index + 1) % len(strategies)]
            self.coordination_manager.active_strategy = next_strategy
            
            self.logger.info(f"Adapted coordination strategy to: {next_strategy.value}")
    
    def _update_system_metrics(self):
        """Update system metrics"""
        # Update active agents
        active_agents = sum(1 for agent in self.agents.values() if agent.state.value != "error")
        self.metrics.active_agents = active_agents
        
        # Update network efficiency
        if self.topology_manager:
            self.metrics.network_efficiency = self.topology_manager.topology_metrics.efficiency_score
        
        # Update coordination efficiency
        if self.coordination_manager:
            self.metrics.coordination_efficiency = self.coordination_manager.metrics.coordination_efficiency
            
            # Update processing throughput
            total_tasks = self.coordination_manager.metrics.total_tasks_completed
            uptime = time.time() - self.start_time
            self.metrics.processing_throughput = total_tasks / max(1, uptime)
            
            # Update error rate
            total_created = self.coordination_manager.metrics.total_tasks_created
            if total_created > 0:
                self.metrics.error_rate = (total_created - total_tasks) / total_created
        
        # Update domain performance
        self._update_domain_performance()
        
        # Update resource utilization
        self._update_resource_utilization()
        
        # Update collaboration metrics
        self._update_collaboration_metrics()
        
        # Update system health
        self._update_system_health()
    
    def _update_domain_performance(self):
        """Update domain performance metrics"""
        domain_performance = defaultdict(list)
        
        for agent in self.agents.values():
            domain = agent.specialization.domain.value
            performance_score = (
                agent.metrics.efficiency_score * 0.5 +
                agent.metrics.success_rate * 0.3 +
                (1.0 - agent.load.queue_size / 20.0) * 0.2
            )
            domain_performance[domain].append(performance_score)
        
        # Calculate average performance per domain
        for domain, scores in domain_performance.items():
            if scores:
                self.metrics.domain_performance[domain] = sum(scores) / len(scores)
    
    def _update_resource_utilization(self):
        """Update resource utilization metrics"""
        total_cpu = sum(agent.load.cpu_usage for agent in self.agents.values())
        total_memory = sum(agent.load.memory_usage for agent in self.agents.values())
        total_network = sum(agent.load.network_usage for agent in self.agents.values())
        
        self.metrics.resource_utilization["cpu"] = min(1.0, total_cpu / len(self.agents))
        self.metrics.resource_utilization["memory"] = min(1.0, total_memory / len(self.agents))
        self.metrics.resource_utilization["network"] = min(1.0, total_network / len(self.agents))
        
        # Coordination overhead
        if self.coordination_manager:
            coord_overhead = self.coordination_manager.metrics.coordination_overhead
            self.metrics.resource_utilization["coordination"] = min(1.0, coord_overhead / 20.0)
    
    def _update_collaboration_metrics(self):
        """Update collaboration metrics"""
        if not self.coordination_manager:
            return
        
        # Count total collaborations
        total_collaborations = sum(self.coordination_manager.metrics.collaboration_frequency.values())
        self.metrics.collaboration_metrics["total_collaborations"] = total_collaborations
        
        # Count successful collaborations (estimated)
        successful_collaborations = int(total_collaborations * 0.9)  # Assume 90% success rate
        self.metrics.collaboration_metrics["successful_collaborations"] = successful_collaborations
        
        # Calculate average collaboration size
        if total_collaborations > 0:
            total_participants = sum(
                len(key.split("->")) for key in self.coordination_manager.metrics.collaboration_frequency.keys()
            )
            self.metrics.collaboration_metrics["average_collaboration_size"] = total_participants / total_collaborations
        
        # Calculate collaboration efficiency
        if total_collaborations > 0:
            efficiency = successful_collaborations / total_collaborations
            self.metrics.collaboration_metrics["collaboration_efficiency"] = efficiency
    
    def _update_system_health(self):
        """Update overall system health"""
        health_factors = []
        
        # Agent health
        agent_health = self.metrics.active_agents / self.metrics.total_agents
        health_factors.append(agent_health)
        
        # Network health
        if self.topology_manager:
            network_health = self.topology_manager.topology_metrics.robustness_score
            health_factors.append(network_health)
        
        # Coordination health
        if self.coordination_manager:
            coord_health = self.coordination_manager.metrics.coordination_efficiency
            health_factors.append(coord_health)
        
        # Resource health
        resource_health = 1.0 - sum(self.metrics.resource_utilization.values()) / len(self.metrics.resource_utilization)
        health_factors.append(resource_health)
        
        # Calculate overall health
        if health_factors:
            self.metrics.system_health = sum(health_factors) / len(health_factors)
        else:
            self.metrics.system_health = 1.0
    
    def process_task(self, task_description: str, required_capabilities: List[str],
                    priority: TaskPriority = TaskPriority.MEDIUM,
                    complexity: float = 0.5, **kwargs) -> str:
        """Process a task through the 250-agent system"""
        if self.state != BrainSystemState.PROCESSING:
            self.logger.error(f"Cannot process task in state: {self.state}")
            return None
        
        self.logger.info(f"Processing task: {task_description}")
        
        try:
            # Create coordination task
            task_id = self.coordination_manager.create_coordination_task(
                description=task_description,
                required_capabilities=required_capabilities,
                priority=priority,
                complexity=complexity,
                **kwargs
            )
            
            # Trigger task processing event
            self._trigger_event("task_processed", {
                "task_id": task_id,
                "description": task_description,
                "priority": priority.value,
                "complexity": complexity
            })
            
            return task_id
            
        except Exception as e:
            self.logger.error(f"Error processing task: {e}")
            self._trigger_event("task_error", {"error": str(e)})
            return None
    
    def coordinate_collaboration(self, initiator_id: str, target_ids: List[str],
                              collaboration_type: str, content: Any) -> bool:
        """Coordinate collaboration between agents"""
        if not self.coordination_manager:
            return False
        
        return self.coordination_manager.coordinate_agent_collaboration(
            initiator_id, target_ids, collaboration_type, content
        )
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        status = {
            "system_id": self.id,
            "state": self.state.value,
            "mode": self.mode.value,
            "is_running": self.is_running,
            "uptime": time.time() - self.start_time,
            "metrics": self.metrics.__dict__,
            "components": {}
        }
        
        # Add component status
        if self.agents:
            status["components"]["agents"] = {
                "total": len(self.agents),
                "active": self.metrics.active_agents,
                "domain_distribution": self._get_domain_distribution()
            }
        
        if self.cluster_manager:
            status["components"]["clusters"] = {
                "total": len(self.cluster_manager.clusters),
                "average_size": sum(len(cluster) for cluster in self.cluster_manager.clusters.values()) / len(self.cluster_manager.clusters)
            }
        
        if self.topology_manager:
            status["components"]["network"] = self.topology_manager.get_topology_status()
        
        if self.coordination_manager:
            status["components"]["coordination"] = self.coordination_manager.get_coordination_status()
        
        return status
    
    def _get_domain_distribution(self) -> Dict[str, int]:
        """Get agent distribution by domain"""
        domain_distribution = defaultdict(int)
        
        for agent in self.agents.values():
            domain = agent.specialization.domain.value
            domain_distribution[domain] += 1
        
        return dict(domain_distribution)
    
    def get_agent_details(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific agent"""
        if agent_id not in self.agents:
            return None
        
        agent = self.agents[agent_id]
        return agent.get_status()
    
    def get_cluster_info(self, cluster_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific cluster"""
        if not self.cluster_manager:
            return None
        
        if cluster_name not in self.cluster_manager.clusters:
            return None
        
        members = self.cluster_manager.get_cluster_members(cluster_name)
        leader = self.cluster_manager.cluster_leaders.get(cluster_name)
        
        return {
            "cluster_name": cluster_name,
            "members": list(members),
            "leader": leader,
            "size": len(members),
            "domain_distribution": self._get_cluster_domain_distribution(members)
        }
    
    def _get_cluster_domain_distribution(self, members: Set[str]) -> Dict[str, int]:
        """Get domain distribution within a cluster"""
        domain_distribution = defaultdict(int)
        
        for member_id in members:
            if member_id in self.agents:
                agent = self.agents[member_id]
                domain = agent.specialization.domain.value
                domain_distribution[domain] += 1
        
        return dict(domain_distribution)
    
    def get_network_visualization_data(self) -> Optional[Dict[str, Any]]:
        """Get network visualization data"""
        if not self.topology_manager:
            return None
        
        return self.topology_manager.get_network_visualization_data()
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        stats = {
            "system_info": {
                "id": self.id,
                "state": self.state.value,
                "mode": self.mode.value,
                "uptime": time.time() - self.start_time,
                "last_optimization": time.time() - self.last_optimization
            },
            "agent_statistics": self.agent_factory.get_network_statistics(),
            "performance_metrics": self.metrics.__dict__,
            "coordination_stats": self.coordination_manager.get_coordination_status() if self.coordination_manager else {},
            "network_stats": self.topology_manager.get_topology_status() if self.topology_manager else {}
        }
        
        return stats
    
    def optimize_system(self):
        """Manually trigger system optimization"""
        self.logger.info("Manual system optimization triggered")
        
        # Optimize network topology
        if self.topology_manager:
            self.topology_manager.optimize_topology()
        
        # Optimize coordination
        if self.coordination_manager:
            self.coordination_manager.performance_optimizer.optimize_coordination()
        
        # Update optimization time
        self.last_optimization = time.time()
        
        # Trigger optimization event
        self._trigger_event("system_optimized", {
            "optimization_time": self.last_optimization
        })
        
        self.logger.info("System optimization completed")
    
    def adapt_system(self):
        """Manually trigger system adaptation"""
        self.logger.info("Manual system adaptation triggered")
        
        # Adapt network topology
        if self.topology_manager:
            self.topology_manager.adapt_to_conditions()
        
        # Adapt coordination strategy
        self._adapt_coordination_strategy()
        
        # Update adaptation score
        self.metrics.adaptation_score = min(1.0, self.metrics.adaptation_score + 0.05)
        
        # Trigger adaptation event
        self._trigger_event("system_adapted", {
            "adaptation_score": self.metrics.adaptation_score
        })
        
        self.logger.info("System adaptation completed")
    
    def set_system_mode(self, mode: BrainSystemMode):
        """Set system operational mode"""
        old_mode = self.mode
        self.mode = mode
        
        self.logger.info(f"System mode changed from {old_mode.value} to {mode.value}")
        
        # Trigger mode change event
        self._trigger_event("mode_changed", {
            "old_mode": old_mode.value,
            "new_mode": mode.value,
            "timestamp": time.time()
        })
    
    def add_event_handler(self, event_type: str, handler: callable):
        """Add event handler"""
        self.event_handlers[event_type].append(handler)
    
    def _trigger_event(self, event_type: str, data: Dict[str, Any]):
        """Trigger system event"""
        # Add to system history
        self.system_history.append({
            "timestamp": time.time(),
            "type": event_type,
            "data": data
        })
        
        # Call event handlers
        for handler in self.event_handlers.get(event_type, []):
            try:
                handler(data)
            except Exception as e:
                self.logger.error(f"Error in event handler for {event_type}: {e}")
    
    def save_system_state(self, filename: str):
        """Save system state to file"""
        system_state = {
            "system_id": self.id,
            "state": self.state.value,
            "mode": self.mode.value,
            "config": self.config,
            "metrics": self.metrics.__dict__,
            "start_time": self.start_time,
            "last_optimization": self.last_optimization
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(system_state, f, indent=2, default=str)
            
            self.logger.info(f"System state saved to {filename}")
            
        except Exception as e:
            self.logger.error(f"Error saving system state: {e}")
    
    def load_system_state(self, filename: str):
        """Load system state from file"""
        try:
            with open(filename, 'r') as f:
                system_state = json.load(f)
            
            self.id = system_state["system_id"]
            self.state = BrainSystemState(system_state["state"])
            self.mode = BrainSystemMode(system_state["mode"])
            self.config = system_state["config"]
            
            # Restore metrics
            for key, value in system_state["metrics"].items():
                if hasattr(self.metrics, key):
                    setattr(self.metrics, key, value)
            
            self.start_time = system_state["start_time"]
            self.last_optimization = system_state["last_optimization"]
            
            self.logger.info(f"System state loaded from {filename}")
            
        except Exception as e:
            self.logger.error(f"Error loading system state: {e}")

# Factory function to create a 250-agent brain system
def create_massive_brain_system(brain_id: str = "massive_brain_250") -> MassiveBrainSystem:
    """Create a 250-agent brain system"""
    return MassiveBrainSystem(brain_id)

# Global instance
massive_brain_system = None