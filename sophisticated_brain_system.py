"""
Sophisticated 250-Agent Brain System

This module implements a sophisticated brain system with 250 specialized AI agents
working together collaboratively with advanced coordination, communication, and management.
"""

import time
import threading
import logging
import json
from typing import Dict, List, Any, Optional, Set, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
import random
from collections import defaultdict, deque

# Import existing systems
from hrm_system import HRMSystem, Task, TaskPriority
from communication_system import CommunicationNetwork, Signal, SignalType, SignalGenerator
from agent_specializations import AgentSpecializationRegistry, AgentDomain, AgentComplexity
from advanced_agent_factory import (
    AdvancedAgentFactory, AgentProfile, AgentState, AgentHealth, 
    NetworkTopologyManager
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SystemMode(Enum):
    """System operational modes"""
    EMERGENCY = "emergency"      # Crisis management mode
    MAINTENANCE = "maintenance"  # System maintenance mode
    NORMAL = "normal"           # Normal operation mode
    OPTIMIZATION = "optimization"  # Performance optimization mode
    LEARNING = "learning"       # Active learning mode
    COLLABORATIVE = "collaborative"  # Enhanced collaboration mode

class CoordinationStrategy(Enum):
    """Coordination strategies for agent interaction"""
    CENTRALIZED = "centralized"        # Master coordinator controls all
    DISTRIBUTED = "distributed"        # Agents coordinate locally
    HIERARCHICAL = "hierarchical"      # Multi-level coordination
    MARKET_BASED = "market_based"      # Economic/resource-based coordination
    SWARM = "swarm"                   # Emergent coordination
    HYBRID = "hybrid"                 # Adaptive combination

@dataclass
class SystemMetrics:
    """Comprehensive metrics for the 250-agent system"""
    total_agents: int = 0
    active_agents: int = 0
    total_signals_processed: int = 0
    successful_signals: int = 0
    failed_signals: int = 0
    average_processing_time: float = 0.0
    system_efficiency: float = 1.0
    collaboration_index: float = 0.0
    learning_progress: float = 0.0
    adaptation_score: float = 0.0
    resource_utilization: Dict[str, float] = field(default_factory=dict)
    domain_performance: Dict[str, float] = field(default_factory=dict)
    network_health: float = 1.0
    system_uptime: float = 0.0
    last_major_event: str = ""
    coordination_efficiency: float = 0.0

@dataclass
class CoordinationEvent:
    """Event for coordination system"""
    event_id: str
    event_type: str
    source_agent: str
    target_agents: List[str]
    content: Any
    priority: int
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

class SophisticatedBrainSystem:
    """Sophisticated brain system with 250 specialized agents"""
    
    def __init__(self, system_id: str = "sophisticated_brain_250"):
        self.system_id = system_id
        self.mode = SystemMode.NORMAL
        self.coordination_strategy = CoordinationStrategy.HYBRID
        
        # Core systems
        from agent_specializations import AgentSpecializationRegistry
        self.specialization_registry = AgentSpecializationRegistry()
        self.agent_factory = AdvancedAgentFactory(self.specialization_registry)
        self.communication_network = CommunicationNetwork()
        self.topology_manager = NetworkTopologyManager(self.agent_factory)
        
        # System state
        self.is_running = False
        self.start_time = time.time()
        self.metrics = SystemMetrics()
        
        # Coordination systems
        self.coordination_queue = queue.PriorityQueue()
        self.coordination_events: List[CoordinationEvent] = []
        self.coordination_handlers: Dict[str, Callable] = {}
        
        # Resource management
        self.resource_pool = {
            "cpu": 100.0,
            "memory": 100.0,
            "network": 100.0,
            "storage": 100.0
        }
        self.resource_allocations: Dict[str, Dict[str, float]] = {}
        
        # Task management
        self.task_queue = queue.PriorityQueue()
        self.active_tasks: Dict[str, Dict[str, Any]] = {}
        self.completed_tasks: List[Dict[str, Any]] = []
        
        # Learning and adaptation
        self.learning_data = deque(maxlen=10000)
        self.adaptation_history = []
        self.performance_patterns = defaultdict(list)
        
        # Threading and concurrency
        self.executor = ThreadPoolExecutor(max_workers=50)
        self.coordination_thread = None
        self.monitoring_thread = None
        self.learning_thread = None
        self.resource_thread = None
        
        # Event handlers
        self.event_handlers = {
            "agent_created": [],
            "agent_error": [],
            "coordination_event": [],
            "resource_alert": [],
            "system_mode_change": [],
            "performance_update": []
        }
        
        # Configuration
        self.config = {
            "max_concurrent_tasks": 1000,
            "coordination_interval": 0.1,
            "monitoring_interval": 1.0,
            "learning_interval": 5.0,
            "resource_interval": 0.5,
            "optimization_interval": 60.0,
            "max_coordination_events": 10000,
            "resource_thresholds": {
                "cpu": 0.8,
                "memory": 0.8,
                "network": 0.8,
                "storage": 0.8
            },
            "performance_targets": {
                "min_efficiency": 0.7,
                "max_error_rate": 0.1,
                "min_collaboration": 0.6
            }
        }
        
        self.logger = logging.getLogger(f"{__name__}.SophisticatedBrainSystem.{system_id}")
        
        # Initialize system
        self._initialize_system()
    
    def _initialize_system(self):
        """Initialize the sophisticated brain system"""
        self.logger.info(f"Initializing sophisticated brain system {self.system_id}")
        
        # Create all 250 agents
        self.logger.info("Creating 250 specialized agents...")
        agent_profiles = self.agent_factory.create_all_agents()
        
        # Add agents to communication network
        for agent_id, profile in agent_profiles.items():
            self.communication_network.add_agent(profile.instance)
        
        # Create network topology
        self.logger.info("Creating network topology...")
        connections = self.topology_manager.create_topology("hybrid")
        self.topology_manager.apply_topology(connections, self.communication_network)
        
        # Update metrics
        self.metrics.total_agents = len(agent_profiles)
        self.metrics.active_agents = len(agent_profiles)
        self.metrics.domain_performance = self._calculate_domain_performance()
        
        # Initialize coordination handlers
        self._initialize_coordination_handlers()
        
        # Set up resource allocations
        self._initialize_resource_allocations()
        
        self.logger.info(f"Sophisticated brain system initialized with {len(agent_profiles)} agents")
    
    def _initialize_coordination_handlers(self):
        """Initialize coordination event handlers"""
        self.coordination_handlers = {
            "task_assignment": self._handle_task_assignment,
            "resource_request": self._handle_resource_request,
            "collaboration_request": self._handle_collaboration_request,
            "learning_update": self._handle_learning_update,
            "error_report": self._handle_error_report,
            "performance_alert": self._handle_performance_alert,
            "coordination_sync": self._handle_coordination_sync
        }
    
    def _initialize_resource_allocations(self):
        """Initialize resource allocations for agents"""
        for agent_id, profile in self.agent_factory.agent_profiles.items():
            base_allocation = {
                "cpu": 0.1,
                "memory": 0.1,
                "network": 0.1,
                "storage": 0.1
            }
            
            # Adjust based on agent complexity
            complexity_multiplier = {
                AgentComplexity.BASIC: 0.5,
                AgentComplexity.INTERMEDIATE: 0.75,
                AgentComplexity.ADVANCED: 1.0,
                AgentComplexity.EXPERT: 1.5
            }
            
            multiplier = complexity_multiplier[profile.specialization.complexity]
            for resource in base_allocation:
                base_allocation[resource] *= multiplier
            
            self.resource_allocations[agent_id] = base_allocation
    
    def start_system(self):
        """Start the sophisticated brain system"""
        if self.is_running:
            self.logger.warning("System is already running")
            return
        
        self.logger.info("Starting sophisticated brain system")
        self.is_running = True
        self.start_time = time.time()
        
        # Start coordination thread
        self.coordination_thread = threading.Thread(target=self._coordination_loop)
        self.coordination_thread.daemon = True
        self.coordination_thread.start()
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        # Start learning thread
        self.learning_thread = threading.Thread(target=self._learning_loop)
        self.learning_thread.daemon = True
        self.learning_thread.start()
        
        # Start resource management thread
        self.resource_thread = threading.Thread(target=self._resource_management_loop)
        self.resource_thread.daemon = True
        self.resource_thread.start()
        
        self.logger.info("Sophisticated brain system started")
    
    def stop_system(self):
        """Stop the sophisticated brain system"""
        if not self.is_running:
            self.logger.warning("System is not running")
            return
        
        self.logger.info("Stopping sophisticated brain system")
        self.is_running = False
        
        # Stop threads
        if self.coordination_thread:
            self.coordination_thread.join(timeout=5.0)
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        
        if self.learning_thread:
            self.learning_thread.join(timeout=5.0)
        
        if self.resource_thread:
            self.resource_thread.join(timeout=5.0)
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        # Shutdown agent factory
        self.agent_factory.shutdown()
        
        self.logger.info("Sophisticated brain system stopped")
    
    def _coordination_loop(self):
        """Main coordination loop"""
        self.logger.debug("Starting coordination loop")
        
        while self.is_running:
            try:
                # Process coordination events
                self._process_coordination_events()
                
                # Handle task assignments
                self._handle_task_assignments()
                
                # Synchronize agent activities
                self._synchronize_agents()
                
                # Small delay
                time.sleep(self.config["coordination_interval"])
                
            except Exception as e:
                self.logger.error(f"Error in coordination loop: {e}")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        self.logger.debug("Starting monitoring loop")
        
        while self.is_running:
            try:
                # Update system metrics
                self._update_system_metrics()
                
                # Monitor agent health
                self._monitor_agent_health()
                
                # Check for performance issues
                self._check_performance_issues()
                
                # Generate alerts if needed
                self._generate_alerts()
                
                # Update system uptime
                self.metrics.system_uptime = time.time() - self.start_time
                
                # Small delay
                time.sleep(self.config["monitoring_interval"])
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
    
    def _learning_loop(self):
        """Main learning loop"""
        self.logger.debug("Starting learning loop")
        
        while self.is_running:
            try:
                if self.mode in [SystemMode.LEARNING, SystemMode.OPTIMIZATION]:
                    # Collect learning data
                    self._collect_learning_data()
                    
                    # Analyze performance patterns
                    self._analyze_performance_patterns()
                    
                    # Adapt system behavior
                    self._adapt_system_behavior()
                    
                    # Update learning progress
                    self._update_learning_progress()
                
                # Small delay
                time.sleep(self.config["learning_interval"])
                
            except Exception as e:
                self.logger.error(f"Error in learning loop: {e}")
    
    def _resource_management_loop(self):
        """Main resource management loop"""
        self.logger.debug("Starting resource management loop")
        
        while self.is_running:
            try:
                # Monitor resource usage
                self._monitor_resource_usage()
                
                # Reallocate resources if needed
                self._reallocate_resources()
                
                # Handle resource requests
                self._handle_resource_requests()
                
                # Small delay
                time.sleep(self.config["resource_interval"])
                
            except Exception as e:
                self.logger.error(f"Error in resource management loop: {e}")
    
    def _process_coordination_events(self):
        """Process coordination events from queue"""
        events_processed = 0
        
        while not self.coordination_queue.empty() and events_processed < 10:
            try:
                # Get event from queue
                event_priority, event = self.coordination_queue.get_nowait()
                
                # Process event
                handler = self.coordination_handlers.get(event.event_type)
                if handler:
                    handler(event)
                
                # Store event
                self.coordination_events.append(event)
                
                # Limit event history
                if len(self.coordination_events) > self.config["max_coordination_events"]:
                    self.coordination_events = self.coordination_events[-self.config["max_coordination_events"]:]
                
                events_processed += 1
                
            except queue.Empty:
                break
            except Exception as e:
                self.logger.error(f"Error processing coordination event: {e}")
    
    def _handle_task_assignment(self, event: CoordinationEvent):
        """Handle task assignment coordination event"""
        task_data = event.content
        
        # Find suitable agents
        suitable_agents = self._find_suitable_agents(task_data)
        
        # Assign task to best agent
        if suitable_agents:
            best_agent = self._select_best_agent(suitable_agents, task_data)
            
            # Create task signal
            signal = SignalGenerator.create_data_signal(
                task_data,
                source_id=event.source_agent,
                strength=0.8
            )
            
            # Send to agent
            self.communication_network.send_signal_to_agent(signal, best_agent)
            
            # Track task
            task_id = f"task_{int(time.time())}"
            self.active_tasks[task_id] = {
                "agent_id": best_agent,
                "task_data": task_data,
                "start_time": time.time(),
                "priority": event.priority
            }
            
            self.logger.debug(f"Assigned task {task_id} to agent {best_agent}")
    
    def _handle_resource_request(self, event: CoordinationEvent):
        """Handle resource request coordination event"""
        request_data = event.content
        agent_id = event.source_agent
        
        # Check resource availability
        available_resources = self._check_resource_availability(request_data)
        
        if available_resources:
            # Allocate resources
            self._allocate_resources(agent_id, request_data)
            
            # Send approval
            response_event = CoordinationEvent(
                event_id=f"resource_response_{int(time.time())}",
                event_type="resource_allocation",
                source_agent="system",
                target_agents=[agent_id],
                content={"status": "approved", "resources": available_resources},
                priority=1
            )
            
            self.coordination_queue.put((1, response_event))
        else:
            # Queue request or deny
            response_event = CoordinationEvent(
                event_id=f"resource_response_{int(time.time())}",
                event_type="resource_allocation",
                source_agent="system",
                target_agents=[agent_id],
                content={"status": "denied", "reason": "insufficient_resources"},
                priority=1
            )
            
            self.coordination_queue.put((1, response_event))
    
    def _handle_collaboration_request(self, event: CoordinationEvent):
        """Handle collaboration request coordination event"""
        request_data = event.content
        source_agent = event.source_agent
        
        # Find collaboration partners
        partners = self._find_collaboration_partners(source_agent, request_data)
        
        if partners:
            # Create collaboration group
            collaboration_id = f"collab_{int(time.time())}"
            
            # Notify partners
            for partner in partners:
                signal = SignalGenerator.create_data_signal(
                    {
                        "collaboration_id": collaboration_id,
                        "initiator": source_agent,
                        "task": request_data
                    },
                    source_id="coordination_system",
                    strength=0.7
                )
                
                self.communication_network.send_signal_to_agent(signal, partner)
            
            self.logger.debug(f"Created collaboration {collaboration_id} with {len(partners)} partners")
    
    def _handle_learning_update(self, event: CoordinationEvent):
        """Handle learning update coordination event"""
        learning_data = event.content
        agent_id = event.source_agent
        
        # Store learning data
        self.learning_data.append({
            "agent_id": agent_id,
            "timestamp": time.time(),
            "learning_data": learning_data
        })
        
        # Propagate learning to relevant agents
        relevant_agents = self._find_relevant_agents_for_learning(agent_id, learning_data)
        
        for target_agent in relevant_agents:
            signal = SignalGenerator.create_data_signal(
                learning_data,
                source_id=agent_id,
                strength=0.6
            )
            
            self.communication_network.send_signal_to_agent(signal, target_agent)
    
    def _handle_error_report(self, event: CoordinationEvent):
        """Handle error report coordination event"""
        error_data = event.content
        agent_id = event.source_agent
        
        # Log error
        self.logger.error(f"Error reported by agent {agent_id}: {error_data}")
        
        # Update agent health
        self.agent_factory.update_agent_health(agent_id, AgentHealth.POOR)
        
        # Trigger recovery procedures
        self._trigger_agent_recovery(agent_id, error_data)
        
        # Notify dependent agents
        self._notify_dependent_agents(agent_id, error_data)
    
    def _handle_performance_alert(self, event: CoordinationEvent):
        """Handle performance alert coordination event"""
        alert_data = event.content
        agent_id = event.source_agent
        
        # Log alert
        self.logger.warning(f"Performance alert from agent {agent_id}: {alert_data}")
        
        # Analyze performance issue
        issue_analysis = self._analyze_performance_issue(agent_id, alert_data)
        
        # Apply optimization if needed
        if issue_analysis["requires_optimization"]:
            self._optimize_agent_performance(agent_id, issue_analysis)
    
    def _handle_coordination_sync(self, event: CoordinationEvent):
        """Handle coordination synchronization event"""
        sync_data = event.content
        agent_id = event.source_agent
        
        # Synchronize agent state
        self._synchronize_agent_state(agent_id, sync_data)
        
        # Update coordination metrics
        self.metrics.coordination_efficiency = self._calculate_coordination_efficiency()
    
    def _find_suitable_agents(self, task_data: Dict[str, Any]) -> List[str]:
        """Find suitable agents for a task"""
        suitable_agents = []
        
        # Extract task requirements
        task_type = task_data.get("type", "general")
        required_capabilities = task_data.get("capabilities", [])
        domain = task_data.get("domain")
        
        # Find agents matching criteria
        for agent_id, profile in self.agent_factory.agent_profiles.items():
            if profile.state == AgentState.ERROR:
                continue
            
            # Check domain match
            if domain and profile.specialization.domain.value != domain:
                continue
            
            # Check capability match
            agent_capabilities = [cap.name for cap in profile.specialization.capabilities]
            if required_capabilities and not any(cap in agent_capabilities for cap in required_capabilities):
                continue
            
            # Check load
            if profile.metrics.current_load > 0.8:
                continue
            
            suitable_agents.append(agent_id)
        
        return suitable_agents
    
    def _select_best_agent(self, suitable_agents: List[str], task_data: Dict[str, Any]) -> str:
        """Select the best agent from suitable candidates"""
        if not suitable_agents:
            return None
        
        # Score agents based on multiple criteria
        agent_scores = {}
        
        for agent_id in suitable_agents:
            profile = self.agent_factory.agent_profiles[agent_id]
            score = 0.0
            
            # Efficiency score
            score += profile.metrics.efficiency_score * 0.3
            
            # Load score (lower load is better)
            score += (1.0 - profile.metrics.current_load) * 0.3
            
            # Health score
            health_scores = {
                AgentHealth.EXCELLENT: 1.0,
                AgentHealth.GOOD: 0.8,
                AgentHealth.FAIR: 0.6,
                AgentHealth.POOR: 0.3,
                AgentHealth.CRITICAL: 0.1
            }
            score += health_scores[profile.health] * 0.2
            
            # Collaboration score
            score += profile.metrics.collaboration_score * 0.2
            
            agent_scores[agent_id] = score
        
        # Return agent with highest score
        return max(agent_scores, key=agent_scores.get)
    
    def _find_collaboration_partners(self, source_agent: str, task_data: Dict[str, Any]) -> List[str]:
        """Find collaboration partners for an agent"""
        partners = []
        
        # Get agent profile
        source_profile = self.agent_factory.agent_profiles.get(source_agent)
        if not source_profile:
            return partners
        
        # Find agents with complementary capabilities
        source_capabilities = {cap.name for cap in source_profile.specialization.capabilities}
        
        for agent_id, profile in self.agent_factory.agent_profiles.items():
            if agent_id == source_agent or profile.state == AgentState.ERROR:
                continue
            
            # Check for complementary capabilities
            agent_capabilities = {cap.name for cap in profile.specialization.capabilities}
            
            # Calculate complementarity (how much they add)
            complementarity = len(agent_capabilities - source_capabilities)
            
            if complementarity > 0:
                partners.append(agent_id)
        
        # Limit number of partners
        return partners[:5]
    
    def _find_relevant_agents_for_learning(self, source_agent: str, learning_data: Dict[str, Any]) -> List[str]:
        """Find agents relevant for learning propagation"""
        relevant_agents = []
        
        # Get source agent profile
        source_profile = self.agent_factory.agent_profiles.get(source_agent)
        if not source_profile:
            return relevant_agents
        
        # Find agents in same domain or with similar specializations
        for agent_id, profile in self.agent_factory.agent_profiles.items():
            if agent_id == source_agent:
                continue
            
            # Same domain agents
            if (profile.specialization.domain == source_profile.specialization.domain and
                profile.state != AgentState.ERROR):
                relevant_agents.append(agent_id)
        
        return relevant_agents
    
    def _trigger_agent_recovery(self, agent_id: str, error_data: Dict[str, Any]):
        """Trigger recovery procedures for an agent"""
        self.logger.info(f"Triggering recovery for agent {agent_id}")
        
        # Get agent profile
        profile = self.agent_factory.agent_profiles.get(agent_id)
        if not profile:
            return
        
        # Attempt recovery through HRM
        recovery_directive = f"Recover from error: {error_data.get('error_type', 'unknown')}"
        
        try:
            result = profile.instance.hrm_system.process_directive(recovery_directive)
            
            if result.get("success", False):
                self.agent_factory.update_agent_health(agent_id, AgentHealth.GOOD)
                self.logger.info(f"Agent {agent_id} recovery successful")
            else:
                self.logger.warning(f"Agent {agent_id} recovery failed")
                
        except Exception as e:
            self.logger.error(f"Error during agent {agent_id} recovery: {e}")
    
    def _notify_dependent_agents(self, agent_id: str, error_data: Dict[str, Any]):
        """Notify agents that depend on the failed agent"""
        # Find agents that have connections to the failed agent
        dependent_agents = []
        
        for other_id, profile in self.agent_factory.agent_profiles.items():
            if agent_id in profile.connections:
                dependent_agents.append(other_id)
        
        # Send notification
        for dependent_id in dependent_agents:
            signal = SignalGenerator.create_data_signal(
                {
                    "type": "agent_error_notification",
                    "failed_agent": agent_id,
                    "error_data": error_data
                },
                source_id="coordination_system",
                strength=0.9
            )
            
            self.communication_network.send_signal_to_agent(signal, dependent_id)
    
    def _analyze_performance_issue(self, agent_id: str, alert_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance issue for an agent"""
        profile = self.agent_factory.agent_profiles.get(agent_id)
        if not profile:
            return {"requires_optimization": False}
        
        analysis = {
            "requires_optimization": False,
            "issues": [],
            "recommendations": []
        }
        
        # Check efficiency
        if profile.metrics.efficiency_score < self.config["performance_targets"]["min_efficiency"]:
            analysis["issues"].append("low_efficiency")
            analysis["recommendations"].append("Optimize processing algorithms")
            analysis["requires_optimization"] = True
        
        # Check error rate
        if profile.metrics.error_rate > self.config["performance_targets"]["max_error_rate"]:
            analysis["issues"].append("high_error_rate")
            analysis["recommendations"].append("Improve error handling")
            analysis["requires_optimization"] = True
        
        # Check collaboration
        if profile.metrics.collaboration_score < self.config["performance_targets"]["min_collaboration"]:
            analysis["issues"].append("poor_collaboration")
            analysis["recommendations"].append("Enhance collaboration mechanisms")
            analysis["requires_optimization"] = True
        
        return analysis
    
    def _optimize_agent_performance(self, agent_id: str, analysis: Dict[str, Any]):
        """Optimize agent performance based on analysis"""
        self.logger.info(f"Optimizing performance for agent {agent_id}")
        
        profile = self.agent_factory.agent_profiles.get(agent_id)
        if not profile:
            return
        
        # Apply optimizations based on analysis
        for issue in analysis["issues"]:
            if issue == "low_efficiency":
                # Optimize threshold
                profile.instance.threshold = min(1.0, profile.instance.threshold * 1.1)
                
            elif issue == "high_error_rate":
                # Reduce threshold to be more selective
                profile.instance.threshold = max(0.1, profile.instance.threshold * 0.9)
                
            elif issue == "poor_collaboration":
                # Add more connections
                self._enhance_agent_connections(agent_id)
        
        self.logger.info(f"Applied optimizations for agent {agent_id}")
    
    def _enhance_agent_connections(self, agent_id: str):
        """Enhance connections for an agent"""
        profile = self.agent_factory.agent_profiles.get(agent_id)
        if not profile:
            return
        
        # Find potential new connections
        current_connections = set(profile.connections)
        all_agents = set(self.agent_factory.agent_profiles.keys())
        potential_connections = all_agents - current_connections - {agent_id}
        
        # Add new connections
        new_connections = random.sample(list(potential_connections), min(3, len(potential_connections)))
        
        for new_connection in new_connections:
            new_profile = self.agent_factory.agent_profiles[new_connection]
            
            # Create connection
            weight = self.topology_manager._calculate_connection_weight(profile, new_profile)
            profile.instance.add_connection(new_profile.instance, weight)
            
            # Update connection lists
            profile.connections.append(new_connection)
            new_profile.connections.append(agent_id)
        
        self.logger.info(f"Enhanced connections for agent {agent_id}: +{len(new_connections)} connections")
    
    def _update_system_metrics(self):
        """Update system-wide metrics"""
        # Get agent performance summary
        agent_summary = self.agent_factory.get_agent_performance_summary()
        
        # Update basic metrics
        self.metrics.total_agents = agent_summary["total_agents"]
        self.metrics.total_signals_processed = agent_summary["total_signals_processed"]
        self.metrics.successful_signals = agent_summary["successful_signals"]
        self.metrics.failed_signals = agent_summary["total_signals_processed"] - agent_summary["successful_signals"]
        
        # Calculate success rate
        if self.metrics.total_signals_processed > 0:
            self.metrics.successful_signals / self.metrics.total_signals_processed
        
        # Update domain performance
        self.metrics.domain_performance = self._calculate_domain_performance()
        
        # Calculate collaboration index
        self.metrics.collaboration_index = self._calculate_collaboration_index()
        
        # Update network health
        self.metrics.network_health = self._calculate_network_health()
    
    def _calculate_domain_performance(self) -> Dict[str, float]:
        """Calculate performance metrics for each domain"""
        domain_performance = {}
        
        for domain in AgentDomain:
            domain_agents = self.agent_factory.get_domain_agents(domain)
            
            if not domain_agents:
                domain_performance[domain.value] = 0.0
                continue
            
            # Calculate average performance for domain
            total_efficiency = 0.0
            active_agents = 0
            
            for agent_id in domain_agents:
                profile = self.agent_factory.agent_profiles.get(agent_id)
                if profile and profile.state != AgentState.ERROR:
                    total_efficiency += profile.metrics.efficiency_score
                    active_agents += 1
            
            if active_agents > 0:
                domain_performance[domain.value] = total_efficiency / active_agents
            else:
                domain_performance[domain.value] = 0.0
        
        return domain_performance
    
    def _calculate_collaboration_index(self) -> float:
        """Calculate system collaboration index"""
        total_collaboration = 0.0
        active_agents = 0
        
        for profile in self.agent_factory.agent_profiles.values():
            if profile.state != AgentState.ERROR:
                total_collaboration += profile.metrics.collaboration_score
                active_agents += 1
        
        return total_collaboration / active_agents if active_agents > 0 else 0.0
    
    def _calculate_network_health(self) -> float:
        """Calculate network health score"""
        # Get network status
        network_status = self.communication_network.get_network_status()
        
        # Calculate based on agent health and connectivity
        total_health = 0.0
        total_agents = 0
        
        for agent_id, agent_status in network_status["agent_statuses"].items():
            profile = self.agent_factory.agent_profiles.get(agent_id)
            if profile:
                health_score = {
                    AgentHealth.EXCELLENT: 1.0,
                    AgentHealth.GOOD: 0.8,
                    AgentHealth.FAIR: 0.6,
                    AgentHealth.POOR: 0.3,
                    AgentHealth.CRITICAL: 0.1
                }[profile.health]
                
                total_health += health_score
                total_agents += 1
        
        return total_health / total_agents if total_agents > 0 else 0.0
    
    def _monitor_agent_health(self):
        """Monitor health of all agents"""
        for agent_id, profile in self.agent_factory.agent_profiles.items():
            # Check for health degradation
            if profile.health in [AgentHealth.POOR, AgentHealth.CRITICAL]:
                self.logger.warning(f"Agent {agent_id} health is {profile.health.value}")
                
                # Trigger health check
                self._perform_agent_health_check(agent_id)
    
    def _perform_agent_health_check(self, agent_id: str):
        """Perform health check on an agent"""
        profile = self.agent_factory.agent_profiles.get(agent_id)
        if not profile:
            return
        
        # Send health check signal
        signal = SignalGenerator.create_data_signal(
            {"type": "health_check"},
            source_id="system_monitor",
            strength=1.0
        )
        
        try:
            # Process signal and measure response time
            start_time = time.time()
            self.agent_factory.process_agent_signal(agent_id, signal)
            response_time = time.time() - start_time
            
            # Update health based on response
            if response_time < 1.0:
                self.agent_factory.update_agent_health(agent_id, AgentHealth.GOOD)
            elif response_time < 5.0:
                self.agent_factory.update_agent_health(agent_id, AgentHealth.FAIR)
            else:
                self.agent_factory.update_agent_health(agent_id, AgentHealth.POOR)
                
        except Exception as e:
            self.logger.error(f"Health check failed for agent {agent_id}: {e}")
            self.agent_factory.update_agent_health(agent_id, AgentHealth.CRITICAL)
    
    def _check_performance_issues(self):
        """Check for system-wide performance issues"""
        # Check overall efficiency
        if self.metrics.system_efficiency < 0.7:
            self.logger.warning("System efficiency below threshold")
            self._trigger_system_optimization()
        
        # Check error rates
        if self.metrics.failed_signals / max(1, self.metrics.total_signals_processed) > 0.1:
            self.logger.warning("High system error rate detected")
            self._trigger_error_analysis()
        
        # Check resource utilization
        for resource, usage in self.metrics.resource_utilization.items():
            if usage > 0.9:
                self.logger.warning(f"High {resource} utilization: {usage:.2%}")
                self._trigger_resource_alert(resource, usage)
    
    def _trigger_system_optimization(self):
        """Trigger system-wide optimization"""
        self.logger.info("Triggering system optimization")
        
        # Optimize all agents
        optimization_results = self.agent_factory.optimize_agents()
        
        # Update metrics
        self.metrics.system_efficiency += 0.1  # Assume improvement
        
        self.logger.info(f"System optimization completed: {optimization_results['agents_optimized']} agents optimized")
    
    def _trigger_error_analysis(self):
        """Trigger error analysis and recovery"""
        self.logger.info("Triggering error analysis")
        
        # Analyze error patterns
        error_patterns = self._analyze_error_patterns()
        
        # Apply corrective actions
        for pattern in error_patterns:
            self._apply_error_correction(pattern)
        
        self.logger.info("Error analysis completed")
    
    def _analyze_error_patterns(self) -> List[Dict[str, Any]]:
        """Analyze error patterns in the system"""
        patterns = []
        
        # Group errors by type
        error_types = defaultdict(int)
        error_agents = defaultdict(int)
        
        for profile in self.agent_factory.agent_profiles.values():
            if profile.state == AgentState.ERROR:
                error_types[profile.specialization.domain.value] += 1
                error_agents[profile.agent_id] += 1
        
        # Identify patterns
        for error_type, count in error_types.items():
            if count > 2:  # More than 2 errors in domain
                patterns.append({
                    "type": "domain_error_cluster",
                    "domain": error_type,
                    "severity": count / len(self.agent_factory.get_domain_agents(AgentDomain(error_type))),
                    "affected_agents": count
                })
        
        return patterns
    
    def _apply_error_correction(self, pattern: Dict[str, Any]):
        """Apply error correction for a pattern"""
        if pattern["type"] == "domain_error_cluster":
            domain = pattern["domain"]
            
            # Apply domain-wide corrections
            domain_agents = self.agent_factory.get_domain_agents(AgentDomain(domain))
            
            for agent_id in domain_agents:
                profile = self.agent_factory.agent_profiles.get(agent_id)
                if profile and profile.state == AgentState.ERROR:
                    self._trigger_agent_recovery(agent_id, {"error_type": "domain_cluster"})
    
    def _trigger_resource_alert(self, resource: str, usage: float):
        """Trigger resource alert"""
        self.logger.warning(f"Resource alert: {resource} usage at {usage:.2%}")
        
        # Create resource alert event
        alert_event = CoordinationEvent(
            event_id=f"resource_alert_{int(time.time())}",
            event_type="resource_alert",
            source_agent="system",
            target_agents=[],
            content={"resource": resource, "usage": usage},
            priority=3
        )
        
        self.coordination_queue.put((3, alert_event))
    
    def _generate_alerts(self):
        """Generate system alerts based on current state"""
        alerts = []
        
        # Check for critical issues
        critical_agents = self.agent_factory.get_agents_by_health(AgentHealth.CRITICAL)
        if critical_agents:
            alerts.append({
                "type": "critical_agents",
                "message": f"{len(critical_agents)} agents in critical state",
                "severity": "critical"
            })
        
        # Check system mode
        if self.mode == SystemMode.EMERGENCY:
            alerts.append({
                "type": "emergency_mode",
                "message": "System in emergency mode",
                "severity": "critical"
            })
        
        # Generate alerts
        for alert in alerts:
            self._handle_alert(alert)
    
    def _handle_alert(self, alert: Dict[str, Any]):
        """Handle system alert"""
        self.logger.warning(f"System alert: {alert['message']}")
        
        # Update metrics
        self.metrics.last_major_event = alert["message"]
        
        # Trigger appropriate response
        if alert["severity"] == "critical":
            self.mode = SystemMode.EMERGENCY
            self._handle_emergency_mode()
    
    def _handle_emergency_mode(self):
        """Handle emergency mode operations"""
        self.logger.info("Entering emergency mode")
        
        # Prioritize critical tasks
        self._prioritize_critical_tasks()
        
        # Allocate emergency resources
        self._allocate_emergency_resources()
        
        # Reduce non-essential operations
        self._reduce_non_essential_operations()
    
    def _prioritize_critical_tasks(self):
        """Prioritize critical tasks during emergency"""
        # Move critical tasks to front of queue
        critical_tasks = []
        
        for task_id, task_data in self.active_tasks.items():
            if task_data.get("priority", 0) >= 3:
                critical_tasks.append((task_id, task_data))
        
        # Re-queue critical tasks
        for task_id, task_data in critical_tasks:
            self.task_queue.put((1, task_data))
    
    def _allocate_emergency_resources(self):
        """Allocate emergency resources"""
        # Redirect resources to critical agents
        critical_agents = self.agent_factory.get_agents_by_health(AgentHealth.CRITICAL)
        
        for agent_id in critical_agents:
            # Increase resource allocation
            if agent_id in self.resource_allocations:
                for resource in self.resource_allocations[agent_id]:
                    self.resource_allocations[agent_id][resource] *= 1.5
    
    def _reduce_non_essential_operations(self):
        """Reduce non-essential operations during emergency"""
        # Pause non-essential agents
        non_essential_domains = [AgentDomain.CREATIVITY, AgentDomain.LEARNING]
        
        for domain in non_essential_domains:
            domain_agents = self.agent_factory.get_domain_agents(domain)
            
            for agent_id in domain_agents:
                profile = self.agent_factory.agent_profiles.get(agent_id)
                if profile and profile.state == AgentState.IDLE:
                    self.agent_factory.update_agent_state(agent_id, AgentState.SLEEPING)
    
    def _collect_learning_data(self):
        """Collect learning data from agents"""
        # Collect performance metrics
        for agent_id, profile in self.agent_factory.agent_profiles.items():
            learning_point = {
                "agent_id": agent_id,
                "timestamp": time.time(),
                "efficiency": profile.metrics.efficiency_score,
                "error_rate": profile.metrics.error_rate,
                "collaboration_score": profile.metrics.collaboration_score,
                "load": profile.metrics.current_load,
                "health": profile.health.value
            }
            
            self.learning_data.append(learning_point)
    
    def _analyze_performance_patterns(self):
        """Analyze performance patterns for learning"""
        if len(self.learning_data) < 100:
            return
        
        # Group data by agent
        agent_data = defaultdict(list)
        for data_point in self.learning_data:
            agent_data[data_point["agent_id"]].append(data_point)
        
        # Analyze patterns for each agent
        for agent_id, data in agent_data.items():
            if len(data) < 10:
                continue
            
            # Calculate trends
            efficiency_trend = self._calculate_trend([d["efficiency"] for d in data])
            error_trend = self._calculate_trend([d["error_rate"] for d in data])
            
            # Store patterns
            self.performance_patterns[agent_id] = {
                "efficiency_trend": efficiency_trend,
                "error_trend": error_trend,
                "stability": self._calculate_stability([d["efficiency"] for d in data])
            }
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend in a series of values"""
        if len(values) < 2:
            return 0.0
        
        # Simple linear trend calculation
        n = len(values)
        x_sum = sum(range(n))
        y_sum = sum(values)
        xy_sum = sum(i * v for i, v in enumerate(values))
        x2_sum = sum(i * i for i in range(n))
        
        slope = (n * xy_sum - x_sum * y_sum) / (n * x2_sum - x_sum * x_sum)
        return slope
    
    def _calculate_stability(self, values: List[float]) -> float:
        """Calculate stability of a series of values"""
        if len(values) < 2:
            return 1.0
        
        # Calculate coefficient of variation
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        std_dev = variance ** 0.5
        
        # Lower coefficient of variation means higher stability
        cv = std_dev / mean if mean != 0 else 0
        return 1.0 / (1.0 + cv)
    
    def _adapt_system_behavior(self):
        """Adapt system behavior based on learning"""
        # Analyze overall system patterns
        overall_efficiency = self.metrics.system_efficiency
        overall_error_rate = self.metrics.failed_signals / max(1, self.metrics.total_signals_processed)
        
        # Adapt coordination strategy
        if overall_efficiency < 0.6:
            self.coordination_strategy = CoordinationStrategy.CENTRALIZED
        elif overall_error_rate > 0.15:
            self.coordination_strategy = CoordinationStrategy.DISTRIBUTED
        else:
            self.coordination_strategy = CoordinationStrategy.HYBRID
        
        # Adapt system mode
        if overall_efficiency > 0.9:
            self.mode = SystemMode.OPTIMIZATION
        elif overall_error_rate > 0.2:
            self.mode = SystemMode.MAINTENANCE
        else:
            self.mode = SystemMode.NORMAL
        
        # Record adaptation
        self.adaptation_history.append({
            "timestamp": time.time(),
            "efficiency": overall_efficiency,
            "error_rate": overall_error_rate,
            "coordination_strategy": self.coordination_strategy.value,
            "system_mode": self.mode.value
        })
    
    def _update_learning_progress(self):
        """Update learning progress metrics"""
        if len(self.learning_data) < 10:
            return
        
        # Calculate learning progress based on improvement trends
        recent_data = list(self.learning_data)[-100:]  # Last 100 data points
        
        # Calculate improvement metrics
        efficiency_improvement = self._calculate_improvement(
            [d["efficiency"] for d in recent_data]
        )
        
        error_reduction = self._calculate_improvement(
            [-d["error_rate"] for d in recent_data]  # Negative because lower error is better
        )
        
        # Update learning progress
        self.metrics.learning_progress = (efficiency_improvement + error_reduction) / 2
        self.metrics.adaptation_score = len(self.adaptation_history) / 100.0  # Normalize
    
    def _calculate_improvement(self, values: List[float]) -> float:
        """Calculate improvement in a series of values"""
        if len(values) < 10:
            return 0.0
        
        # Compare first and last quartiles
        q1_size = len(values) // 4
        q3_start = 3 * len(values) // 4
        
        q1_avg = sum(values[:q1_size]) / q1_size
        q3_avg = sum(values[q3_start:]) / (len(values) - q3_start)
        
        # Calculate improvement as percentage change
        if q1_avg != 0:
            improvement = (q3_avg - q1_avg) / abs(q1_avg)
        else:
            improvement = 0.0
        
        return max(0.0, min(1.0, improvement + 0.5))  # Normalize to [0, 1]
    
    def _monitor_resource_usage(self):
        """Monitor resource usage across the system"""
        # Calculate current resource usage
        current_usage = {
            "cpu": 0.0,
            "memory": 0.0,
            "network": 0.0,
            "storage": 0.0
        }
        
        # Sum up allocations
        for agent_id, allocation in self.resource_allocations.items():
            profile = self.agent_factory.agent_profiles.get(agent_id)
            if profile and profile.state != AgentState.ERROR:
                for resource, amount in allocation.items():
                    current_usage[resource] += amount
        
        # Update metrics
        self.metrics.resource_utilization = current_usage
    
    def _reallocate_resources(self):
        """Reallocate resources based on current needs"""
        # Check for resource constraints
        for resource, usage in self.metrics.resource_utilization.items():
            if usage > self.config["resource_thresholds"][resource]:
                self._optimize_resource_allocation(resource)
    
    def _optimize_resource_allocation(self, resource: str):
        """Optimize allocation of a specific resource"""
        self.logger.info(f"Optimizing {resource} allocation")
        
        # Get current allocations
        allocations = [(agent_id, alloc[resource]) 
                      for agent_id, alloc in self.resource_allocations.items()]
        
        # Sort by priority and efficiency
        allocations.sort(key=lambda x: self._get_agent_priority(x[0]), reverse=True)
        
        # Reduce allocations for lower priority agents
        total_reduction = 0.0
        target_reduction = self.metrics.resource_utilization[resource] - self.config["resource_thresholds"][resource]
        
        for agent_id, current_alloc in allocations:
            if total_reduction >= target_reduction:
                break
            
            # Reduce allocation for this agent
            reduction = min(current_alloc * 0.2, target_reduction - total_reduction)
            self.resource_allocations[agent_id][resource] -= reduction
            total_reduction += reduction
        
        self.logger.info(f"Reduced {resource} allocation by {total_reduction:.2f}")
    
    def _get_agent_priority(self, agent_id: str) -> float:
        """Get priority score for an agent"""
        profile = self.agent_factory.agent_profiles.get(agent_id)
        if not profile:
            return 0.0
        
        # Calculate priority based on multiple factors
        priority = 0.0
        
        # Health
        health_scores = {
            AgentHealth.EXCELLENT: 1.0,
            AgentHealth.GOOD: 0.8,
            AgentHealth.FAIR: 0.6,
            AgentHealth.POOR: 0.3,
            AgentHealth.CRITICAL: 0.1
        }
        priority += health_scores[profile.health] * 0.3
        
        # Efficiency
        priority += profile.metrics.efficiency_score * 0.3
        
        # Load (lower load gets higher priority)
        priority += (1.0 - profile.metrics.current_load) * 0.2
        
        # Criticality (based on domain)
        domain_criticality = {
            AgentDomain.SECURITY: 1.0,
            AgentDomain.COORDINATION: 0.9,
            AgentDomain.MONITORING: 0.8,
            AgentDomain.DATA_PROCESSING: 0.7,
            AgentDomain.ANALYSIS: 0.6,
            AgentDomain.OPTIMIZATION: 0.5,
            AgentDomain.COMMUNICATION: 0.4,
            AgentDomain.LEARNING: 0.3,
            AgentDomain.CREATIVITY: 0.2,
            AgentDomain.SPECIALIZED: 0.1
        }
        priority += domain_criticality.get(profile.specialization.domain, 0.5) * 0.2
        
        return priority
    
    def _handle_resource_requests(self):
        """Handle pending resource requests"""
        # This would handle resource requests from agents
        # For now, it's a placeholder for future implementation
        pass
    
    def _handle_task_assignments(self):
        """Handle task assignments from queue"""
        # Process tasks from queue
        tasks_processed = 0
        
        while not self.task_queue.empty() and tasks_processed < 10:
            try:
                priority, task_data = self.task_queue.get_nowait()
                
                # Create coordination event for task assignment
                event = CoordinationEvent(
                    event_id=f"task_assignment_{int(time.time())}",
                    event_type="task_assignment",
                    source_agent="task_queue",
                    target_agents=[],
                    content=task_data,
                    priority=priority
                )
                
                self.coordination_queue.put((priority, event))
                tasks_processed += 1
                
            except queue.Empty:
                break
    
    def _synchronize_agents(self):
        """Synchronize agent activities"""
        # Send synchronization signals to all agents
        sync_data = {
            "system_time": time.time(),
            "system_mode": self.mode.value,
            "coordination_strategy": self.coordination_strategy.value,
            "system_metrics": {
                "efficiency": self.metrics.system_efficiency,
                "collaboration_index": self.metrics.collaboration_index,
                "network_health": self.metrics.network_health
            }
        }
        
        # Send to domain leaders
        for domain, leader in self.agent_factory.domain_leaders.items():
            signal = SignalGenerator.create_data_signal(
                sync_data,
                source_id="coordination_system",
                strength=0.7
            )
            
            self.communication_network.send_signal_to_agent(signal, leader)
    
    def _synchronize_agent_state(self, agent_id: str, sync_data: Dict[str, Any]):
        """Synchronize individual agent state"""
        profile = self.agent_factory.agent_profiles.get(agent_id)
        if not profile:
            return
        
        # Update agent state based on sync data
        if "performance_metrics" in sync_data:
            profile.metrics.efficiency_score = sync_data["performance_metrics"].get("efficiency", profile.metrics.efficiency_score)
            profile.metrics.collaboration_score = sync_data["performance_metrics"].get("collaboration", profile.metrics.collaboration_score)
    
    def _calculate_coordination_efficiency(self) -> float:
        """Calculate coordination efficiency"""
        if not self.coordination_events:
            return 1.0
        
        # Calculate efficiency based on successful coordination events
        total_events = len(self.coordination_events)
        successful_events = sum(1 for event in self.coordination_events 
                             if event.metadata.get("success", True))
        
        return successful_events / total_events
    
    def _check_resource_availability(self, request: Dict[str, float]) -> Dict[str, float]:
        """Check if requested resources are available"""
        available = {}
        
        for resource, amount in request.items():
            current_usage = self.metrics.resource_utilization.get(resource, 0.0)
            available_capacity = 1.0 - current_usage
            
            if available_capacity >= amount:
                available[resource] = amount
            else:
                available[resource] = available_capacity
        
        return available
    
    def _allocate_resources(self, agent_id: str, request: Dict[str, float]):
        """Allocate resources to an agent"""
        if agent_id not in self.resource_allocations:
            self.resource_allocations[agent_id] = {"cpu": 0.0, "memory": 0.0, "network": 0.0, "storage": 0.0}
        
        for resource, amount in request.items():
            self.resource_allocations[agent_id][resource] += amount
    
    def submit_task(self, task_data: Dict[str, Any], priority: int = 2):
        """Submit a task to the system"""
        task_id = f"task_{int(time.time())}"
        task_data["task_id"] = task_id
        
        # Add to task queue
        self.task_queue.put((priority, task_data))
        
        self.logger.info(f"Task {task_id} submitted with priority {priority}")
        return task_id
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            "system_id": self.system_id,
            "is_running": self.is_running,
            "mode": self.mode.value,
            "coordination_strategy": self.coordination_strategy.value,
            "uptime": time.time() - self.start_time,
            "metrics": {
                "total_agents": self.metrics.total_agents,
                "active_agents": self.metrics.active_agents,
                "total_signals_processed": self.metrics.total_signals_processed,
                "successful_signals": self.metrics.successful_signals,
                "failed_signals": self.metrics.failed_signals,
                "system_efficiency": self.metrics.system_efficiency,
                "collaboration_index": self.metrics.collaboration_index,
                "learning_progress": self.metrics.learning_progress,
                "adaptation_score": self.metrics.adaptation_score,
                "network_health": self.metrics.network_health,
                "coordination_efficiency": self.metrics.coordination_efficiency
            },
            "resource_utilization": self.metrics.resource_utilization,
            "domain_performance": self.metrics.domain_performance,
            "active_tasks": len(self.active_tasks),
            "completed_tasks": len(self.completed_tasks),
            "coordination_events": len(self.coordination_events),
            "learning_data_points": len(self.learning_data),
            "adaptation_history": len(self.adaptation_history)
        }
    
    def get_detailed_agent_status(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed status for a specific agent"""
        profile = self.agent_factory.agent_profiles.get(agent_id)
        if not profile:
            return None
        
        return {
            "agent_id": agent_id,
            "name": profile.specialization.name,
            "domain": profile.specialization.domain.value,
            "complexity": profile.specialization.complexity.value,
            "state": profile.state.value,
            "health": profile.health.value,
            "metrics": {
                "total_signals_processed": profile.metrics.total_signals_processed,
                "successful_signals": profile.metrics.successful_signals,
                "failed_signals": profile.metrics.failed_signals,
                "average_processing_time": profile.metrics.average_processing_time,
                "current_load": profile.metrics.current_load,
                "efficiency_score": profile.metrics.efficiency_score,
                "error_rate": profile.metrics.error_rate,
                "collaboration_score": profile.metrics.collaboration_score
            },
            "connections": len(profile.connections),
            "current_tasks": len(profile.current_tasks),
            "uptime": time.time() - profile.created_at,
            "last_activity": profile.metrics.last_activity
        }
    
    def get_network_topology(self) -> Dict[str, Any]:
        """Get network topology information"""
        connections = {}
        
        for agent_id, profile in self.agent_factory.agent_profiles.items():
            connections[agent_id] = profile.connections.copy()
        
        return {
            "total_agents": len(connections),
            "connections": connections,
            "topology_metrics": self.topology_manager.get_topology_metrics(connections)
        }
    
    def save_system_state(self, filename: str):
        """Save system state to file"""
        state = {
            "system_id": self.system_id,
            "timestamp": time.time(),
            "mode": self.mode.value,
            "coordination_strategy": self.coordination_strategy.value,
            "metrics": self.metrics.__dict__,
            "resource_allocations": self.resource_allocations,
            "learning_data": list(self.learning_data)[-1000:],  # Save last 1000 points
            "adaptation_history": self.adaptation_history[-100:],  # Save last 100 adaptations
            "agent_profiles": {
                agent_id: {
                    "state": profile.state.value,
                    "health": profile.health.value,
                    "metrics": profile.metrics.__dict__,
                    "connections": profile.connections
                }
                for agent_id, profile in self.agent_factory.agent_profiles.items()
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        self.logger.info(f"System state saved to {filename}")
    
    def load_system_state(self, filename: str):
        """Load system state from file"""
        try:
            with open(filename, 'r') as f:
                state = json.load(f)
            
            # Restore basic state
            self.mode = SystemMode(state["mode"])
            self.coordination_strategy = CoordinationStrategy(state["coordination_strategy"])
            
            # Restore metrics
            for key, value in state["metrics"].items():
                if hasattr(self.metrics, key):
                    setattr(self.metrics, key, value)
            
            # Restore resource allocations
            self.resource_allocations = state["resource_allocations"]
            
            # Restore learning data
            self.learning_data = deque(state["learning_data"], maxlen=10000)
            
            # Restore adaptation history
            self.adaptation_history = state["adaptation_history"]
            
            # Restore agent profiles
            for agent_id, profile_data in state["agent_profiles"].items():
                profile = self.agent_factory.agent_profiles.get(agent_id)
                if profile:
                    profile.state = AgentState(profile_data["state"])
                    profile.health = AgentHealth(profile_data["health"])
                    
                    # Restore metrics
                    for key, value in profile_data["metrics"].items():
                        if hasattr(profile.metrics, key):
                            setattr(profile.metrics, key, value)
                    
                    # Restore connections
                    profile.connections = profile_data["connections"]
            
            self.logger.info(f"System state loaded from {filename}")
            
        except Exception as e:
            self.logger.error(f"Error loading system state: {e}")

# Factory function for easy initialization
def create_sophisticated_brain_system(system_id: str = "sophisticated_brain_250") -> SophisticatedBrainSystem:
    """Create and initialize a sophisticated brain system"""
    system = SophisticatedBrainSystem(system_id)
    return system