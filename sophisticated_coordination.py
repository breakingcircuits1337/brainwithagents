"""
Sophisticated Coordination Mechanisms for 250-Agent Brain System

This module implements advanced coordination mechanisms that enable
250 specialized agents to work together collaboratively and efficiently.
"""

import time
import random
import threading
from typing import Dict, List, Any, Optional, Set, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import logging
import heapq
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import existing components
from massive_agent_factory import AdvancedAgent, AgentClusterManager, massive_agent_factory
from advanced_network_topology import AdvancedNetworkTopology, NetworkTopology, TopologyConfig
from communication_system import Signal, SignalType
from agent_specializations import AgentDomain, agent_registry

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CoordinationStrategy(Enum):
    """Coordination strategies for agent collaboration"""
    CENTRALIZED = "centralized"
    DISTRIBUTED = "distributed"
    HIERARCHICAL = "hierarchical"
    MARKET_BASED = "market_based"
    SWARM = "swarm"
    HYBRID = "hybrid"

class CoordinationMode(Enum):
    """Coordination modes for different scenarios"""
    COLLABORATIVE = "collaborative"
    COMPETITIVE = "competitive"
    COOPERATIVE = "cooperative"
    ADAPTIVE = "adaptive"
    EMERGENT = "emergent"

class TaskPriority(Enum):
    """Task priority levels for coordination"""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    BACKGROUND = 5

@dataclass
class CoordinationTask:
    """Task for coordination among agents"""
    id: str
    description: str
    required_capabilities: List[str]
    priority: TaskPriority
    complexity: float
    estimated_duration: float
    resource_requirements: Dict[str, float]
    dependencies: List[str] = field(default_factory=list)
    assigned_agents: Set[str] = field(default_factory=set)
    status: str = "pending"
    progress: float = 0.0
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[Any] = None
    coordination_metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CoordinationEvent:
    """Event in the coordination system"""
    id: str
    type: str
    timestamp: float
    source_agent: str
    target_agents: List[str]
    content: Any
    priority: TaskPriority
    coordination_strategy: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CoordinationMetrics:
    """Metrics for coordination system performance"""
    total_tasks_created: int = 0
    total_tasks_completed: int = 0
    average_completion_time: float = 0.0
    coordination_efficiency: float = 0.0
    agent_utilization: Dict[str, float] = field(default_factory=dict)
    strategy_performance: Dict[str, float] = field(default_factory=dict)
    collaboration_frequency: Dict[str, int] = field(default_factory=dict)
    load_balance_score: float = 0.0
    coordination_overhead: float = 0.0

class CoordinationManager:
    """Main coordination manager for 250-agent system"""
    
    def __init__(self, agents: Dict[str, AdvancedAgent], 
                 topology_manager: AdvancedNetworkTopology = None):
        self.agents = agents
        self.topology_manager = topology_manager
        
        # Coordination state
        self.active_tasks: Dict[str, CoordinationTask] = {}
        self.completed_tasks: Dict[str, CoordinationTask] = {}
        self.task_queue = []
        self.coordination_events = deque(maxlen=10000)
        self.coordination_history = deque(maxlen=1000)
        
        # Coordination strategies
        self.active_strategy = CoordinationStrategy.HYBRID
        self.strategy_performance = defaultdict(float)
        self.strategy_usage = defaultdict(int)
        
        # Coordination mechanisms
        self.task_allocator = TaskAllocator(self)
        self.resource_manager = ResourceManager(self)
        self.collaboration_engine = CollaborationEngine(self)
        self.conflict_resolver = ConflictResolver(self)
        self.performance_optimizer = PerformanceOptimizer(self)
        
        # Metrics and monitoring
        self.metrics = CoordinationMetrics()
        self.coordination_stats = defaultdict(int)
        self.performance_history = deque(maxlen=1000)
        
        # Configuration
        self.config = {
            "max_concurrent_tasks": 50,
            "task_allocation_strategy": "capability_based",
            "resource_allocation_strategy": "fair_share",
            "collaboration_threshold": 0.7,
            "conflict_resolution_strategy": "negotiation",
            "optimization_interval": 100,
            "coordination_timeout": 300.0
        }
        
        # Threading and execution
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.coordination_lock = threading.Lock()
        self.is_running = False
        
        self.logger = logging.getLogger(__name__)
    
    def start_coordination(self):
        """Start the coordination system"""
        self.logger.info("Starting coordination system")
        self.is_running = True
        
        # Start coordination threads
        self._start_coordination_threads()
        
        self.logger.info("Coordination system started")
    
    def stop_coordination(self):
        """Stop the coordination system"""
        self.logger.info("Stopping coordination system")
        self.is_running = False
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        self.logger.info("Coordination system stopped")
    
    def _start_coordination_threads(self):
        """Start coordination management threads"""
        # Task processing thread
        threading.Thread(target=self._task_processing_loop, daemon=True).start()
        
        # Coordination optimization thread
        threading.Thread(target=self._coordination_optimization_loop, daemon=True).start()
        
        # Performance monitoring thread
        threading.Thread(target=self._performance_monitoring_loop, daemon=True).start()
    
    def _task_processing_loop(self):
        """Main task processing loop"""
        while self.is_running:
            try:
                with self.coordination_lock:
                    # Process task queue
                    if self.task_queue and len(self.active_tasks) < self.config["max_concurrent_tasks"]:
                        task = heapq.heappop(self.task_queue)
                        self._start_task_execution(task)
                
                time.sleep(0.1)  # Small delay to prevent excessive CPU usage
                
            except Exception as e:
                self.logger.error(f"Error in task processing loop: {e}")
    
    def _coordination_optimization_loop(self):
        """Coordination optimization loop"""
        optimization_count = 0
        
        while self.is_running:
            try:
                time.sleep(self.config["optimization_interval"])
                
                if optimization_count % 10 == 0:  # Every 10th optimization
                    self.performance_optimizer.optimize_coordination()
                
                # Optimize task allocation
                self.task_allocator.optimize_allocation()
                
                # Optimize resource allocation
                self.resource_manager.optimize_resources()
                
                optimization_count += 1
                
            except Exception as e:
                self.logger.error(f"Error in coordination optimization loop: {e}")
    
    def _performance_monitoring_loop(self):
        """Performance monitoring loop"""
        while self.is_running:
            try:
                time.sleep(5.0)  # Monitor every 5 seconds
                
                # Update metrics
                self._update_coordination_metrics()
                
                # Check for performance issues
                self._check_performance_issues()
                
            except Exception as e:
                self.logger.error(f"Error in performance monitoring loop: {e}")
    
    def create_coordination_task(self, description: str, required_capabilities: List[str],
                               priority: TaskPriority, complexity: float = 0.5,
                               estimated_duration: float = 10.0,
                               resource_requirements: Dict[str, float] = None) -> str:
        """Create a new coordination task"""
        task_id = f"task_{int(time.time())}_{random.randint(1000, 9999)}"
        
        task = CoordinationTask(
            id=task_id,
            description=description,
            required_capabilities=required_capabilities,
            priority=priority,
            complexity=complexity,
            estimated_duration=estimated_duration,
            resource_requirements=resource_requirements or {}
        )
        
        with self.coordination_lock:
            # Add to task queue (priority queue)
            heapq.heappush(self.task_queue, (priority.value, task))
            self.active_tasks[task_id] = task
        
        # Update metrics
        self.metrics.total_tasks_created += 1
        
        # Create coordination event
        event = CoordinationEvent(
            id=f"event_{int(time.time())}",
            type="task_created",
            timestamp=time.time(),
            source_agent="coordination_manager",
            target_agents=[],
            content=task,
            priority=priority,
            coordination_strategy=self.active_strategy.value,
            metadata={"task_id": task_id}
        )
        
        self.coordination_events.append(event)
        
        self.logger.info(f"Created coordination task: {task_id} - {description}")
        
        return task_id
    
    def _start_task_execution(self, task: CoordinationTask):
        """Start execution of a coordination task"""
        task.status = "executing"
        task.started_at = time.time()
        
        # Allocate agents to task
        allocated_agents = self.task_allocator.allocate_agents(task)
        task.assigned_agents = set(allocated_agents)
        
        # Allocate resources
        self.resource_manager.allocate_resources(task)
        
        # Create coordination event
        event = CoordinationEvent(
            id=f"event_{int(time.time())}",
            type="task_started",
            timestamp=time.time(),
            source_agent="coordination_manager",
            target_agents=allocated_agents,
            content=task,
            priority=task.priority,
            coordination_strategy=self.active_strategy.value,
            metadata={"allocation_strategy": self.config["task_allocation_strategy"]}
        )
        
        self.coordination_events.append(event)
        
        # Submit task for execution
        future = self.executor.submit(self._execute_coordination_task, task)
        
        self.logger.info(f"Started execution of task: {task.id} with {len(allocated_agents)} agents")
    
    def _execute_coordination_task(self, task: CoordinationTask) -> Any:
        """Execute a coordination task with assigned agents"""
        try:
            # Create coordination signal
            signal = Signal(
                id=f"coord_signal_{task.id}",
                type=SignalType.CONTROL,
                content={
                    "type": "coordination_task",
                    "task_id": task.id,
                    "description": task.description,
                    "required_capabilities": task.required_capabilities,
                    "complexity": task.complexity,
                    "coordination_strategy": self.active_strategy.value
                },
                source_agent_id="coordination_manager",
                strength=0.9
            )
            
            # Send signal to assigned agents
            for agent_id in task.assigned_agents:
                if agent_id in self.agents:
                    self.agents[agent_id].receive_signal(signal)
            
            # Monitor task progress
            result = self._monitor_task_progress(task)
            
            # Complete task
            task.status = "completed"
            task.completed_at = time.time()
            task.progress = 1.0
            task.result = result
            
            # Move to completed tasks
            with self.coordination_lock:
                if task.id in self.active_tasks:
                    del self.active_tasks[task.id]
                self.completed_tasks[task.id] = task
            
            # Update metrics
            self.metrics.total_tasks_completed += 1
            completion_time = task.completed_at - task.started_at
            self._update_completion_time_metrics(completion_time)
            
            # Create completion event
            event = CoordinationEvent(
                id=f"event_{int(time.time())}",
                type="task_completed",
                timestamp=time.time(),
                source_agent="coordination_manager",
                target_agents=list(task.assigned_agents),
                content=task,
                priority=task.priority,
                coordination_strategy=self.active_strategy.value,
                metadata={"completion_time": completion_time, "success": True}
            )
            
            self.coordination_events.append(event)
            
            self.logger.info(f"Completed task: {task.id} in {completion_time:.2f} seconds")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error executing task {task.id}: {e}")
            
            # Handle task failure
            task.status = "failed"
            task.completed_at = time.time()
            
            # Create failure event
            event = CoordinationEvent(
                id=f"event_{int(time.time())}",
                type="task_failed",
                timestamp=time.time(),
                source_agent="coordination_manager",
                target_agents=list(task.assigned_agents),
                content=task,
                priority=task.priority,
                coordination_strategy=self.active_strategy.value,
                metadata={"error": str(e)}
            )
            
            self.coordination_events.append(event)
            
            return None
    
    def _monitor_task_progress(self, task: CoordinationTask) -> Any:
        """Monitor progress of task execution"""
        start_time = time.time()
        timeout = self.config["coordination_timeout"]
        
        while time.time() - start_time < timeout:
            # Check agent progress
            total_progress = 0
            active_agents = 0
            
            for agent_id in task.assigned_agents:
                if agent_id in self.agents:
                    agent = self.agents[agent_id]
                    # Estimate progress based on agent state and load
                    agent_progress = self._estimate_agent_progress(agent, task)
                    total_progress += agent_progress
                    active_agents += 1
            
            # Update task progress
            if active_agents > 0:
                task.progress = total_progress / active_agents
            
            # Check if task is complete
            if task.progress >= 1.0:
                return {"status": "completed", "progress": task.progress}
            
            # Adapt coordination strategy if needed
            if task.progress < 0.1 and time.time() - start_time > timeout * 0.3:
                self._adapt_coordination_strategy(task)
            
            time.sleep(1.0)  # Check progress every second
        
        # Task timed out
        return {"status": "timeout", "progress": task.progress}
    
    def _estimate_agent_progress(self, agent: AdvancedAgent, task: CoordinationTask) -> float:
        """Estimate progress of an agent on a task"""
        # Base progress on agent state and load
        if agent.state.value == "processing":
            base_progress = 0.5
        elif agent.state.value == "idle":
            base_progress = 0.1
        else:
            base_progress = 0.0
        
        # Adjust based on agent expertise
        expertise_bonus = agent.specialization.expertise_level * 0.3
        
        # Adjust based on task complexity match
        complexity_match = 1.0 - abs(agent.specialization.expertise_level - task.complexity)
        complexity_bonus = complexity_match * 0.2
        
        # Adjust based on load
        load_penalty = agent.load.queue_size * 0.05
        
        progress = base_progress + expertise_bonus + complexity_bonus - load_penalty
        
        return max(0.0, min(1.0, progress))
    
    def _adapt_coordination_strategy(self, task: CoordinationTask):
        """Adapt coordination strategy for struggling task"""
        current_strategy = self.active_strategy
        
        # Try different strategies based on task characteristics
        if task.complexity > 0.8:
            new_strategy = CoordinationStrategy.HIERARCHICAL
        elif len(task.assigned_agents) > 10:
            new_strategy = CoordinationStrategy.DISTRIBUTED
        elif task.priority == TaskPriority.CRITICAL:
            new_strategy = CoordinationStrategy.CENTRALIZED
        else:
            new_strategy = CoordinationStrategy.SWARM
        
        if new_strategy != current_strategy:
            self.active_strategy = new_strategy
            self.logger.info(f"Adapted coordination strategy to {new_strategy.value} for task {task.id}")
            
            # Create adaptation event
            event = CoordinationEvent(
                id=f"event_{int(time.time())}",
                type="strategy_adapted",
                timestamp=time.time(),
                source_agent="coordination_manager",
                target_agents=list(task.assigned_agents),
                content=task,
                priority=task.priority,
                coordination_strategy=new_strategy.value,
                metadata={"previous_strategy": current_strategy.value, "reason": "slow_progress"}
            )
            
            self.coordination_events.append(event)
    
    def coordinate_agent_collaboration(self, initiator_id: str, target_ids: List[str],
                                      collaboration_type: str, content: Any) -> bool:
        """Coordinate collaboration between agents"""
        if initiator_id not in self.agents:
            return False
        
        initiator = self.agents[initiator_id]
        
        # Validate target agents
        valid_targets = [tid for tid in target_ids if tid in self.agents]
        if not valid_targets:
            return False
        
        # Create collaboration event
        event = CoordinationEvent(
            id=f"event_{int(time.time())}",
            type="collaboration_initiated",
            timestamp=time.time(),
            source_agent=initiator_id,
            target_agents=valid_targets,
            content=content,
            priority=TaskPriority.MEDIUM,
            coordination_strategy=self.active_strategy.value,
            metadata={"collaboration_type": collaboration_type}
        )
        
        self.coordination_events.append(event)
        
        # Execute collaboration
        success = self.collaboration_engine.execute_collaboration(
            initiator, valid_targets, collaboration_type, content
        )
        
        # Update collaboration metrics
        for target_id in valid_targets:
            collaboration_key = f"{initiator_id}->{target_id}"
            self.metrics.collaboration_frequency[collaboration_key] += 1
        
        return success
    
    def resolve_coordination_conflict(self, conflict_type: str, involved_agents: List[str],
                                   conflict_details: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve conflicts in coordination"""
        # Create conflict event
        event = CoordinationEvent(
            id=f"event_{int(time.time())}",
            type="conflict_detected",
            timestamp=time.time(),
            source_agent="conflict_detector",
            target_agents=involved_agents,
            content=conflict_details,
            priority=TaskPriority.HIGH,
            coordination_strategy=self.active_strategy.value,
            metadata={"conflict_type": conflict_type}
        )
        
        self.coordination_events.append(event)
        
        # Resolve conflict
        resolution = self.conflict_resolver.resolve_conflict(
            conflict_type, involved_agents, conflict_details
        )
        
        # Create resolution event
        event = CoordinationEvent(
            id=f"event_{int(time.time())}",
            type="conflict_resolved",
            timestamp=time.time(),
            source_agent="conflict_resolver",
            target_agents=involved_agents,
            content=resolution,
            priority=TaskPriority.HIGH,
            coordination_strategy=self.active_strategy.value,
            metadata={"resolution_strategy": self.config["conflict_resolution_strategy"]}
        )
        
        self.coordination_events.append(event)
        
        return resolution
    
    def _update_completion_time_metrics(self, completion_time: float):
        """Update completion time metrics"""
        if self.metrics.total_tasks_completed == 1:
            self.metrics.average_completion_time = completion_time
        else:
            self.metrics.average_completion_time = (
                (self.metrics.average_completion_time * (self.metrics.total_tasks_completed - 1) + completion_time) /
                self.metrics.total_tasks_completed
            )
    
    def _update_coordination_metrics(self):
        """Update coordination metrics"""
        # Calculate agent utilization
        for agent_id, agent in self.agents.items():
            if agent.metrics.activation_count > 0:
                utilization = agent.metrics.processing_time / max(1, time.time() - agent.metrics.last_activity)
                self.metrics.agent_utilization[agent_id] = min(1.0, utilization)
        
        # Calculate coordination efficiency
        if self.metrics.total_tasks_created > 0:
            self.metrics.coordination_efficiency = (
                self.metrics.total_tasks_completed / self.metrics.total_tasks_created
            )
        
        # Calculate load balance score
        utilizations = list(self.metrics.agent_utilization.values())
        if utilizations:
            avg_utilization = sum(utilizations) / len(utilizations)
            utilization_variance = sum((u - avg_utilization) ** 2 for u in utilizations) / len(utilizations)
            self.metrics.load_balance_score = 1.0 - min(1.0, utilization_variance)
        
        # Calculate coordination overhead
        total_events = len(self.coordination_events)
        total_tasks = self.metrics.total_tasks_created
        if total_tasks > 0:
            self.metrics.coordination_overhead = total_events / total_tasks
    
    def _check_performance_issues(self):
        """Check for performance issues in coordination"""
        # Check for low efficiency
        if self.metrics.coordination_efficiency < 0.7:
            self.logger.warning(f"Low coordination efficiency: {self.metrics.coordination_efficiency:.2f}")
        
        # Check for poor load balance
        if self.metrics.load_balance_score < 0.6:
            self.logger.warning(f"Poor load balance: {self.metrics.load_balance_score:.2f}")
        
        # Check for high overhead
        if self.metrics.coordination_overhead > 10.0:
            self.logger.warning(f"High coordination overhead: {self.metrics.coordination_overhead:.2f}")
        
        # Check for stuck tasks
        current_time = time.time()
        stuck_tasks = [
            task for task in self.active_tasks.values()
            if task.started_at and (current_time - task.started_at) > self.config["coordination_timeout"]
        ]
        
        if stuck_tasks:
            self.logger.warning(f"Found {len(stuck_tasks)} stuck tasks")
            for task in stuck_tasks:
                self._handle_stuck_task(task)
    
    def _handle_stuck_task(self, task: CoordinationTask):
        """Handle a stuck task"""
        self.logger.warning(f"Handling stuck task: {task.id}")
        
        # Try to restart task with different strategy
        task.status = "restarting"
        task.started_at = time.time()
        
        # Reallocate agents
        new_agents = self.task_allocator.reallocate_agents(task)
        task.assigned_agents = set(new_agents)
        
        # Create restart event
        event = CoordinationEvent(
            id=f"event_{int(time.time())}",
            type="task_restarted",
            timestamp=time.time(),
            source_agent="coordination_manager",
            target_agents=new_agents,
            content=task,
            priority=task.priority,
            coordination_strategy=self.active_strategy.value,
            metadata={"reason": "timeout", "restart_count": task.coordination_metadata.get("restart_count", 0) + 1}
        )
        
        self.coordination_events.append(event)
    
    def get_coordination_status(self) -> Dict[str, Any]:
        """Get comprehensive coordination status"""
        return {
            "active_strategy": self.active_strategy.value,
            "active_tasks": len(self.active_tasks),
            "completed_tasks": len(self.completed_tasks),
            "task_queue_size": len(self.task_queue),
            "coordination_events": len(self.coordination_events),
            "metrics": {
                "total_tasks_created": self.metrics.total_tasks_created,
                "total_tasks_completed": self.metrics.total_tasks_completed,
                "average_completion_time": self.metrics.average_completion_time,
                "coordination_efficiency": self.metrics.coordination_efficiency,
                "load_balance_score": self.metrics.load_balance_score,
                "coordination_overhead": self.metrics.coordination_overhead
            },
            "agent_utilization": {
                agent_id: utilization for agent_id, utilization in self.metrics.agent_utilization.items()
                if utilization > 0.1  # Only show significantly utilized agents
            },
            "strategy_performance": dict(self.strategy_performance),
            "configuration": self.config
        }

class TaskAllocator:
    """Handles task allocation to agents"""
    
    def __init__(self, coordination_manager: CoordinationManager):
        self.coordination_manager = coordination_manager
        self.allocation_history = deque(maxlen=1000)
        self.allocation_performance = defaultdict(float)
    
    def allocate_agents(self, task: CoordinationTask) -> List[str]:
        """Allocate agents to a task"""
        strategy = self.coordination_manager.config["task_allocation_strategy"]
        
        if strategy == "capability_based":
            agents = self._capability_based_allocation(task)
        elif strategy == "load_balanced":
            agents = self._load_balanced_allocation(task)
        elif strategy == "expertise_based":
            agents = self._expertise_based_allocation(task)
        else:
            agents = self._hybrid_allocation(task)
        
        # Record allocation
        self.allocation_history.append({
            "task_id": task.id,
            "allocated_agents": agents,
            "strategy": strategy,
            "timestamp": time.time()
        })
        
        return agents
    
    def _capability_based_allocation(self, task: CoordinationTask) -> List[str]:
        """Allocate agents based on capability matching"""
        suitable_agents = []
        
        for agent_id, agent in self.coordination_manager.agents.items():
            # Calculate capability match score
            match_score = self._calculate_capability_match(agent, task)
            
            if match_score > 0.5:  # Minimum threshold
                suitable_agents.append((agent_id, match_score))
        
        # Sort by match score and select top agents
        suitable_agents.sort(key=lambda x: x[1], reverse=True)
        
        # Select number of agents based on task complexity
        num_agents = max(1, int(task.complexity * 5))
        selected_agents = [agent_id for agent_id, score in suitable_agents[:num_agents]]
        
        return selected_agents
    
    def _load_balanced_allocation(self, task: CoordinationTask) -> List[str]:
        """Allocate agents based on load balancing"""
        agent_loads = []
        
        for agent_id, agent in self.coordination_manager.agents.items():
            # Calculate load score
            load_score = (
                agent.load.queue_size * 0.5 +
                agent.load.cpu_usage * 0.3 +
                agent.load.memory_usage * 0.2
            )
            
            # Calculate capability match
            capability_match = self._calculate_capability_match(agent, task)
            
            if capability_match > 0.3:  # Minimum capability threshold
                agent_loads.append((agent_id, load_score, capability_match))
        
        # Sort by load (ascending) and capability (descending)
        agent_loads.sort(key=lambda x: (x[1], -x[2]))
        
        # Select least loaded agents with sufficient capability
        num_agents = max(1, int(task.complexity * 5))
        selected_agents = [agent_id for agent_id, load, capability in agent_loads[:num_agents]]
        
        return selected_agents
    
    def _expertise_based_allocation(self, task: CoordinationTask) -> List[str]:
        """Allocate agents based on expertise matching"""
        expert_agents = []
        
        for agent_id, agent in self.coordination_manager.agents.items():
            # Calculate expertise match
            expertise_match = self._calculate_expertise_match(agent, task)
            
            if expertise_match > 0.6:  # High expertise threshold
                expert_agents.append((agent_id, expertise_match))
        
        # Sort by expertise and select top experts
        expert_agents.sort(key=lambda x: x[1], reverse=True)
        
        # Select number of agents based on task complexity
        num_agents = max(1, int(task.complexity * 3))
        selected_agents = [agent_id for agent_id, score in expert_agents[:num_agents]]
        
        return selected_agents
    
    def _hybrid_allocation(self, task: CoordinationTask) -> List[str]:
        """Hybrid allocation considering multiple factors"""
        agent_scores = []
        
        for agent_id, agent in self.coordination_manager.agents.items():
            # Calculate composite score
            capability_score = self._calculate_capability_match(agent, task)
            load_score = 1.0 - (agent.load.queue_size / 20.0)  # Inverse load
            expertise_score = self._calculate_expertise_match(agent, task)
            
            # Weighted composite score
            composite_score = (
                capability_score * 0.4 +
                load_score * 0.3 +
                expertise_score * 0.3
            )
            
            if composite_score > 0.4:  # Minimum threshold
                agent_scores.append((agent_id, composite_score))
        
        # Sort by composite score
        agent_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select top agents
        num_agents = max(1, int(task.complexity * 4))
        selected_agents = [agent_id for agent_id, score in agent_scores[:num_agents]]
        
        return selected_agents
    
    def _calculate_capability_match(self, agent: AdvancedAgent, task: CoordinationTask) -> float:
        """Calculate capability match score between agent and task"""
        if not task.required_capabilities:
            return 0.5
        
        # Count matching capabilities
        matching_capabilities = 0
        for required_cap in task.required_capabilities:
            for agent_cap in agent.specialization.capabilities:
                if required_cap.lower() in agent_cap.name.lower():
                    matching_capabilities += 1
                    break
        
        # Calculate match ratio
        match_ratio = matching_capabilities / len(task.required_capabilities)
        
        return match_ratio
    
    def _calculate_expertise_match(self, agent: AdvancedAgent, task: CoordinationTask) -> float:
        """Calculate expertise match score between agent and task"""
        # Base expertise level
        expertise_match = agent.specialization.expertise_level
        
        # Adjust based on task complexity
        complexity_diff = abs(agent.specialization.expertise_level - task.complexity)
        expertise_match *= (1.0 - complexity_diff * 0.5)
        
        return expertise_match
    
    def reallocate_agents(self, task: CoordinationTask) -> List[str]:
        """Reallocate agents for a task"""
        # Remove current assignments
        task.assigned_agents.clear()
        
        # Try different allocation strategy
        current_strategy = self.coordination_manager.config["task_allocation_strategy"]
        
        # Cycle through strategies
        strategies = ["capability_based", "load_balanced", "expertise_based", "hybrid"]
        current_index = strategies.index(current_strategy)
        new_strategy = strategies[(current_index + 1) % len(strategies)]
        
        # Temporarily change strategy
        original_strategy = self.coordination_manager.config["task_allocation_strategy"]
        self.coordination_manager.config["task_allocation_strategy"] = new_strategy
        
        # Allocate with new strategy
        new_agents = self.allocate_agents(task)
        
        # Restore original strategy
        self.coordination_manager.config["task_allocation_strategy"] = original_strategy
        
        return new_agents
    
    def optimize_allocation(self):
        """Optimize task allocation strategies"""
        # Analyze allocation performance
        for allocation in list(self.allocation_history)[-100:]:  # Last 100 allocations
            task_id = allocation["task_id"]
            strategy = allocation["strategy"]
            
            # Check if task was completed successfully
            if task_id in self.coordination_manager.completed_tasks:
                task = self.coordination_manager.completed_tasks[task_id]
                if task.status == "completed":
                    # Calculate performance score
                    performance_score = 1.0 / (1.0 + (task.completed_at - task.started_at))
                    self.allocation_performance[strategy] += performance_score
        
        # Update strategy performance
        for strategy, performance in self.allocation_performance.items():
            if strategy in self.coordination_manager.strategy_performance:
                self.coordination_manager.strategy_performance[strategy] = performance

class ResourceManager:
    """Handles resource allocation and management"""
    
    def __init__(self, coordination_manager: CoordinationManager):
        self.coordination_manager = coordination_manager
        self.resource_pools = defaultdict(float)
        self.allocation_history = deque(maxlen=1000)
        self.resource_usage = defaultdict(float)
    
    def allocate_resources(self, task: CoordinationTask):
        """Allocate resources to a task"""
        # Calculate required resources
        required_resources = task.resource_requirements.copy()
        
        # Add default resources if not specified
        default_resources = {"cpu": 0.1, "memory": 0.1, "network": 0.1}
        for resource, amount in default_resources.items():
            if resource not in required_resources:
                required_resources[resource] = amount
        
        # Check resource availability
        for resource, required_amount in required_resources.items():
            available_amount = self.resource_pools[resource]
            
            if available_amount < required_amount:
                # Try to reclaim resources from other tasks
                self._reclaim_resources(resource, required_amount - available_amount)
        
        # Allocate resources
        for resource, amount in required_resources.items():
            self.resource_pools[resource] -= amount
            self.resource_usage[f"{task.id}_{resource}"] = amount
        
        # Record allocation
        self.allocation_history.append({
            "task_id": task.id,
            "resources": required_resources,
            "timestamp": time.time()
        })
    
    def _reclaim_resources(self, resource: str, required_amount: float):
        """Reclaim resources from other tasks"""
        # Find tasks using the resource
        resource_users = []
        
        for usage_key, amount in self.resource_usage.items():
            if resource in usage_key:
                task_id = usage_key.split("_")[0]
                if task_id in self.coordination_manager.active_tasks:
                    resource_users.append((task_id, amount))
        
        # Sort by usage (ascending) - reclaim from least critical tasks first
        resource_users.sort(key=lambda x: x[1])
        
        # Reclaim resources
        reclaimed_amount = 0.0
        for task_id, amount in resource_users:
            if reclaimed_amount >= required_amount:
                break
            
            # Reclaim a portion of the resources
            reclaim_amount = min(amount * 0.5, required_amount - reclaimed_amount)
            self.resource_pools[resource] += reclaim_amount
            self.resource_usage[f"{task_id}_{resource}"] -= reclaim_amount
            reclaimed_amount += reclaim_amount
    
    def release_resources(self, task: CoordinationTask):
        """Release resources allocated to a task"""
        # Find and release all resources used by the task
        resources_to_release = []
        
        for usage_key, amount in list(self.resource_usage.items()):
            if task.id in usage_key:
                resource = usage_key.split("_")[1]
                resources_to_release.append((resource, amount))
                del self.resource_usage[usage_key]
        
        # Release resources back to pools
        for resource, amount in resources_to_release:
            self.resource_pools[resource] += amount
    
    def optimize_resources(self):
        """Optimize resource allocation"""
        # Calculate resource utilization
        total_resources = {
            "cpu": 10.0,  # Total available CPU
            "memory": 16.0,  # Total available memory
            "network": 5.0   # Total available network bandwidth
        }
        
        # Calculate current utilization
        utilization = {}
        for resource, total in total_resources.items():
            used = sum(amount for key, amount in self.resource_usage.items() if resource in key)
            utilization[resource] = used / total if total > 0 else 0.0
        
        # Adjust resource pools based on utilization
        for resource, util in utilization.items():
            if util > 0.8:  # High utilization
                # Increase resource pool
                self.resource_pools[resource] += 1.0
            elif util < 0.3:  # Low utilization
                # Decrease resource pool
                self.resource_pools[resource] = max(0.0, self.resource_pools[resource] - 0.5)

class CollaborationEngine:
    """Handles collaboration between agents"""
    
    def __init__(self, coordination_manager: CoordinationManager):
        self.coordination_manager = coordination_manager
        self.collaboration_history = deque(maxlen=1000)
        self.collaboration_patterns = defaultdict(list)
    
    def execute_collaboration(self, initiator: AdvancedAgent, targets: List[AdvancedAgent],
                             collaboration_type: str, content: Any) -> bool:
        """Execute collaboration between agents"""
        # Establish coordination relationships
        for target in targets:
            initiator.coordinate_with(target, collaboration_type)
        
        # Create collaboration signal
        signal = Signal(
            id=f"collab_signal_{int(time.time())}",
            type=SignalType.CONTROL,
            content={
                "type": "collaboration",
                "collaboration_type": collaboration_type,
                "content": content,
                "initiator": initiator.id
            },
            source_agent_id=initiator.id,
            strength=0.8
        )
        
        # Send collaboration signals
        for target in targets:
            target.receive_signal(signal)
        
        # Record collaboration
        self.collaboration_history.append({
            "initiator": initiator.id,
            "targets": [t.id for t in targets],
            "type": collaboration_type,
            "timestamp": time.time(),
            "success": True
        })
        
        # Analyze collaboration patterns
        self._analyze_collaboration_patterns(initiator.id, [t.id for t in targets], collaboration_type)
        
        return True
    
    def _analyze_collaboration_patterns(self, initiator: str, targets: List[str], collaboration_type: str):
        """Analyze collaboration patterns for optimization"""
        pattern_key = f"{collaboration_type}_{len(targets)}"
        self.collaboration_patterns[pattern_key].append({
            "initiator": initiator,
            "targets": targets,
            "timestamp": time.time()
        })
        
        # Keep only recent patterns
        if len(self.collaboration_patterns[pattern_key]) > 100:
            self.collaboration_patterns[pattern_key] = self.collaboration_patterns[pattern_key][-100:]

class ConflictResolver:
    """Handles conflict resolution in coordination"""
    
    def __init__(self, coordination_manager: CoordinationManager):
        self.coordination_manager = coordination_manager
        self.conflict_history = deque(maxlen=1000)
        self.resolution_strategies = {
            "resource_conflict": "negotiation",
            "task_conflict": "prioritization",
            "communication_conflict": "mediation",
            "goal_conflict": "compromise"
        }
    
    def resolve_conflict(self, conflict_type: str, involved_agents: List[str],
                        conflict_details: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve conflict between agents"""
        # Get resolution strategy
        strategy = self.resolution_strategies.get(conflict_type, "negotiation")
        
        # Execute resolution strategy
        if strategy == "negotiation":
            resolution = self._negotiation_resolution(involved_agents, conflict_details)
        elif strategy == "prioritization":
            resolution = self._prioritization_resolution(involved_agents, conflict_details)
        elif strategy == "mediation":
            resolution = self._mediation_resolution(involved_agents, conflict_details)
        elif strategy == "compromise":
            resolution = self._compromise_resolution(involved_agents, conflict_details)
        else:
            resolution = {"strategy": "default", "resolution": "escalate_to_coordinator"}
        
        # Record conflict resolution
        self.conflict_history.append({
            "type": conflict_type,
            "involved_agents": involved_agents,
            "strategy": strategy,
            "resolution": resolution,
            "timestamp": time.time()
        })
        
        return resolution
    
    def _negotiation_resolution(self, involved_agents: List[str], conflict_details: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve conflict through negotiation"""
        # Simple negotiation: find middle ground
        resolution = {
            "strategy": "negotiation",
            "resolution": "compromise",
            "details": {
                "agreed_terms": "mutual_benefits",
                "compromise_level": 0.5
            }
        }
        
        return resolution
    
    def _prioritization_resolution(self, involved_agents: List[str], conflict_details: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve conflict through prioritization"""
        # Prioritize based on agent priority and expertise
        agent_priorities = []
        
        for agent_id in involved_agents:
            if agent_id in self.coordination_manager.agents:
                agent = self.coordination_manager.agents[agent_id]
                priority_score = (
                    agent.priority.value * 0.5 +
                    agent.specialization.expertise_level * 0.5
                )
                agent_priorities.append((agent_id, priority_score))
        
        # Sort by priority
        agent_priorities.sort(key=lambda x: x[1], reverse=True)
        
        resolution = {
            "strategy": "prioritization",
            "resolution": "priority_based",
            "details": {
                "priority_order": [agent_id for agent_id, score in agent_priorities],
                "selected_agent": agent_priorities[0][0] if agent_priorities else None
            }
        }
        
        return resolution
    
    def _mediation_resolution(self, involved_agents: List[str], conflict_details: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve conflict through mediation"""
        # Find neutral mediator
        mediator_id = self._find_neutral_mediator(involved_agents)
        
        resolution = {
            "strategy": "mediation",
            "resolution": "mediated",
            "details": {
                "mediator": mediator_id,
                "mediated_terms": "fair_distribution"
            }
        }
        
        return resolution
    
    def _compromise_resolution(self, involved_agents: List[str], conflict_details: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve conflict through compromise"""
        # Find compromise solution
        resolution = {
            "strategy": "compromise",
            "resolution": "shared_resources",
            "details": {
                "sharing_ratio": "equal",
                "compromise_terms": "mutual_concessions"
            }
        }
        
        return resolution
    
    def _find_neutral_mediator(self, involved_agents: List[str]) -> Optional[str]:
        """Find a neutral mediator for conflict resolution"""
        # Look for agents not involved in conflict
        neutral_agents = [
            agent_id for agent_id in self.coordination_manager.agents.keys()
            if agent_id not in involved_agents
        ]
        
        if neutral_agents:
            # Select agent with high expertise and low load
            best_mediator = None
            best_score = -1
            
            for agent_id in neutral_agents:
                agent = self.coordination_manager.agents[agent_id]
                score = (
                    agent.specialization.expertise_level * 0.7 +
                    (1.0 - agent.load.queue_size / 20.0) * 0.3
                )
                
                if score > best_score:
                    best_score = score
                    best_mediator = agent_id
            
            return best_mediator
        
        return None

class PerformanceOptimizer:
    """Handles performance optimization for coordination"""
    
    def __init__(self, coordination_manager: CoordinationManager):
        self.coordination_manager = coordination_manager
        self.optimization_history = deque(maxlen=100)
        self.performance_thresholds = {
            "efficiency": 0.7,
            "load_balance": 0.6,
            "response_time": 5.0,
            "success_rate": 0.8
        }
    
    def optimize_coordination(self):
        """Optimize overall coordination performance"""
        # Analyze current performance
        performance_issues = self._analyze_performance_issues()
        
        # Apply optimizations based on issues
        for issue in performance_issues:
            if issue["type"] == "low_efficiency":
                self._optimize_efficiency()
            elif issue["type"] == "poor_load_balance":
                self._optimize_load_balance()
            elif issue["type"] == "high_response_time":
                self._optimize_response_time()
            elif issue["type"] == "low_success_rate":
                self._optimize_success_rate()
        
        # Record optimization
        optimization_event = {
            "timestamp": time.time(),
            "issues_addressed": len(performance_issues),
            "optimizations_applied": len(performance_issues)
        }
        
        self.optimization_history.append(optimization_event)
    
    def _analyze_performance_issues(self) -> List[Dict[str, Any]]:
        """Analyze performance issues"""
        issues = []
        
        # Check efficiency
        if self.coordination_manager.metrics.coordination_efficiency < self.performance_thresholds["efficiency"]:
            issues.append({"type": "low_efficiency", "severity": "high"})
        
        # Check load balance
        if self.coordination_manager.metrics.load_balance_score < self.performance_thresholds["load_balance"]:
            issues.append({"type": "poor_load_balance", "severity": "medium"})
        
        # Check response time
        if self.coordination_manager.metrics.average_completion_time > self.performance_thresholds["response_time"]:
            issues.append({"type": "high_response_time", "severity": "medium"})
        
        # Check success rate
        success_rate = self.coordination_manager.metrics.coordination_efficiency
        if success_rate < self.performance_thresholds["success_rate"]:
            issues.append({"type": "low_success_rate", "severity": "high"})
        
        return issues
    
    def _optimize_efficiency(self):
        """Optimize coordination efficiency"""
        # Adjust task allocation strategy
        current_strategy = self.coordination_manager.config["task_allocation_strategy"]
        
        # Try different strategies
        strategies = ["capability_based", "load_balanced", "expertise_based", "hybrid"]
        
        for strategy in strategies:
            if strategy != current_strategy:
                # Test strategy performance
                test_performance = self._test_strategy_performance(strategy)
                
                if test_performance > self.coordination_manager.metrics.coordination_efficiency:
                    self.coordination_manager.config["task_allocation_strategy"] = strategy
                    break
    
    def _optimize_load_balance(self):
        """Optimize load balance across agents"""
        # Identify overloaded and underloaded agents
        overloaded = []
        underloaded = []
        
        for agent_id, utilization in self.coordination_manager.metrics.agent_utilization.items():
            if utilization > 0.8:
                overloaded.append(agent_id)
            elif utilization < 0.3:
                underloaded.append(agent_id)
        
        # Redistribute tasks if possible
        if overloaded and underloaded:
            # This would involve more complex task redistribution logic
            pass
    
    def _optimize_response_time(self):
        """Optimize response time"""
        # Reduce coordination timeout for faster failure detection
        current_timeout = self.coordination_manager.config["coordination_timeout"]
        if current_timeout > 60.0:
            self.coordination_manager.config["coordination_timeout"] = current_timeout * 0.8
    
    def _optimize_success_rate(self):
        """Optimize success rate"""
        # Increase collaboration threshold for better coordination
        current_threshold = self.coordination_manager.config["collaboration_threshold"]
        if current_threshold < 0.9:
            self.coordination_manager.config["collaboration_threshold"] = min(0.9, current_threshold * 1.1)
    
    def _test_strategy_performance(self, strategy: str) -> float:
        """Test performance of a strategy"""
        # This would involve more sophisticated testing
        # For now, return a random performance score
        return random.uniform(0.6, 0.9)

# Global coordination manager instance
coordination_manager = None