"""
Scalable Multi-Agent System: Support for 250+ Specialized Agents

This module implements a sophisticated multi-agent architecture that can scale to support
250+ specialized AI agents working collaboratively. It builds upon the existing brain system,
neural networks, and quantum integration to create a massive, intelligent ecosystem.

Key Features:
- Agent registry and discovery system
- Domain-specific agent specializations
- Load balancing and resource management
- Optimized communication networks
- Performance monitoring and analytics
- Hierarchical organization for scalability
"""

import asyncio
import threading
import time
import uuid
from typing import Dict, List, Set, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import heapq
import json

# Import existing components
from brain import Brain, Agent
from hierarchical_reasoning import FourLevelHRM
from communication_system import CommunicationSystem, Message
from neural_processing_agent import NeuralProcessingAgent
from quantum_neural_integration import QuantumNeuralNetwork, QuantumState


class AgentDomain(Enum):
    """Domain classifications for specialized agents"""
    VISION = "vision"
    NLP = "nlp"
    AUDIO = "audio"
    REASONING = "reasoning"
    PLANNING = "planning"
    LEARNING = "learning"
    MEMORY = "memory"
    COMMUNICATION = "communication"
    QUANTUM = "quantum"
    NEURAL = "neural"
    OPTIMIZATION = "optimization"
    ANALYSIS = "analysis"
    SYNTHESIS = "synthesis"
    COORDINATION = "coordination"
    MONITORING = "monitoring"
    SECURITY = "security"
    ETHICS = "ethics"
    CREATIVITY = "creativity"
    DECISION = "decision"
    PREDICTION = "prediction"
    CLASSIFICATION = "classification"
    GENERATION = "generation"
    TRANSLATION = "translation"
    SUMMARIZATION = "summarization"
    EXTRACTION = "extraction"
    VALIDATION = "validation"
    OPTIMIZATION_ADVANCED = "optimization_advanced"
    SIMULATION = "simulation"
    MODELING = "modeling"
    VISUALIZATION = "visualization"
    INTERFACE = "interface"
    ADAPTATION = "adaptation"
    EVOLUTION = "evolution"
    META_LEARNING = "meta_learning"
    TRANSFER_LEARNING = "transfer_learning"


class AgentPriority(Enum):
    """Priority levels for agent execution"""
    CRITICAL = 0
    HIGH = 1
    MEDIUM = 2
    LOW = 3
    BACKGROUND = 4


class AgentState(Enum):
    """Operational states of agents"""
    IDLE = "idle"
    ACTIVE = "active"
    BUSY = "busy"
    OVERLOADED = "overloaded"
    ERROR = "error"
    MAINTENANCE = "maintenance"
    LEARNING = "learning"


@dataclass
class AgentMetrics:
    """Performance metrics for individual agents"""
    messages_processed: int = 0
    average_response_time: float = 0.0
    success_rate: float = 1.0
    error_count: int = 0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    last_active: float = field(default_factory=time.time)
    load_factor: float = 0.0
    
    def update_response_time(self, response_time: float):
        """Update average response time"""
        if self.messages_processed == 0:
            self.average_response_time = response_time
        else:
            self.average_response_time = (
                0.9 * self.average_response_time + 0.1 * response_time
            )
        self.messages_processed += 1
    
    def update_success(self, success: bool):
        """Update success rate"""
        if success:
            self.success_rate = 0.99 * self.success_rate + 0.01
        else:
            self.success_rate = 0.99 * self.success_rate
            self.error_count += 1


@dataclass
class AgentCapability:
    """Defines the capabilities and expertise of an agent"""
    domain: AgentDomain
    subdomains: List[str]
    complexity_level: int  # 1-10 scale
    expertise_score: float  # 0.0-1.0
    supported_tasks: List[str]
    performance_characteristics: Dict[str, float]
    
    def can_handle(self, task_type: str, complexity: int) -> float:
        """Calculate compatibility score for a given task"""
        if task_type not in self.supported_tasks:
            return 0.0
        
        complexity_match = 1.0 - abs(complexity - self.complexity_level) / 10.0
        return self.expertise_score * complexity_match


class ScalableAgent(Agent):
    """Enhanced agent with scalability features and advanced capabilities"""
    
    def __init__(self, agent_id: str, name: str, domain: AgentDomain, 
                 capabilities: AgentCapability, priority: AgentPriority = AgentPriority.MEDIUM):
        super().__init__(agent_id, name)
        self.domain = domain
        self.capabilities = capabilities
        self.priority = priority
        self.state = AgentState.IDLE
        self.metrics = AgentMetrics()
        self.load_history = deque(maxlen=100)
        self.task_queue = asyncio.Queue()
        self.max_concurrent_tasks = 3
        self.active_tasks = 0
        self.specialized_models = {}
        self.quantum_enhanced = False
        self.neural_enhanced = False
        
        # Initialize hierarchical reasoning
        self.hrm = FourLevelHRM()
        
        # Performance monitoring
        self.response_times = deque(maxlen=50)
        self.error_log = deque(maxlen=20)
        
    async def process_message(self, message: Message) -> Optional[Message]:
        """Process incoming message with enhanced scalability"""
        start_time = time.time()
        
        try:
            # Check if agent can handle this task
            if not self._can_handle_message(message):
                return None
            
            # Update state
            self.state = AgentState.BUSY
            self.active_tasks += 1
            
            # Process using hierarchical reasoning
            result = await self._process_with_hrm(message)
            
            # Update metrics
            response_time = time.time() - start_time
            self.metrics.update_response_time(response_time)
            self.metrics.update_success(result is not None)
            self.response_times.append(response_time)
            
            # Update load factor
            self._update_load_factor()
            
            return result
            
        except Exception as e:
            self.metrics.update_success(False)
            self.error_log.append(str(e))
            self.state = AgentState.ERROR
            return None
        finally:
            self.active_tasks -= 1
            self.metrics.last_active = time.time()
            if self.active_tasks == 0:
                self.state = AgentState.IDLE
    
    def _can_handle_message(self, message: Message) -> bool:
        """Check if agent can handle the message based on capabilities"""
        task_type = message.content.get("task_type", "")
        complexity = message.content.get("complexity", 5)
        
        compatibility = self.capabilities.can_handle(task_type, complexity)
        return compatibility > 0.5
    
    async def _process_with_hrm(self, message: Message) -> Optional[Message]:
        """Process message using four-level hierarchical reasoning"""
        # Visionary level: Set high-level goals
        visionary_goal = self.hrm.visionary.process(message)
        
        # Architect level: Create strategic plan
        architect_plan = self.hrm.architect.process(visionary_goal)
        
        # Foreman level: Coordinate execution
        foreman_tasks = self.hrm.foreman.process(architect_plan)
        
        # Technician level: Execute specific actions
        technician_result = await self.hrm.technician.execute(foreman_tasks)
        
        return technician_result
    
    def _update_load_factor(self):
        """Update load factor based on current activity"""
        base_load = self.active_tasks / self.max_concurrent_tasks
        recent_load = sum(self.load_history) / len(self.load_history) if self.load_history else 0
        self.metrics.load_factor = 0.7 * base_load + 0.3 * recent_load
        self.load_history.append(self.metrics.load_factor)
    
    def get_health_score(self) -> float:
        """Calculate overall health score (0.0-1.0)"""
        load_score = 1.0 - self.metrics.load_factor
        success_score = self.metrics.success_rate
        response_score = 1.0 / (1.0 + self.metrics.average_response_time)
        
        return (load_score + success_score + response_score) / 3.0
    
    def enable_quantum_enhancement(self, quantum_network: QuantumNeuralNetwork):
        """Enable quantum processing capabilities"""
        self.quantum_enhanced = True
        self.quantum_network = quantum_network
    
    def enable_neural_enhancement(self, neural_models: Dict[str, Any]):
        """Enable neural processing capabilities"""
        self.neural_enhanced = True
        self.specialized_models.update(neural_models)


class AgentRegistry:
    """Central registry for agent discovery and management"""
    
    def __init__(self):
        self.agents: Dict[str, ScalableAgent] = {}
        self.domain_index: Dict[AgentDomain, Set[str]] = defaultdict(set)
        self.capability_index: Dict[str, Set[str]] = defaultdict(set)
        self.priority_index: Dict[AgentPriority, Set[str]] = defaultdict(set)
        self.state_index: Dict[AgentState, Set[str]] = defaultdict(set)
        
        # Performance tracking
        self.registration_time = {}
        self.last_heartbeat = {}
        
    def register_agent(self, agent: ScalableAgent) -> bool:
        """Register a new agent"""
        if agent.agent_id in self.agents:
            return False
        
        self.agents[agent.agent_id] = agent
        self.domain_index[agent.domain].add(agent.agent_id)
        self.priority_index[agent.priority].add(agent.agent_id)
        self.state_index[agent.state].add(agent.agent_id)
        
        # Index capabilities
        for task in agent.capabilities.supported_tasks:
            self.capability_index[task].add(agent.agent_id)
        
        self.registration_time[agent.agent_id] = time.time()
        self.last_heartbeat[agent.agent_id] = time.time()
        
        return True
    
    def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent"""
        if agent_id not in self.agents:
            return False
        
        agent = self.agents[agent_id]
        
        # Remove from indexes
        self.domain_index[agent.domain].discard(agent_id)
        self.priority_index[agent.priority].discard(agent_id)
        self.state_index[agent.state].discard(agent_id)
        
        for task in agent.capabilities.supported_tasks:
            self.capability_index[task].discard(agent_id)
        
        # Remove from registries
        del self.agents[agent_id]
        del self.registration_time[agent_id]
        del self.last_heartbeat[agent_id]
        
        return True
    
    def find_agents_by_domain(self, domain: AgentDomain) -> List[ScalableAgent]:
        """Find agents by domain"""
        agent_ids = self.domain_index.get(domain, set())
        return [self.agents[aid] for aid in agent_ids if aid in self.agents]
    
    def find_agents_by_capability(self, capability: str) -> List[ScalableAgent]:
        """Find agents by capability"""
        agent_ids = self.capability_index.get(capability, set())
        return [self.agents[aid] for aid in agent_ids if aid in self.agents]
    
    def find_agents_by_priority(self, priority: AgentPriority) -> List[ScalableAgent]:
        """Find agents by priority"""
        agent_ids = self.priority_index.get(priority, set())
        return [self.agents[aid] for aid in agent_ids if aid in self.agents]
    
    def find_available_agents(self, max_load: float = 0.7) -> List[ScalableAgent]:
        """Find agents with available capacity"""
        available = []
        for agent in self.agents.values():
            if (agent.state in [AgentState.IDLE, AgentState.ACTIVE] and 
                agent.metrics.load_factor < max_load):
                available.append(agent)
        return available
    
    def get_agent_by_id(self, agent_id: str) -> Optional[ScalableAgent]:
        """Get agent by ID"""
        return self.agents.get(agent_id)
    
    def update_agent_state(self, agent_id: str, new_state: AgentState):
        """Update agent state"""
        if agent_id not in self.agents:
            return
        
        agent = self.agents[agent_id]
        self.state_index[agent.state].discard(agent_id)
        agent.state = new_state
        self.state_index[new_state].add(agent_id)
        self.last_heartbeat[agent_id] = time.time()
    
    def heartbeat(self, agent_id: str):
        """Update agent heartbeat"""
        if agent_id in self.agents:
            self.last_heartbeat[agent_id] = time.time()
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system-wide statistics"""
        total_agents = len(self.agents)
        active_agents = len(self.state_index[AgentState.ACTIVE])
        busy_agents = len(self.state_index[AgentState.BUSY])
        overloaded_agents = len(self.state_index[AgentState.OVERLOADED])
        error_agents = len(self.state_index[AgentState.ERROR])
        
        domain_distribution = {
            domain.value: len(agents) 
            for domain, agents in self.domain_index.items()
        }
        
        return {
            "total_agents": total_agents,
            "active_agents": active_agents,
            "busy_agents": busy_agents,
            "overloaded_agents": overloaded_agents,
            "error_agents": error_agents,
            "domain_distribution": domain_distribution,
            "average_load": self._calculate_average_load(),
            "system_health": self._calculate_system_health()
        }
    
    def _calculate_average_load(self) -> float:
        """Calculate average system load"""
        if not self.agents:
            return 0.0
        return sum(agent.metrics.load_factor for agent in self.agents.values()) / len(self.agents)
    
    def _calculate_system_health(self) -> float:
        """Calculate overall system health"""
        if not self.agents:
            return 0.0
        return sum(agent.get_health_score() for agent in self.agents.values()) / len(self.agents)


class LoadBalancer:
    """Intelligent load balancing for multi-agent systems"""
    
    def __init__(self, registry: AgentRegistry):
        self.registry = registry
        self.task_history = deque(maxlen=1000)
        self.load_distribution = defaultdict(float)
        self.optimization_interval = 30  # seconds
        self.last_optimization = time.time()
        
    async def assign_task(self, task_type: str, complexity: int, 
                         content: Dict[str, Any]) -> Optional[ScalableAgent]:
        """Assign task to best available agent"""
        # Find candidate agents
        candidates = self._find_candidate_agents(task_type, complexity)
        
        if not candidates:
            return None
        
        # Select best agent using load balancing algorithm
        best_agent = self._select_best_agent(candidates, task_type, complexity)
        
        if best_agent:
            # Create and assign task
            message = Message(
                sender_id="load_balancer",
                receiver_id=best_agent.agent_id,
                content={
                    "task_type": task_type,
                    "complexity": complexity,
                    **content
                }
            )
            
            # Record task assignment
            self.task_history.append({
                "task_type": task_type,
                "complexity": complexity,
                "agent_id": best_agent.agent_id,
                "timestamp": time.time()
            })
            
            # Update load distribution
            self.load_distribution[best_agent.agent_id] += complexity / 10.0
            
            return best_agent
        
        return None
    
    def _find_candidate_agents(self, task_type: str, complexity: int) -> List[ScalableAgent]:
        """Find candidate agents for the task"""
        candidates = []
        
        # Find agents by capability
        capable_agents = self.registry.find_agents_by_capability(task_type)
        
        for agent in capable_agents:
            # Check if agent can handle the complexity
            compatibility = agent.capabilities.can_handle(task_type, complexity)
            if compatibility > 0.5 and agent.metrics.load_factor < 0.8:
                candidates.append((agent, compatibility))
        
        # Sort by compatibility
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [agent for agent, _ in candidates]
    
    def _select_best_agent(self, candidates: List[ScalableAgent], 
                          task_type: str, complexity: int) -> Optional[ScalableAgent]:
        """Select best agent using multi-criteria decision"""
        if not candidates:
            return None
        
        # Score each candidate
        scored_agents = []
        for agent in candidates:
            score = self._calculate_agent_score(agent, task_type, complexity)
            scored_agents.append((agent, score))
        
        # Select highest scoring agent
        scored_agents.sort(key=lambda x: x[1], reverse=True)
        return scored_agents[0][0]
    
    def _calculate_agent_score(self, agent: ScalableAgent, 
                             task_type: str, complexity: int) -> float:
        """Calculate agent suitability score"""
        # Compatibility score
        compatibility = agent.capabilities.can_handle(task_type, complexity)
        
        # Load score (prefer less loaded agents)
        load_score = 1.0 - agent.metrics.load_factor
        
        # Performance score
        performance_score = agent.get_health_score()
        
        # Priority score (prefer higher priority agents)
        priority_score = 1.0 - (agent.priority.value / 5.0)
        
        # Combine scores with weights
        total_score = (
            0.4 * compatibility +
            0.3 * load_score +
            0.2 * performance_score +
            0.1 * priority_score
        )
        
        return total_score
    
    def optimize_load_distribution(self):
        """Optimize load distribution across agents"""
        current_time = time.time()
        if current_time - self.last_optimization < self.optimization_interval:
            return
        
        self.last_optimization = current_time
        
        # Analyze load patterns
        overloaded_agents = [
            agent for agent in self.registry.agents.values()
            if agent.metrics.load_factor > 0.8
        ]
        
        underloaded_agents = [
            agent for agent in self.registry.agents.values()
            if agent.metrics.load_factor < 0.3
        ]
        
        # Suggest load balancing actions
        if overloaded_agents and underloaded_agents:
            print(f"Load balancing: {len(overloaded_agents)} overloaded, {len(underloaded_agents)} underloaded")
            # In a real implementation, this would trigger task migration


class MultiAgentSystem:
    """Main multi-agent system orchestrator"""
    
    def __init__(self, target_agent_count: int = 250):
        self.target_agent_count = target_agent_count
        self.registry = AgentRegistry()
        self.load_balancer = LoadBalancer(self.registry)
        self.communication_system = CommunicationSystem()
        
        # System components
        self.brain = Brain()
        self.quantum_network = QuantumNeuralNetwork()
        
        # Performance monitoring
        self.performance_history = deque(maxlen=1000)
        self.system_metrics = defaultdict(list)
        
        # Background tasks
        self.monitoring_task = None
        self.optimization_task = None
        
    async def initialize_system(self):
        """Initialize the multi-agent system"""
        print(f"Initializing multi-agent system with {self.target_agent_count} agents...")
        
        # Create specialized agents
        await self._create_specialized_agents()
        
        # Start background tasks
        self.monitoring_task = asyncio.create_task(self._monitor_system_performance())
        self.optimization_task = asyncio.create_task(self._optimize_system())
        
        print(f"System initialized with {len(self.registry.agents)} agents")
    
    async def _create_specialized_agents(self):
        """Create 250+ specialized agents across different domains"""
        agent_configs = self._generate_agent_configurations()
        
        for config in agent_configs:
            agent = await self._create_agent_from_config(config)
            self.registry.register_agent(agent)
            
            # Enable enhancements based on domain
            if agent.domain in [AgentDomain.QUANTUM, AgentDomain.OPTIMIZATION]:
                agent.enable_quantum_enhancement(self.quantum_network)
            
            if agent.domain in [AgentDomain.NEURAL, AgentDomain.VISION, AgentDomain.NLP]:
                neural_models = await self._create_neural_models(agent.domain)
                agent.enable_neural_enhancement(neural_models)
    
    def _generate_agent_configurations(self) -> List[Dict[str, Any]]:
        """Generate configurations for 250+ specialized agents"""
        configs = []
        
        # Domain distribution
        domain_weights = {
            AgentDomain.VISION: 20,
            AgentDomain.NLP: 25,
            AgentDomain.AUDIO: 15,
            AgentDomain.REASONING: 30,
            AgentDomain.PLANNING: 20,
            AgentDomain.LEARNING: 25,
            AgentDomain.MEMORY: 15,
            AgentDomain.COMMUNICATION: 20,
            AgentDomain.QUANTUM: 10,
            AgentDomain.NEURAL: 15,
            AgentDomain.OPTIMIZATION: 15,
            AgentDomain.ANALYSIS: 20,
            AgentDomain.SYNTHESIS: 10,
            AgentDomain.COORDINATION: 15,
            AgentDomain.MONITORING: 10,
            AgentDomain.SECURITY: 10,
            AgentDomain.ETHICS: 5,
            AgentDomain.CREATIVITY: 10,
            AgentDomain.DECISION: 15,
            AgentDomain.PREDICTION: 15,
            AgentDomain.CLASSIFICATION: 20,
            AgentDomain.GENERATION: 15,
            AgentDomain.TRANSLATION: 10,
            AgentDomain.SUMMARIZATION: 10,
            AgentDomain.EXTRACTION: 15,
            AgentDomain.VALIDATION: 10,
            AgentDomain.OPTIMIZATION_ADVANCED: 10,
            AgentDomain.SIMULATION: 10,
            AgentDomain.MODELING: 10,
            AgentDomain.VISUALIZATION: 10,
            AgentDomain.INTERFACE: 10,
            AgentDomain.ADAPTATION: 10,
            AgentDomain.EVOLUTION: 5,
            AgentDomain.META_LEARNING: 5,
            AgentDomain.TRANSFER_LEARNING: 5
        }
        
        agent_id_counter = 0
        
        for domain, count in domain_weights.items():
            for i in range(count):
                agent_id = f"agent_{agent_id_counter:03d}"
                agent_id_counter += 1
                
                config = {
                    "agent_id": agent_id,
                    "name": f"{domain.value.title()}Agent_{i}",
                    "domain": domain,
                    "priority": self._assign_priority(domain),
                    "capabilities": self._generate_capabilities(domain, i)
                }
                
                configs.append(config)
        
        return configs
    
    def _assign_priority(self, domain: AgentDomain) -> AgentPriority:
        """Assign priority based on domain importance"""
        high_priority_domains = {
            AgentDomain.REASONING, AgentDomain.PLANNING, AgentDomain.DECISION,
            AgentDomain.SECURITY, AgentDomain.ETHICS, AgentDomain.COORDINATION
        }
        
        medium_priority_domains = {
            AgentDomain.VISION, AgentDomain.NLP, AgentDomain.LEARNING,
            AgentDomain.MEMORY, AgentDomain.COMMUNICATION, AgentDomain.QUANTUM
        }
        
        if domain in high_priority_domains:
            return AgentPriority.HIGH
        elif domain in medium_priority_domains:
            return AgentPriority.MEDIUM
        else:
            return AgentPriority.LOW
    
    def _generate_capabilities(self, domain: AgentDomain, index: int) -> AgentCapability:
        """Generate capabilities for an agent based on domain"""
        capability_map = {
            AgentDomain.VISION: {
                "subdomains": ["image_recognition", "object_detection", "scene_analysis"],
                "tasks": ["image_classification", "face_detection", "object_tracking"],
                "complexity": 7
            },
            AgentDomain.NLP: {
                "subdomains": ["text_understanding", "sentiment_analysis", "language_generation"],
                "tasks": ["text_classification", "sentiment_analysis", "text_generation"],
                "complexity": 8
            },
            AgentDomain.REASONING: {
                "subdomains": ["logical_reasoning", "causal_inference", "decision_making"],
                "tasks": ["logical_analysis", "causal_reasoning", "decision_support"],
                "complexity": 9
            },
            AgentDomain.QUANTUM: {
                "subdomains": ["quantum_computing", "quantum_algorithms", "quantum_optimization"],
                "tasks": ["quantum_simulation", "quantum_optimization", "quantum_analysis"],
                "complexity": 10
            }
        }
        
        # Default capabilities
        default_caps = {
            "subdomains": [f"{domain.value}_processing"],
            "tasks": [f"{domain.value}_task"],
            "complexity": 5
        }
        
        caps = capability_map.get(domain, default_caps)
        
        return AgentCapability(
            domain=domain,
            subdomains=caps["subdomains"],
            complexity_level=caps["complexity"],
            expertise_score=0.7 + (index % 3) * 0.1,  # Vary expertise
            supported_tasks=caps["tasks"],
            performance_characteristics={
                "speed": 0.8 + (index % 5) * 0.04,
                "accuracy": 0.85 + (index % 4) * 0.03,
                "efficiency": 0.75 + (index % 6) * 0.04
            }
        )
    
    async def _create_agent_from_config(self, config: Dict[str, Any]) -> ScalableAgent:
        """Create an agent from configuration"""
        return ScalableAgent(
            agent_id=config["agent_id"],
            name=config["name"],
            domain=config["domain"],
            capabilities=config["capabilities"],
            priority=config["priority"]
        )
    
    async def _create_neural_models(self, domain: AgentDomain) -> Dict[str, Any]:
        """Create neural models for agent enhancement"""
        # Mock neural model creation
        return {
            "model_type": f"{domain.value}_model",
            "parameters": {"layers": 3, "units": 128},
            "trained": True
        }
    
    async def process_task(self, task_type: str, complexity: int, 
                          content: Dict[str, Any]) -> Dict[str, Any]:
        """Process a task using the multi-agent system"""
        # Find best agent for the task
        agent = await self.load_balancer.assign_task(task_type, complexity, content)
        
        if not agent:
            return {"error": "No suitable agent found"}
        
        # Create message
        message = Message(
            sender_id="system",
            receiver_id=agent.agent_id,
            content={
                "task_type": task_type,
                "complexity": complexity,
                **content
            }
        )
        
        # Process message
        result = await agent.process_message(message)
        
        return {
            "agent_id": agent.agent_id,
            "result": result.content if result else None,
            "processing_time": agent.metrics.average_response_time,
            "agent_load": agent.metrics.load_factor
        }
    
    async def _monitor_system_performance(self):
        """Continuously monitor system performance"""
        while True:
            try:
                stats = self.registry.get_system_stats()
                self.performance_history.append(stats)
                
                # Check for system health issues
                if stats["system_health"] < 0.5:
                    print(f"Warning: System health low ({stats['system_health']:.2f})")
                
                # Log performance metrics
                for metric, value in stats.items():
                    if isinstance(value, (int, float)):
                        self.system_metrics[metric].append(value)
                
                await asyncio.sleep(5)  # Monitor every 5 seconds
                
            except Exception as e:
                print(f"Monitoring error: {e}")
                await asyncio.sleep(10)
    
    async def _optimize_system(self):
        """Continuously optimize system performance"""
        while True:
            try:
                # Optimize load distribution
                self.load_balancer.optimize_load_distribution()
                
                # Check for overloaded agents
                overloaded_agents = [
                    agent for agent in self.registry.agents.values()
                    if agent.metrics.load_factor > 0.9
                ]
                
                if overloaded_agents:
                    print(f"Found {len(overloaded_agents)} overloaded agents")
                    # Implement load balancing strategies
                
                await asyncio.sleep(30)  # Optimize every 30 seconds
                
            except Exception as e:
                print(f"Optimization error: {e}")
                await asyncio.sleep(60)
    
    def get_system_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive system dashboard"""
        stats = self.registry.get_system_stats()
        
        # Calculate additional metrics
        total_tasks = sum(agent.metrics.messages_processed for agent in self.registry.agents.values())
        average_response_time = sum(agent.metrics.average_response_time for agent in self.registry.agents.values()) / len(self.registry.agents) if self.registry.agents else 0
        
        # Agent distribution by state
        state_distribution = {
            state.value: len(agents)
            for state, agents in self.registry.state_index.items()
        }
        
        # Top performing agents
        top_agents = sorted(
            self.registry.agents.values(),
            key=lambda a: a.get_health_score(),
            reverse=True
        )[:10]
        
        return {
            "system_stats": stats,
            "total_tasks_processed": total_tasks,
            "average_response_time": average_response_time,
            "state_distribution": state_distribution,
            "top_performing_agents": [
                {
                    "id": agent.agent_id,
                    "name": agent.name,
                    "domain": agent.domain.value,
                    "health_score": agent.get_health_score(),
                    "tasks_processed": agent.metrics.messages_processed
                }
                for agent in top_agents
            ],
            "recent_performance": list(self.performance_history)[-10:] if self.performance_history else []
        }
    
    async def shutdown(self):
        """Shutdown the multi-agent system"""
        print("Shutting down multi-agent system...")
        
        # Cancel background tasks
        if self.monitoring_task:
            self.monitoring_task.cancel()
        if self.optimization_task:
            self.optimization_task.cancel()
        
        print("Multi-agent system shutdown complete")


# Factory functions for creating specialized agent types
def create_vision_agent(agent_id: str, specialization: str = "general") -> ScalableAgent:
    """Create a specialized vision processing agent"""
    capabilities = AgentCapability(
        domain=AgentDomain.VISION,
        subdomains=["image_recognition", "object_detection", "scene_analysis"],
        complexity_level=7,
        expertise_score=0.8,
        supported_tasks=["image_classification", "face_detection", "object_tracking"],
        performance_characteristics={
            "speed": 0.85,
            "accuracy": 0.92,
            "efficiency": 0.78
        }
    )
    
    return ScalableAgent(
        agent_id=agent_id,
        name=f"VisionAgent_{specialization}",
        domain=AgentDomain.VISION,
        capabilities=capabilities,
        priority=AgentPriority.MEDIUM
    )


def create_reasoning_agent(agent_id: str, reasoning_type: str = "logical") -> ScalableAgent:
    """Create a specialized reasoning agent"""
    capabilities = AgentCapability(
        domain=AgentDomain.REASONING,
        subdomains=["logical_reasoning", "causal_inference", "decision_making"],
        complexity_level=9,
        expertise_score=0.9,
        supported_tasks=["logical_analysis", "causal_reasoning", "decision_support"],
        performance_characteristics={
            "speed": 0.75,
            "accuracy": 0.95,
            "efficiency": 0.82
        }
    )
    
    return ScalableAgent(
        agent_id=agent_id,
        name=f"ReasoningAgent_{reasoning_type}",
        domain=AgentDomain.REASONING,
        capabilities=capabilities,
        priority=AgentPriority.HIGH
    )


def create_quantum_agent(agent_id: str, quantum_specialty: str = "optimization") -> ScalableAgent:
    """Create a specialized quantum processing agent"""
    capabilities = AgentCapability(
        domain=AgentDomain.QUANTUM,
        subdomains=["quantum_computing", "quantum_algorithms", "quantum_optimization"],
        complexity_level=10,
        expertise_score=0.85,
        supported_tasks=["quantum_simulation", "quantum_optimization", "quantum_analysis"],
        performance_characteristics={
            "speed": 0.95,  # Quantum speedup
            "accuracy": 0.88,
            "efficiency": 0.90
        }
    )
    
    return ScalableAgent(
        agent_id=agent_id,
        name=f"QuantumAgent_{quantum_specialty}",
        domain=AgentDomain.QUANTUM,
        capabilities=capabilities,
        priority=AgentPriority.HIGH
    )


# Main execution
async def main():
    """Main function to demonstrate the scalable multi-agent system"""
    print("🚀 Initializing Scalable Multi-Agent System")
    print("=" * 60)
    
    # Create multi-agent system
    multi_agent_system = MultiAgentSystem(target_agent_count=250)
    
    # Initialize system
    await multi_agent_system.initialize_system()
    
    # Display system dashboard
    dashboard = multi_agent_system.get_system_dashboard()
    print(f"\n📊 System Dashboard:")
    print(f"Total Agents: {dashboard['system_stats']['total_agents']}")
    print(f"System Health: {dashboard['system_stats']['system_health']:.2f}")
    print(f"Average Load: {dashboard['system_stats']['average_load']:.2f}")
    print(f"Domain Distribution: {dashboard['system_stats']['domain_distribution']}")
    
    # Demonstrate task processing
    print(f"\n🔄 Processing Sample Tasks:")
    
    # Sample tasks
    sample_tasks = [
        ("image_classification", 6, {"image_data": "sample_image_1.jpg"}),
        ("logical_analysis", 8, {"problem": "complex_logic_problem"}),
        ("quantum_optimization", 9, {"problem": "traveling_salesman", "cities": 50}),
        ("text_generation", 7, {"prompt": "Write a story about AI", "length": 500}),
        ("decision_support", 8, {"options": ["A", "B", "C"], "criteria": ["cost", "quality"]})
    ]
    
    for task_type, complexity, content in sample_tasks:
        print(f"\nProcessing {task_type} (complexity: {complexity})...")
        result = await multi_agent_system.process_task(task_type, complexity, content)
        
        if "error" not in result:
            print(f"✅ Task completed by {result['agent_id']}")
            print(f"   Processing time: {result['processing_time']:.3f}s")
            print(f"   Agent load: {result['agent_load']:.2f}")
        else:
            print(f"❌ Task failed: {result['error']}")
    
    # Display final dashboard
    print(f"\n📈 Final System Status:")
    final_dashboard = multi_agent_system.get_system_dashboard()
    print(f"Total Tasks Processed: {final_dashboard['total_tasks_processed']}")
    print(f"Average Response Time: {final_dashboard['average_response_time']:.3f}s")
    print(f"Top Performing Agents:")
    for agent in final_dashboard['top_performing_agents'][:5]:
        print(f"  - {agent['name']} ({agent['domain']}): Health {agent['health_score']:.2f}, Tasks {agent['tasks_processed']}")
    
    # Shutdown
    await multi_agent_system.shutdown()
    print(f"\n🏁 Multi-Agent System Demo Complete")


if __name__ == "__main__":
    asyncio.run(main())