"""
Neuron-style Weighted Propagation Communication System

This module implements a communication system inspired by biological neurons,
where agents act as specialized neurons that communicate through weighted
connections, threshold-based activation, and signal propagation.
"""

import random
import time
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import math
from collections import defaultdict, deque

# Import HRM system for integration
from hrm_system import HRMSystem, Task, TaskPriority

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SignalType(Enum):
    """Types of signals that can propagate through the network"""
    ACTIVATION = "activation"
    DATA = "data"
    CONTROL = "control"
    FEEDBACK = "feedback"
    ERROR = "error"

@dataclass
class Signal:
    """Represents a signal propagating through the network"""
    id: str
    type: SignalType
    content: Any
    source_agent_id: str
    target_agent_id: Optional[str] = None
    strength: float = 1.0
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    propagation_path: List[str] = field(default_factory=list)
    
    def decay(self, decay_factor: float):
        """Apply decay to signal strength"""
        self.strength *= decay_factor
        
    def add_to_path(self, agent_id: str):
        """Add agent to propagation path"""
        if agent_id not in self.propagation_path:
            self.propagation_path.append(agent_id)

@dataclass
class Connection:
    """Represents a connection between two agents"""
    source_id: str
    target_id: str
    weight: float
    connection_type: str = "excitatory"  # or "inhibitory"
    delay: float = 0.0  # transmission delay
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def propagate_signal(self, signal: Signal) -> Signal:
        """Propagate a signal through this connection"""
        # Create a new signal for the target
        new_signal = Signal(
            id=f"{signal.id}_{self.target_id}",
            type=signal.type,
            content=signal.content,
            source_agent_id=signal.source_agent_id,
            target_agent_id=self.target_id,
            strength=signal.strength * self.weight,
            timestamp=time.time() + self.delay,
            metadata=signal.metadata.copy(),
            propagation_path=signal.propagation_path.copy()
        )
        
        # Add current connection to path
        new_signal.add_to_path(self.source_id)
        
        return new_signal

class Agent:
    """Represents an agent as a neuron in the communication network"""
    
    def __init__(self, agent_id: str, specialization: str, threshold: float = 0.5):
        self.id = agent_id
        self.specialization = specialization
        self.threshold = threshold
        self.connections: List[Connection] = []
        self.input_buffer: deque = deque()
        self.output_buffer: deque = deque()
        self.is_active = False
        self.last_activation_time = 0.0
        self.activation_count = 0
        self.hrm_system = HRMSystem()  # Each agent has its own HRM system
        self.processing_history = []
        self.logger = logging.getLogger(f"{__name__}.Agent.{agent_id}")
        
        # Agent-specific properties
        self.capabilities = self._define_capabilities()
        self.performance_metrics = {
            "signals_processed": 0,
            "average_processing_time": 0.0,
            "success_rate": 1.0,
            "resource_usage": 0.0
        }
        
    def _define_capabilities(self) -> List[str]:
        """Define agent capabilities based on specialization"""
        capability_map = {
            "PythonSyntaxFixer": ["parse_python", "fix_syntax", "validate_code"],
            "SQLQueryCrafter": ["generate_query", "optimize_query", "validate_sql"],
            "CreativeWriter": ["generate_text", "edit_content", "style_adaptation"],
            "DataAnalyzer": ["process_data", "generate_insights", "create_reports"],
            "GeneralProcessor": ["process_task", "analyze_input", "generate_output"]
        }
        
        return capability_map.get(self.specialization, ["process_task"])
    
    def add_connection(self, target_agent: 'Agent', weight: float, connection_type: str = "excitatory"):
        """Add a connection to another agent"""
        connection = Connection(
            source_id=self.id,
            target_id=target_agent.id,
            weight=weight,
            connection_type=connection_type
        )
        self.connections.append(connection)
        self.logger.debug(f"Added connection to {target_agent.id} with weight {weight}")
    
    def receive_signal(self, signal: Signal):
        """Receive a signal and add it to input buffer"""
        self.input_buffer.append(signal)
        self.logger.debug(f"Received signal {signal.id} with strength {signal.strength}")
        
        # Check if signal exceeds threshold
        if signal.strength >= self.threshold:
            self._activate(signal)
    
    def _activate(self, signal: Signal):
        """Activate the agent and process the signal"""
        self.is_active = True
        self.last_activation_time = time.time()
        self.activation_count += 1
        
        self.logger.info(f"Agent {self.id} activated by signal {signal.id}")
        
        # Process the signal using HRM
        try:
            result = self._process_with_hrm(signal)
            
            # Create output signal
            output_signal = Signal(
                id=f"output_{self.id}_{int(time.time())}",
                type=SignalType.DATA,
                content=result,
                source_agent_id=self.id,
                strength=signal.strength * 0.8,  # Slight decay
                metadata={"processing_time": time.time() - self.last_activation_time}
            )
            
            self.output_buffer.append(output_signal)
            
            # Update performance metrics
            self._update_performance_metrics(True, time.time() - self.last_activation_time)
            
        except Exception as e:
            self.logger.error(f"Error processing signal: {e}")
            
            # Create error signal
            error_signal = Signal(
                id=f"error_{self.id}_{int(time.time())}",
                type=SignalType.ERROR,
                content={"error": str(e)},
                source_agent_id=self.id,
                strength=signal.strength * 0.5
            )
            
            self.output_buffer.append(error_signal)
            self._update_performance_metrics(False, time.time() - self.last_activation_time)
        
        # Reset active state
        self.is_active = False
    
    def _process_with_hrm(self, signal: Signal) -> Any:
        """Process signal using the agent's HRM system"""
        # Convert signal to HRM directive
        directive = self._signal_to_directive(signal)
        
        # Process through HRM
        result = self.hrm_system.process_directive(directive, signal.metadata)
        
        # Record processing
        self.processing_history.append({
            "signal_id": signal.id,
            "directive": directive,
            "result": result,
            "timestamp": time.time()
        })
        
        return result
    
    def _signal_to_directive(self, signal: Signal) -> str:
        """Convert signal to HRM directive"""
        if signal.type == SignalType.ACTIVATION:
            return f"Process activation request: {signal.content}"
        elif signal.type == SignalType.DATA:
            return f"Analyze data: {signal.content}"
        elif signal.type == SignalType.CONTROL:
            return f"Execute control command: {signal.content}"
        elif signal.type == SignalType.FEEDBACK:
            return f"Process feedback: {signal.content}"
        else:
            return f"Process signal: {signal.content}"
    
    def _update_performance_metrics(self, success: bool, processing_time: float):
        """Update agent performance metrics"""
        self.performance_metrics["signals_processed"] += 1
        
        # Update average processing time
        current_avg = self.performance_metrics["average_processing_time"]
        total_signals = self.performance_metrics["signals_processed"]
        self.performance_metrics["average_processing_time"] = (
            (current_avg * (total_signals - 1) + processing_time) / total_signals
        )
        
        # Update success rate
        if success:
            current_rate = self.performance_metrics["success_rate"]
            self.performance_metrics["success_rate"] = (
                (current_rate * (total_signals - 1) + 1.0) / total_signals
            )
        else:
            current_rate = self.performance_metrics["success_rate"]
            self.performance_metrics["success_rate"] = (
                (current_rate * (total_signals - 1)) / total_signals
            )
    
    def propagate_signals(self) -> List[Signal]:
        """Propagate output signals to connected agents"""
        propagated_signals = []
        
        while self.output_buffer:
            signal = self.output_buffer.popleft()
            
            for connection in self.connections:
                if connection.connection_type == "excitatory":
                    propagated_signal = connection.propagate_signal(signal)
                    propagated_signals.append(propagated_signal)
                    
                    self.logger.debug(f"Propagated signal to {connection.target_id} "
                                    f"with strength {propagated_signal.strength}")
        
        return propagated_signals
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status information"""
        return {
            "id": self.id,
            "specialization": self.specialization,
            "threshold": self.threshold,
            "is_active": self.is_active,
            "activation_count": self.activation_count,
            "last_activation_time": self.last_activation_time,
            "input_buffer_size": len(self.input_buffer),
            "output_buffer_size": len(self.output_buffer),
            "connection_count": len(self.connections),
            "performance_metrics": self.performance_metrics
        }

class CommunicationNetwork:
    """Manages the network of communicating agents"""
    
    def __init__(self):
        self.agents: Dict[str, Agent] = {}
        self.global_signal_history = []
        self.network_metrics = {
            "total_signals": 0,
            "average_signal_strength": 0.0,
            "network_activity": 0.0,
            "agent_activity": defaultdict(int)
        }
        self.logger = logging.getLogger(__name__)
        
    def add_agent(self, agent: Agent):
        """Add an agent to the network"""
        self.agents[agent.id] = agent
        self.logger.info(f"Added agent {agent.id} with specialization {agent.specialization}")
    
    def remove_agent(self, agent_id: str):
        """Remove an agent from the network"""
        if agent_id in self.agents:
            # Remove connections to this agent
            for agent in self.agents.values():
                agent.connections = [conn for conn in agent.connections 
                                   if conn.target_id != agent_id]
            
            # Remove the agent
            del self.agents[agent_id]
            self.logger.info(f"Removed agent {agent_id}")
    
    def connect_agents(self, source_id: str, target_id: str, weight: float, 
                      connection_type: str = "excitatory"):
        """Connect two agents in the network"""
        if source_id in self.agents and target_id in self.agents:
            source_agent = self.agents[source_id]
            target_agent = self.agents[target_id]
            source_agent.add_connection(target_agent, weight, connection_type)
            self.logger.info(f"Connected {source_id} to {target_id} with weight {weight}")
        else:
            self.logger.error(f"Cannot connect: agent {source_id} or {target_id} not found")
    
    def broadcast_signal(self, signal: Signal):
        """Broadcast a signal to all agents"""
        self.logger.info(f"Broadcasting signal {signal.id} to all agents")
        
        for agent in self.agents.values():
            # Create a copy of the signal for each agent
            agent_signal = Signal(
                id=f"{signal.id}_{agent.id}",
                type=signal.type,
                content=signal.content,
                source_agent_id=signal.source_agent_id,
                target_agent_id=agent.id,
                strength=signal.strength,
                timestamp=signal.timestamp,
                metadata=signal.metadata.copy(),
                propagation_path=signal.propagation_path.copy()
            )
            
            agent.receive_signal(agent_signal)
        
        self.global_signal_history.append(signal)
        self._update_network_metrics(signal)
    
    def send_signal_to_agent(self, signal: Signal, target_agent_id: str):
        """Send a signal to a specific agent"""
        if target_agent_id in self.agents:
            target_agent = self.agents[target_agent_id]
            signal.target_agent_id = target_agent_id
            target_agent.receive_signal(signal)
            
            self.global_signal_history.append(signal)
            self._update_network_metrics(signal)
            
            self.logger.info(f"Sent signal {signal.id} to agent {target_agent_id}")
        else:
            self.logger.error(f"Target agent {target_agent_id} not found")
    
    def process_network_cycle(self):
        """Process one cycle of network activity"""
        self.logger.debug("Processing network cycle")
        
        # Collect all signals to be propagated
        all_propagated_signals = []
        
        # Process each agent
        for agent in self.agents.values():
            propagated_signals = agent.propagate_signals()
            all_propagated_signals.extend(propagated_signals)
        
        # Deliver propagated signals
        for signal in all_propagated_signals:
            if signal.target_agent_id in self.agents:
                target_agent = self.agents[signal.target_agent_id]
                target_agent.receive_signal(signal)
            else:
                self.logger.warning(f"Signal target {signal.target_agent_id} not found")
        
        # Update network metrics
        self._update_network_activity()
    
    def _update_network_metrics(self, signal: Signal):
        """Update network metrics based on signal activity"""
        self.network_metrics["total_signals"] += 1
        
        # Update average signal strength
        current_avg = self.network_metrics["average_signal_strength"]
        total_signals = self.network_metrics["total_signals"]
        self.network_metrics["average_signal_strength"] = (
            (current_avg * (total_signals - 1) + signal.strength) / total_signals
        )
        
        # Track agent activity
        if signal.target_agent_id:
            self.network_metrics["agent_activity"][signal.target_agent_id] += 1
    
    def _update_network_activity(self):
        """Update overall network activity metrics"""
        total_activity = sum(agent.activation_count for agent in self.agents.values())
        self.network_metrics["network_activity"] = total_activity
    
    def get_network_status(self) -> Dict[str, Any]:
        """Get overall network status"""
        agent_statuses = {agent_id: agent.get_status() 
                         for agent_id, agent in self.agents.items()}
        
        return {
            "agent_count": len(self.agents),
            "agent_statuses": agent_statuses,
            "network_metrics": self.network_metrics,
            "total_signals_processed": len(self.global_signal_history)
        }
    
    def optimize_connections(self):
        """Optimize network connections based on activity patterns"""
        self.logger.info("Optimizing network connections")
        
        # Analyze connection usage patterns
        connection_usage = defaultdict(int)
        
        for agent in self.agents.values():
            for connection in agent.connections:
                connection_key = f"{connection.source_id}->{connection.target_id}"
                connection_usage[connection_key] += 1
        
        # Adjust weights based on usage
        for agent in self.agents.values():
            for connection in agent.connections:
                connection_key = f"{connection.source_id}->{connection.target_id}"
                usage = connection_usage.get(connection_key, 0)
                
                # Increase weight for frequently used connections
                if usage > 10:
                    connection.weight = min(connection.weight * 1.1, 1.0)
                # Decrease weight for rarely used connections
                elif usage < 2:
                    connection.weight = max(connection.weight * 0.9, 0.1)
        
        self.logger.info("Connection optimization completed")

class SignalGenerator:
    """Generates various types of signals for testing and operation"""
    
    @staticmethod
    def create_activation_signal(content: Any, source_id: str = "system", 
                               strength: float = 1.0) -> Signal:
        """Create an activation signal"""
        return Signal(
            id=f"activation_{int(time.time())}",
            type=SignalType.ACTIVATION,
            content=content,
            source_agent_id=source_id,
            strength=strength
        )
    
    @staticmethod
    def create_data_signal(data: Any, source_id: str = "system", 
                          strength: float = 1.0) -> Signal:
        """Create a data signal"""
        return Signal(
            id=f"data_{int(time.time())}",
            type=SignalType.DATA,
            content=data,
            source_agent_id=source_id,
            strength=strength
        )
    
    @staticmethod
    def create_control_signal(command: str, source_id: str = "system", 
                             strength: float = 1.0) -> Signal:
        """Create a control signal"""
        return Signal(
            id=f"control_{int(time.time())}",
            type=SignalType.CONTROL,
            content=command,
            source_agent_id=source_id,
            strength=strength
        )
    
    @staticmethod
    def create_feedback_signal(feedback: Any, source_id: str = "system", 
                              strength: float = 1.0) -> Signal:
        """Create a feedback signal"""
        return Signal(
            id=f"feedback_{int(time.time())}",
            type=SignalType.FEEDBACK,
            content=feedback,
            source_agent_id=source_id,
            strength=strength
        )

# Factory function to create specialized agents
def create_specialized_agent(agent_id: str, specialization: str, 
                           threshold: float = 0.5) -> Agent:
    """Create a specialized agent with predefined capabilities"""
    
    # Set specialization-specific thresholds
    specialization_thresholds = {
        "PythonSyntaxFixer": 0.6,
        "SQLQueryCrafter": 0.7,
        "CreativeWriter": 0.4,
        "DataAnalyzer": 0.8,
        "GeneralProcessor": 0.5
    }
    
    threshold = specialization_thresholds.get(specialization, threshold)
    
    agent = Agent(agent_id, specialization, threshold)
    
    # Set up HRM system based on specialization
    if specialization == "PythonSyntaxFixer":
        agent.hrm_system.visionary.set_core_mission("Ensure Python code syntax correctness")
        agent.hrm_system.visionary.add_ethical_guideline("Preserve code functionality")
        agent.hrm_system.visionary.add_fundamental_objective("Fix syntax errors")
        
    elif specialization == "SQLQueryCrafter":
        agent.hrm_system.visionary.set_core_mission("Generate efficient SQL queries")
        agent.hrm_system.visionary.add_ethical_guideline("Ensure data security")
        agent.hrm_system.visionary.add_fundamental_objective("Optimize query performance")
        
    elif specialization == "CreativeWriter":
        agent.hrm_system.visionary.set_core_mission("Generate creative and engaging content")
        agent.hrm_system.visionary.add_ethical_guideline("Create original content")
        agent.hrm_system.visionary.add_fundamental_objective("Adapt to different styles")
        
    elif specialization == "DataAnalyzer":
        agent.hrm_system.visionary.set_core_mission("Extract meaningful insights from data")
        agent.hrm_system.visionary.add_ethical_guideline("Ensure data privacy")
        agent.hrm_system.visionary.add_fundamental_objective("Provide accurate analysis")
    
    return agent