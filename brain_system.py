"""
Brain Architecture: Integration of HRM and Neuron-style Communication System

This module implements the main Brain class that combines the Four-Level Hierarchical
Reasoning Model (HRM) with the Neuron-style Weighted Propagation Communication System
to create a complete, working artificial intelligence brain architecture.
"""

import time
import logging
import threading
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import json

# Import the component systems
from hrm_system import HRMSystem, Task, TaskPriority, Goal, Plan, ExecutionResult
from communication_system import (
    CommunicationNetwork, Agent, Signal, SignalType, SignalGenerator,
    create_specialized_agent
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BrainState(Enum):
    """Brain operational states"""
    IDLE = "idle"
    PROCESSING = "processing"
    LEARNING = "learning"
    OPTIMIZING = "optimizing"
    ERROR = "error"

class BrainMode(Enum):
    """Brain operational modes"""
    REACTIVE = "reactive"      # Respond to immediate stimuli
    PROACTIVE = "proactive"    # Plan ahead and anticipate
    LEARNING = "learning"      # Focus on learning and adaptation
    CREATIVE = "creative"      # Focus on generation and innovation

@dataclass
class BrainMetrics:
    """Comprehensive metrics for brain performance"""
    processing_cycles: int = 0
    total_signals_processed: int = 0
    average_processing_time: float = 0.0
    success_rate: float = 1.0
    network_efficiency: float = 1.0
    learning_rate: float = 0.0
    adaptation_score: float = 0.0
    resource_usage: Dict[str, float] = field(default_factory=dict)
    agent_performance: Dict[str, Dict[str, float]] = field(default_factory=dict)
    system_health: float = 1.0
    
class Brain:
    """Main Brain class integrating HRM and Communication System"""
    
    def __init__(self, brain_id: str = "brain_001"):
        self.id = brain_id
        self.state = BrainState.IDLE
        self.mode = BrainMode.REACTIVE
        
        # Core systems
        self.communication_network = CommunicationNetwork()
        self.central_hrm = HRMSystem()  # Central HRM for high-level coordination
        
        # Brain configuration
        self.config = {
            "max_processing_cycles": 1000,
            "signal_decay_rate": 0.95,
            "learning_threshold": 0.7,
            "optimization_interval": 100,
            "max_agents": 50,
            "default_agent_threshold": 0.5
        }
        
        # Performance tracking
        self.metrics = BrainMetrics()
        self.processing_history = []
        self.learning_history = []
        self.optimization_history = []
        
        # Threading for continuous operation
        self.is_running = False
        self.processing_thread = None
        self.learning_thread = None
        
        # Event handlers
        self.event_handlers = {
            "signal_received": [],
            "task_completed": [],
            "error_occurred": [],
            "state_changed": []
        }
        
        self.logger = logging.getLogger(f"{__name__}.Brain.{brain_id}")
        
        # Initialize brain
        self._initialize_brain()
    
    def _initialize_brain(self):
        """Initialize the brain with basic structure"""
        self.logger.info(f"Initializing brain {self.id}")
        
        # Set up central HRM with core mission
        self.central_hrm.visionary.set_core_mission(
            "Integrate and coordinate multiple AI systems to achieve complex goals"
        )
        self.central_hrm.visionary.add_ethical_guideline("Ensure system safety and reliability")
        self.central_hrm.visionary.add_fundamental_objective("Maximize overall system efficiency")
        
        # Create essential agents
        self._create_essential_agents()
        
        # Set up initial connections
        self._setup_initial_connections()
        
        self.logger.info("Brain initialization completed")
    
    def _create_essential_agents(self):
        """Create essential agents for basic brain functionality"""
        essential_agents = [
            ("coordinator", "GeneralProcessor", 0.6),
            ("monitor", "DataAnalyzer", 0.4),
            ("executor", "GeneralProcessor", 0.7),
            ("learner", "DataAnalyzer", 0.8),
            ("optimizer", "DataAnalyzer", 0.9)
        ]
        
        for agent_id, specialization, threshold in essential_agents:
            agent = create_specialized_agent(agent_id, specialization, threshold)
            self.communication_network.add_agent(agent)
    
    def _setup_initial_connections(self):
        """Set up initial connections between agents"""
        # Basic network topology
        connections = [
            ("coordinator", "monitor", 0.9),
            ("coordinator", "executor", 0.8),
            ("monitor", "learner", 0.7),
            ("executor", "optimizer", 0.6),
            ("learner", "coordinator", 0.5),
            ("optimizer", "coordinator", 0.5)
        ]
        
        for source, target, weight in connections:
            self.communication_network.connect_agents(source, target, weight)
    
    def add_specialized_agent(self, agent_id: str, specialization: str, 
                            threshold: Optional[float] = None):
        """Add a specialized agent to the brain"""
        if len(self.communication_network.agents) >= self.config["max_agents"]:
            self.logger.warning("Maximum agent limit reached")
            return False
        
        if threshold is None:
            threshold = self.config["default_agent_threshold"]
        
        agent = create_specialized_agent(agent_id, specialization, threshold)
        self.communication_network.add_agent(agent)
        
        # Connect to coordinator
        if "coordinator" in self.communication_network.agents:
            self.communication_network.connect_agents("coordinator", agent_id, 0.8)
            self.communication_network.connect_agents(agent_id, "coordinator", 0.6)
        
        self.logger.info(f"Added specialized agent: {agent_id} ({specialization})")
        return True
    
    def process_input(self, input_data: Any, input_type: str = "general") -> Dict[str, Any]:
        """Process input through the brain system"""
        self.logger.info(f"Processing {input_type} input")
        
        start_time = time.time()
        self.state = BrainState.PROCESSING
        
        try:
            # Create input signal
            signal = SignalGenerator.create_data_signal(
                {
                    "content": input_data,
                    "type": input_type,
                    "timestamp": time.time()
                },
                source_id="external_input",
                strength=1.0
            )
            
            # Send to coordinator agent
            result = self._process_through_brain(signal)
            
            # Update metrics
            processing_time = time.time() - start_time
            self._update_processing_metrics(result, processing_time)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing input: {e}")
            self.state = BrainState.ERROR
            self._trigger_event("error_occurred", {"error": str(e)})
            return {"success": False, "error": str(e)}
    
    def _process_through_brain(self, signal: Signal) -> Dict[str, Any]:
        """Process a signal through the brain system"""
        self.logger.debug(f"Processing signal {signal.id} through brain")
        
        # Step 1: Central HRM analyzes the input
        central_result = self.central_hrm.process_directive(
            f"Process input: {signal.content}",
            {"signal_id": signal.id, "signal_type": signal.type.value}
        )
        
        # Step 2: Route to appropriate agents
        if "coordinator" in self.communication_network.agents:
            # Send to coordinator for distribution
            self.communication_network.send_signal_to_agent(signal, "coordinator")
            
            # Process network cycles
            for _ in range(3):  # Process multiple cycles for propagation
                self.communication_network.process_network_cycle()
                time.sleep(0.01)  # Small delay for realistic processing
        
        # Step 3: Collect results
        results = self._collect_processing_results(signal)
        
        return {
            "success": True,
            "signal_id": signal.id,
            "central_processing": central_result,
            "agent_results": results,
            "processing_summary": self._generate_processing_summary(results)
        }
    
    def _collect_processing_results(self, original_signal: Signal) -> List[Dict[str, Any]]:
        """Collect results from agent processing"""
        results = []
        
        for agent_id, agent in self.communication_network.agents.items():
            # Check if agent processed the signal
            for processing_event in agent.processing_history:
                if (processing_event["signal_id"] == original_signal.id or
                    original_signal.id in processing_event["signal_id"]):
                    
                    results.append({
                        "agent_id": agent_id,
                        "specialization": agent.specialization,
                        "result": processing_event["result"],
                        "processing_time": processing_event["timestamp"] - original_signal.timestamp,
                        "performance": agent.performance_metrics
                    })
        
        return results
    
    def _generate_processing_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a summary of processing results"""
        if not results:
            return {"status": "no_results", "agents_processed": 0}
        
        total_agents = len(results)
        successful_agents = sum(1 for r in results if r["result"].get("success", False))
        average_time = sum(r["processing_time"] for r in results) / total_agents
        
        return {
            "status": "completed",
            "agents_processed": total_agents,
            "successful_agents": successful_agents,
            "success_rate": successful_agents / total_agents,
            "average_processing_time": average_time,
            "specializations_involved": list(set(r["specialization"] for r in results))
        }
    
    def start_continuous_operation(self):
        """Start continuous brain operation"""
        if self.is_running:
            self.logger.warning("Brain is already running")
            return
        
        self.logger.info("Starting continuous brain operation")
        self.is_running = True
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        # Start learning thread
        self.learning_thread = threading.Thread(target=self._learning_loop)
        self.learning_thread.daemon = True
        self.learning_thread.start()
    
    def stop_continuous_operation(self):
        """Stop continuous brain operation"""
        self.logger.info("Stopping continuous brain operation")
        self.is_running = False
        
        if self.processing_thread:
            self.processing_thread.join(timeout=5.0)
        
        if self.learning_thread:
            self.learning_thread.join(timeout=5.0)
        
        self.state = BrainState.IDLE
    
    def _processing_loop(self):
        """Main processing loop for continuous operation"""
        cycle_count = 0
        
        while self.is_running:
            try:
                cycle_count += 1
                self.logger.debug(f"Processing cycle {cycle_count}")
                
                # Process network cycle
                self.communication_network.process_network_cycle()
                
                # Update metrics
                self.metrics.processing_cycles += 1
                
                # Periodic optimization
                if cycle_count % self.config["optimization_interval"] == 0:
                    self._optimize_brain()
                
                # Small delay to prevent excessive CPU usage
                time.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Error in processing loop: {e}")
                self.state = BrainState.ERROR
                self._trigger_event("error_occurred", {"error": str(e)})
    
    def _learning_loop(self):
        """Learning loop for continuous adaptation"""
        while self.is_running:
            try:
                if self.mode == BrainMode.LEARNING:
                    self._perform_learning()
                
                # Learning happens less frequently
                time.sleep(1.0)
                
            except Exception as e:
                self.logger.error(f"Error in learning loop: {e}")
    
    def _perform_learning(self):
        """Perform learning and adaptation"""
        self.logger.debug("Performing learning cycle")
        
        # Analyze processing history
        if len(self.processing_history) > 10:
            recent_performance = self._analyze_recent_performance()
            
            # Adapt agent thresholds based on performance
            self._adapt_agent_thresholds(recent_performance)
            
            # Record learning
            learning_event = {
                "timestamp": time.time(),
                "performance_analysis": recent_performance,
                "adaptations_made": "threshold_adjustments"
            }
            self.learning_history.append(learning_event)
    
    def _analyze_recent_performance(self) -> Dict[str, Any]:
        """Analyze recent performance metrics"""
        recent_events = self.processing_history[-10:] if self.processing_history else []
        
        if not recent_events:
            return {"status": "insufficient_data"}
        
        # Calculate performance metrics
        success_rate = sum(1 for e in recent_events if e.get("success", False)) / len(recent_events)
        avg_processing_time = sum(e.get("processing_time", 0) for e in recent_events) / len(recent_events)
        
        return {
            "success_rate": success_rate,
            "average_processing_time": avg_processing_time,
            "total_events": len(recent_events),
            "performance_trend": "improving" if success_rate > 0.8 else "needs_attention"
        }
    
    def _adapt_agent_thresholds(self, performance_analysis: Dict[str, Any]):
        """Adapt agent thresholds based on performance"""
        if performance_analysis["success_rate"] < 0.7:
            # Lower thresholds to increase sensitivity
            for agent in self.communication_network.agents.values():
                agent.threshold = max(agent.threshold * 0.95, 0.1)
        elif performance_analysis["success_rate"] > 0.9:
            # Raise thresholds to reduce false positives
            for agent in self.communication_network.agents.values():
                agent.threshold = min(agent.threshold * 1.05, 0.9)
    
    def _optimize_brain(self):
        """Optimize brain performance"""
        self.logger.info("Performing brain optimization")
        
        # Optimize communication network
        self.communication_network.optimize_connections()
        
        # Update metrics
        optimization_event = {
            "timestamp": time.time(),
            "network_optimization": True,
            "performance_metrics": self.get_comprehensive_metrics()
        }
        self.optimization_history.append(optimization_event)
        
        self.logger.info("Brain optimization completed")
    
    def _update_processing_metrics(self, result: Dict[str, Any], processing_time: float):
        """Update processing metrics"""
        success = result.get("success", False)
        
        # Update brain metrics
        self.metrics.total_signals_processed += 1
        
        # Update average processing time
        current_avg = self.metrics.average_processing_time
        total_signals = self.metrics.total_signals_processed
        self.metrics.average_processing_time = (
            (current_avg * (total_signals - 1) + processing_time) / total_signals
        )
        
        # Update success rate
        if success:
            current_rate = self.metrics.success_rate
            self.metrics.success_rate = (
                (current_rate * (total_signals - 1) + 1.0) / total_signals
            )
        else:
            current_rate = self.metrics.success_rate
            self.metrics.success_rate = (
                (current_rate * (total_signals - 1)) / total_signals
            )
        
        # Record processing event
        processing_event = {
            "timestamp": time.time(),
            "result": result,
            "processing_time": processing_time,
            "success": success
        }
        self.processing_history.append(processing_event)
        
        # Trigger event
        self._trigger_event("task_completed", processing_event)
    
    def set_mode(self, mode: BrainMode):
        """Set the brain's operational mode"""
        old_mode = self.mode
        self.mode = mode
        
        self.logger.info(f"Brain mode changed from {old_mode.value} to {mode.value}")
        self._trigger_event("state_changed", {
            "old_mode": old_mode.value,
            "new_mode": mode.value,
            "timestamp": time.time()
        })
    
    def add_event_handler(self, event_type: str, handler: Callable):
        """Add an event handler"""
        if event_type in self.event_handlers:
            self.event_handlers[event_type].append(handler)
        else:
            self.logger.warning(f"Unknown event type: {event_type}")
    
    def _trigger_event(self, event_type: str, data: Dict[str, Any]):
        """Trigger an event"""
        if event_type in self.event_handlers:
            for handler in self.event_handlers[event_type]:
                try:
                    handler(data)
                except Exception as e:
                    self.logger.error(f"Error in event handler: {e}")
    
    def get_comprehensive_metrics(self) -> Dict[str, Any]:
        """Get comprehensive brain metrics"""
        network_status = self.communication_network.get_network_status()
        
        return {
            "brain_id": self.id,
            "state": self.state.value,
            "mode": self.mode.value,
            "metrics": {
                "processing_cycles": self.metrics.processing_cycles,
                "total_signals_processed": self.metrics.total_signals_processed,
                "average_processing_time": self.metrics.average_processing_time,
                "success_rate": self.metrics.success_rate,
                "network_efficiency": self.metrics.network_efficiency,
                "learning_rate": self.metrics.learning_rate,
                "adaptation_score": self.metrics.adaptation_score,
                "system_health": self.metrics.system_health
            },
            "network_status": network_status,
            "agent_count": len(self.communication_network.agents),
            "processing_history_size": len(self.processing_history),
            "learning_history_size": len(self.learning_history),
            "optimization_history_size": len(self.optimization_history)
        }
    
    def get_brain_status(self) -> Dict[str, Any]:
        """Get current brain status"""
        return {
            "id": self.id,
            "state": self.state.value,
            "mode": self.mode.value,
            "is_running": self.is_running,
            "agent_count": len(self.communication_network.agents),
            "processing_cycles": self.metrics.processing_cycles,
            "total_signals_processed": self.metrics.total_signals_processed,
            "success_rate": self.metrics.success_rate,
            "system_health": self.metrics.system_health
        }
    
    def save_brain_state(self, filename: str):
        """Save brain state to file"""
        state = {
            "id": self.id,
            "config": self.config,
            "metrics": self.metrics.__dict__,
            "processing_history": self.processing_history[-100:],  # Save last 100 events
            "learning_history": self.learning_history[-50:] if self.learning_history else [],  # Save last 50 events
            "network_state": self.communication_network.get_network_status()
        }
        
        with open(filename, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        self.logger.info(f"Brain state saved to {filename}")
    
    def load_brain_state(self, filename: str):
        """Load brain state from file"""
        try:
            with open(filename, 'r') as f:
                state = json.load(f)
            
            self.id = state["id"]
            self.config = state["config"]
            
            # Restore metrics
            for key, value in state["metrics"].items():
                if hasattr(self.metrics, key):
                    setattr(self.metrics, key, value)
            
            # Restore histories
            self.processing_history = state.get("processing_history", [])
            self.learning_history = state.get("learning_history", [])
            
            self.logger.info(f"Brain state loaded from {filename}")
            
        except Exception as e:
            self.logger.error(f"Error loading brain state: {e}")
    
    def shutdown(self):
        """Shutdown the brain system"""
        self.logger.info("Shutting down brain system")
        
        # Stop continuous operation
        self.stop_continuous_operation()
        
        # Save final state
        self.save_brain_state(f"{self.id}_final_state.json")
        
        self.logger.info("Brain system shutdown completed")

# Factory function to create a pre-configured brain
def create_enhanced_brain(brain_id: str = "enhanced_brain") -> Brain:
    """Create an enhanced brain with additional specialized agents"""
    brain = Brain(brain_id)
    
    # Add specialized agents
    specialized_agents = [
        ("python_fixer", "PythonSyntaxFixer", 0.6),
        ("sql_crafter", "SQLQueryCrafter", 0.7),
        ("creative_writer", "CreativeWriter", 0.4),
        ("data_analyzer", "DataAnalyzer", 0.8)
    ]
    
    for agent_id, specialization, threshold in specialized_agents:
        brain.add_specialized_agent(agent_id, specialization, threshold)
    
    # Set up additional connections
    additional_connections = [
        ("python_fixer", "executor", 0.8),
        ("sql_crafter", "executor", 0.8),
        ("creative_writer", "executor", 0.7),
        ("data_analyzer", "monitor", 0.9),
        ("data_analyzer", "learner", 0.8)
    ]
    
    for source, target, weight in additional_connections:
        brain.communication_network.connect_agents(source, target, weight)
    
    return brain