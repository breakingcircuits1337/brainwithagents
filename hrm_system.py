"""
Four-Level Hierarchical Reasoning Model (HRM) Implementation

This module implements the four-level HRM architecture as described in the document:
- Level 1: Visionary - Sets long-term objectives and ethical guidelines
- Level 2: Architect - Creates strategic plans based on Visionary's goals
- Level 3: Foreman - Breaks down plans into tactical tasks and coordinates execution
- Level 4: Technician - Executes specific actions and interacts with the environment
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TaskPriority(Enum):
    """Priority levels for tasks"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class Task:
    """Represents a task in the HRM system"""
    id: str
    description: str
    priority: TaskPriority
    parameters: Dict[str, Any]
    deadline: Optional[float] = None
    dependencies: List[str] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []

@dataclass
class Goal:
    """Represents a goal in the HRM system"""
    id: str
    description: str
    priority: TaskPriority
    constraints: List[str]
    success_criteria: List[str]

@dataclass
class Plan:
    """Represents a strategic plan"""
    id: str
    description: str
    goals: List[Goal]
    phases: List[Dict[str, Any]]
    resource_requirements: Dict[str, Any]
    contingency_plans: List[Dict[str, Any]]

@dataclass
class ExecutionResult:
    """Represents the result of task execution"""
    task_id: str
    success: bool
    result: Any
    execution_time: float
    error_message: Optional[str] = None

class HRMLevel(ABC):
    """Base class for all HRM levels"""
    
    def __init__(self, name: str, abstraction_level: str, time_horizon: str):
        self.name = name
        self.abstraction_level = abstraction_level
        self.time_horizon = time_horizon
        self.logger = logging.getLogger(f"{__name__}.{name}")
        
    @abstractmethod
    def process(self, input_data: Any) -> Any:
        """Process input data and return output"""
        pass
    
    def log_decision(self, decision: str, details: Dict[str, Any] = None):
        """Log a decision made at this level"""
        self.logger.info(f"Decision at {self.name} level: {decision}")
        if details:
            self.logger.debug(f"Details: {details}")

class Visionary(HRMLevel):
    """Level 1: Sets long-term objectives and ethical guidelines"""
    
    def __init__(self):
        super().__init__("Visionary", "Extremely High", "Long-term (years/lifespan)")
        self.ethical_guidelines = []
        self.core_mission = ""
        self.fundamental_objectives = []
        
    def set_core_mission(self, mission: str):
        """Set the core mission of the AI agent"""
        self.core_mission = mission
        self.log_decision("Core mission established", {"mission": mission})
        
    def add_ethical_guideline(self, guideline: str):
        """Add an ethical guideline"""
        self.ethical_guidelines.append(guideline)
        self.log_decision("Ethical guideline added", {"guideline": guideline})
        
    def add_fundamental_objective(self, objective: str):
        """Add a fundamental objective"""
        self.fundamental_objectives.append(objective)
        self.log_decision("Fundamental objective added", {"objective": objective})
        
    def process(self, input_data: Dict[str, Any]) -> Goal:
        """Process high-level directives and create overarching goals"""
        self.logger.info("Visionary processing high-level directives")
        
        # Extract directives from input
        directives = input_data.get("directives", [])
        context = input_data.get("context", {})
        
        # Create a goal based on the core mission and directives
        goal_id = f"goal_{int(time.time())}"
        goal_description = self._synthesize_goal(directives, context)
        
        goal = Goal(
            id=goal_id,
            description=goal_description,
            priority=TaskPriority.HIGH,
            constraints=self.ethical_guidelines.copy(),
            success_criteria=self._generate_success_criteria(goal_description)
        )
        
        self.log_decision("Goal created", {
            "goal_id": goal_id,
            "description": goal_description,
            "priority": goal.priority.name
        })
        
        return goal
    
    def _synthesize_goal(self, directives: List[str], context: Dict[str, Any]) -> str:
        """Synthesize a goal description from directives and context"""
        if not self.core_mission:
            return "Establish core mission first"
            
        # Combine core mission with directives
        goal_parts = [self.core_mission]
        goal_parts.extend(directives)
        
        # Add context-aware elements
        if "environment" in context:
            goal_parts.append(f"Operating in {context['environment']}")
            
        return " | ".join(goal_parts)
    
    def _generate_success_criteria(self, goal_description: str) -> List[str]:
        """Generate success criteria for a goal"""
        criteria = [
            "All ethical guidelines must be followed",
            "Core mission must be advanced",
            "Actions must align with fundamental objectives"
        ]
        
        # Add specific criteria based on goal content
        if "safety" in goal_description.lower():
            criteria.append("Safety must be maintained at all times")
        if "efficiency" in goal_description.lower():
            criteria.append("Resources must be used efficiently")
            
        return criteria

class Architect(HRMLevel):
    """Level 2: Creates strategic plans based on Visionary's goals"""
    
    def __init__(self):
        super().__init__("Architect", "High", "Medium-to-long-term (hours/days/weeks)")
        self.current_plans = {}
        self.resource_availability = {}
        
    def set_resource_availability(self, resources: Dict[str, Any]):
        """Set available resources for planning"""
        self.resource_availability = resources
        self.log_decision("Resource availability updated", {"resources": resources})
        
    def process(self, input_data: Dict[str, Any]) -> Plan:
        """Process goals and create strategic plans"""
        self.logger.info("Architect processing goals and creating strategic plans")
        
        goal = input_data.get("goal")
        if not goal:
            raise ValueError("Architect requires a goal to create a plan")
            
        environmental_data = input_data.get("environmental_data", {})
        
        # Create a strategic plan
        plan_id = f"plan_{int(time.time())}"
        plan_description = f"Strategic plan for: {goal.description}"
        
        # Break down into phases
        phases = self._create_phases(goal, environmental_data)
        
        # Calculate resource requirements
        resource_requirements = self._calculate_resource_requirements(phases)
        
        # Create contingency plans
        contingency_plans = self._create_contingency_plans(phases)
        
        plan = Plan(
            id=plan_id,
            description=plan_description,
            goals=[goal],
            phases=phases,
            resource_requirements=resource_requirements,
            contingency_plans=contingency_plans
        )
        
        self.current_plans[plan_id] = plan
        
        self.log_decision("Strategic plan created", {
            "plan_id": plan_id,
            "phases": len(phases),
            "resource_requirements": resource_requirements
        })
        
        return plan
    
    def _create_phases(self, goal: Goal, environmental_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create phases for the strategic plan"""
        phases = []
        
        # Basic phase structure
        phase_templates = [
            {"name": "Analysis", "duration": "short", "focus": "understanding requirements"},
            {"name": "Preparation", "duration": "medium", "focus": "resource allocation"},
            {"name": "Execution", "duration": "long", "focus": "main task completion"},
            {"name": "Review", "duration": "short", "focus": "evaluation and refinement"}
        ]
        
        for i, template in enumerate(phase_templates):
            phase = {
                "id": f"phase_{i+1}",
                "name": template["name"],
                "duration": template["duration"],
                "focus": template["focus"],
                "tasks": self._generate_phase_tasks(template["name"], goal),
                "dependencies": [f"phase_{i}"] if i > 0 else []
            }
            phases.append(phase)
            
        return phases
    
    def _generate_phase_tasks(self, phase_name: str, goal: Goal) -> List[str]:
        """Generate tasks for a specific phase"""
        task_templates = {
            "Analysis": [
                "Gather requirements",
                "Assess environmental conditions",
                "Identify constraints"
            ],
            "Preparation": [
                "Allocate resources",
                "Prepare tools and systems",
                "Establish monitoring protocols"
            ],
            "Execution": [
                "Execute main tasks",
                "Monitor progress",
                "Handle deviations"
            ],
            "Review": [
                "Evaluate outcomes",
                "Document lessons learned",
                "Prepare reports"
            ]
        }
        
        return task_templates.get(phase_name, ["Generic task"])
    
    def _calculate_resource_requirements(self, phases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate resource requirements for the plan"""
        requirements = {
            "computational": "medium",
            "memory": "medium",
            "time": "medium",
            "specialized_tools": []
        }
        
        # Adjust based on phases
        for phase in phases:
            if phase["name"] == "Execution":
                requirements["computational"] = "high"
                requirements["memory"] = "high"
                
        return requirements
    
    def _create_contingency_plans(self, phases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create contingency plans for potential issues"""
        return [
            {
                "trigger": "Resource shortage",
                "action": "Reallocate resources or request additional resources",
                "priority": "high"
            },
            {
                "trigger": "Environmental disruption",
                "action": "Pause execution and reassess plan",
                "priority": "high"
            },
            {
                "trigger": "Task failure",
                "action": "Retry with alternative approach",
                "priority": "medium"
            }
        ]

class Foreman(HRMLevel):
    """Level 3: Breaks down plans into tactical tasks and coordinates execution"""
    
    def __init__(self):
        super().__init__("Foreman", "Medium", "Medium-to-short-term (minutes/hours)")
        self.current_tasks = {}
        self.task_queue = []
        self.execution_status = {}
        
    def process(self, input_data: Dict[str, Any]) -> List[Task]:
        """Process plans and break them down into executable tasks"""
        self.logger.info("Foreman processing plans and creating tasks")
        
        plan = input_data.get("plan")
        if not plan:
            raise ValueError("Foreman requires a plan to create tasks")
            
        current_phase = input_data.get("current_phase", 0)
        real_time_data = input_data.get("real_time_data", {})
        
        # Get the current phase from the plan
        if current_phase < len(plan.phases):
            phase = plan.phases[current_phase]
        else:
            raise ValueError("Invalid phase specified")
            
        # Create executable tasks for the current phase
        tasks = self._create_executable_tasks(phase, real_time_data)
        
        # Add tasks to queue
        for task in tasks:
            self.task_queue.append(task)
            self.current_tasks[task.id] = task
            
        self.log_decision("Tasks created for execution", {
            "phase": phase["name"],
            "task_count": len(tasks),
            "queue_size": len(self.task_queue)
        })
        
        return tasks
    
    def _create_executable_tasks(self, phase: Dict[str, Any], real_time_data: Dict[str, Any]) -> List[Task]:
        """Create executable tasks from phase description"""
        tasks = []
        
        # Convert phase tasks to executable tasks
        for i, task_description in enumerate(phase.get("tasks", [])):
            task_id = f"task_{phase['id']}_{i+1}"
            
            task = Task(
                id=task_id,
                description=task_description,
                priority=self._determine_task_priority(task_description, real_time_data),
                parameters=self._generate_task_parameters(task_description, real_time_data),
                deadline=time.time() + self._estimate_task_duration(task_description),
                dependencies=self._get_task_dependencies(phase, i)
            )
            
            tasks.append(task)
            
        return tasks
    
    def _determine_task_priority(self, task_description: str, real_time_data: Dict[str, Any]) -> TaskPriority:
        """Determine priority of a task based on description and real-time data"""
        # High priority for safety-critical or urgent tasks
        if any(keyword in task_description.lower() for keyword in ["safety", "emergency", "critical"]):
            return TaskPriority.CRITICAL
            
        # Medium priority for important but non-urgent tasks
        if any(keyword in task_description.lower() for keyword in ["monitor", "check", "assess"]):
            return TaskPriority.HIGH
            
        # Low priority for routine tasks
        return TaskPriority.MEDIUM
    
    def _generate_task_parameters(self, task_description: str, real_time_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate parameters for a task"""
        parameters = {
            "description": task_description,
            "real_time_context": real_time_data,
            "retry_count": 0,
            "max_retries": 3
        }
        
        # Add specific parameters based on task type
        if "monitor" in task_description.lower():
            parameters["monitoring_interval"] = 5.0  # seconds
        elif "execute" in task_description.lower():
            parameters["execution_timeout"] = 30.0  # seconds
            
        return parameters
    
    def _estimate_task_duration(self, task_description: str) -> float:
        """Estimate task duration in seconds"""
        # Simple estimation based on task type
        if "monitor" in task_description.lower():
            return 300.0  # 5 minutes
        elif "execute" in task_description.lower():
            return 600.0  # 10 minutes
        elif "review" in task_description.lower():
            return 900.0  # 15 minutes
        else:
            return 450.0  # 7.5 minutes default
    
    def _get_task_dependencies(self, phase: Dict[str, Any], task_index: int) -> List[str]:
        """Get dependencies for a task"""
        dependencies = []
        
        # Add phase dependencies
        dependencies.extend(phase.get("dependencies", []))
        
        # Add task dependencies (tasks must be executed in order)
        if task_index > 0:
            dependencies.append(f"task_{phase['id']}_{task_index}")
            
        return dependencies
    
    def monitor_execution(self, task_id: str) -> Dict[str, Any]:
        """Monitor the execution of a specific task"""
        status = self.execution_status.get(task_id, {"status": "unknown"})
        self.log_decision("Task monitoring", {"task_id": task_id, "status": status})
        return status
    
    def handle_deviation(self, task_id: str, deviation: Dict[str, Any]) -> Dict[str, Any]:
        """Handle deviations in task execution"""
        self.logger.warning(f"Deviation detected in task {task_id}: {deviation}")
        
        # Simple deviation handling strategy
        response = {
            "action": "retry",
            "reason": "Standard deviation handling",
            "new_parameters": {}
        }
        
        # Adjust response based on deviation type
        if deviation.get("type") == "resource_shortage":
            response["action"] = "reallocate_resources"
        elif deviation.get("type") == "environmental_disruption":
            response["action"] = "pause_and_reassess"
            
        self.log_decision("Deviation handled", {
            "task_id": task_id,
            "deviation": deviation,
            "response": response
        })
        
        return response

class Technician(HRMLevel):
    """Level 4: Executes specific actions and interacts with the environment"""
    
    def __init__(self):
        super().__init__("Technician", "Very Low", "Short-term (milliseconds/seconds)")
        self.execution_history = []
        self.current_operations = {}
        
    def process(self, input_data: Dict[str, Any]) -> ExecutionResult:
        """Process tasks and execute specific actions"""
        self.logger.info("Technician processing tasks for execution")
        
        task = input_data.get("task")
        if not task:
            raise ValueError("Technician requires a task to execute")
            
        environmental_input = input_data.get("environmental_input", {})
        
        # Execute the task
        start_time = time.time()
        
        try:
            result = self._execute_task(task, environmental_input)
            success = True
            error_message = None
        except Exception as e:
            result = None
            success = False
            error_message = str(e)
            
        execution_time = time.time() - start_time
        
        execution_result = ExecutionResult(
            task_id=task.id,
            success=success,
            result=result,
            execution_time=execution_time,
            error_message=error_message
        )
        
        # Record execution
        self.execution_history.append(execution_result)
        
        self.log_decision("Task execution completed", {
            "task_id": task.id,
            "success": success,
            "execution_time": execution_time
        })
        
        return execution_result
    
    def _execute_task(self, task: Task, environmental_input: Dict[str, Any]) -> Any:
        """Execute a specific task"""
        self.logger.debug(f"Executing task: {task.description}")
        
        # Simulate task execution based on task description
        if "monitor" in task.description.lower():
            return self._execute_monitoring_task(task, environmental_input)
        elif "execute" in task.description.lower():
            return self._execute_main_task(task, environmental_input)
        elif "review" in task.description.lower():
            return self._execute_review_task(task, environmental_input)
        else:
            return self._execute_generic_task(task, environmental_input)
    
    def _execute_monitoring_task(self, task: Task, environmental_input: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a monitoring task"""
        # Simulate monitoring by analyzing environmental input
        monitoring_result = {
            "status": "normal",
            "metrics": {
                "cpu_usage": environmental_input.get("cpu_usage", 50.0),
                "memory_usage": environmental_input.get("memory_usage", 60.0),
                "network_activity": environmental_input.get("network_activity", 30.0)
            },
            "anomalies": [],
            "timestamp": time.time()
        }
        
        # Check for anomalies
        for metric, value in monitoring_result["metrics"].items():
            if value > 80.0:
                monitoring_result["anomalies"].append({
                    "metric": metric,
                    "value": value,
                    "severity": "high"
                })
                
        return monitoring_result
    
    def _execute_main_task(self, task: Task, environmental_input: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a main task"""
        # Simulate main task execution
        result = {
            "status": "completed",
            "output": f"Executed: {task.description}",
            "resources_used": {
                "cpu_time": 2.5,
                "memory_mb": 128,
                "network_io": 1024
            },
            "timestamp": time.time()
        }
        
        # Simulate some processing time
        time.sleep(0.1)
        
        return result
    
    def _execute_review_task(self, task: Task, environmental_input: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a review task"""
        # Simulate review by analyzing previous results
        result = {
            "status": "reviewed",
            "findings": [
                "Task execution was successful",
                "Resource usage was within expected limits",
                "No critical issues detected"
            ],
            "recommendations": [
                "Continue monitoring system performance",
                "Consider optimizing resource allocation"
            ],
            "timestamp": time.time()
        }
        
        return result
    
    def _execute_generic_task(self, task: Task, environmental_input: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a generic task"""
        return {
            "status": "completed",
            "output": f"Generic task executed: {task.description}",
            "timestamp": time.time()
        }
    
    def get_sensor_data(self) -> Dict[str, Any]:
        """Get current sensor/environmental data"""
        # Simulate sensor data collection
        return {
            "timestamp": time.time(),
            "sensors": {
                "temperature": 22.5,
                "humidity": 45.0,
                "pressure": 1013.25,
                "light_level": 750
            },
            "system_metrics": {
                "cpu_usage": 45.2,
                "memory_usage": 62.8,
                "disk_usage": 78.3,
                "network_activity": 25.6
            }
        }
    
    def send_control_signal(self, target: str, signal: Dict[str, Any]) -> bool:
        """Send control signal to external systems"""
        self.logger.debug(f"Sending control signal to {target}: {signal}")
        
        # Simulate signal transmission
        time.sleep(0.01)
        
        # Return success status
        return True

class HRMSystem:
    """Complete Four-Level HRM System"""
    
    def __init__(self):
        self.visionary = Visionary()
        self.architect = Architect()
        self.foreman = Foreman()
        self.technician = Technician()
        
        self.logger = logging.getLogger(__name__)
        
    def process_directive(self, directive: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process a high-level directive through all HRM levels"""
        if context is None:
            context = {}
            
        self.logger.info(f"Processing directive: {directive}")
        
        try:
            # Level 1: Visionary creates goals
            visionary_input = {
                "directives": [directive],
                "context": context
            }
            goal = self.visionary.process(visionary_input)
            
            # Level 2: Architect creates strategic plan
            architect_input = {
                "goal": goal,
                "environmental_data": context.get("environment", {})
            }
            plan = self.architect.process(architect_input)
            
            # Level 3: Foreman creates tasks (for first phase)
            foreman_input = {
                "plan": plan,
                "current_phase": 0,
                "real_time_data": {}
            }
            tasks = self.foreman.process(foreman_input)
            
            # Level 4: Technician executes tasks
            results = []
            for task in tasks:
                technician_input = {
                    "task": task,
                    "environmental_input": self.technician.get_sensor_data()
                }
                result = self.technician.process(technician_input)
                results.append(result)
                
            return {
                "success": True,
                "goal": goal,
                "plan": plan,
                "tasks": tasks,
                "results": results,
                "summary": self._generate_summary(results)
            }
            
        except Exception as e:
            self.logger.error(f"Error processing directive: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _generate_summary(self, results: List[ExecutionResult]) -> Dict[str, Any]:
        """Generate a summary of execution results"""
        total_tasks = len(results)
        successful_tasks = sum(1 for r in results if r.success)
        total_execution_time = sum(r.execution_time for r in results)
        
        return {
            "total_tasks": total_tasks,
            "successful_tasks": successful_tasks,
            "failed_tasks": total_tasks - successful_tasks,
            "success_rate": successful_tasks / total_tasks if total_tasks > 0 else 0,
            "total_execution_time": total_execution_time,
            "average_execution_time": total_execution_time / total_tasks if total_tasks > 0 else 0
        }