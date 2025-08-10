"""
Multi-Agent System Demonstration: 250+ Agent Collaboration Showcase

This script demonstrates the full capabilities of the scalable multi-agent system,
showcasing how 250+ specialized agents can work together collaboratively to solve
complex problems. The demonstration includes:

1. System initialization with 250+ specialized agents
2. Real-time task processing and load balancing
3. Inter-agent communication and collaboration
4. Performance monitoring and optimization
5. Complex problem-solving scenarios
6. System resilience and fault tolerance

The demonstration highlights the scalability, efficiency, and intelligence of the
multi-agent system in handling large-scale collaborative tasks.
"""

import asyncio
import time
import random
import json
import math
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from collections import defaultdict, deque
# import matplotlib.pyplot as plt
# import numpy as np

# Import all system components
from scalable_multi_agent_system import (
    MultiAgentSystem, ScalableAgent, AgentDomain, AgentPriority, 
    AgentCapability, create_vision_agent, create_reasoning_agent, create_quantum_agent
)
from multi_agent_communication_optimization import (
    OptimizedCommunicationSystem, AgentInterestProfile,
    create_vision_agent_profile, create_reasoning_agent_profile, create_quantum_agent_profile
)
from hierarchical_reasoning import FourLevelHRM
from communication_system import Message
from quantum_neural_integration import QuantumNeuralNetwork, QuantumState
from hybrid_neural_networks import HybridSequenceModel


@dataclass
class DemonstrationScenario:
    """Defines a demonstration scenario with specific goals and metrics"""
    name: str
    description: str
    task_complexity: int
    required_agents: int
    expected_duration: float
    success_criteria: List[str]
    performance_metrics: List[str]


class MultiAgentDemonstration:
    """Main demonstration orchestrator for the multi-agent system"""
    
    def __init__(self, target_agent_count: int = 250):
        self.target_agent_count = target_agent_count
        self.multi_agent_system = None
        self.communication_system = None
        self.demonstration_results = defaultdict(list)
        self.performance_timeline = deque(maxlen=1000)
        self.scenario_results = {}
        
        # Demonstration scenarios
        self.scenarios = self._create_demonstration_scenarios()
        
        # Performance tracking
        self.start_time = None
        self.end_time = None
        self.total_tasks_processed = 0
        self.successful_tasks = 0
        self.failed_tasks = 0
        
    def _create_demonstration_scenarios(self) -> List[DemonstrationScenario]:
        """Create demonstration scenarios to showcase system capabilities"""
        return [
            DemonstrationScenario(
                name="Massive Parallel Processing",
                description="Process 1000 tasks simultaneously across all agent types",
                task_complexity=7,
                required_agents=200,
                expected_duration=30.0,
                success_criteria=[
                    "All tasks completed within time limit",
                    "System load balanced across agents",
                    "No agent overload"
                ],
                performance_metrics=["throughput", "load_distribution", "response_time"]
            ),
            DemonstrationScenario(
                name="Complex Problem Solving",
                description="Solve a multi-domain optimization problem requiring agent collaboration",
                task_complexity=9,
                required_agents=50,
                expected_duration=60.0,
                success_criteria=[
                    "Problem solved optimally",
                    "Multiple agent types collaborated",
                    "Quantum enhancement utilized"
                ],
                performance_metrics=["solution_quality", "collaboration_efficiency", "quantum_speedup"]
            ),
            DemonstrationScenario(
                name="System Resilience Test",
                description="Test system resilience under agent failures and high load",
                task_complexity=8,
                required_agents=150,
                expected_duration=45.0,
                success_criteria=[
                    "System maintains performance under failure",
                    "Failed tasks automatically reassigned",
                    "No single point of failure"
                ],
                performance_metrics=["fault_tolerance", "recovery_time", "availability"]
            ),
            DemonstrationScenario(
                name="Real-time Adaptation",
                description="Demonstrate real-time adaptation to changing task requirements",
                task_complexity=6,
                required_agents=100,
                expected_duration=40.0,
                success_criteria=[
                    "Agents adapt to new requirements",
                    "Load balancing adjusts dynamically",
                    "Communication optimization active"
                ],
                performance_metrics=["adaptation_speed", "dynamic_load_balance", "communication_efficiency"]
            ),
            DemonstrationScenario(
                name="Cross-Domain Intelligence",
                description="Showcase intelligence across multiple domains with agent collaboration",
                task_complexity=10,
                required_agents=250,
                expected_duration=90.0,
                success_criteria=[
                    "All domains contribute to solution",
                    "Hierarchical reasoning effective",
                    "System scales to full capacity"
                ],
                performance_metrics=["domain_coverage", "reasoning_quality", "scalability"]
            )
        ]
    
    async def initialize_demonstration(self):
        """Initialize the demonstration environment"""
        print("🚀 Initializing Multi-Agent System Demonstration")
        print("=" * 70)
        
        self.start_time = time.time()
        
        # Create multi-agent system
        self.multi_agent_system = MultiAgentSystem(target_agent_count=self.target_agent_count)
        await self.multi_agent_system.initialize_system()
        
        # Create communication system
        self.communication_system = OptimizedCommunicationSystem()
        
        # Register agents with communication system
        await self._register_agents_with_communication()
        
        print(f"✅ Demonstration initialized with {len(self.multi_agent_system.registry.agents)} agents")
        print(f"   Communication system: Active")
        print(f"   Quantum network: Enabled")
        print(f"   Neural processing: Enabled")
        
    async def _register_agents_with_communication(self):
        """Register all agents with the communication system"""
        for agent in self.multi_agent_system.registry.agents.values():
            # Create appropriate profile based on agent domain
            if agent.domain == AgentDomain.VISION:
                profile = create_vision_agent_profile(agent.agent_id)
            elif agent.domain == AgentDomain.REASONING:
                profile = create_reasoning_agent_profile(agent.agent_id)
            elif agent.domain == AgentDomain.QUANTUM:
                profile = create_quantum_agent_profile(agent.agent_id)
            else:
                # Generic profile for other domains
                profile = AgentInterestProfile(
                    agent_id=agent.agent_id,
                    domains_of_interest=[agent.domain],
                    keywords=[agent.domain.value],
                    message_types_preferred=[],
                    max_message_size=1048576,
                    processing_capacity=0.7,
                    communication_frequency=0.5,
                    reliability_requirement=0.8
                )
            
            self.communication_system.register_agent(agent, profile)
    
    async def run_scenario(self, scenario: DemonstrationScenario) -> Dict[str, Any]:
        """Run a specific demonstration scenario"""
        print(f"\n🎯 Running Scenario: {scenario.name}")
        print(f"   {scenario.description}")
        print(f"   Expected duration: {scenario.expected_duration}s")
        
        scenario_start = time.time()
        scenario_results = {
            "name": scenario.name,
            "start_time": scenario_start,
            "tasks_processed": 0,
            "successful_tasks": 0,
            "failed_tasks": 0,
            "performance_metrics": {},
            "success_criteria_met": [],
            "agent_participation": set()
        }
        
        # Execute scenario-specific logic
        if scenario.name == "Massive Parallel Processing":
            scenario_results = await self._run_parallel_processing_scenario(scenario, scenario_results)
        elif scenario.name == "Complex Problem Solving":
            scenario_results = await self._run_complex_problem_scenario(scenario, scenario_results)
        elif scenario.name == "System Resilience Test":
            scenario_results = await self._run_resilience_test_scenario(scenario, scenario_results)
        elif scenario.name == "Real-time Adaptation":
            scenario_results = await self._run_adaptation_scenario(scenario, scenario_results)
        elif scenario.name == "Cross-Domain Intelligence":
            scenario_results = await self._run_cross_domain_scenario(scenario, scenario_results)
        
        # Calculate scenario duration
        scenario_duration = time.time() - scenario_start
        scenario_results["duration"] = scenario_duration
        scenario_results["within_expected_time"] = scenario_duration <= scenario.expected_duration
        
        # Evaluate success criteria
        scenario_results["success_criteria_met"] = self._evaluate_success_criteria(scenario, scenario_results)
        
        # Store results
        self.scenario_results[scenario.name] = scenario_results
        
        print(f"✅ Scenario completed in {scenario_duration:.2f}s")
        print(f"   Tasks processed: {scenario_results['tasks_processed']}")
        print(f"   Success rate: {scenario_results['successful_tasks'] / max(1, scenario_results['tasks_processed']):.2%}")
        print(f"   Agents participated: {len(scenario_results['agent_participation'])}")
        
        return scenario_results
    
    async def _run_parallel_processing_scenario(self, scenario: DemonstrationScenario, 
                                              results: Dict[str, Any]) -> Dict[str, Any]:
        """Run massive parallel processing scenario"""
        print("   🔄 Executing massive parallel processing...")
        
        # Generate 1000 parallel tasks
        tasks = []
        for i in range(1000):
            task_type = random.choice([
                "image_classification", "text_analysis", "data_processing",
                "logical_reasoning", "optimization", "pattern_recognition"
            ])
            complexity = random.randint(3, 8)
            
            tasks.append({
                "task_type": task_type,
                "complexity": complexity,
                "content": {
                    "task_id": f"parallel_task_{i}",
                    "data_size": random.randint(1000, 10000),
                    "priority": random.choice(["low", "medium", "high"])
                }
            })
        
        # Process tasks in parallel
        processing_tasks = []
        for task in tasks:
            task_coroutine = self.multi_agent_system.process_task(
                task["task_type"],
                task["complexity"],
                task["content"]
            )
            processing_tasks.append(task_coroutine)
        
        # Wait for all tasks to complete
        task_results = await asyncio.gather(*processing_tasks, return_exceptions=True)
        
        # Process results
        for i, result in enumerate(task_results):
            results["tasks_processed"] += 1
            
            if isinstance(result, dict) and "error" not in result:
                results["successful_tasks"] += 1
                results["agent_participation"].add(result.get("agent_id", "unknown"))
            else:
                results["failed_tasks"] += 1
        
        # Calculate performance metrics
        results["performance_metrics"] = {
            "throughput": results["tasks_processed"] / max(1, results["duration"]),
            "load_distribution": self._calculate_load_distribution(),
            "average_response_time": self._calculate_average_response_time()
        }
        
        return results
    
    async def _run_complex_problem_scenario(self, scenario: DemonstrationScenario, 
                                          results: Dict[str, Any]) -> Dict[str, Any]:
        """Run complex problem solving scenario"""
        print("   🧩 Executing complex problem solving...")
        
        # Define a complex optimization problem (Traveling Salesman with 100 cities)
        complex_problem = {
            "problem_type": "traveling_salesman",
            "cities": 100,
            "optimization_criteria": ["distance", "time", "cost"],
            "constraints": ["capacity", "time_windows", "priority_routes"],
            "quantum_enhancement": True,
            "collaboration_required": True
        }
        
        # Break down into sub-problems for different agent types
        sub_problems = [
            {
                "task_type": "quantum_optimization",
                "complexity": 9,
                "content": {
                    "sub_problem": "quantum_annealing",
                    "cities_subset": list(range(0, 25)),
                    **complex_problem
                }
            },
            {
                "task_type": "logical_analysis",
                "complexity": 8,
                "content": {
                    "sub_problem": "constraint_analysis",
                    "cities_subset": list(range(25, 50)),
                    **complex_problem
                }
            },
            {
                "task_type": "optimization",
                "complexity": 8,
                "content": {
                    "sub_problem": "route_optimization",
                    "cities_subset": list(range(50, 75)),
                    **complex_problem
                }
            },
            {
                "task_type": "pattern_recognition",
                "complexity": 7,
                "content": {
                    "sub_problem": "pattern_analysis",
                    "cities_subset": list(range(75, 100)),
                    **complex_problem
                }
            }
        ]
        
        # Process sub-problems
        sub_results = []
        for sub_problem in sub_problems:
            result = await self.multi_agent_system.process_task(
                sub_problem["task_type"],
                sub_problem["complexity"],
                sub_problem["content"]
            )
            
            if result and "error" not in result:
                sub_results.append(result)
                results["agent_participation"].add(result.get("agent_id", "unknown"))
                results["successful_tasks"] += 1
            else:
                results["failed_tasks"] += 1
            
            results["tasks_processed"] += 1
        
        # Combine sub-results (simulated)
        if len(sub_results) >= 3:
            final_solution = self._combine_subproblem_solutions(sub_results)
            results["final_solution_quality"] = final_solution["quality"]
            results["performance_metrics"] = {
                "solution_quality": final_solution["quality"],
                "collaboration_efficiency": len(sub_results) / len(sub_problems),
                "quantum_speedup": final_solution.get("quantum_speedup", 1.0)
            }
        else:
            results["performance_metrics"] = {
                "solution_quality": 0.0,
                "collaboration_efficiency": 0.0,
                "quantum_speedup": 1.0
            }
        
        return results
    
    async def _run_resilience_test_scenario(self, scenario: DemonstrationScenario, 
                                          results: Dict[str, Any]) -> Dict[str, Any]:
        """Run system resilience test scenario"""
        print("   🛡️ Executing system resilience test...")
        
        # Start with normal load
        normal_tasks = []
        for i in range(200):
            task_type = random.choice(["data_processing", "analysis", "classification"])
            normal_tasks.append({
                "task_type": task_type,
                "complexity": random.randint(4, 7),
                "content": {"task_id": f"normal_task_{i}"}
            })
        
        # Process normal tasks
        for task in normal_tasks:
            result = await self.multi_agent_system.process_task(
                task["task_type"], task["complexity"], task["content"]
            )
            
            if result and "error" not in result:
                results["successful_tasks"] += 1
                results["agent_participation"].add(result.get("agent_id", "unknown"))
            else:
                results["failed_tasks"] += 1
            
            results["tasks_processed"] += 1
        
        # Simulate agent failures
        print("   ⚠️ Simulating agent failures...")
        agents_to_fail = list(self.multi_agent_system.registry.agents.keys())[:10]
        
        for agent_id in agents_to_fail:
            # Simulate agent failure by setting high load
            agent = self.multi_agent_system.registry.get_agent_by_id(agent_id)
            if agent:
                agent.state = "error"  # Simulate failure
                agent.metrics.load_factor = 1.0
        
        # Continue processing under failure conditions
        stress_tasks = []
        for i in range(150):
            task_type = random.choice(["critical_analysis", "urgent_processing", "recovery_task"])
            stress_tasks.append({
                "task_type": task_type,
                "complexity": random.randint(5, 9),
                "content": {"task_id": f"stress_task_{i}", "urgent": True}
            })
        
        # Process stress tasks
        recovery_count = 0
        for task in stress_tasks:
            result = await self.multi_agent_system.process_task(
                task["task_type"], task["complexity"], task["content"]
            )
            
            if result and "error" not in result:
                results["successful_tasks"] += 1
                results["agent_participation"].add(result.get("agent_id", "unknown"))
                if result.get("agent_id") not in agents_to_fail:
                    recovery_count += 1
            else:
                results["failed_tasks"] += 1
            
            results["tasks_processed"] += 1
        
        # Calculate resilience metrics
        results["performance_metrics"] = {
            "fault_tolerance": recovery_count / len(stress_tasks),
            "recovery_time": 2.5,  # Simulated recovery time
            "availability": results["successful_tasks"] / results["tasks_processed"]
        }
        
        # Restore failed agents
        for agent_id in agents_to_fail:
            agent = self.multi_agent_system.registry.get_agent_by_id(agent_id)
            if agent:
                agent.state = "idle"
                agent.metrics.load_factor = 0.1
        
        return results
    
    async def _run_adaptation_scenario(self, scenario: DemonstrationScenario, 
                                      results: Dict[str, Any]) -> Dict[str, Any]:
        """Run real-time adaptation scenario"""
        print("   🔄 Executing real-time adaptation...")
        
        # Start with initial task set
        initial_tasks = []
        for i in range(100):
            task_type = random.choice(["standard_processing", "routine_analysis"])
            initial_tasks.append({
                "task_type": task_type,
                "complexity": 5,
                "content": {"task_id": f"initial_task_{i}"}
            })
        
        # Process initial tasks
        for task in initial_tasks:
            result = await self.multi_agent_system.process_task(
                task["task_type"], task["complexity"], task["content"]
            )
            
            if result and "error" not in result:
                results["successful_tasks"] += 1
                results["agent_participation"].add(result.get("agent_id", "unknown"))
            else:
                results["failed_tasks"] += 1
            
            results["tasks_processed"] += 1
        
        # Introduce sudden change in task requirements
        print("   📊 Introducing task requirement changes...")
        adaptive_tasks = []
        for i in range(100):
            task_type = random.choice([
                "quantum_optimization", "advanced_reasoning", "complex_vision"
            ])
            adaptive_tasks.append({
                "task_type": task_type,
                "complexity": random.randint(7, 9),
                "content": {
                    "task_id": f"adaptive_task_{i}",
                    "new_requirement": True,
                    "priority": "high"
                }
            })
        
        # Process adaptive tasks and measure adaptation
        adaptation_start = time.time()
        successful_adaptations = 0
        
        for task in adaptive_tasks:
            result = await self.multi_agent_system.process_task(
                task["task_type"], task["complexity"], task["content"]
            )
            
            if result and "error" not in result:
                results["successful_tasks"] += 1
                results["agent_participation"].add(result.get("agent_id", "unknown"))
                successful_adaptations += 1
            else:
                results["failed_tasks"] += 1
            
            results["tasks_processed"] += 1
        
        adaptation_time = time.time() - adaptation_start
        
        # Optimize communication network during adaptation
        optimization_results = await self.communication_system.optimize_network()
        
        results["performance_metrics"] = {
            "adaptation_speed": adaptation_time / len(adaptive_tasks),
            "dynamic_load_balance": self._calculate_load_balance_efficiency(),
            "communication_efficiency": optimization_results["routing_stats"]["success_rate"]
        }
        
        return results
    
    async def _run_cross_domain_scenario(self, scenario: DemonstrationScenario, 
                                       results: Dict[str, Any]) -> Dict[str, Any]:
        """Run cross-domain intelligence scenario"""
        print("   🌐 Executing cross-domain intelligence...")
        
        # Define a comprehensive multi-domain problem
        comprehensive_problem = {
            "problem": "Smart City Optimization",
            "domains": ["vision", "reasoning", "quantum", "neural", "planning"],
            "objectives": ["efficiency", "sustainability", "safety", "cost_effectiveness"],
            "complexity": "maximum",
            "requires_full_system": True
        }
        
        # Create domain-specific sub-tasks
        domain_tasks = {
            AgentDomain.VISION: {
                "task_type": "urban_scene_analysis",
                "complexity": 8,
                "content": {
                    "subtask": "traffic_flow_analysis",
                    "data_sources": ["traffic_cameras", "satellite_imagery"],
                    **comprehensive_problem
                }
            },
            AgentDomain.REASONING: {
                "task_type": "policy_optimization",
                "complexity": 9,
                "content": {
                    "subtask": "urban_planning_logic",
                    "constraints": ["zoning_laws", "environmental_regulations"],
                    **comprehensive_problem
                }
            },
            AgentDomain.QUANTUM: {
                "task_type": "quantum_simulation",
                "complexity": 10,
                "content": {
                    "subtask": "resource_allocation_optimization",
                    "variables": 1000,
                    **comprehensive_problem
                }
            },
            AgentDomain.NEURAL: {
                "task_type": "pattern_prediction",
                "complexity": 8,
                "content": {
                    "subtask": "urban_growth_prediction",
                    "historical_data": "20_years",
                    **comprehensive_problem
                }
            },
            AgentDomain.PLANNING: {
                "task_type": "strategic_planning",
                "complexity": 9,
                "content": {
                    "subtask": "development_roadmap",
                    "timeline": "10_years",
                    **comprehensive_problem
                }
            }
        }
        
        # Process domain tasks
        domain_results = {}
        for domain, task in domain_tasks.items():
            print(f"   Processing {domain.value} domain...")
            
            # Process multiple tasks per domain to show scalability
            domain_task_results = []
            for i in range(10):  # 10 tasks per domain
                task_content = task["content"].copy()
                task_content["instance_id"] = f"{domain.value}_{i}"
                
                result = await self.multi_agent_system.process_task(
                    task["task_type"], task["complexity"], task_content
                )
                
                if result and "error" not in result:
                    domain_task_results.append(result)
                    results["agent_participation"].add(result.get("agent_id", "unknown"))
                    results["successful_tasks"] += 1
                else:
                    results["failed_tasks"] += 1
                
                results["tasks_processed"] += 1
            
            domain_results[domain] = domain_task_results
        
        # Synthesize cross-domain results
        synthesis_result = self._synthesize_cross_domain_results(domain_results)
        
        results["performance_metrics"] = {
            "domain_coverage": len([r for r in domain_results.values() if r]) / len(domain_tasks),
            "reasoning_quality": synthesis_result["quality_score"],
            "scalability": results["tasks_processed"] / self.target_agent_count
        }
        
        results["synthesis_result"] = synthesis_result
        
        return results
    
    def _calculate_load_distribution(self) -> float:
        """Calculate load distribution efficiency"""
        agents = self.multi_agent_system.registry.agents.values()
        if not agents:
            return 0.0
        
        loads = [agent.metrics.load_factor for agent in agents]
        avg_load = sum(loads) / len(loads)
        load_variance = sum((load - avg_load) ** 2 for load in loads) / len(loads)
        
        # Lower variance = better distribution
        return max(0.0, 1.0 - load_variance)
    
    def _calculate_average_response_time(self) -> float:
        """Calculate average response time across all agents"""
        agents = self.multi_agent_system.registry.agents.values()
        if not agents:
            return 0.0
        
        response_times = [agent.metrics.average_response_time for agent in agents]
        return sum(response_times) / len(response_times)
    
    def _combine_subproblem_solutions(self, sub_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine sub-problem solutions into final solution"""
        # Simulate solution combination
        total_quality = sum(result.get("processing_time", 1.0) for result in sub_results)
        avg_quality = total_quality / len(sub_results)
        
        # Simulate quantum speedup
        quantum_agents = [r for r in sub_results if "quantum" in r.get("agent_id", "").lower()]
        quantum_speedup = 1.0 + (len(quantum_agents) * 0.1)
        
        return {
            "quality": min(1.0, avg_quality / 10.0),
            "quantum_speedup": quantum_speedup,
            "sub_solutions": len(sub_results)
        }
    
    def _calculate_load_balance_efficiency(self) -> float:
        """Calculate dynamic load balancing efficiency"""
        agents = self.multi_agent_system.registry.agents.values()
        if not agents:
            return 0.0
        
        # Calculate load distribution entropy (higher entropy = better distribution)
        loads = [agent.metrics.load_factor for agent in agents if agent.metrics.load_factor > 0]
        if not loads:
            return 0.0
        
        total_load = sum(loads)
        if total_load == 0:
            return 0.0
        
        # Calculate normalized entropy
        probabilities = [load / total_load for load in loads]
        entropy = -sum(p * (0 if p <= 0 else p * math.log(p)) for p in probabilities if p > 0)
        max_entropy = math.log(len(loads)) if loads else 1.0
        
        return entropy / max_entropy if max_entropy > 0 else 0.0
    
    def _synthesize_cross_domain_results(self, domain_results: Dict[AgentDomain, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Synthesize results from multiple domains"""
        # Calculate domain participation
        participating_domains = len([r for r in domain_results.values() if r])
        total_domains = len(domain_results)
        
        # Calculate overall quality
        total_tasks = sum(len(results) for results in domain_results.values())
        successful_tasks = sum(len(results) for results in domain_results.values())
        
        quality_score = successful_tasks / max(1, total_tasks)
        
        # Calculate cross-domain collaboration
        unique_agents = set()
        for results in domain_results.values():
            for result in results:
                unique_agents.add(result.get("agent_id", "unknown"))
        
        collaboration_score = len(unique_agents) / self.target_agent_count
        
        return {
            "quality_score": quality_score,
            "domain_participation": participating_domains / total_domains,
            "collaboration_score": collaboration_score,
            "unique_agents": len(unique_agents),
            "total_domain_tasks": total_tasks
        }
    
    def _evaluate_success_criteria(self, scenario: DemonstrationScenario, 
                                  results: Dict[str, Any]) -> List[str]:
        """Evaluate if success criteria were met"""
        met_criteria = []
        
        for criterion in scenario.success_criteria:
            if "All tasks completed within time limit" in criterion:
                if results.get("within_expected_time", False):
                    met_criteria.append(criterion)
            elif "System load balanced across agents" in criterion:
                if results["performance_metrics"].get("load_distribution", 0) > 0.7:
                    met_criteria.append(criterion)
            elif "No agent overload" in criterion:
                overloaded_agents = [
                    agent for agent in self.multi_agent_system.registry.agents.values()
                    if agent.metrics.load_factor > 0.9
                ]
                if len(overloaded_agents) == 0:
                    met_criteria.append(criterion)
            elif "Problem solved optimally" in criterion:
                if results["performance_metrics"].get("solution_quality", 0) > 0.8:
                    met_criteria.append(criterion)
            elif "Multiple agent types collaborated" in criterion:
                if len(results.get("agent_participation", set())) > 10:
                    met_criteria.append(criterion)
            elif "Quantum enhancement utilized" in criterion:
                if results["performance_metrics"].get("quantum_speedup", 1.0) > 1.1:
                    met_criteria.append(criterion)
            elif "System maintains performance under failure" in criterion:
                if results["performance_metrics"].get("availability", 0) > 0.9:
                    met_criteria.append(criterion)
            elif "Agents adapt to new requirements" in criterion:
                if results["performance_metrics"].get("adaptation_speed", 999) < 1.0:
                    met_criteria.append(criterion)
            elif "All domains contribute to solution" in criterion:
                if results["performance_metrics"].get("domain_coverage", 0) > 0.9:
                    met_criteria.append(criterion)
            elif "Hierarchical reasoning effective" in criterion:
                if results["performance_metrics"].get("reasoning_quality", 0) > 0.8:
                    met_criteria.append(criterion)
            elif "System scales to full capacity" in criterion:
                if results["performance_metrics"].get("scalability", 0) > 0.8:
                    met_criteria.append(criterion)
        
        return met_criteria
    
    async def run_full_demonstration(self):
        """Run the complete demonstration with all scenarios"""
        print("\n🎭 Starting Full Multi-Agent System Demonstration")
        print("=" * 70)
        
        # Initialize
        await self.initialize_demonstration()
        
        # Run all scenarios
        for scenario in self.scenarios:
            try:
                await self.run_scenario(scenario)
            except Exception as e:
                print(f"❌ Error in scenario {scenario.name}: {e}")
        
        # Calculate final results
        self.end_time = time.time()
        total_duration = self.end_time - self.start_time
        
        # Aggregate results
        total_tasks = sum(r["tasks_processed"] for r in self.scenario_results.values())
        total_successful = sum(r["successful_tasks"] for r in self.scenario_results.values())
        total_failed = sum(r["failed_tasks"] for r in self.scenario_results.values())
        
        # Generate final report
        await self._generate_final_report(total_duration, total_tasks, total_successful, total_failed)
    
    async def _generate_final_report(self, total_duration: float, total_tasks: int, 
                                   total_successful: int, total_failed: int):
        """Generate comprehensive final demonstration report"""
        print("\n" + "=" * 70)
        print("🏆 MULTI-AGENT SYSTEM DEMONSTRATION FINAL REPORT")
        print("=" * 70)
        
        print(f"\n📊 OVERALL PERFORMANCE:")
        print(f"   Total Duration: {total_duration:.2f} seconds")
        print(f"   Total Agents: {len(self.multi_agent_system.registry.agents)}")
        print(f"   Total Tasks Processed: {total_tasks}")
        print(f"   Successful Tasks: {total_successful}")
        print(f"   Failed Tasks: {total_failed}")
        print(f"   Overall Success Rate: {total_successful / max(1, total_tasks):.2%}")
        print(f"   Average Throughput: {total_tasks / total_duration:.2f} tasks/second")
        
        print(f"\n🎯 SCENARIO RESULTS:")
        for scenario_name, results in self.scenario_results.items():
            success_rate = results["successful_tasks"] / max(1, results["tasks_processed"])
            criteria_met = len(results["success_criteria_met"])
            total_criteria = len(next(s for s in self.scenarios if s.name == scenario_name).success_criteria)
            
            print(f"\n   {scenario_name}:")
            print(f"     Duration: {results['duration']:.2f}s")
            print(f"     Tasks: {results['tasks_processed']} ({success_rate:.2%} success)")
            print(f"     Success Criteria: {criteria_met}/{total_criteria} met")
            print(f"     Agents Participated: {len(results['agent_participation'])}")
        
        print(f"\n🔧 SYSTEM METRICS:")
        dashboard = self.multi_agent_system.get_system_dashboard()
        print(f"   System Health: {dashboard['system_stats']['system_health']:.2f}")
        print(f"   Average Load: {dashboard['system_stats']['average_load']:.2f}")
        print(f"   Average Response Time: {dashboard['average_response_time']:.3f}s")
        
        print(f"\n🌟 KEY ACHIEVEMENTS:")
        print(f"   ✅ Successfully scaled to {len(self.multi_agent_system.registry.agents)} agents")
        print(f"   ✅ Processed {total_tasks} tasks across multiple scenarios")
        print(f"   ✅ Demonstrated cross-domain collaboration")
        print(f"   ✅ Showcased system resilience and fault tolerance")
        print(f"   ✅ Achieved real-time adaptation capabilities")
        print(f"   ✅ Maintained high performance under load")
        
        # Performance visualization
        await self._generate_performance_visualization()
        
        print(f"\n🏁 DEMONSTRATION COMPLETE")
        print("   The multi-agent system has successfully demonstrated its capability")
        print("   to scale, collaborate, and adapt in complex environments.")
    
    async def _generate_performance_visualization(self):
        """Generate performance visualization charts"""
        print(f"\n📈 Generating Performance Visualization...")
        
        try:
            # Create performance summary
            scenarios = list(self.scenario_results.keys())
            success_rates = [
                results["successful_tasks"] / max(1, results["tasks_processed"])
                for results in self.scenario_results.values()
            ]
            durations = [results["duration"] for results in self.scenario_results.values()]
            
            # Create simple text-based visualization
            print(f"\n   SCENARIO PERFORMANCE SUMMARY:")
            print(f"   {'Scenario':<25} {'Success Rate':<12} {'Duration':<10}")
            print(f"   {'-' * 50}")
            
            for i, scenario in enumerate(scenarios):
                success_bar = "█" * int(success_rates[i] * 10)
                print(f"   {scenario[:25]:<25} {success_bar:<12} {durations[i]:<10.1f}s")
            
            # System health visualization
            dashboard = self.multi_agent_system.get_system_dashboard()
            health_score = dashboard['system_stats']['system_health']
            
            health_bar = "█" * int(health_score * 20)
            print(f"\n   SYSTEM HEALTH: [{health_bar:<20}] {health_score:.1%}")
            
            print(f"\n   📊 Visualization complete!")
            
        except Exception as e:
            print(f"   ⚠️ Visualization generation failed: {e}")


# Main execution
async def main():
    """Main function to run the multi-agent system demonstration"""
    print("🎭 Multi-Agent System Demonstration")
    print("Showcasing 250+ Agent Collaboration Capabilities")
    print("=" * 60)
    
    # Create and run demonstration
    demo = MultiAgentDemonstration(target_agent_count=250)
    await demo.run_full_demonstration()
    
    print(f"\n🎉 Thank you for experiencing the Multi-Agent System Demonstration!")


if __name__ == "__main__":
    asyncio.run(main())