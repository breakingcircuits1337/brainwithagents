"""
Advanced Agent Specializations for 250-Agent Brain System

This module defines 250 specialized agent types with unique capabilities,
organized into functional domains for comprehensive AI collaboration.
"""

from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum
import random
import json

class AgentDomain(Enum):
    """Functional domains for agent organization"""
    DATA_PROCESSING = "data_processing"
    ANALYSIS = "analysis"
    COMMUNICATION = "communication"
    LEARNING = "learning"
    CREATIVITY = "creativity"
    OPTIMIZATION = "optimization"
    MONITORING = "monitoring"
    SECURITY = "security"
    COORDINATION = "coordination"
    SPECIALIZED = "specialized"

class AgentComplexity(Enum):
    """Complexity levels for agents"""
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"

@dataclass
class AgentCapability:
    """Defines a specific agent capability"""
    name: str
    description: str
    input_types: List[str]
    output_types: List[str]
    complexity: AgentComplexity
    resource_requirements: Dict[str, float]
    success_rate: float = 0.95
    processing_time: float = 1.0

@dataclass
class AgentSpecialization:
    """Complete agent specialization definition"""
    id: str
    name: str
    domain: AgentDomain
    capabilities: List[AgentCapability]
    preferred_connections: List[str]
    threshold_range: Tuple[float, float]
    expertise_level: float
    collaboration_style: str
    resource_profile: Dict[str, float]

class AgentSpecializationRegistry:
    """Registry for all 250 specialized agent types"""
    
    def __init__(self):
        self.specializations: Dict[str, AgentSpecialization] = {}
        self.domain_mapping: Dict[AgentDomain, List[str]] = {}
        self._create_all_specializations()
    
    def _create_all_specializations(self):
        """Create all 250 specialized agent types"""
        
        # DATA PROCESSING DOMAIN (50 agents)
        data_agents = self._create_data_processing_agents()
        for agent in data_agents:
            self._add_specialization(agent)
        
        # ANALYSIS DOMAIN (40 agents)
        analysis_agents = self._create_analysis_agents()
        for agent in analysis_agents:
            self._add_specialization(agent)
        
        # COMMUNICATION DOMAIN (30 agents)
        comm_agents = self._create_communication_agents()
        for agent in comm_agents:
            self._add_specialization(agent)
        
        # LEARNING DOMAIN (35 agents)
        learning_agents = self._create_learning_agents()
        for agent in learning_agents:
            self._add_specialization(agent)
        
        # CREATIVITY DOMAIN (25 agents)
        creativity_agents = self._create_creativity_agents()
        for agent in creativity_agents:
            self._add_specialization(agent)
        
        # OPTIMIZATION DOMAIN (20 agents)
        optimization_agents = self._create_optimization_agents()
        for agent in optimization_agents:
            self._add_specialization(agent)
        
        # MONITORING DOMAIN (20 agents)
        monitoring_agents = self._create_monitoring_agents()
        for agent in monitoring_agents:
            self._add_specialization(agent)
        
        # SECURITY DOMAIN (15 agents)
        security_agents = self._create_security_agents()
        for agent in security_agents:
            self._add_specialization(agent)
        
        # COORDINATION DOMAIN (10 agents)
        coordination_agents = self._create_coordination_agents()
        for agent in coordination_agents:
            self._add_specialization(agent)
        
        # SPECIALIZED DOMAIN (5 agents)
        specialized_agents = self._create_specialized_agents()
        for agent in specialized_agents:
            self._add_specialization(agent)
    
    def _create_data_processing_agents(self) -> List[AgentSpecialization]:
        """Create 50 data processing agents"""
        agents = []
        
        # Data Ingestion Agents (10)
        for i in range(10):
            agent = AgentSpecialization(
                id=f"data_ingestor_{i:02d}",
                name=f"Data Ingestion Specialist {i+1}",
                domain=AgentDomain.DATA_PROCESSING,
                capabilities=[
                    AgentCapability(
                        name="data_ingestion",
                        description=f"Ingest data from source type {i+1}",
                        input_types=["raw_data", "stream"],
                        output_types=["structured_data", "metadata"],
                        complexity=AgentComplexity.INTERMEDIATE,
                        resource_requirements={"cpu": 0.3, "memory": 0.4, "network": 0.6}
                    )
                ],
                preferred_connections=["data_validator", "data_cleaner"],
                threshold_range=(0.4, 0.7),
                expertise_level=0.7 + (i * 0.03),
                collaboration_style="sequential",
                resource_profile={"throughput": 100 + i * 10, "latency": 0.1}
            )
            agents.append(agent)
        
        # Data Validation Agents (10)
        for i in range(10):
            agent = AgentSpecialization(
                id=f"data_validator_{i:02d}",
                name=f"Data Validation Specialist {i+1}",
                domain=AgentDomain.DATA_PROCESSING,
                capabilities=[
                    AgentCapability(
                        name="data_validation",
                        description=f"Validate data type {i+1}",
                        input_types=["structured_data"],
                        output_types=["validated_data", "validation_report"],
                        complexity=AgentComplexity.INTERMEDIATE,
                        resource_requirements={"cpu": 0.4, "memory": 0.3, "network": 0.2}
                    )
                ],
                preferred_connections=["data_cleaner", "data_transformer"],
                threshold_range=(0.5, 0.8),
                expertise_level=0.8 + (i * 0.02),
                collaboration_style="parallel",
                resource_profile={"accuracy": 0.95 + i * 0.005, "speed": 50 + i * 5}
            )
            agents.append(agent)
        
        # Data Cleaning Agents (10)
        for i in range(10):
            agent = AgentSpecialization(
                id=f"data_cleaner_{i:02d}",
                name=f"Data Cleaning Specialist {i+1}",
                domain=AgentDomain.DATA_PROCESSING,
                capabilities=[
                    AgentCapability(
                        name="data_cleaning",
                        description=f"Clean data using method {i+1}",
                        input_types=["validated_data", "raw_data"],
                        output_types=["clean_data", "cleaning_log"],
                        complexity=AgentComplexity.ADVANCED,
                        resource_requirements={"cpu": 0.6, "memory": 0.5, "network": 0.1}
                    )
                ],
                preferred_connections=["data_transformer", "data_enricher"],
                threshold_range=(0.6, 0.9),
                expertise_level=0.75 + (i * 0.025),
                collaboration_style="adaptive",
                resource_profile={"quality": 0.9 + i * 0.01, "efficiency": 0.8 + i * 0.02}
            )
            agents.append(agent)
        
        # Data Transformation Agents (10)
        for i in range(10):
            agent = AgentSpecialization(
                id=f"data_transformer_{i:02d}",
                name=f"Data Transformation Specialist {i+1}",
                domain=AgentDomain.DATA_PROCESSING,
                capabilities=[
                    AgentCapability(
                        name="data_transformation",
                        description=f"Transform data using technique {i+1}",
                        input_types=["clean_data"],
                        output_types=["transformed_data", "transformation_metadata"],
                        complexity=AgentComplexity.ADVANCED,
                        resource_requirements={"cpu": 0.5, "memory": 0.6, "network": 0.3}
                    )
                ],
                preferred_connections=["data_enricher", "data_aggregator"],
                threshold_range=(0.5, 0.8),
                expertise_level=0.8 + (i * 0.02),
                collaboration_style="transformative",
                resource_profile={"flexibility": 0.9 + i * 0.01, "precision": 0.85 + i * 0.015}
            )
            agents.append(agent)
        
        # Data Enrichment Agents (10)
        for i in range(10):
            agent = AgentSpecialization(
                id=f"data_enricher_{i:02d}",
                name=f"Data Enrichment Specialist {i+1}",
                domain=AgentDomain.DATA_PROCESSING,
                capabilities=[
                    AgentCapability(
                        name="data_enrichment",
                        description=f"Enrich data with external source {i+1}",
                        input_types=["transformed_data"],
                        output_types=["enriched_data", "enrichment_sources"],
                        complexity=AgentComplexity.EXPERT,
                        resource_requirements={"cpu": 0.4, "memory": 0.7, "network": 0.8}
                    )
                ],
                preferred_connections=["data_aggregator", "data_analyzer"],
                threshold_range=(0.7, 0.95),
                expertise_level=0.85 + (i * 0.015),
                collaboration_style="integrative",
                resource_profile={"comprehensiveness": 0.95 + i * 0.005, "relevance": 0.9 + i * 0.01}
            )
            agents.append(agent)
        
        return agents
    
    def _create_analysis_agents(self) -> List[AgentSpecialization]:
        """Create 40 analysis agents"""
        agents = []
        
        # Statistical Analysis Agents (8)
        for i in range(8):
            agent = AgentSpecialization(
                id=f"stat_analyzer_{i:02d}",
                name=f"Statistical Analysis Specialist {i+1}",
                domain=AgentDomain.ANALYSIS,
                capabilities=[
                    AgentCapability(
                        name="statistical_analysis",
                        description=f"Perform statistical analysis type {i+1}",
                        input_types=["numerical_data", "categorical_data"],
                        output_types=["statistics", "confidence_intervals"],
                        complexity=AgentComplexity.ADVANCED,
                        resource_requirements={"cpu": 0.8, "memory": 0.6, "network": 0.1}
                    )
                ],
                preferred_connections=["pattern_detector", "insight_generator"],
                threshold_range=(0.6, 0.9),
                expertise_level=0.8 + (i * 0.025),
                collaboration_style="analytical",
                resource_profile={"depth": 0.9 + i * 0.0125, "breadth": 0.7 + i * 0.0375}
            )
            agents.append(agent)
        
        # Pattern Detection Agents (8)
        for i in range(8):
            agent = AgentSpecialization(
                id=f"pattern_detector_{i:02d}",
                name=f"Pattern Detection Specialist {i+1}",
                domain=AgentDomain.ANALYSIS,
                capabilities=[
                    AgentCapability(
                        name="pattern_detection",
                        description=f"Detect patterns using algorithm {i+1}",
                        input_types=["time_series", "sequential_data"],
                        output_types=["patterns", "pattern_significance"],
                        complexity=AgentComplexity.EXPERT,
                        resource_requirements={"cpu": 0.9, "memory": 0.8, "network": 0.2}
                    )
                ],
                preferred_connections=["anomaly_detector", "trend_analyzer"],
                threshold_range=(0.7, 0.95),
                expertise_level=0.85 + (i * 0.01875),
                collaboration_style="discovery",
                resource_profile={"sensitivity": 0.85 + i * 0.01875, "specificity": 0.8 + i * 0.025}
            )
            agents.append(agent)
        
        # Anomaly Detection Agents (8)
        for i in range(8):
            agent = AgentSpecialization(
                id=f"anomaly_detector_{i:02d}",
                name=f"Anomaly Detection Specialist {i+1}",
                domain=AgentDomain.ANALYSIS,
                capabilities=[
                    AgentCapability(
                        name="anomaly_detection",
                        description=f"Detect anomalies using method {i+1}",
                        input_types=["data_stream", "historical_data"],
                        output_types=["anomalies", "anomaly_scores"],
                        complexity=AgentComplexity.ADVANCED,
                        resource_requirements={"cpu": 0.7, "memory": 0.5, "network": 0.3}
                    )
                ],
                preferred_connections=["alert_generator", "security_analyzer"],
                threshold_range=(0.8, 0.98),
                expertise_level=0.9 + (i * 0.0125),
                collaboration_style="vigilant",
                resource_profile={"accuracy": 0.95 + i * 0.00625, "speed": 0.7 + i * 0.0375}
            )
            agents.append(agent)
        
        # Trend Analysis Agents (8)
        for i in range(8):
            agent = AgentSpecialization(
                id=f"trend_analyzer_{i:02d}",
                name=f"Trend Analysis Specialist {i+1}",
                domain=AgentDomain.ANALYSIS,
                capabilities=[
                    AgentCapability(
                        name="trend_analysis",
                        description=f"Analyze trends using technique {i+1}",
                        input_types=["time_series", "historical_data"],
                        output_types=["trends", "forecasts"],
                        complexity=AgentComplexity.ADVANCED,
                        resource_requirements={"cpu": 0.6, "memory": 0.7, "network": 0.1}
                    )
                ],
                preferred_connections=["predictor", "insight_generator"],
                threshold_range=(0.6, 0.85),
                expertise_level=0.8 + (i * 0.025),
                collaboration_style="predictive",
                resource_profile={"horizon": 10 + i * 5, "accuracy": 0.8 + i * 0.025}
            )
            agents.append(agent)
        
        # Insight Generation Agents (8)
        for i in range(8):
            agent = AgentSpecialization(
                id=f"insight_generator_{i:02d}",
                name=f"Insight Generation Specialist {i+1}",
                domain=AgentDomain.ANALYSIS,
                capabilities=[
                    AgentCapability(
                        name="insight_generation",
                        description=f"Generate insights using method {i+1}",
                        input_types=["analysis_results", "patterns"],
                        output_types=["insights", "recommendations"],
                        complexity=AgentComplexity.EXPERT,
                        resource_requirements={"cpu": 0.5, "memory": 0.9, "network": 0.2}
                    )
                ],
                preferred_connections=["decision_maker", "report_generator"],
                threshold_range=(0.7, 0.9),
                expertise_level=0.85 + (i * 0.01875),
                collaboration_style="synthetic",
                resource_profile={"depth": 0.95 + i * 0.00625, "novelty": 0.8 + i * 0.025}
            )
            agents.append(agent)
        
        return agents
    
    def _create_communication_agents(self) -> List[AgentSpecialization]:
        """Create 30 communication agents"""
        agents = []
        
        # Message Routing Agents (6)
        for i in range(6):
            agent = AgentSpecialization(
                id=f"msg_router_{i:02d}",
                name=f"Message Routing Specialist {i+1}",
                domain=AgentDomain.COMMUNICATION,
                capabilities=[
                    AgentCapability(
                        name="message_routing",
                        description=f"Route messages using protocol {i+1}",
                        input_types=["messages", "routing_requests"],
                        output_types=["routed_messages", "routing_logs"],
                        complexity=AgentComplexity.INTERMEDIATE,
                        resource_requirements={"cpu": 0.3, "memory": 0.2, "network": 0.9}
                    )
                ],
                preferred_connections=["protocol_handler", "message_validator"],
                threshold_range=(0.4, 0.7),
                expertise_level=0.75 + (i * 0.0417),
                collaboration_style="distributive",
                resource_profile={"throughput": 1000 + i * 200, "latency": 0.01 + i * 0.005}
            )
            agents.append(agent)
        
        # Protocol Handler Agents (6)
        for i in range(6):
            agent = AgentSpecialization(
                id=f"protocol_handler_{i:02d}",
                name=f"Protocol Handler Specialist {i+1}",
                domain=AgentDomain.COMMUNICATION,
                capabilities=[
                    AgentCapability(
                        name="protocol_handling",
                        description=f"Handle protocol {i+1}",
                        input_types=["raw_messages", "protocol_specs"],
                        output_types=["parsed_messages", "protocol_metadata"],
                        complexity=AgentComplexity.INTERMEDIATE,
                        resource_requirements={"cpu": 0.4, "memory": 0.3, "network": 0.7}
                    )
                ],
                preferred_connections=["msg_router", "data_serializer"],
                threshold_range=(0.5, 0.8),
                expertise_level=0.8 + (i * 0.0333),
                collaboration_style="translational",
                resource_profile={"compatibility": 0.95 + i * 0.0083, "efficiency": 0.8 + i * 0.0333}
            )
            agents.append(agent)
        
        # Data Serialization Agents (6)
        for i in range(6):
            agent = AgentSpecialization(
                id=f"data_serializer_{i:02d}",
                name=f"Data Serialization Specialist {i+1}",
                domain=AgentDomain.COMMUNICATION,
                capabilities=[
                    AgentCapability(
                        name="data_serialization",
                        description=f"Serialize data using format {i+1}",
                        input_types=["structured_data", "serialization_specs"],
                        output_types=["serialized_data", "serialization_metadata"],
                        complexity=AgentComplexity.BASIC,
                        resource_requirements={"cpu": 0.2, "memory": 0.4, "network": 0.5}
                    )
                ],
                preferred_connections=["protocol_handler", "compressor"],
                threshold_range=(0.3, 0.6),
                expertise_level=0.7 + (i * 0.05),
                collaboration_style="conversion",
                resource_profile={"compression_ratio": 0.7 + i * 0.05, "speed": 0.9 + i * 0.0167}
            )
            agents.append(agent)
        
        # Message Validator Agents (6)
        for i in range(6):
            agent = AgentSpecialization(
                id=f"msg_validator_{i:02d}",
                name=f"Message Validation Specialist {i+1}",
                domain=AgentDomain.COMMUNICATION,
                capabilities=[
                    AgentCapability(
                        name="message_validation",
                        description=f"Validate messages using schema {i+1}",
                        input_types=["messages", "validation_schemas"],
                        output_types=["validated_messages", "validation_reports"],
                        complexity=AgentComplexity.INTERMEDIATE,
                        resource_requirements={"cpu": 0.3, "memory": 0.2, "network": 0.4}
                    )
                ],
                preferred_connections=["msg_router", "security_scanner"],
                threshold_range=(0.6, 0.85),
                expertise_level=0.85 + (i * 0.025),
                collaboration_style="verification",
                resource_profile={"thoroughness": 0.95 + i * 0.0083, "speed": 0.6 + i * 0.0667}
            )
            agents.append(agent)
        
        # Compression Agents (6)
        for i in range(6):
            agent = AgentSpecialization(
                id=f"compressor_{i:02d}",
                name=f"Data Compression Specialist {i+1}",
                domain=AgentDomain.COMMUNICATION,
                capabilities=[
                    AgentCapability(
                        name="data_compression",
                        description=f"Compress data using algorithm {i+1}",
                        input_types=["serialized_data"],
                        output_types=["compressed_data", "compression_stats"],
                        complexity=AgentComplexity.INTERMEDIATE,
                        resource_requirements={"cpu": 0.5, "memory": 0.3, "network": 0.2}
                    )
                ],
                preferred_connections=["data_serializer", "encryptor"],
                threshold_range=(0.4, 0.7),
                expertise_level=0.75 + (i * 0.0417),
                collaboration_style="optimization",
                resource_profile={"ratio": 0.5 + i * 0.0833, "speed": 0.7 + i * 0.05}
            )
            agents.append(agent)
        
        return agents
    
    def _create_learning_agents(self) -> List[AgentSpecialization]:
        """Create 35 learning agents"""
        agents = []
        
        # Neural Network Trainers (7)
        for i in range(7):
            agent = AgentSpecialization(
                id=f"nn_trainer_{i:02d}",
                name=f"Neural Network Trainer {i+1}",
                domain=AgentDomain.LEARNING,
                capabilities=[
                    AgentCapability(
                        name="neural_network_training",
                        description=f"Train neural networks using architecture {i+1}",
                        input_types=["training_data", "model_specs"],
                        output_types=["trained_model", "training_metrics"],
                        complexity=AgentComplexity.EXPERT,
                        resource_requirements={"cpu": 0.9, "memory": 0.95, "network": 0.3}
                    )
                ],
                preferred_connections=["data_preprocessor", "model_evaluator"],
                threshold_range=(0.7, 0.95),
                expertise_level=0.9 + (i * 0.0143),
                collaboration_style="iterative",
                resource_profile={"convergence_rate": 0.8 + i * 0.0286, "accuracy": 0.95 + i * 0.0071}
            )
            agents.append(agent)
        
        # Feature Extractors (7)
        for i in range(7):
            agent = AgentSpecialization(
                id=f"feature_extractor_{i:02d}",
                name=f"Feature Extraction Specialist {i+1}",
                domain=AgentDomain.LEARNING,
                capabilities=[
                    AgentCapability(
                        name="feature_extraction",
                        description=f"Extract features using technique {i+1}",
                        input_types=["raw_data", "feature_specs"],
                        output_types=["features", "feature_importance"],
                        complexity=AgentComplexity.ADVANCED,
                        resource_requirements={"cpu": 0.7, "memory": 0.6, "network": 0.1}
                    )
                ],
                preferred_connections=["data_preprocessor", "model_trainer"],
                threshold_range=(0.6, 0.85),
                expertise_level=0.85 + (i * 0.0214),
                collaboration_style="extractive",
                resource_profile={"relevance": 0.9 + i * 0.0143, "dimensionality": 100 - i * 10}
            )
            agents.append(agent)
        
        # Model Evaluators (7)
        for i in range(7):
            agent = AgentSpecialization(
                id=f"model_evaluator_{i:02d}",
                name=f"Model Evaluation Specialist {i+1}",
                domain=AgentDomain.LEARNING,
                capabilities=[
                    AgentCapability(
                        name="model_evaluation",
                        description=f"Evaluate models using metric set {i+1}",
                        input_types=["trained_model", "test_data"],
                        output_types=["evaluation_metrics", "model_report"],
                        complexity=AgentComplexity.ADVANCED,
                        resource_requirements={"cpu": 0.5, "memory": 0.4, "network": 0.2}
                    )
                ],
                preferred_connections=["nn_trainer", "model_optimizer"],
                threshold_range=(0.65, 0.9),
                expertise_level=0.88 + (i * 0.0171),
                collaboration_style="assessment",
                resource_profile={"comprehensiveness": 0.95 + i * 0.0071, "objectivity": 0.9 + i * 0.0143}
            )
            agents.append(agent)
        
        # Hyperparameter Optimizers (7)
        for i in range(7):
            agent = AgentSpecialization(
                id=f"hyperopt_{i:02d}",
                name=f"Hyperparameter Optimization Specialist {i+1}",
                domain=AgentDomain.LEARNING,
                capabilities=[
                    AgentCapability(
                        name="hyperparameter_optimization",
                        description=f"Optimize hyperparameters using method {i+1}",
                        input_types=["model_specs", "performance_data"],
                        output_types=["optimized_params", "optimization_report"],
                        complexity=AgentComplexity.EXPERT,
                        resource_requirements={"cpu": 0.95, "memory": 0.7, "network": 0.3}
                    )
                ],
                preferred_connections=["model_evaluator", "nn_trainer"],
                threshold_range=(0.75, 0.98),
                expertise_level=0.92 + (i * 0.0114),
                collaboration_style="optimizing",
                resource_profile={"efficiency": 0.9 + i * 0.0143, "effectiveness": 0.95 + i * 0.0071}
            )
            agents.append(agent)
        
        # Knowledge Integrators (7)
        for i in range(7):
            agent = AgentSpecialization(
                id=f"knowledge_integrator_{i:02d}",
                name=f"Knowledge Integration Specialist {i+1}",
                domain=AgentDomain.LEARNING,
                capabilities=[
                    AgentCapability(
                        name="knowledge_integration",
                        description=f"Integrate knowledge using method {i+1}",
                        input_types=["learned_models", "domain_knowledge"],
                        output_types=["integrated_knowledge", "integration_report"],
                        complexity=AgentComplexity.EXPERT,
                        resource_requirements={"cpu": 0.6, "memory": 0.9, "network": 0.4}
                    )
                ],
                preferred_connections=["model_evaluator", "reasoning_engine"],
                threshold_range=(0.8, 0.96),
                expertise_level=0.93 + (i * 0.01),
                collaboration_style="synthetic",
                resource_profile={"coherence": 0.95 + i * 0.0071, "completeness": 0.9 + i * 0.0143}
            )
            agents.append(agent)
        
        return agents
    
    def _create_creativity_agents(self) -> List[AgentSpecialization]:
        """Create 25 creativity agents"""
        agents = []
        
        # Creative Writers (5)
        for i in range(5):
            agent = AgentSpecialization(
                id=f"creative_writer_{i:02d}",
                name=f"Creative Writing Specialist {i+1}",
                domain=AgentDomain.CREATIVITY,
                capabilities=[
                    AgentCapability(
                        name="creative_writing",
                        description=f"Generate creative content in style {i+1}",
                        input_types=["prompts", "style_guidelines"],
                        output_types=["creative_content", "creativity_metrics"],
                        complexity=AgentComplexity.ADVANCED,
                        resource_requirements={"cpu": 0.4, "memory": 0.8, "network": 0.1}
                    )
                ],
                preferred_connections=["idea_generator", "content_editor"],
                threshold_range=(0.5, 0.8),
                expertise_level=0.8 + (i * 0.04),
                collaboration_style="expressive",
                resource_profile={"originality": 0.9 + i * 0.02, "coherence": 0.85 + i * 0.03}
            )
            agents.append(agent)
        
        # Idea Generators (5)
        for i in range(5):
            agent = AgentSpecialization(
                id=f"idea_generator_{i:02d}",
                name=f"Idea Generation Specialist {i+1}",
                domain=AgentDomain.CREATIVITY,
                capabilities=[
                    AgentCapability(
                        name="idea_generation",
                        description=f"Generate ideas using technique {i+1}",
                        input_types=["problem_statements", "constraints"],
                        output_types=["ideas", "idea_evaluations"],
                        complexity=AgentComplexity.EXPERT,
                        resource_requirements={"cpu": 0.6, "memory": 0.7, "network": 0.2}
                    )
                ],
                preferred_connections=["creative_writer", "innovation_analyst"],
                threshold_range=(0.6, 0.9),
                expertise_level=0.85 + (i * 0.03),
                collaboration_style="generative",
                resource_profile={"novelty": 0.95 + i * 0.01, "feasibility": 0.7 + i * 0.06}
            )
            agents.append(agent)
        
        # Innovation Analysts (5)
        for i in range(5):
            agent = AgentSpecialization(
                id=f"innovation_analyst_{i:02d}",
                name=f"Innovation Analysis Specialist {i+1}",
                domain=AgentDomain.CREATIVITY,
                capabilities=[
                    AgentCapability(
                        name="innovation_analysis",
                        description=f"Analyze innovation potential using method {i+1}",
                        input_types=["ideas", "market_data"],
                        output_types=["innovation_scores", "market_analysis"],
                        complexity=AgentComplexity.ADVANCED,
                        resource_requirements={"cpu": 0.5, "memory": 0.6, "network": 0.5}
                    )
                ],
                preferred_connections=["idea_generator", "trend_forecaster"],
                threshold_range=(0.7, 0.85),
                expertise_level=0.88 + (i * 0.024),
                collaboration_style="analytical",
                resource_profile={"insight_depth": 0.9 + i * 0.02, "practicality": 0.8 + i * 0.04}
            )
            agents.append(agent)
        
        # Content Editors (5)
        for i in range(5):
            agent = AgentSpecialization(
                id=f"content_editor_{i:02d}",
                name=f"Content Editing Specialist {i+1}",
                domain=AgentDomain.CREATIVITY,
                capabilities=[
                    AgentCapability(
                        name="content_editing",
                        description=f"Edit content using style {i+1}",
                        input_types=["draft_content", "editing_guidelines"],
                        output_types=["edited_content", "editing_feedback"],
                        complexity=AgentComplexity.INTERMEDIATE,
                        resource_requirements={"cpu": 0.3, "memory": 0.5, "network": 0.1}
                    )
                ],
                preferred_connections=["creative_writer", "quality_assurer"],
                threshold_range=(0.55, 0.75),
                expertise_level=0.82 + (i * 0.036),
                collaboration_style="refining",
                resource_profile={"clarity": 0.95 + i * 0.01, "style_consistency": 0.9 + i * 0.02}
            )
            agents.append(agent)
        
        # Trend Forecasters (5)
        for i in range(5):
            agent = AgentSpecialization(
                id=f"trend_forecaster_{i:02d}",
                name=f"Trend Forecasting Specialist {i+1}",
                domain=AgentDomain.CREATIVITY,
                capabilities=[
                    AgentCapability(
                        name="trend_forecasting",
                        description=f"Forecast trends using model {i+1}",
                        input_types=["historical_data", "market_signals"],
                        output_types=["trend_predictions", "confidence_intervals"],
                        complexity=AgentComplexity.EXPERT,
                        resource_requirements={"cpu": 0.7, "memory": 0.8, "network": 0.6}
                    )
                ],
                preferred_connections=["innovation_analyst", "strategic_planner"],
                threshold_range=(0.75, 0.92),
                expertise_level=0.9 + (i * 0.02),
                collaboration_style="predictive",
                resource_profile={"accuracy": 0.85 + i * 0.03, "horizon": 12 + i * 3}
            )
            agents.append(agent)
        
        return agents
    
    def _create_optimization_agents(self) -> List[AgentSpecialization]:
        """Create 20 optimization agents"""
        agents = []
        
        # Performance Optimizers (5)
        for i in range(5):
            agent = AgentSpecialization(
                id=f"perf_optimizer_{i:02d}",
                name=f"Performance Optimization Specialist {i+1}",
                domain=AgentDomain.OPTIMIZATION,
                capabilities=[
                    AgentCapability(
                        name="performance_optimization",
                        description=f"Optimize performance using technique {i+1}",
                        input_types=["system_metrics", "bottlenecks"],
                        output_types=["optimization_plan", "performance_gains"],
                        complexity=AgentComplexity.EXPERT,
                        resource_requirements={"cpu": 0.8, "memory": 0.6, "network": 0.3}
                    )
                ],
                preferred_connections=["system_monitor", "resource_manager"],
                threshold_range=(0.7, 0.9),
                expertise_level=0.88 + (i * 0.024),
                collaboration_style="improving",
                resource_profile={"efficiency_gain": 0.2 + i * 0.1, "risk_level": 0.1 + i * 0.05}
            )
            agents.append(agent)
        
        # Resource Managers (5)
        for i in range(5):
            agent = AgentSpecialization(
                id=f"resource_manager_{i:02d}",
                name=f"Resource Management Specialist {i+1}",
                domain=AgentDomain.OPTIMIZATION,
                capabilities=[
                    AgentCapability(
                        name="resource_management",
                        description=f"Manage resources using strategy {i+1}",
                        input_types=["resource_usage", "allocation_requests"],
                        output_types=["resource_allocation", "utilization_reports"],
                        complexity=AgentComplexity.ADVANCED,
                        resource_requirements={"cpu": 0.4, "memory": 0.5, "network": 0.4}
                    )
                ],
                preferred_connections=["perf_optimizer", "load_balancer"],
                threshold_range=(0.6, 0.85),
                expertise_level=0.85 + (i * 0.03),
                collaboration_style="allocative",
                resource_profile={"utilization": 0.9 + i * 0.02, "fairness": 0.95 + i * 0.01}
            )
            agents.append(agent)
        
        # Load Balancers (5)
        for i in range(5):
            agent = AgentSpecialization(
                id=f"load_balancer_{i:02d}",
                name=f"Load Balancing Specialist {i+1}",
                domain=AgentDomain.OPTIMIZATION,
                capabilities=[
                    AgentCapability(
                        name="load_balancing",
                        description=f"Balance load using algorithm {i+1}",
                        input_types=["workload_data", "capacity_info"],
                        output_types=["load_distribution", "balance_metrics"],
                        complexity=AgentComplexity.INTERMEDIATE,
                        resource_requirements={"cpu": 0.3, "memory": 0.3, "network": 0.8}
                    )
                ],
                preferred_connections=["resource_manager", "task_scheduler"],
                threshold_range=(0.5, 0.75),
                expertise_level=0.8 + (i * 0.04),
                collaboration_style="distributive",
                resource_profile={"balance_quality": 0.95 + i * 0.01, "response_time": 0.1 + i * 0.02}
            )
            agents.append(agent)
        
        # Task Schedulers (5)
        for i in range(5):
            agent = AgentSpecialization(
                id=f"task_scheduler_{i:02d}",
                name=f"Task Scheduling Specialist {i+1}",
                domain=AgentDomain.OPTIMIZATION,
                capabilities=[
                    AgentCapability(
                        name="task_scheduling",
                        description=f"Schedule tasks using algorithm {i+1}",
                        input_types=["task_queue", "priority_data"],
                        output_types=["schedule", "scheduling_metrics"],
                        complexity=AgentComplexity.ADVANCED,
                        resource_requirements={"cpu": 0.6, "memory": 0.4, "network": 0.2}
                    )
                ],
                preferred_connections=["load_balancer", "workflow_optimizer"],
                threshold_range=(0.65, 0.8),
                expertise_level=0.86 + (i * 0.028),
                collaboration_style="organizational",
                resource_profile={"efficiency": 0.9 + i * 0.02, "adaptability": 0.8 + i * 0.04}
            )
            agents.append(agent)
        
        return agents
    
    def _create_monitoring_agents(self) -> List[AgentSpecialization]:
        """Create 20 monitoring agents"""
        agents = []
        
        # System Monitors (5)
        for i in range(5):
            agent = AgentSpecialization(
                id=f"system_monitor_{i:02d}",
                name=f"System Monitoring Specialist {i+1}",
                domain=AgentDomain.MONITORING,
                capabilities=[
                    AgentCapability(
                        name="system_monitoring",
                        description=f"Monitor system using method {i+1}",
                        input_types=["system_logs", "performance_metrics"],
                        output_types=["monitoring_data", "health_status"],
                        complexity=AgentComplexity.INTERMEDIATE,
                        resource_requirements={"cpu": 0.2, "memory": 0.3, "network": 0.5}
                    )
                ],
                preferred_connections=["alert_generator", "perf_optimizer"],
                threshold_range=(0.4, 0.7),
                expertise_level=0.78 + (i * 0.044),
                collaboration_style="observational",
                resource_profile={"coverage": 0.95 + i * 0.01, "frequency": 1.0 + i * 0.5}
            )
            agents.append(agent)
        
        # Alert Generators (5)
        for i in range(5):
            agent = AgentSpecialization(
                id=f"alert_generator_{i:02d}",
                name=f"Alert Generation Specialist {i+1}",
                domain=AgentDomain.MONITORING,
                capabilities=[
                    AgentCapability(
                        name="alert_generation",
                        description=f"Generate alerts using criteria {i+1}",
                        input_types=["monitoring_data", "threshold_rules"],
                        output_types=["alerts", "alert_statistics"],
                        complexity=AgentComplexity.BASIC,
                        resource_requirements={"cpu": 0.1, "memory": 0.2, "network": 0.8}
                    )
                ],
                preferred_connections=["system_monitor", "incident_responder"],
                threshold_range=(0.8, 0.95),
                expertise_level=0.92 + (i * 0.016),
                collaboration_style="responsive",
                resource_profile={"accuracy": 0.98 + i * 0.004, "speed": 0.05 + i * 0.01}
            )
            agents.append(agent)
        
        # Health Checkers (5)
        for i in range(5):
            agent = AgentSpecialization(
                id=f"health_checker_{i:02d}",
                name=f"Health Checking Specialist {i+1}",
                domain=AgentDomain.MONITORING,
                capabilities=[
                    AgentCapability(
                        name="health_checking",
                        description=f"Perform health checks using protocol {i+1}",
                        input_types=["system_components", "check_specs"],
                        output_types=["health_reports", "component_status"],
                        complexity=AgentComplexity.INTERMEDIATE,
                        resource_requirements={"cpu": 0.3, "memory": 0.2, "network": 0.6}
                    )
                ],
                preferred_connections=["system_monitor", "diagnostic_agent"],
                threshold_range=(0.6, 0.8),
                expertise_level=0.84 + (i * 0.032),
                collaboration_style="diagnostic",
                resource_profile={"thoroughness": 0.95 + i * 0.01, "reliability": 0.99 + i * 0.002}
            )
            agents.append(agent)
        
        # Diagnostic Agents (5)
        for i in range(5):
            agent = AgentSpecialization(
                id=f"diagnostic_agent_{i:02d}",
                name=f"Diagnostic Specialist {i+1}",
                domain=AgentDomain.MONITORING,
                capabilities=[
                    AgentCapability(
                        name="diagnostics",
                        description=f"Diagnose issues using method {i+1}",
                        input_types=["error_logs", "symptom_data"],
                        output_types=["diagnosis", "repair_recommendations"],
                        complexity=AgentComplexity.ADVANCED,
                        resource_requirements={"cpu": 0.5, "memory": 0.6, "network": 0.3}
                    )
                ],
                preferred_connections=["health_checker", "incident_responder"],
                threshold_range=(0.7, 0.9),
                expertise_level=0.88 + (i * 0.024),
                collaboration_style="analytical",
                resource_profile={"accuracy": 0.9 + i * 0.02, "speed": 0.8 + i * 0.04}
            )
            agents.append(agent)
        
        return agents
    
    def _create_security_agents(self) -> List[AgentSpecialization]:
        """Create 15 security agents"""
        agents = []
        
        # Security Scanners (5)
        for i in range(5):
            agent = AgentSpecialization(
                id=f"security_scanner_{i:02d}",
                name=f"Security Scanning Specialist {i+1}",
                domain=AgentDomain.SECURITY,
                capabilities=[
                    AgentCapability(
                        name="security_scanning",
                        description=f"Scan for vulnerabilities using tool {i+1}",
                        input_types=["system_configs", "scan_targets"],
                        output_types=["vulnerability_report", "security_metrics"],
                        complexity=AgentComplexity.ADVANCED,
                        resource_requirements={"cpu": 0.6, "memory": 0.4, "network": 0.7}
                    )
                ],
                preferred_connections=["threat_detector", "incident_responder"],
                threshold_range=(0.8, 0.98),
                expertise_level=0.94 + (i * 0.012),
                collaboration_style="protective",
                resource_profile={"thoroughness": 0.98 + i * 0.004, "false_positive_rate": 0.05 - i * 0.01}
            )
            agents.append(agent)
        
        # Threat Detectors (5)
        for i in range(5):
            agent = AgentSpecialization(
                id=f"threat_detector_{i:02d}",
                name=f"Threat Detection Specialist {i+1}",
                domain=AgentDomain.SECURITY,
                capabilities=[
                    AgentCapability(
                        name="threat_detection",
                        description=f"Detect threats using algorithm {i+1}",
                        input_types=["network_traffic", "behavioral_data"],
                        output_types=["threat_alerts", "risk_assessment"],
                        complexity=AgentComplexity.EXPERT,
                        resource_requirements={"cpu": 0.8, "memory": 0.7, "network": 0.9}
                    )
                ],
                preferred_connections=["security_scanner", "incident_responder"],
                threshold_range=(0.85, 0.99),
                expertise_level=0.96 + (i * 0.008),
                collaboration_style="vigilant",
                resource_profile={"sensitivity": 0.95 + i * 0.01, "specificity": 0.9 + i * 0.02}
            )
            agents.append(agent)
        
        # Incident Responders (5)
        for i in range(5):
            agent = AgentSpecialization(
                id=f"incident_responder_{i:02d}",
                name=f"Incident Response Specialist {i+1}",
                domain=AgentDomain.SECURITY,
                capabilities=[
                    AgentCapability(
                        name="incident_response",
                        description=f"Respond to incidents using protocol {i+1}",
                        input_types=["incident_reports", "response_templates"],
                        output_types=["response_actions", "incident_logs"],
                        complexity=AgentComplexity.EXPERT,
                        resource_requirements={"cpu": 0.5, "memory": 0.3, "network": 0.8}
                    )
                ],
                preferred_connections=["threat_detector", "recovery_agent"],
                threshold_range=(0.9, 0.99),
                expertise_level=0.98 + (i * 0.004),
                collaboration_style="reactive",
                resource_profile={"response_time": 0.1 + i * 0.02, "effectiveness": 0.95 + i * 0.01}
            )
            agents.append(agent)
        
        return agents
    
    def _create_coordination_agents(self) -> List[AgentSpecialization]:
        """Create 10 coordination agents"""
        agents = []
        
        # Master Coordinators (3)
        for i in range(3):
            agent = AgentSpecialization(
                id=f"master_coordinator_{i:02d}",
                name=f"Master Coordinator {i+1}",
                domain=AgentDomain.COORDINATION,
                capabilities=[
                    AgentCapability(
                        name="master_coordination",
                        description=f"Coordinate system-wide operations using strategy {i+1}",
                        input_types=["system_state", "coordination_requests"],
                        output_types=["coordination_plan", "system_directives"],
                        complexity=AgentComplexity.EXPERT,
                        resource_requirements={"cpu": 0.4, "memory": 0.8, "network": 0.9}
                    )
                ],
                preferred_connections=["task_scheduler", "resource_manager"],
                threshold_range=(0.85, 0.99),
                expertise_level=0.97 + (i * 0.01),
                collaboration_style="directive",
                resource_profile={"scope": "system", "efficiency": 0.95 + i * 0.0167}
            )
            agents.append(agent)
        
        # Workflow Optimizers (3)
        for i in range(3):
            agent = AgentSpecialization(
                id=f"workflow_optimizer_{i:02d}",
                name=f"Workflow Optimization Specialist {i+1}",
                domain=AgentDomain.COORDINATION,
                capabilities=[
                    AgentCapability(
                        name="workflow_optimization",
                        description=f"Optimize workflows using method {i+1}",
                        input_types=["workflow_data", "performance_metrics"],
                        output_types=["optimized_workflows", "optimization_metrics"],
                        complexity=AgentComplexity.EXPERT,
                        resource_requirements={"cpu": 0.7, "memory": 0.6, "network": 0.5}
                    )
                ],
                preferred_connections=["master_coordinator", "task_scheduler"],
                threshold_range=(0.8, 0.95),
                expertise_level=0.95 + (i * 0.0167),
                collaboration_style="streamlining",
                resource_profile={"improvement_rate": 0.3 + i * 0.1, "adaptability": 0.9 + i * 0.0333}
            )
            agents.append(agent)
        
        # Integration Managers (4)
        for i in range(4):
            agent = AgentSpecialization(
                id=f"integration_manager_{i:02d}",
                name=f"Integration Management Specialist {i+1}",
                domain=AgentDomain.COORDINATION,
                capabilities=[
                    AgentCapability(
                        name="integration_management",
                        description=f"Manage integrations using framework {i+1}",
                        input_types=["integration_specs", "component_data"],
                        output_types=["integration_plan", "compatibility_matrix"],
                        complexity=AgentComplexity.ADVANCED,
                        resource_requirements={"cpu": 0.5, "memory": 0.7, "network": 0.8}
                    )
                ],
                preferred_connections=["workflow_optimizer", "protocol_handler"],
                threshold_range=(0.75, 0.9),
                expertise_level=0.92 + (i * 0.02),
                collaboration_style="integrative",
                resource_profile={"compatibility": 0.95 + i * 0.0125, "scalability": 0.9 + i * 0.025}
            )
            agents.append(agent)
        
        return agents
    
    def _create_specialized_agents(self) -> List[AgentSpecialization]:
        """Create 5 specialized agents for unique tasks"""
        agents = []
        
        # Quantum Computing Specialist
        agent = AgentSpecialization(
            id="quantum_specialist",
            name="Quantum Computing Specialist",
            domain=AgentDomain.SPECIALIZED,
            capabilities=[
                AgentCapability(
                    name="quantum_computing",
                    description="Perform quantum computing operations",
                    input_types=["quantum_circuits", "quantum_algorithms"],
                    output_types=["quantum_results", "quantum_metrics"],
                    complexity=AgentComplexity.EXPERT,
                    resource_requirements={"cpu": 0.3, "memory": 0.2, "network": 0.1, "quantum": 0.9}
                )
            ],
            preferred_connections=["nn_trainer", "algorithm_developer"],
            threshold_range=(0.9, 0.99),
            expertise_level=0.99,
            collaboration_style="quantum",
            resource_profile={"qubit_count": 1000, "coherence_time": 100.0}
        )
        agents.append(agent)
        
        # Blockchain Specialist
        agent = AgentSpecialization(
            id="blockchain_specialist",
            name="Blockchain Specialist",
            domain=AgentDomain.SPECIALIZED,
            capabilities=[
                AgentCapability(
                    name="blockchain_operations",
                    description="Perform blockchain operations and smart contracts",
                    input_types=["transactions", "smart_contracts"],
                    output_types=["blockchain_results", "consensus_data"],
                    complexity=AgentComplexity.EXPERT,
                    resource_requirements={"cpu": 0.4, "memory": 0.3, "network": 0.8}
                )
            ],
            preferred_connections=["security_scanner", "data_validator"],
            threshold_range=(0.85, 0.98),
            expertise_level=0.97,
            collaboration_style="decentralized",
            resource_profile={"throughput": 100, "security_level": 0.99}
        )
        agents.append(agent)
        
        # AR/VR Specialist
        agent = AgentSpecialization(
            id="arvr_specialist",
            name="AR/VR Specialist",
            domain=AgentDomain.SPECIALIZED,
            capabilities=[
                AgentCapability(
                    name="arvr_processing",
                    description="Process AR/VR content and interactions",
                    input_types=["vr_content", "ar_data"],
                    output_types=["immersive_experiences", "interaction_data"],
                    complexity=AgentComplexity.EXPERT,
                    resource_requirements={"cpu": 0.9, "memory": 0.8, "network": 0.7, "gpu": 0.95}
                )
            ],
            preferred_connections=["creative_writer", "data_processor"],
            threshold_range=(0.8, 0.95),
            expertise_level=0.95,
            collaboration_style="immersive",
            resource_profile={"fps": 120, "latency": 0.02}
        )
        agents.append(agent)
        
        # IoT Specialist
        agent = AgentSpecialization(
            id="iot_specialist",
            name="IoT Specialist",
            domain=AgentDomain.SPECIALIZED,
            capabilities=[
                AgentCapability(
                    name="iot_management",
                    description="Manage IoT devices and sensor networks",
                    input_types=["sensor_data", "iot_configs"],
                    output_types=["iot_insights", "device_commands"],
                    complexity=AgentComplexity.ADVANCED,
                    resource_requirements={"cpu": 0.3, "memory": 0.4, "network": 0.9}
                )
            ],
            preferred_connections=["data_ingestor", "system_monitor"],
            threshold_range=(0.7, 0.9),
            expertise_level=0.93,
            collaboration_style="distributed",
            resource_profile={"device_count": 10000, "data_rate": 1000}
        )
        agents.append(agent)
        
        # AI Ethics Specialist
        agent = AgentSpecialization(
            id="ethics_specialist",
            name="AI Ethics Specialist",
            domain=AgentDomain.SPECIALIZED,
            capabilities=[
                AgentCapability(
                    name="ethics_evaluation",
                    description="Evaluate AI decisions and systems for ethical compliance",
                    input_types=["ai_decisions", "ethical_frameworks"],
                    output_types=["ethics_reports", "compliance_scores"],
                    complexity=AgentComplexity.EXPERT,
                    resource_requirements={"cpu": 0.2, "memory": 0.6, "network": 0.3}
                )
            ],
            preferred_connections=["master_coordinator", "decision_maker"],
            threshold_range=(0.95, 0.99),
            expertise_level=0.99,
            collaboration_style="evaluative",
            resource_profile={"thoroughness": 1.0, "objectivity": 0.99}
        )
        agents.append(agent)
        
        return agents
    
    def _add_specialization(self, specialization: AgentSpecialization):
        """Add a specialization to the registry"""
        self.specializations[specialization.id] = specialization
        
        if specialization.domain not in self.domain_mapping:
            self.domain_mapping[specialization.domain] = []
        
        self.domain_mapping[specialization.domain].append(specialization.id)
    
    def get_specialization(self, agent_id: str) -> Optional[AgentSpecialization]:
        """Get specialization by agent ID"""
        return self.specializations.get(agent_id)
    
    def get_specializations_by_domain(self, domain: AgentDomain) -> List[AgentSpecialization]:
        """Get all specializations in a domain"""
        agent_ids = self.domain_mapping.get(domain, [])
        return [self.specializations[agent_id] for agent_id in agent_ids]
    
    def get_domain_distribution(self) -> Dict[AgentDomain, int]:
        """Get distribution of agents across domains"""
        return {domain: len(agent_ids) for domain, agent_ids in self.domain_mapping.items()}
    
    def get_domain_agents(self, domain: AgentDomain) -> List[str]:
        """Get all agent IDs in a domain"""
        return self.domain_mapping.get(domain, [])
    
    def get_all_specializations(self) -> List[AgentSpecialization]:
        """Get all specializations"""
        return list(self.specializations.values())
    
    def get_compatible_connections(self, agent_id: str) -> List[str]:
        """Get compatible connection targets for an agent"""
        specialization = self.get_specialization(agent_id)
        if not specialization:
            return []
        
        # Start with preferred connections
        compatible = set(specialization.preferred_connections)
        
        # Add agents in the same domain
        domain_agents = self.domain_mapping.get(specialization.domain, [])
        compatible.update(domain_agents)
        
        # Add agents with compatible capabilities
        for other_id, other_spec in self.specializations.items():
            if other_id != agent_id:
                # Check for complementary capabilities
                for cap in specialization.capabilities:
                    for other_cap in other_spec.capabilities:
                        if (cap.output_types and other_cap.input_types and
                            any(out_type in other_cap.input_types for out_type in cap.output_types)):
                            compatible.add(other_id)
        
        return list(compatible)
    
    def get_agent_statistics(self) -> Dict[str, Any]:
        """Get statistics about all agent specializations"""
        stats = {
            "total_agents": len(self.specializations),
            "domain_distribution": {},
            "complexity_distribution": {
                "basic": 0,
                "intermediate": 0,
                "advanced": 0,
                "expert": 0
            },
            "average_expertise": 0.0,
            "connection_density": 0.0
        }
        
        # Count domain distribution
        for domain, agent_ids in self.domain_mapping.items():
            stats["domain_distribution"][domain.value] = len(agent_ids)
        
        # Count complexity distribution and calculate average expertise
        total_expertise = 0
        total_possible_connections = 0
        total_actual_connections = 0
        
        for spec in self.specializations.values():
            # Complexity distribution
            for cap in spec.capabilities:
                complexity_level = cap.complexity.value
                stats["complexity_distribution"][complexity_level] += 1
            
            # Expertise
            total_expertise += spec.expertise
            
            # Connections
            total_possible_connections += len(self.specializations) - 1
            total_actual_connections += len(spec.preferred_connections)
        
        stats["average_expertise"] = total_expertise / len(self.specializations)
        stats["connection_density"] = total_actual_connections / total_possible_connections if total_possible_connections > 0 else 0
        
        return stats
    
    def to_json(self) -> str:
        """Export registry to JSON"""
        data = {
            "specializations": {},
            "domain_mapping": {domain.value: agent_ids for domain, agent_ids in self.domain_mapping.items()},
            "statistics": self.get_agent_statistics()
        }
        
        for agent_id, spec in self.specializations.items():
            data["specializations"][agent_id] = {
                "id": spec.id,
                "name": spec.name,
                "domain": spec.domain.value,
                "capabilities": [
                    {
                        "name": cap.name,
                        "description": cap.description,
                        "input_types": cap.input_types,
                        "output_types": cap.output_types,
                        "complexity": cap.complexity.value,
                        "resource_requirements": cap.resource_requirements,
                        "success_rate": cap.success_rate,
                        "processing_time": cap.processing_time
                    }
                    for cap in spec.capabilities
                ],
                "preferred_connections": spec.preferred_connections,
                "threshold_range": spec.threshold_range,
                "expertise_level": spec.expertise_level,
                "collaboration_style": spec.collaboration_style,
                "resource_profile": spec.resource_profile
            }
        
        return json.dumps(data, indent=2)

# Global registry instance
agent_registry = AgentSpecializationRegistry()