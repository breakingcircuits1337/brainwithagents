"""
Performance Monitoring and Analytics for Multi-Agent Systems

This module provides comprehensive performance monitoring and analytics capabilities
for large-scale multi-agent systems. It enables real-time monitoring, historical analysis,
predictive analytics, and performance optimization for systems with 250+ agents.

Key Features:
- Real-time performance monitoring
- Historical data analysis and trending
- Predictive analytics and anomaly detection
- Performance optimization recommendations
- Comprehensive dashboards and reporting
- Agent-level and system-level metrics
- Resource utilization tracking
- Communication efficiency analysis
"""

import asyncio
import time
import threading
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import json
import statistics
import math
from datetime import datetime, timedelta
import logging

# Import existing components (commented out for standalone demo)
# from scalable_multi_agent_system import ScalableAgent, AgentDomain, AgentState
# from multi_agent_communication_optimization import OptimizedCommunicationSystem


class MetricType(Enum):
    """Types of metrics that can be monitored"""
    COUNTER = "counter"          # Incrementing values
    GAUGE = "gauge"             # Current values
    HISTOGRAM = "histogram"     # Distribution of values
    PERCENTILE = "percentile"   # Percentile calculations
    RATE = "rate"              # Rate of change over time
    AVERAGE = "average"        # Moving averages


class AlertSeverity(Enum):
    """Severity levels for alerts"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class MetricDefinition:
    """Definition of a performance metric"""
    name: str
    description: str
    metric_type: MetricType
    unit: str
    tags: Dict[str, str] = field(default_factory=dict)
    thresholds: Dict[str, float] = field(default_factory=dict)  # warning, critical thresholds


@dataclass
class MetricValue:
    """Value of a metric at a specific time"""
    metric_name: str
    value: float
    timestamp: float
    tags: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "metric_name": self.metric_name,
            "value": self.value,
            "timestamp": self.timestamp,
            "tags": self.tags
        }


@dataclass
class Alert:
    """Performance alert"""
    alert_id: str
    metric_name: str
    severity: AlertSeverity
    message: str
    timestamp: float
    value: float
    threshold: float
    tags: Dict[str, str] = field(default_factory=dict)
    resolved: bool = False
    resolved_timestamp: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "alert_id": self.alert_id,
            "metric_name": self.metric_name,
            "severity": self.severity.value,
            "message": self.message,
            "timestamp": self.timestamp,
            "value": self.value,
            "threshold": self.threshold,
            "tags": self.tags,
            "resolved": self.resolved,
            "resolved_timestamp": self.resolved_timestamp
        }


class PerformanceMonitor:
    """Core performance monitoring system for multi-agent systems"""
    
    def __init__(self, retention_period: int = 3600):  # 1 hour retention
        self.retention_period = retention_period
        self.current_time = time.time()
        
        # Metric storage
        self.metric_definitions: Dict[str, MetricDefinition] = {}
        self.metric_values: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.current_values: Dict[str, float] = {}
        
        # Alerting system
        self.alerts: deque = deque(maxlen=1000)
        self.alert_rules: Dict[str, Dict] = {}
        self.alert_handlers: List[Callable] = []
        
        # Aggregation and analysis
        self.aggregated_metrics: Dict[str, Dict] = defaultdict(dict)
        self.trend_analysis: Dict[str, Dict] = defaultdict(dict)
        
        # Background tasks
        self.monitoring_active = False
        self.analysis_thread = None
        self.cleanup_thread = None
        
        # Initialize standard metrics
        self._initialize_standard_metrics()
        
    def _initialize_standard_metrics(self):
        """Initialize standard performance metrics"""
        standard_metrics = [
            MetricDefinition(
                name="agent_count",
                description="Total number of active agents",
                metric_type=MetricType.GAUGE,
                unit="count",
                thresholds={"warning": 200, "critical": 180}
            ),
            MetricDefinition(
                name="system_load",
                description="Average system load across all agents",
                metric_type=MetricType.GAUGE,
                unit="percentage",
                thresholds={"warning": 0.8, "critical": 0.9}
            ),
            MetricDefinition(
                name="task_throughput",
                description="Number of tasks processed per second",
                metric_type=MetricType.RATE,
                unit="tasks/sec",
                thresholds={"warning": 10, "critical": 5}
            ),
            MetricDefinition(
                name="response_time",
                description="Average response time for task processing",
                metric_type=MetricType.AVERAGE,
                unit="seconds",
                thresholds={"warning": 2.0, "critical": 5.0}
            ),
            MetricDefinition(
                name="error_rate",
                description="Percentage of failed tasks",
                metric_type=MetricType.PERCENTILE,
                unit="percentage",
                thresholds={"warning": 5.0, "critical": 10.0}
            ),
            MetricDefinition(
                name="communication_efficiency",
                description="Efficiency of inter-agent communication",
                metric_type=MetricType.GAUGE,
                unit="ratio",
                thresholds={"warning": 0.7, "critical": 0.5}
            ),
            MetricDefinition(
                name="resource_utilization",
                description="Average resource utilization across agents",
                metric_type=MetricType.GAUGE,
                unit="percentage",
                thresholds={"warning": 0.85, "critical": 0.95}
            ),
            MetricDefinition(
                name="agent_health",
                description="Average health score of all agents",
                metric_type=MetricType.GAUGE,
                unit="score",
                thresholds={"warning": 0.6, "critical": 0.4}
            )
        ]
        
        for metric in standard_metrics:
            self.register_metric(metric)
    
    def register_metric(self, metric_def: MetricDefinition):
        """Register a new metric definition"""
        self.metric_definitions[metric_def.name] = metric_def
        self.current_values[metric_def.name] = 0.0
        
        # Create alert rule if thresholds are defined
        if metric_def.thresholds:
            self.alert_rules[metric_def.name] = {
                "warning": metric_def.thresholds.get("warning"),
                "critical": metric_def.thresholds.get("critical")
            }
    
    def record_metric(self, metric_name: str, value: float, 
                     tags: Optional[Dict[str, str]] = None):
        """Record a metric value"""
        if metric_name not in self.metric_definitions:
            logging.warning(f"Unknown metric: {metric_name}")
            return
        
        timestamp = time.time()
        metric_def = self.metric_definitions[metric_name]
        
        # Store metric value
        metric_value = MetricValue(
            metric_name=metric_name,
            value=value,
            timestamp=timestamp,
            tags=tags or {}
        )
        
        self.metric_values[metric_name].append(metric_value)
        self.current_values[metric_name] = value
        
        # Check for alerts
        self._check_alerts(metric_name, value, timestamp, tags or {})
    
    def _check_alerts(self, metric_name: str, value: float, 
                      timestamp: float, tags: Dict[str, str]):
        """Check if metric value triggers any alerts"""
        if metric_name not in self.alert_rules:
            return
        
        rules = self.alert_rules[metric_name]
        
        # Check critical threshold
        if rules.get("critical") and value >= rules["critical"]:
            self._create_alert(
                metric_name=metric_name,
                severity=AlertSeverity.CRITICAL,
                message=f"Critical threshold exceeded for {metric_name}: {value} >= {rules['critical']}",
                value=value,
                threshold=rules["critical"],
                timestamp=timestamp,
                tags=tags
            )
        
        # Check warning threshold
        elif rules.get("warning") and value >= rules["warning"]:
            self._create_alert(
                metric_name=metric_name,
                severity=AlertSeverity.WARNING,
                message=f"Warning threshold exceeded for {metric_name}: {value} >= {rules['warning']}",
                value=value,
                threshold=rules["warning"],
                timestamp=timestamp,
                tags=tags
            )
    
    def _create_alert(self, metric_name: str, severity: AlertSeverity, 
                      message: str, value: float, threshold: float,
                      timestamp: float, tags: Dict[str, str]):
        """Create and process a new alert"""
        alert = Alert(
            alert_id=f"alert_{int(timestamp * 1000)}",
            metric_name=metric_name,
            severity=severity,
            message=message,
            timestamp=timestamp,
            value=value,
            threshold=threshold,
            tags=tags
        )
        
        self.alerts.append(alert)
        
        # Notify alert handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logging.error(f"Error in alert handler: {e}")
    
    def add_alert_handler(self, handler: Callable[[Alert], None]):
        """Add an alert handler function"""
        self.alert_handlers.append(handler)
    
    def get_metric_value(self, metric_name: str, 
                         time_range: Optional[Tuple[float, float]] = None) -> List[MetricValue]:
        """Get metric values within a time range"""
        if metric_name not in self.metric_values:
            return []
        
        values = list(self.metric_values[metric_name])
        
        if time_range:
            start_time, end_time = time_range
            values = [v for v in values if start_time <= v.timestamp <= end_time]
        
        return values
    
    def get_current_metrics(self) -> Dict[str, float]:
        """Get current values of all metrics"""
        return self.current_values.copy()
    
    def get_alerts(self, severity: Optional[AlertSeverity] = None,
                   resolved: Optional[bool] = None) -> List[Alert]:
        """Get alerts with optional filtering"""
        alerts = list(self.alerts)
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        if resolved is not None:
            alerts = [a for a in alerts if a.resolved == resolved]
        
        return alerts
    
    def start_monitoring(self):
        """Start background monitoring tasks"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        
        # Start analysis thread
        self.analysis_thread = threading.Thread(
            target=self._analysis_loop,
            daemon=True
        )
        self.analysis_thread.start()
        
        # Start cleanup thread
        self.cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            daemon=True
        )
        self.cleanup_thread.start()
        
        logging.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop background monitoring tasks"""
        self.monitoring_active = False
        
        if self.analysis_thread:
            self.analysis_thread.join(timeout=5)
        if self.cleanup_thread:
            self.cleanup_thread.join(timeout=5)
        
        logging.info("Performance monitoring stopped")
    
    def _analysis_loop(self):
        """Background loop for metric analysis"""
        while self.monitoring_active:
            try:
                self._perform_metric_analysis()
                self._update_aggregations()
                time.sleep(10)  # Analyze every 10 seconds
            except Exception as e:
                logging.error(f"Error in analysis loop: {e}")
                time.sleep(30)
    
    def _cleanup_loop(self):
        """Background loop for data cleanup"""
        while self.monitoring_active:
            try:
                self._cleanup_old_data()
                time.sleep(60)  # Cleanup every minute
            except Exception as e:
                logging.error(f"Error in cleanup loop: {e}")
                time.sleep(120)
    
    def _perform_metric_analysis(self):
        """Perform analysis on current metrics"""
        current_time = time.time()
        
        for metric_name, values in self.metric_values.items():
            if len(values) < 2:
                continue
            
            # Calculate trend
            recent_values = [v.value for v in values if current_time - v.timestamp < 300]  # Last 5 minutes
            if len(recent_values) >= 2:
                trend = self._calculate_trend(recent_values)
                self.trend_analysis[metric_name]["trend"] = trend
                self.trend_analysis[metric_name]["timestamp"] = current_time
            
            # Calculate statistics
            all_values = [v.value for v in values]
            stats = {
                "count": len(all_values),
                "min": min(all_values),
                "max": max(all_values),
                "mean": statistics.mean(all_values),
                "median": statistics.median(all_values),
                "std_dev": statistics.stdev(all_values) if len(all_values) > 1 else 0,
            "percentile_95": sorted_values[int(len(sorted_values) * 0.95)] if sorted_values else 0,
            "timestamp": current_time
            }
            
            self.trend_analysis[metric_name]["statistics"] = stats
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend coefficient using linear regression"""
        if len(values) < 2:
            return 0.0
        
        x = list(range(len(values)))
        y = values
        
        # Simple linear regression
        x_mean = statistics.mean(x)
        y_mean = statistics.mean(y)
        
        numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(len(x)))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(len(x)))
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    def _update_aggregations(self):
        """Update aggregated metrics"""
        current_time = time.time()
        
        # Calculate 1-minute, 5-minute, and 15-minute aggregations
        for window in [60, 300, 900]:
            for metric_name, values in self.metric_values.items():
                window_values = [
                    v.value for v in values
                    if current_time - v.timestamp <= window
                ]
                
                if window_values:
                    aggregation = {
                        "window": window,
                        "count": len(window_values),
                        "sum": sum(window_values),
                        "avg": statistics.mean(window_values),
                        "min": min(window_values),
                        "max": max(window_values),
                        "timestamp": current_time
                    }
                    
                    self.aggregated_metrics[metric_name][f"{window}s"] = aggregation
    
    def _cleanup_old_data(self):
        """Clean up old metric data"""
        cutoff_time = time.time() - self.retention_period
        
        for metric_name in list(self.metric_values.keys()):
            # Keep only recent values
            recent_values = deque(
                [v for v in self.metric_values[metric_name] if v.timestamp > cutoff_time],
                maxlen=10000
            )
            self.metric_values[metric_name] = recent_values
        
        # Clean up old alerts
        recent_alerts = deque(
            [a for a in self.alerts if a.timestamp > cutoff_time],
            maxlen=1000
        )
        self.alerts = recent_alerts
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        current_time = time.time()
        
        # Current metrics
        current_metrics = self.get_current_metrics()
        
        # Recent alerts
        recent_alerts = [
            alert for alert in self.alerts
            if current_time - alert.timestamp < 3600 and not alert.resolved
        ]
        
        # System health summary
        health_score = self._calculate_system_health()
        
        # Performance trends
        trends = {}
        for metric_name, analysis in self.trend_analysis.items():
            if "trend" in analysis:
                trends[metric_name] = {
                    "trend": analysis["trend"],
                    "direction": "increasing" if analysis["trend"] > 0.01 else "decreasing" if analysis["trend"] < -0.01 else "stable"
                }
        
        return {
            "timestamp": current_time,
            "system_health": health_score,
            "current_metrics": current_metrics,
            "active_alerts": len(recent_alerts),
            "alert_summary": {
                "critical": len([a for a in recent_alerts if a.severity == AlertSeverity.CRITICAL]),
                "warning": len([a for a in recent_alerts if a.severity == AlertSeverity.WARNING]),
                "info": len([a for a in recent_alerts if a.severity == AlertSeverity.INFO])
            },
            "performance_trends": trends,
            "metric_count": len(self.metric_definitions),
            "data_points": sum(len(values) for values in self.metric_values.values())
        }
    
    def _calculate_system_health(self) -> float:
        """Calculate overall system health score"""
        health_factors = []
        
        # Agent health
        agent_health = self.current_values.get("agent_health", 0.5)
        health_factors.append(agent_health)
        
        # System load (inverse)
        system_load = self.current_values.get("system_load", 0.5)
        health_factors.append(1.0 - system_load)
        
        # Error rate (inverse)
        error_rate = self.current_values.get("error_rate", 0.0)
        health_factors.append(1.0 - min(error_rate / 100.0, 1.0))
        
        # Response time (inverse, normalized)
        response_time = self.current_values.get("response_time", 1.0)
        health_factors.append(1.0 / (1.0 + response_time))
        
        # Communication efficiency
        comm_efficiency = self.current_values.get("communication_efficiency", 0.7)
        health_factors.append(comm_efficiency)
        
        return statistics.mean(health_factors) if health_factors else 0.0


class MultiAgentAnalytics:
    """Advanced analytics for multi-agent system performance"""
    
    def __init__(self, monitor: PerformanceMonitor):
        self.monitor = monitor
        self.predictive_models = {}
        self.baseline_metrics = {}
        self.anomaly_detector = AnomalyDetector()
        
    def collect_agent_metrics(self, agents: List[ScalableAgent]):
        """Collect metrics from individual agents"""
        if not agents:
            return
        
        # Calculate agent-level metrics
        total_agents = len(agents)
        active_agents = len([a for a in agents if a.state == AgentState.ACTIVE])
        busy_agents = len([a for a in agents if a.state == AgentState.BUSY])
        overloaded_agents = len([a for a in agents if a.state == AgentState.OVERLOADED])
        
        # Record agent metrics
        self.monitor.record_metric("agent_count", total_agents)
        self.monitor.record_metric("active_agents", active_agents)
        self.monitor.record_metric("busy_agents", busy_agents)
        self.monitor.record_metric("overloaded_agents", overloaded_agents)
        
        # Calculate average metrics
        if total_agents > 0:
            avg_load = sum(a.metrics.load_factor for a in agents) / total_agents
            avg_health = sum(a.get_health_score() for a in agents) / total_agents
            avg_response_time = sum(a.metrics.average_response_time for a in agents) / total_agents
            
            self.monitor.record_metric("system_load", avg_load)
            self.monitor.record_metric("agent_health", avg_health)
            self.monitor.record_metric("response_time", avg_response_time)
            
            # Calculate domain distribution
            domain_counts = defaultdict(int)
            for agent in agents:
                domain_counts[agent.domain] += 1
            
            for domain, count in domain_counts.items():
                self.monitor.record_metric(
                    f"domain_{domain.value}_count",
                    count,
                    tags={"domain": domain.value}
                )
    
    def collect_communication_metrics(self, comm_system: OptimizedCommunicationSystem):
        """Collect communication system metrics"""
        if not comm_system:
            return
        
        # Get communication stats
        routing_stats = comm_system.message_router.get_routing_stats()
        optimization_stats = comm_system.message_optimizer.get_optimization_stats()
        
        # Record communication metrics
        self.monitor.record_metric(
            "communication_efficiency",
            routing_stats.get("success_rate", 0.0)
        )
        
        self.monitor.record_metric(
            "message_compression_ratio",
            optimization_stats.get("average_compression_ratio", 1.0)
        )
        
        self.monitor.record_metric(
            "registered_agents",
            routing_stats.get("registered_agents", 0)
        )
        
        self.monitor.record_metric(
            "network_connections",
            routing_stats.get("network_connections", 0)
        )
    
    def analyze_performance_patterns(self) -> Dict[str, Any]:
        """Analyze performance patterns and trends"""
        current_time = time.time()
        
        # Get recent performance data
        analysis_window = current_time - 1800  # Last 30 minutes
        
        patterns = {
            "timestamp": current_time,
            "performance_patterns": {},
            "anomalies": [],
            "predictions": {},
            "recommendations": []
        }
        
        # Analyze each metric
        for metric_name in self.monitor.metric_definitions.keys():
            values = self.monitor.get_metric_value(metric_name, (analysis_window, current_time))
            
            if len(values) < 5:
                continue
            
            # Extract value series
            value_series = [v.value for v in values]
            timestamps = [v.timestamp for v in values]
            
            # Detect patterns
            pattern = self._detect_pattern(value_series, timestamps)
            patterns["performance_patterns"][metric_name] = pattern
            
            # Detect anomalies
            anomalies = self.anomaly_detector.detect_anomalies(value_series, timestamps)
            for anomaly in anomalies:
                patterns["anomalies"].append({
                    "metric": metric_name,
                    "timestamp": anomaly["timestamp"],
                    "value": anomaly["value"],
                    "severity": anomaly["severity"]
                })
            
            # Generate predictions
            prediction = self._predict_next_value(value_series, timestamps)
            if prediction:
                patterns["predictions"][metric_name] = prediction
        
        # Generate recommendations
        patterns["recommendations"] = self._generate_recommendations(patterns)
        
        return patterns
    
    def _detect_pattern(self, values: List[float], timestamps: List[float]) -> Dict[str, Any]:
        """Detect performance patterns in metric data"""
        if len(values) < 5:
            return {"pattern": "insufficient_data"}
        
        # Calculate basic statistics
        mean_val = statistics.mean(values)
        std_val = statistics.stdev(values) if len(values) > 1 else 0
        
        # Detect trend
        trend = self._calculate_trend_coefficient(values)
        
        # Detect volatility
        volatility = std_val / mean_val if mean_val != 0 else 0
        
        # Detect periodicity (simple check)
        periodicity = self._detect_periodicity(values, timestamps)
        
        return {
            "pattern": self._classify_pattern(trend, volatility, periodicity),
            "trend": trend,
            "volatility": volatility,
            "periodicity": periodicity,
            "mean": mean_val,
            "std_dev": std_val
        }
    
    def _calculate_trend_coefficient(self, values: List[float]) -> float:
        """Calculate trend coefficient"""
        if len(values) < 2:
            return 0.0
        
        x = list(range(len(values)))
        y = values
        
        x_mean = statistics.mean(x)
        y_mean = statistics.mean(y)
        
        numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(len(x)))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(len(x)))
        
        return numerator / denominator if denominator != 0 else 0.0
    
    def _detect_periodicity(self, values: List[float], timestamps: List[float]) -> Optional[float]:
        """Simple periodicity detection"""
        if len(values) < 10:
            return None
        
        # Check for repeating patterns using autocorrelation
        max_lag = min(len(values) // 2, 20)
        correlations = []
        
        for lag in range(1, max_lag):
            if lag >= len(values):
                break
            
            # Calculate autocorrelation
            original = values[:-lag]
            shifted = values[lag:]
            
            if len(original) != len(shifted):
                continue
            
            # Calculate correlation manually
            if len(original) < 2:
                continue
            
            mean_orig = statistics.mean(original)
            mean_shift = statistics.mean(shifted)
            
            numerator = sum((o - mean_orig) * (s - mean_shift) for o, s in zip(original, shifted))
            denom_orig = math.sqrt(sum((o - mean_orig) ** 2 for o in original))
            denom_shift = math.sqrt(sum((s - mean_shift) ** 2 for s in shifted))
            
            if denom_orig == 0 or denom_shift == 0:
                correlation = 0
            else:
                correlation = numerator / (denom_orig * denom_shift)
            
            correlations.append((lag, correlation))
        
        # Find peak correlation
        if correlations:
            best_lag, best_corr = max(correlations, key=lambda x: x[1])
            if best_corr > 0.5:  # Significant correlation
                return best_lag
        
        return None
    
    def _classify_pattern(self, trend: float, volatility: float, 
                         periodicity: Optional[float]) -> str:
        """Classify the pattern type"""
        if periodicity:
            return "periodic"
        elif abs(trend) > 0.1:
            return "increasing" if trend > 0 else "decreasing"
        elif volatility > 0.3:
            return "volatile"
        else:
            return "stable"
    
    def _predict_next_value(self, values: List[float], timestamps: List[float]) -> Optional[float]:
        """Simple linear prediction for next value"""
        if len(values) < 3:
            return None
        
        # Use simple linear regression on recent values
        recent_values = values[-10:]  # Last 10 values
        recent_timestamps = timestamps[-10:]
        
        if len(recent_values) < 3:
            return None
        
        # Normalize timestamps
        min_time = min(recent_timestamps)
        normalized_times = [t - min_time for t in recent_timestamps]
        
        # Linear regression
        x = normalized_times
        y = recent_values
        
        x_mean = statistics.mean(x)
        y_mean = statistics.mean(y)
        
        numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(len(x)))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(len(x)))
        
        if denominator == 0:
            return None
        
        slope = numerator / denominator
        intercept = y_mean - slope * x_mean
        
        # Predict next value (assuming same time interval)
        if len(normalized_times) >= 2:
            time_interval = normalized_times[-1] - normalized_times[-2]
            next_time = normalized_times[-1] + time_interval
            prediction = slope * next_time + intercept
            return max(0, prediction)  # Ensure non-negative
        
        return None
    
    def _generate_recommendations(self, patterns: Dict[str, Any]) -> List[str]:
        """Generate performance optimization recommendations"""
        recommendations = []
        
        # Analyze patterns and generate recommendations
        for metric_name, pattern in patterns["performance_patterns"].items():
            pattern_type = pattern.get("pattern", "unknown")
            
            if pattern_type == "increasing" and "load" in metric_name:
                recommendations.append(f"Consider scaling resources for {metric_name} - showing increasing trend")
            elif pattern_type == "decreasing" and "health" in metric_name:
                recommendations.append(f"Investigate declining health in {metric_name} - immediate attention needed")
            elif pattern_type == "volatile":
                recommendations.append(f"High volatility detected in {metric_name} - consider stabilization measures")
            elif pattern_type == "periodic":
                recommendations.append(f"Periodic pattern detected in {metric_name} - optimize for peak loads")
        
        # Check for anomalies
        if patterns["anomalies"]:
            critical_anomalies = [a for a in patterns["anomalies"] if a.get("severity") == "critical"]
            if critical_anomalies:
                recommendations.append(f"Critical anomalies detected - immediate investigation required")
        
        # Check predictions
        for metric_name, prediction in patterns["predictions"].items():
            if prediction and "load" in metric_name and prediction > 0.8:
                recommendations.append(f"High load predicted for {metric_name} - prepare scaling measures")
        
        return recommendations[:5]  # Return top 5 recommendations


class AnomalyDetector:
    """Simple anomaly detection for performance metrics"""
    
    def __init__(self, sensitivity: float = 2.0):
        self.sensitivity = sensitivity  # Number of standard deviations
        self.baseline_window = 100  # Number of points for baseline
    
    def detect_anomalies(self, values: List[float], timestamps: List[float]) -> List[Dict[str, Any]]:
        """Detect anomalies in metric values"""
        if len(values) < self.baseline_window:
            return []
        
        anomalies = []
        
        # Calculate baseline statistics from recent values
        baseline_values = values[-self.baseline_window:]
        baseline_mean = statistics.mean(baseline_values)
        baseline_std = statistics.stdev(baseline_values) if len(baseline_values) > 1 else 0
        
        if baseline_std == 0:
            return []
        
        # Check each value for anomalies
        for i, (value, timestamp) in enumerate(zip(values, timestamps)):
            # Skip baseline period
            if i < len(values) - self.baseline_window:
                continue
            
            # Calculate z-score
            z_score = abs(value - baseline_mean) / baseline_std
            
            # Check if anomaly
            if z_score > self.sensitivity:
                severity = "critical" if z_score > 3 * self.sensitivity else "warning"
                
                anomalies.append({
                    "timestamp": timestamp,
                    "value": value,
                    "z_score": z_score,
                    "severity": severity,
                    "baseline_mean": baseline_mean,
                    "baseline_std": baseline_std
                })
        
        return anomalies


class PerformanceDashboard:
    """Real-time performance dashboard for multi-agent systems"""
    
    def __init__(self, monitor: PerformanceMonitor, analytics: MultiAgentAnalytics):
        self.monitor = monitor
        self.analytics = analytics
        self.dashboard_data = defaultdict(dict)
        self.update_interval = 5  # seconds
        
    async def start_dashboard(self):
        """Start the dashboard update loop"""
        print("📊 Starting Performance Dashboard...")
        print("   Press Ctrl+C to stop the dashboard")
        
        try:
            while True:
                await self._update_dashboard()
                await self._display_dashboard()
                await asyncio.sleep(self.update_interval)
        except KeyboardInterrupt:
            print("\n📊 Dashboard stopped")
    
    async def _update_dashboard(self):
        """Update dashboard data"""
        # Get current metrics
        current_metrics = self.monitor.get_current_metrics()
        
        # Get performance report
        performance_report = self.monitor.get_performance_report()
        
        # Get analytics
        patterns = self.analytics.analyze_performance_patterns()
        
        # Update dashboard data
        self.dashboard_data["current_metrics"] = current_metrics
        self.dashboard_data["performance_report"] = performance_report
        self.dashboard_data["patterns"] = patterns
        self.dashboard_data["last_update"] = time.time()
    
    async def _display_dashboard(self):
        """Display the dashboard"""
        # Clear screen (simple approach)
        print("\033[2J\033[H", end="")
        
        print("🤖 MULTI-AGENT SYSTEM PERFORMANCE DASHBOARD")
        print("=" * 60)
        
        # System health
        health_score = self.dashboard_data["performance_report"].get("system_health", 0.0)
        health_bar = "█" * int(health_score * 20)
        print(f"System Health: [{health_bar:<20}] {health_score:.1%}")
        
        # Current metrics
        print(f"\n📈 Current Metrics:")
        metrics = self.dashboard_data["current_metrics"]
        for metric_name, value in metrics.items():
            print(f"  {metric_name:<25}: {value:>10.3f}")
        
        # Alerts
        alerts = self.monitor.get_alerts(resolved=False)
        if alerts:
            print(f"\n🚨 Active Alerts ({len(alerts)}):")
            for alert in alerts[-5:]:  # Show last 5 alerts
                severity_icon = {"critical": "🔴", "warning": "🟡", "info": "🔵"}.get(alert.severity.value, "⚪")
                print(f"  {severity_icon} {alert.metric_name}: {alert.message}")
        
        # Patterns
        patterns = self.dashboard_data.get("patterns", {})
        if patterns.get("performance_patterns"):
            print(f"\n🔍 Performance Patterns:")
            for metric_name, pattern in patterns["performance_patterns"].items()[:5]:
                pattern_icon = {
                    "increasing": "📈", "decreasing": "📉", "stable": "➡️",
                    "volatile": "🌊", "periodic": "🔄"
                }.get(pattern.get("pattern", "unknown"), "❓")
                print(f"  {pattern_icon} {metric_name:<20}: {pattern.get('pattern', 'unknown')}")
        
        # Recommendations
        recommendations = patterns.get("recommendations", [])
        if recommendations:
            print(f"\n💡 Recommendations:")
            for rec in recommendations[:3]:
                print(f"  • {rec}")
        
        # System info
        report = self.dashboard_data["performance_report"]
        print(f"\n📊 System Summary:")
        print(f"  Active Agents: {metrics.get('agent_count', 0)}")
        print(f"  Total Alerts: {report.get('active_alerts', 0)}")
        print(f"  Metrics Tracked: {report.get('metric_count', 0)}")
        print(f"  Data Points: {report.get('data_points', 0)}")
        
        print(f"\n⏰ Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)


# Main execution
async def main():
    """Main function to demonstrate performance monitoring and analytics"""
    print("🔍 Performance Monitoring and Analytics for Multi-Agent Systems")
    print("=" * 70)
    
    # Create monitoring system
    monitor = PerformanceMonitor()
    monitor.start_monitoring()
    
    # Create analytics system
    analytics = MultiAgentAnalytics(monitor)
    
    # Create dashboard
    dashboard = PerformanceDashboard(monitor, analytics)
    
    # Add custom alert handler
    def handle_alert(alert):
        if alert.severity == AlertSeverity.CRITICAL:
            print(f"🚨 CRITICAL ALERT: {alert.message}")
        elif alert.severity == AlertSeverity.WARNING:
            print(f"⚠️ WARNING ALERT: {alert.message}")
    
    monitor.add_alert_handler(handle_alert)
    
    # Simulate some metrics for demonstration
    print("\n📊 Simulating system metrics...")
    
    for i in range(50):
        # Simulate varying metrics
        agent_count = 200 + (i % 50)
        system_load = 0.3 + 0.4 * (i % 20) / 20
        task_throughput = 15 + 10 * (i % 10) / 10
        response_time = 0.5 + 2.0 * (i % 15) / 15
        error_rate = 1.0 + 4.0 * (i % 8) / 8
        
        # Record metrics
        monitor.record_metric("agent_count", agent_count)
        monitor.record_metric("system_load", system_load)
        monitor.record_metric("task_throughput", task_throughput)
        monitor.record_metric("response_time", response_time)
        monitor.record_metric("error_rate", error_rate)
        
        # Simulate some alerts
        if i % 15 == 0:
            monitor.record_metric("system_load", 0.95)  # Trigger critical alert
        
        await asyncio.sleep(0.1)
    
    # Generate performance report
    print(f"\n📋 Generating Performance Report...")
    report = monitor.get_performance_report()
    
    print(f"System Health: {report['system_health']:.2%}")
    print(f"Active Alerts: {report['active_alerts']}")
    print(f"Metrics Tracked: {report['metric_count']}")
    print(f"Data Points: {report['data_points']}")
    
    # Analyze performance patterns
    print(f"\n🔍 Analyzing Performance Patterns...")
    patterns = analytics.analyze_performance_patterns()
    
    print(f"Patterns Detected: {len(patterns['performance_patterns'])}")
    print(f"Anomalies Found: {len(patterns['anomalies'])}")
    print(f"Predictions Generated: {len(patterns['predictions'])}")
    print(f"Recommendations: {len(patterns['recommendations'])}")
    
    # Display recommendations
    if patterns['recommendations']:
        print(f"\n💡 Optimization Recommendations:")
        for rec in patterns['recommendations']:
            print(f"  • {rec}")
    
    # Start dashboard (commented out for non-interactive demo)
    # await dashboard.start_dashboard()
    
    # Cleanup
    monitor.stop_monitoring()
    print(f"\n🏁 Performance Monitoring Demo Complete")


if __name__ == "__main__":
    asyncio.run(main())