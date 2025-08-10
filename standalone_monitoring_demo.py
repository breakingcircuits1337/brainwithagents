"""
Standalone Performance Monitoring Demonstration

This is a simplified standalone demonstration that showcases the key concepts
of performance monitoring and analytics for multi-agent systems.
"""

import asyncio
import time
import random
import json
import math
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import statistics
from datetime import datetime, timedelta
import logging


class MetricType(Enum):
    """Types of metrics that can be monitored"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    PERCENTILE = "percentile"
    RATE = "rate"
    AVERAGE = "average"


class AlertSeverity(Enum):
    """Severity levels for alerts"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class MetricValue:
    """Value of a metric at a specific time"""
    metric_name: str
    value: float
    timestamp: float
    tags: Dict[str, str] = field(default_factory=dict)


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
    resolved: bool = False


class PerformanceMonitor:
    """Core performance monitoring system"""
    
    def __init__(self, retention_period: int = 3600):
        self.retention_period = retention_period
        self.metric_values: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.current_values: Dict[str, float] = {}
        self.alerts: deque = deque(maxlen=1000)
        self.alert_rules: Dict[str, Dict] = {}
        
        # Initialize standard metrics
        self._initialize_standard_metrics()
    
    def _initialize_standard_metrics(self):
        """Initialize standard performance metrics"""
        standard_metrics = [
            ("agent_count", "Total number of active agents", MetricType.GAUGE, {"warning": 200, "critical": 180}),
            ("system_load", "Average system load", MetricType.GAUGE, {"warning": 0.8, "critical": 0.9}),
            ("task_throughput", "Tasks processed per second", MetricType.RATE, {"warning": 10, "critical": 5}),
            ("response_time", "Average response time", MetricType.AVERAGE, {"warning": 2.0, "critical": 5.0}),
            ("error_rate", "Percentage of failed tasks", MetricType.PERCENTILE, {"warning": 5.0, "critical": 10.0})
        ]
        
        for name, desc, mtype, thresholds in standard_metrics:
            self.alert_rules[name] = thresholds
    
    def record_metric(self, metric_name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record a metric value"""
        timestamp = time.time()
        
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
        self._check_alerts(metric_name, value, timestamp)
    
    def _check_alerts(self, metric_name: str, value: float, timestamp: float):
        """Check if metric value triggers any alerts"""
        if metric_name not in self.alert_rules:
            return
        
        rules = self.alert_rules[metric_name]
        
        # Check thresholds
        if rules.get("critical") and value >= rules["critical"]:
            self._create_alert(metric_name, AlertSeverity.CRITICAL, value, rules["critical"], timestamp)
        elif rules.get("warning") and value >= rules["warning"]:
            self._create_alert(metric_name, AlertSeverity.WARNING, value, rules["warning"], timestamp)
    
    def _create_alert(self, metric_name: str, severity: AlertSeverity, value: float, threshold: float, timestamp: float):
        """Create and process a new alert"""
        alert = Alert(
            alert_id=f"alert_{int(timestamp * 1000)}",
            metric_name=metric_name,
            severity=severity,
            message=f"{severity.value.title()} threshold exceeded for {metric_name}: {value} >= {threshold}",
            timestamp=timestamp,
            value=value,
            threshold=threshold
        )
        
        self.alerts.append(alert)
        print(f"🚨 {alert.message}")
    
    def get_current_metrics(self) -> Dict[str, float]:
        """Get current values of all metrics"""
        return self.current_values.copy()
    
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
            "metric_count": len(self.alert_rules),
            "data_points": sum(len(values) for values in self.metric_values.values())
        }
    
    def _calculate_system_health(self) -> float:
        """Calculate overall system health score"""
        health_factors = []
        
        # System load (inverse)
        system_load = self.current_values.get("system_load", 0.5)
        health_factors.append(1.0 - system_load)
        
        # Error rate (inverse)
        error_rate = self.current_values.get("error_rate", 0.0)
        health_factors.append(1.0 - min(error_rate / 100.0, 1.0))
        
        # Response time (inverse, normalized)
        response_time = self.current_values.get("response_time", 1.0)
        health_factors.append(1.0 / (1.0 + response_time))
        
        return statistics.mean(health_factors) if health_factors else 0.0


class AnomalyDetector:
    """Simple anomaly detection for performance metrics"""
    
    def __init__(self, sensitivity: float = 2.0):
        self.sensitivity = sensitivity
        self.baseline_window = 100
    
    def detect_anomalies(self, values: List[float], timestamps: List[float]) -> List[Dict[str, Any]]:
        """Detect anomalies in metric values"""
        if len(values) < self.baseline_window:
            return []
        
        anomalies = []
        
        # Calculate baseline statistics
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


class MultiAgentAnalytics:
    """Advanced analytics for multi-agent system performance"""
    
    def __init__(self, monitor: PerformanceMonitor):
        self.monitor = monitor
        self.anomaly_detector = AnomalyDetector()
    
    def analyze_performance_patterns(self) -> Dict[str, Any]:
        """Analyze performance patterns and trends"""
        current_time = time.time()
        analysis_window = current_time - 1800  # Last 30 minutes
        
        patterns = {
            "timestamp": current_time,
            "performance_patterns": {},
            "anomalies": [],
            "recommendations": []
        }
        
        # Analyze each metric
        for metric_name in self.monitor.alert_rules.keys():
            values = self.monitor.metric_values.get(metric_name, deque())
            values = [v for v in values if analysis_window <= v.timestamp <= current_time]
            
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
        
        return {
            "pattern": self._classify_pattern(trend, volatility),
            "trend": trend,
            "volatility": volatility,
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
    
    def _classify_pattern(self, trend: float, volatility: float) -> str:
        """Classify the pattern type"""
        if abs(trend) > 0.1:
            return "increasing" if trend > 0 else "decreasing"
        elif volatility > 0.3:
            return "volatile"
        else:
            return "stable"
    
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
        
        # Check for anomalies
        if patterns["anomalies"]:
            critical_anomalies = [a for a in patterns["anomalies"] if a.get("severity") == "critical"]
            if critical_anomalies:
                recommendations.append(f"Critical anomalies detected - immediate investigation required")
        
        return recommendations[:5]  # Return top 5 recommendations


class PerformanceDashboard:
    """Real-time performance dashboard"""
    
    def __init__(self, monitor: PerformanceMonitor, analytics: MultiAgentAnalytics):
        self.monitor = monitor
        self.analytics = analytics
        self.update_interval = 3  # seconds
    
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
        
        # Store dashboard data (in a real implementation, this would be stored in instance variables)
        self.current_metrics = current_metrics
        self.performance_report = performance_report
        self.patterns = patterns
    
    async def _display_dashboard(self):
        """Display the dashboard"""
        # Clear screen (simple approach)
        print("\033[2J\033[H", end="")
        
        print("🤖 PERFORMANCE MONITORING DASHBOARD")
        print("=" * 50)
        
        # System health
        health_score = self.performance_report.get("system_health", 0.0)
        health_bar = "█" * int(health_score * 20)
        print(f"System Health: [{health_bar:<20}] {health_score:.1%}")
        
        # Current metrics
        print(f"\n📈 Current Metrics:")
        for metric_name, value in self.current_metrics.items():
            print(f"  {metric_name:<20}: {value:>10.3f}")
        
        # Alerts
        alerts = self.monitor.alerts
        if alerts:
            print(f"\n🚨 Recent Alerts ({len([a for a in alerts if not a.resolved])}):")
            for alert in list(alerts)[-3:]:  # Show last 3 alerts
                severity_icon = {"critical": "🔴", "warning": "🟡", "info": "🔵"}.get(alert.severity.value, "⚪")
                print(f"  {severity_icon} {alert.metric_name}: {alert.value:.2f}")
        
        # Patterns
        if hasattr(self, 'patterns') and self.patterns.get("performance_patterns"):
            print(f"\n🔍 Performance Patterns:")
            for metric_name, pattern in list(self.patterns["performance_patterns"].items())[:3]:
                pattern_icon = {
                    "increasing": "📈", "decreasing": "📉", "stable": "➡️", "volatile": "🌊"
                }.get(pattern.get("pattern", "unknown"), "❓")
                print(f"  {pattern_icon} {metric_name:<15}: {pattern.get('pattern', 'unknown')}")
        
        # Recommendations
        if hasattr(self, 'patterns') and self.patterns.get("recommendations"):
            print(f"\n💡 Recommendations:")
            for rec in self.patterns["recommendations"][:2]:
                print(f"  • {rec}")
        
        # System info
        report = self.performance_report
        print(f"\n📊 System Summary:")
        print(f"  Active Alerts: {report.get('active_alerts', 0)}")
        print(f"  Metrics Tracked: {report.get('metric_count', 0)}")
        print(f"  Data Points: {report.get('data_points', 0)}")
        
        print(f"\n⏰ Last Update: {datetime.now().strftime('%H:%M:%S')}")
        print("=" * 50)


# Main execution
async def main():
    """Main function to demonstrate performance monitoring and analytics"""
    print("🔍 Performance Monitoring and Analytics Demonstration")
    print("=" * 60)
    
    # Create monitoring system
    monitor = PerformanceMonitor()
    
    # Create analytics system
    analytics = MultiAgentAnalytics(monitor)
    
    # Create dashboard
    dashboard = PerformanceDashboard(monitor, analytics)
    
    # Simulate some metrics for demonstration
    print("\n📊 Simulating system metrics...")
    
    for i in range(100):
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
        
        await asyncio.sleep(0.05)
    
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
    print(f"Recommendations: {len(patterns['recommendations'])}")
    
    # Display recommendations
    if patterns['recommendations']:
        print(f"\n💡 Optimization Recommendations:")
        for rec in patterns['recommendations']:
            print(f"  • {rec}")
    
    # Start dashboard for a short time
    print(f"\n📊 Starting dashboard for 15 seconds...")
    try:
        await asyncio.wait_for(dashboard.start_dashboard(), timeout=15)
    except asyncio.TimeoutError:
        print("\n📊 Dashboard demonstration complete")
    
    print(f"\n🏁 Performance Monitoring Demo Complete")


if __name__ == "__main__":
    asyncio.run(main())