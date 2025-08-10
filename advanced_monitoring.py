"""
Advanced Monitoring and Visualization for 250-Agent Brain System

This module provides comprehensive monitoring and visualization capabilities
for the large-scale 250-agent brain system.
"""

import time
import json
import threading
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import logging
import sqlite3
from datetime import datetime, timedelta
import statistics

# Import existing components
from massive_brain_system import MassiveBrainSystem, BrainSystemState, BrainSystemMode
from massive_agent_factory import AdvancedAgent
from sophisticated_coordination import CoordinationManager
from advanced_network_topology import AdvancedNetworkTopology

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MonitoringLevel(Enum):
    """Monitoring levels for different detail levels"""
    BASIC = "basic"
    DETAILED = "detailed"
    COMPREHENSIVE = "comprehensive"
    DEBUG = "debug"

class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class SystemAlert:
    """System alert for monitoring"""
    id: str
    timestamp: float
    severity: AlertSeverity
    component: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolved_at: Optional[float] = None
    resolution_details: Optional[str] = None

@dataclass
class PerformanceSnapshot:
    """Performance snapshot for monitoring"""
    timestamp: float
    system_metrics: Dict[str, Any]
    agent_metrics: Dict[str, Any]
    network_metrics: Dict[str, Any]
    coordination_metrics: Dict[str, Any]
    resource_metrics: Dict[str, Any]
    alert_count: int

@dataclass
class VisualizationData:
    """Data for system visualization"""
    network_data: Dict[str, Any]
    agent_status_data: List[Dict[str, Any]]
    performance_data: List[Dict[str, Any]]
    alert_data: List[Dict[str, Any]]
    cluster_data: List[Dict[str, Any]]
    timestamp: float

class AdvancedMonitoringSystem:
    """Advanced monitoring system for 250-agent brain system"""
    
    def __init__(self, brain_system: MassiveBrainSystem):
        self.brain_system = brain_system
        
        # Monitoring configuration
        self.config = {
            "monitoring_level": MonitoringLevel.DETAILED,
            "data_retention_hours": 168,  # 7 days
            "snapshot_interval": 5,  # seconds
            "alert_thresholds": {
                "cpu_usage": 0.8,
                "memory_usage": 0.8,
                "network_usage": 0.8,
                "error_rate": 0.1,
                "response_time": 10.0,
                "coordination_efficiency": 0.7,
                "network_efficiency": 0.7,
                "system_health": 0.7
            },
            "visualization_update_interval": 1.0  # seconds
        }
        
        # Data storage
        self.snapshots = deque(maxlen=10000)  # Store last 10k snapshots
        self.alerts = deque(maxlen=5000)  # Store last 5k alerts
        self.performance_history = deque(maxlen=50000)  # Store last 50k data points
        self.agent_history = defaultdict(lambda: deque(maxlen=1000))  # Per-agent history
        
        # Alert management
        self.alert_rules = self._initialize_alert_rules()
        self.active_alerts = set()
        
        # Database for persistent storage
        self.db_path = "brain_system_monitoring.db"
        self._initialize_database()
        
        # Real-time monitoring
        self.is_monitoring = False
        self.monitoring_thread = None
        self.visualization_thread = None
        
        # Visualization data cache
        self.visualization_cache = None
        self.last_visualization_update = 0
        
        # Performance analytics
        self.analytics_cache = {}
        self.analytics_update_interval = 60  # seconds
        self.last_analytics_update = 0
        
        self.logger = logging.getLogger(__name__)
        
        # Start monitoring
        self.start_monitoring()
    
    def _initialize_alert_rules(self):
        """Initialize alert rules"""
        return {
            "high_cpu_usage": {
                "condition": lambda metrics: metrics.get("cpu_usage", 0) > self.config["alert_thresholds"]["cpu_usage"],
                "severity": AlertSeverity.WARNING,
                "message": "High CPU usage detected",
                "component": "system"
            },
            "high_memory_usage": {
                "condition": lambda metrics: metrics.get("memory_usage", 0) > self.config["alert_thresholds"]["memory_usage"],
                "severity": AlertSeverity.WARNING,
                "message": "High memory usage detected",
                "component": "system"
            },
            "high_network_usage": {
                "condition": lambda metrics: metrics.get("network_usage", 0) > self.config["alert_thresholds"]["network_usage"],
                "severity": AlertSeverity.WARNING,
                "message": "High network usage detected",
                "component": "system"
            },
            "high_error_rate": {
                "condition": lambda metrics: metrics.get("error_rate", 0) > self.config["alert_thresholds"]["error_rate"],
                "severity": AlertSeverity.ERROR,
                "message": "High error rate detected",
                "component": "coordination"
            },
            "slow_response_time": {
                "condition": lambda metrics: metrics.get("average_response_time", 0) > self.config["alert_thresholds"]["response_time"],
                "severity": AlertSeverity.WARNING,
                "message": "Slow response time detected",
                "component": "coordination"
            },
            "low_coordination_efficiency": {
                "condition": lambda metrics: metrics.get("coordination_efficiency", 1.0) < self.config["alert_thresholds"]["coordination_efficiency"],
                "severity": AlertSeverity.ERROR,
                "message": "Low coordination efficiency detected",
                "component": "coordination"
            },
            "low_network_efficiency": {
                "condition": lambda metrics: metrics.get("network_efficiency", 1.0) < self.config["alert_thresholds"]["network_efficiency"],
                "severity": AlertSeverity.ERROR,
                "message": "Low network efficiency detected",
                "component": "network"
            },
            "low_system_health": {
                "condition": lambda metrics: metrics.get("system_health", 1.0) < self.config["alert_thresholds"]["system_health"],
                "severity": AlertSeverity.CRITICAL,
                "message": "Low system health detected",
                "component": "system"
            },
            "agent_error_state": {
                "condition": lambda metrics: metrics.get("error_agents", 0) > 0,
                "severity": AlertSeverity.WARNING,
                "message": "Agents in error state detected",
                "component": "agents"
            }
        }
    
    def _initialize_database(self):
        """Initialize SQLite database for persistent storage"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create tables
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    system_metrics TEXT NOT NULL,
                    agent_metrics TEXT NOT NULL,
                    network_metrics TEXT NOT NULL,
                    coordination_metrics TEXT NOT NULL,
                    resource_metrics TEXT NOT NULL,
                    alert_count INTEGER DEFAULT 0
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_id TEXT UNIQUE NOT NULL,
                    timestamp REAL NOT NULL,
                    severity TEXT NOT NULL,
                    component TEXT NOT NULL,
                    message TEXT NOT NULL,
                    details TEXT,
                    resolved BOOLEAN DEFAULT FALSE,
                    resolved_at REAL,
                    resolution_details TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS agent_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_id TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    state TEXT NOT NULL,
                    load INTEGER DEFAULT 0,
                    cpu_usage REAL DEFAULT 0.0,
                    memory_usage REAL DEFAULT 0.0,
                    network_usage REAL DEFAULT 0.0,
                    activation_count INTEGER DEFAULT 0,
                    success_rate REAL DEFAULT 1.0,
                    efficiency_score REAL DEFAULT 1.0
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    component TEXT NOT NULL
                )
            ''')
            
            conn.commit()
            conn.close()
            
            self.logger.info("Database initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing database: {e}")
    
    def start_monitoring(self):
        """Start the monitoring system"""
        if self.is_monitoring:
            return
        
        self.logger.info("Starting advanced monitoring system")
        self.is_monitoring = True
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        # Start visualization update thread
        self.visualization_thread = threading.Thread(target=self._visualization_loop, daemon=True)
        self.visualization_thread.start()
        
        self.logger.info("Advanced monitoring system started")
    
    def stop_monitoring(self):
        """Stop the monitoring system"""
        if not self.is_monitoring:
            return
        
        self.logger.info("Stopping advanced monitoring system")
        self.is_monitoring = False
        
        # Wait for threads to finish
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        
        if self.visualization_thread:
            self.visualization_thread.join(timeout=5.0)
        
        self.logger.info("Advanced monitoring system stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                # Take performance snapshot
                snapshot = self._take_performance_snapshot()
                self.snapshots.append(snapshot)
                
                # Store in database
                self._store_snapshot_in_database(snapshot)
                
                # Check for alerts
                self._check_alerts(snapshot)
                
                # Update agent history
                self._update_agent_history(snapshot)
                
                # Update performance history
                self._update_performance_history(snapshot)
                
                # Clean up old data
                self._cleanup_old_data()
                
                # Sleep until next snapshot
                time.sleep(self.config["snapshot_interval"])
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
    
    def _visualization_loop(self):
        """Visualization update loop"""
        while self.is_monitoring:
            try:
                # Update visualization data cache
                current_time = time.time()
                
                if current_time - self.last_visualization_update >= self.config["visualization_update_interval"]:
                    self.visualization_cache = self._generate_visualization_data()
                    self.last_visualization_update = current_time
                
                # Update analytics cache
                if current_time - self.last_analytics_update >= self.analytics_update_interval:
                    self.analytics_cache = self._generate_analytics()
                    self.last_analytics_update = current_time
                
                time.sleep(0.1)  # Small delay to prevent excessive CPU usage
                
            except Exception as e:
                self.logger.error(f"Error in visualization loop: {e}")
    
    def _take_performance_snapshot(self) -> PerformanceSnapshot:
        """Take a performance snapshot of the system"""
        timestamp = time.time()
        
        # Get system metrics
        system_status = self.brain_system.get_system_status()
        system_metrics = system_status.get("metrics", {})
        
        # Get agent metrics
        agent_metrics = {}
        for agent_id, agent in self.brain_system.agents.items():
            agent_metrics[agent_id] = {
                "state": agent.state.value,
                "load": agent.load.queue_size,
                "cpu_usage": agent.load.cpu_usage,
                "memory_usage": agent.load.memory_usage,
                "network_usage": agent.load.network_usage,
                "activation_count": agent.metrics.activation_count,
                "success_rate": agent.metrics.success_rate,
                "efficiency_score": agent.metrics.efficiency_score,
                "expertise_level": agent.specialization.expertise_level,
                "domain": agent.specialization.domain.value
            }
        
        # Get network metrics
        network_metrics = {}
        if self.brain_system.topology_manager:
            network_metrics = self.brain_system.topology_manager.get_topology_status()
        
        # Get coordination metrics
        coordination_metrics = {}
        if self.brain_system.coordination_manager:
            coordination_metrics = self.brain_system.coordination_manager.get_coordination_status()
        
        # Get resource metrics
        resource_metrics = system_metrics.get("resource_utilization", {})
        
        # Count active alerts
        alert_count = len([alert for alert in self.alerts if not alert.resolved])
        
        return PerformanceSnapshot(
            timestamp=timestamp,
            system_metrics=system_metrics,
            agent_metrics=agent_metrics,
            network_metrics=network_metrics,
            coordination_metrics=coordination_metrics,
            resource_metrics=resource_metrics,
            alert_count=alert_count
        )
    
    def _store_snapshot_in_database(self, snapshot: PerformanceSnapshot):
        """Store snapshot in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO snapshots (timestamp, system_metrics, agent_metrics, network_metrics, 
                                   coordination_metrics, resource_metrics, alert_count)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                snapshot.timestamp,
                json.dumps(snapshot.system_metrics),
                json.dumps(snapshot.agent_metrics),
                json.dumps(snapshot.network_metrics),
                json.dumps(snapshot.coordination_metrics),
                json.dumps(snapshot.resource_metrics),
                snapshot.alert_count
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error storing snapshot in database: {e}")
    
    def _check_alerts(self, snapshot: PerformanceSnapshot):
        """Check for alerts based on snapshot data"""
        # Prepare metrics for alert checking
        metrics = {
            "cpu_usage": snapshot.resource_metrics.get("cpu", 0),
            "memory_usage": snapshot.resource_metrics.get("memory", 0),
            "network_usage": snapshot.resource_metrics.get("network", 0),
            "error_rate": snapshot.system_metrics.get("error_rate", 0),
            "average_response_time": snapshot.system_metrics.get("average_completion_time", 0),
            "coordination_efficiency": snapshot.system_metrics.get("coordination_efficiency", 1.0),
            "network_efficiency": snapshot.system_metrics.get("network_efficiency", 1.0),
            "system_health": snapshot.system_metrics.get("system_health", 1.0),
            "error_agents": sum(1 for agent in snapshot.agent_metrics.values() if agent.get("state") == "error")
        }
        
        # Check each alert rule
        for rule_name, rule_config in self.alert_rules.items():
            try:
                if rule_config["condition"](metrics):
                    # Check if alert already exists
                    alert_key = f"{rule_name}_{int(snapshot.timestamp)}"
                    
                    if alert_key not in self.active_alerts:
                        # Create new alert
                        alert = SystemAlert(
                            id=alert_key,
                            timestamp=snapshot.timestamp,
                            severity=rule_config["severity"],
                            component=rule_config["component"],
                            message=rule_config["message"],
                            details={
                                "rule": rule_name,
                                "metrics": metrics,
                                "threshold": self.config["alert_thresholds"]
                            }
                        )
                        
                        self.alerts.append(alert)
                        self.active_alerts.add(alert_key)
                        
                        # Store in database
                        self._store_alert_in_database(alert)
                        
                        # Log alert
                        self.logger.warning(f"Alert triggered: {rule_config['message']} (severity: {rule_config['severity'].value})")
                        
                        # Trigger alert event
                        self._trigger_alert_event(alert)
                
            except Exception as e:
                self.logger.error(f"Error checking alert rule {rule_name}: {e}")
        
        # Check for resolved alerts
        self._check_resolved_alerts()
    
    def _store_alert_in_database(self, alert: SystemAlert):
        """Store alert in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO alerts (alert_id, timestamp, severity, component, message, details, resolved, resolved_at, resolution_details)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                alert.id,
                alert.timestamp,
                alert.severity.value,
                alert.component,
                alert.message,
                json.dumps(alert.details),
                alert.resolved,
                alert.resolved_at,
                alert.resolution_details
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error storing alert in database: {e}")
    
    def _check_resolved_alerts(self):
        """Check for resolved alerts"""
        current_time = time.time()
        resolved_alerts = []
        
        for alert in self.alerts:
            if not alert.resolved:
                # Check if alert conditions are no longer met
                if self._is_alert_resolved(alert):
                    alert.resolved = True
                    alert.resolved_at = current_time
                    alert.resolution_details = "Automatically resolved"
                    
                    resolved_alerts.append(alert)
                    self.active_alerts.discard(alert.id)
                    
                    # Update in database
                    self._store_alert_in_database(alert)
                    
                    # Log resolution
                    self.logger.info(f"Alert resolved: {alert.message}")
        
        return resolved_alerts
    
    def _is_alert_resolved(self, alert: SystemAlert) -> bool:
        """Check if an alert is resolved"""
        # Get current metrics
        if not self.snapshots:
            return False
        
        latest_snapshot = self.snapshots[-1]
        metrics = {
            "cpu_usage": latest_snapshot.resource_metrics.get("cpu", 0),
            "memory_usage": latest_snapshot.resource_metrics.get("memory", 0),
            "network_usage": latest_snapshot.resource_metrics.get("network", 0),
            "error_rate": latest_snapshot.system_metrics.get("error_rate", 0),
            "average_response_time": latest_snapshot.system_metrics.get("average_completion_time", 0),
            "coordination_efficiency": latest_snapshot.system_metrics.get("coordination_efficiency", 1.0),
            "network_efficiency": latest_snapshot.system_metrics.get("network_efficiency", 1.0),
            "system_health": latest_snapshot.system_metrics.get("system_health", 1.0),
            "error_agents": sum(1 for agent in latest_snapshot.agent_metrics.values() if agent.get("state") == "error")
        }
        
        # Check if the condition that triggered the alert is no longer true
        rule_name = alert.details.get("rule", "")
        if rule_name in self.alert_rules:
            return not self.alert_rules[rule_name]["condition"](metrics)
        
        return False
    
    def _trigger_alert_event(self, alert: SystemAlert):
        """Trigger alert event"""
        # This could be extended to call external alert systems
        # For now, just log the event
        event_data = {
            "alert_id": alert.id,
            "severity": alert.severity.value,
            "component": alert.component,
            "message": alert.message,
            "timestamp": alert.timestamp,
            "details": alert.details
        }
        
        # Store in system history
        self.brain_system.system_history.append({
            "timestamp": time.time(),
            "type": "alert_triggered",
            "data": event_data
        })
    
    def _update_agent_history(self, snapshot: PerformanceSnapshot):
        """Update agent history"""
        for agent_id, agent_data in snapshot.agent_metrics.items():
            history_entry = {
                "timestamp": snapshot.timestamp,
                "state": agent_data.get("state", "unknown"),
                "load": agent_data.get("load", 0),
                "cpu_usage": agent_data.get("cpu_usage", 0.0),
                "memory_usage": agent_data.get("memory_usage", 0.0),
                "network_usage": agent_data.get("network_usage", 0.0),
                "activation_count": agent_data.get("activation_count", 0),
                "success_rate": agent_data.get("success_rate", 1.0),
                "efficiency_score": agent_data.get("efficiency_score", 1.0)
            }
            
            self.agent_history[agent_id].append(history_entry)
            
            # Store in database
            self._store_agent_history_in_database(agent_id, history_entry)
    
    def _store_agent_history_in_database(self, agent_id: str, history_entry: Dict[str, Any]):
        """Store agent history in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO agent_history (agent_id, timestamp, state, load, cpu_usage, memory_usage, 
                                       network_usage, activation_count, success_rate, efficiency_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                agent_id,
                history_entry["timestamp"],
                history_entry["state"],
                history_entry["load"],
                history_entry["cpu_usage"],
                history_entry["memory_usage"],
                history_entry["network_usage"],
                history_entry["activation_count"],
                history_entry["success_rate"],
                history_entry["efficiency_score"]
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error storing agent history in database: {e}")
    
    def _update_performance_history(self, snapshot: PerformanceSnapshot):
        """Update performance history"""
        # Extract key metrics
        metrics_to_track = [
            ("system_health", snapshot.system_metrics.get("system_health", 1.0)),
            ("coordination_efficiency", snapshot.system_metrics.get("coordination_efficiency", 1.0)),
            ("network_efficiency", snapshot.system_metrics.get("network_efficiency", 1.0)),
            ("error_rate", snapshot.system_metrics.get("error_rate", 0.0)),
            ("cpu_usage", snapshot.resource_metrics.get("cpu", 0.0)),
            ("memory_usage", snapshot.resource_metrics.get("memory", 0.0)),
            ("network_usage", snapshot.resource_metrics.get("network", 0.0)),
            ("active_agents", snapshot.system_metrics.get("active_agents", 0)),
            ("processing_throughput", snapshot.system_metrics.get("processing_throughput", 0.0))
        ]
        
        for metric_name, metric_value in metrics_to_track:
            history_entry = {
                "timestamp": snapshot.timestamp,
                "metric_name": metric_name,
                "metric_value": metric_value,
                "component": "system"
            }
            
            self.performance_history.append(history_entry)
            
            # Store in database
            self._store_performance_history_in_database(history_entry)
    
    def _store_performance_history_in_database(self, history_entry: Dict[str, Any]):
        """Store performance history in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO performance_history (timestamp, metric_name, metric_value, component)
                VALUES (?, ?, ?, ?)
            ''', (
                history_entry["timestamp"],
                history_entry["metric_name"],
                history_entry["metric_value"],
                history_entry["component"]
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error storing performance history in database: {e}")
    
    def _cleanup_old_data(self):
        """Clean up old data based on retention policy"""
        try:
            cutoff_time = time.time() - (self.config["data_retention_hours"] * 3600)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Clean up old snapshots
            cursor.execute("DELETE FROM snapshots WHERE timestamp < ?", (cutoff_time,))
            
            # Clean up old alerts (keep resolved alerts longer)
            alert_cutoff = time.time() - (self.config["data_retention_hours"] * 3600 * 2)  # Keep alerts twice as long
            cursor.execute("DELETE FROM alerts WHERE timestamp < ? AND resolved = TRUE", (alert_cutoff,))
            
            # Clean up old agent history
            cursor.execute("DELETE FROM agent_history WHERE timestamp < ?", (cutoff_time,))
            
            # Clean up old performance history
            cursor.execute("DELETE FROM performance_history WHERE timestamp < ?", (cutoff_time,))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error cleaning up old data: {e}")
    
    def _generate_visualization_data(self) -> VisualizationData:
        """Generate data for visualization"""
        # Get network data
        network_data = {}
        if self.brain_system.topology_manager:
            network_data = self.brain_system.topology_manager.get_network_visualization_data()
        
        # Get agent status data
        agent_status_data = []
        if self.snapshots:
            latest_snapshot = self.snapshots[-1]
            for agent_id, agent_data in latest_snapshot.agent_metrics.items():
                agent_status_data.append({
                    "id": agent_id,
                    "state": agent_data.get("state", "unknown"),
                    "load": agent_data.get("load", 0),
                    "cpu_usage": agent_data.get("cpu_usage", 0.0),
                    "memory_usage": agent_data.get("memory_usage", 0.0),
                    "network_usage": agent_data.get("network_usage", 0.0),
                    "domain": agent_data.get("domain", "unknown"),
                    "expertise_level": agent_data.get("expertise_level", 0.0)
                })
        
        # Get performance data
        performance_data = []
        if len(self.performance_history) > 100:
            # Get last 100 performance data points
            recent_performance = list(self.performance_history)[-100:]
            
            # Group by metric name
            metric_groups = defaultdict(list)
            for entry in recent_performance:
                metric_groups[entry["metric_name"]].append(entry)
            
            # Create performance data for visualization
            for metric_name, entries in metric_groups.items():
                performance_data.append({
                    "metric_name": metric_name,
                    "values": [{"timestamp": entry["timestamp"], "value": entry["metric_value"]} for entry in entries],
                    "component": entries[0]["component"]
                })
        
        # Get alert data
        alert_data = []
        for alert in list(self.alerts)[-50:]:  # Last 50 alerts
            alert_data.append({
                "id": alert.id,
                "timestamp": alert.timestamp,
                "severity": alert.severity.value,
                "component": alert.component,
                "message": alert.message,
                "resolved": alert.resolved
            })
        
        # Get cluster data
        cluster_data = []
        if self.brain_system.cluster_manager:
            for cluster_name, cluster_members in self.brain_system.cluster_manager.clusters.items():
                cluster_info = self.brain_system.get_cluster_info(cluster_name)
                if cluster_info:
                    cluster_data.append(cluster_info)
        
        return VisualizationData(
            network_data=network_data,
            agent_status_data=agent_status_data,
            performance_data=performance_data,
            alert_data=alert_data,
            cluster_data=cluster_data,
            timestamp=time.time()
        )
    
    def _generate_analytics(self) -> Dict[str, Any]:
        """Generate analytics data"""
        analytics = {
            "system_health_trend": self._calculate_health_trend(),
            "performance_analysis": self._analyze_performance(),
            "agent_utilization": self._analyze_agent_utilization(),
            "network_analysis": self._analyze_network(),
            "coordination_analysis": self._analyze_coordination(),
            "alert_analysis": self._analyze_alerts(),
            "recommendations": self._generate_recommendations()
        }
        
        return analytics
    
    def _calculate_health_trend(self) -> Dict[str, Any]:
        """Calculate system health trend"""
        if len(self.performance_history) < 10:
            return {"trend": "insufficient_data", "current_health": 1.0}
        
        # Get recent health data
        health_data = [
            entry for entry in self.performance_history
            if entry["metric_name"] == "system_health"
        ][-100:]  # Last 100 data points
        
        if len(health_data) < 2:
            return {"trend": "stable", "current_health": health_data[-1]["metric_value"] if health_data else 1.0}
        
        # Calculate trend
        recent_health = health_data[-10:]  # Last 10 data points
        older_health = health_data[-20:-10] if len(health_data) > 20 else health_data[:-10]
        
        recent_avg = statistics.mean([entry["metric_value"] for entry in recent_health])
        older_avg = statistics.mean([entry["metric_value"] for entry in older_health])
        
        trend_direction = "improving" if recent_avg > older_avg else "declining"
        trend_magnitude = abs(recent_avg - older_avg)
        
        return {
            "trend": trend_direction,
            "magnitude": trend_magnitude,
            "current_health": health_data[-1]["metric_value"],
            "recent_average": recent_avg,
            "older_average": older_avg
        }
    
    def _analyze_performance(self) -> Dict[str, Any]:
        """Analyze system performance"""
        if not self.snapshots:
            return {"status": "no_data"}
        
        # Get recent snapshots
        recent_snapshots = list(self.snapshots)[-100:]
        
        # Calculate performance metrics
        performance_metrics = {
            "average_response_time": statistics.mean([
                snapshot.system_metrics.get("average_completion_time", 0)
                for snapshot in recent_snapshots
                if snapshot.system_metrics.get("average_completion_time", 0) > 0
            ]) or 0,
            "throughput": statistics.mean([
                snapshot.system_metrics.get("processing_throughput", 0)
                for snapshot in recent_snapshots
            ]),
            "error_rate": statistics.mean([
                snapshot.system_metrics.get("error_rate", 0)
                for snapshot in recent_snapshots
            ]),
            "coordination_efficiency": statistics.mean([
                snapshot.system_metrics.get("coordination_efficiency", 1.0)
                for snapshot in recent_snapshots
            ]),
            "network_efficiency": statistics.mean([
                snapshot.system_metrics.get("network_efficiency", 1.0)
                for snapshot in recent_snapshots
            ])
        }
        
        # Calculate performance score
        performance_score = (
            (1.0 - min(performance_metrics["error_rate"], 1.0)) * 0.3 +
            performance_metrics["coordination_efficiency"] * 0.3 +
            performance_metrics["network_efficiency"] * 0.2 +
            (1.0 - min(performance_metrics["average_response_time"] / 30.0, 1.0)) * 0.2
        )
        
        return {
            "metrics": performance_metrics,
            "score": performance_score,
            "status": "excellent" if performance_score > 0.8 else "good" if performance_score > 0.6 else "fair" if performance_score > 0.4 else "poor"
        }
    
    def _analyze_agent_utilization(self) -> Dict[str, Any]:
        """Analyze agent utilization"""
        if not self.snapshots:
            return {"status": "no_data"}
        
        # Get latest snapshot
        latest_snapshot = self.snapshots[-1]
        
        # Calculate utilization metrics
        total_agents = len(latest_snapshot.agent_metrics)
        active_agents = sum(1 for agent in latest_snapshot.agent_metrics.values() if agent.get("state") != "error")
        busy_agents = sum(1 for agent in latest_snapshot.agent_metrics.values() if agent.get("load", 0) > 5)
        
        # Calculate resource utilization
        cpu_utilization = statistics.mean([
            agent.get("cpu_usage", 0) for agent in latest_snapshot.agent_metrics.values()
        ])
        memory_utilization = statistics.mean([
            agent.get("memory_usage", 0) for agent in latest_snapshot.agent_metrics.values()
        ])
        network_utilization = statistics.mean([
            agent.get("network_usage", 0) for agent in latest_snapshot.agent_metrics.values()
        ])
        
        # Calculate domain utilization
        domain_utilization = defaultdict(list)
        for agent_id, agent_data in latest_snapshot.agent_metrics.items():
            domain = agent_data.get("domain", "unknown")
            domain_utilization[domain].append(agent_data.get("load", 0))
        
        domain_avg_utilization = {
            domain: statistics.mean(loads) if loads else 0
            for domain, loads in domain_utilization.items()
        }
        
        return {
            "total_agents": total_agents,
            "active_agents": active_agents,
            "busy_agents": busy_agents,
            "utilization_rate": active_agents / total_agents if total_agents > 0 else 0,
            "resource_utilization": {
                "cpu": cpu_utilization,
                "memory": memory_utilization,
                "network": network_utilization
            },
            "domain_utilization": domain_avg_utilization
        }
    
    def _analyze_network(self) -> Dict[str, Any]:
        """Analyze network performance"""
        if not self.brain_system.topology_manager:
            return {"status": "no_topology_manager"}
        
        # Get network metrics
        network_status = self.brain_system.topology_manager.get_topology_status()
        
        return {
            "total_connections": network_status.get("total_connections", 0),
            "network_density": network_status.get("network_density", 0),
            "average_path_length": network_status.get("average_path_length", 0),
            "clustering_coefficient": network_status.get("clustering_coefficient", 0),
            "efficiency_score": network_status.get("efficiency_score", 0),
            "robustness_score": network_status.get("robustness_score", 0),
            "bandwidth_utilization": network_status.get("bandwidth_utilization", 0),
            "latency_distribution": network_status.get("latency_distribution", {})
        }
    
    def _analyze_coordination(self) -> Dict[str, Any]:
        """Analyze coordination performance"""
        if not self.brain_system.coordination_manager:
            return {"status": "no_coordination_manager"}
        
        # Get coordination metrics
        coordination_status = self.brain_system.coordination_manager.get_coordination_status()
        
        return {
            "active_strategy": coordination_status.get("active_strategy", "unknown"),
            "active_tasks": coordination_status.get("active_tasks", 0),
            "completed_tasks": coordination_status.get("completed_tasks", 0),
            "task_queue_size": coordination_status.get("task_queue_size", 0),
            "coordination_efficiency": coordination_status.get("metrics", {}).get("coordination_efficiency", 0),
            "load_balance_score": coordination_status.get("metrics", {}).get("load_balance_score", 0),
            "coordination_overhead": coordination_status.get("metrics", {}).get("coordination_overhead", 0)
        }
    
    def _analyze_alerts(self) -> Dict[str, Any]:
        """Analyze alert patterns"""
        if not self.alerts:
            return {"status": "no_alerts"}
        
        # Count alerts by severity
        severity_counts = defaultdict(int)
        for alert in self.alerts:
            severity_counts[alert.severity.value] += 1
        
        # Count alerts by component
        component_counts = defaultdict(int)
        for alert in self.alerts:
            component_counts[alert.component] += 1
        
        # Calculate alert rates
        total_alerts = len(self.alerts)
        time_span = time.time() - self.alerts[0].timestamp if self.alerts else 1
        alert_rate = total_alerts / time_span if time_span > 0 else 0
        
        # Calculate resolution rate
        resolved_alerts = sum(1 for alert in self.alerts if alert.resolved)
        resolution_rate = resolved_alerts / total_alerts if total_alerts > 0 else 0
        
        return {
            "total_alerts": total_alerts,
            "alert_rate": alert_rate,
            "resolution_rate": resolution_rate,
            "severity_distribution": dict(severity_counts),
            "component_distribution": dict(component_counts),
            "active_alerts": len(self.active_alerts)
        }
    
    def _generate_recommendations(self) -> List[Dict[str, Any]]:
        """Generate system recommendations based on analytics"""
        recommendations = []
        
        # Get current analytics
        analytics = self._generate_analytics()
        
        # Health trend recommendations
        health_trend = analytics.get("system_health_trend", {})
        if health_trend.get("trend") == "declining":
            recommendations.append({
                "type": "health",
                "priority": "high",
                "message": "System health is declining. Consider optimization or maintenance.",
                "action": "optimize_system"
            })
        
        # Performance recommendations
        performance_analysis = analytics.get("performance_analysis", {})
        if performance_analysis.get("status") == "poor":
            recommendations.append({
                "type": "performance",
                "priority": "high",
                "message": "System performance is poor. Check resource utilization and agent status.",
                "action": "diagnose_performance"
            })
        
        # Agent utilization recommendations
        agent_utilization = analytics.get("agent_utilization", {})
        utilization_rate = agent_utilization.get("utilization_rate", 0)
        if utilization_rate < 0.5:
            recommendations.append({
                "type": "utilization",
                "priority": "medium",
                "message": "Agent utilization is low. Consider load balancing or task redistribution.",
                "action": "optimize_load_balance"
            })
        
        # Resource utilization recommendations
        resource_utilization = agent_utilization.get("resource_utilization", {})
        for resource, utilization in resource_utilization.items():
            if utilization > 0.8:
                recommendations.append({
                    "type": "resource",
                    "priority": "high",
                    "message": f"High {resource} utilization detected. Consider resource optimization.",
                    "action": "optimize_resources"
                })
        
        # Alert recommendations
        alert_analysis = analytics.get("alert_analysis", {})
        if alert_analysis.get("alert_rate", 0) > 0.1:  # More than 1 alert per 10 seconds
            recommendations.append({
                "type": "alerts",
                "priority": "medium",
                "message": "High alert rate detected. Review system configuration and thresholds.",
                "action": "review_alerts"
            })
        
        return recommendations
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get monitoring system status"""
        return {
            "is_monitoring": self.is_monitoring,
            "monitoring_level": self.config["monitoring_level"].value,
            "snapshots_stored": len(self.snapshots),
            "alerts_stored": len(self.alerts),
            "active_alerts": len(self.active_alerts),
            "performance_history_size": len(self.performance_history),
            "database_path": self.db_path,
            "last_snapshot_time": self.snapshots[-1].timestamp if self.snapshots else None,
            "last_visualization_update": self.last_visualization_update,
            "last_analytics_update": self.last_analytics_update
        }
    
    def get_visualization_data(self) -> Optional[VisualizationData]:
        """Get cached visualization data"""
        return self.visualization_cache
    
    def get_analytics(self) -> Dict[str, Any]:
        """Get cached analytics data"""
        return self.analytics_cache
    
    def get_recent_alerts(self, limit: int = 50) -> List[SystemAlert]:
        """Get recent alerts"""
        return list(self.alerts)[-limit:]
    
    def get_agent_history(self, agent_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get history for a specific agent"""
        return list(self.agent_history.get(agent_id, []))[-limit:]
    
    def get_performance_history(self, metric_name: str, limit: int = 1000) -> List[Dict[str, Any]]:
        """Get performance history for a specific metric"""
        return [
            entry for entry in self.performance_history
            if entry["metric_name"] == metric_name
        ][-limit:]
    
    def resolve_alert(self, alert_id: str, resolution_details: str):
        """Manually resolve an alert"""
        for alert in self.alerts:
            if alert.id == alert_id and not alert.resolved:
                alert.resolved = True
                alert.resolved_at = time.time()
                alert.resolution_details = resolution_details
                
                # Update in database
                self._store_alert_in_database(alert)
                
                # Remove from active alerts
                self.active_alerts.discard(alert_id)
                
                self.logger.info(f"Alert {alert_id} resolved manually: {resolution_details}")
                break
    
    def set_alert_threshold(self, threshold_name: str, value: float):
        """Set alert threshold"""
        if threshold_name in self.config["alert_thresholds"]:
            old_value = self.config["alert_thresholds"][threshold_name]
            self.config["alert_thresholds"][threshold_name] = value
            
            self.logger.info(f"Alert threshold updated: {threshold_name} {old_value} -> {value}")
        else:
            self.logger.warning(f"Unknown alert threshold: {threshold_name}")
    
    def export_monitoring_data(self, filename: str, start_time: Optional[float] = None, end_time: Optional[float] = None):
        """Export monitoring data to file"""
        try:
            export_data = {
                "snapshots": [],
                "alerts": [],
                "performance_history": [],
                "export_timestamp": time.time(),
                "system_info": {
                    "system_id": self.brain_system.id,
                    "state": self.brain_system.state.value,
                    "mode": self.brain_system.mode.value
                }
            }
            
            # Filter snapshots by time range
            if start_time or end_time:
                filtered_snapshots = []
                for snapshot in self.snapshots:
                    if start_time and snapshot.timestamp < start_time:
                        continue
                    if end_time and snapshot.timestamp > end_time:
                        continue
                    filtered_snapshots.append(snapshot)
                export_data["snapshots"] = filtered_snapshots
            else:
                export_data["snapshots"] = list(self.snapshots)
            
            # Filter alerts by time range
            if start_time or end_time:
                filtered_alerts = []
                for alert in self.alerts:
                    if start_time and alert.timestamp < start_time:
                        continue
                    if end_time and alert.timestamp > end_time:
                        continue
                    filtered_alerts.append(alert)
                export_data["alerts"] = filtered_alerts
            else:
                export_data["alerts"] = list(self.alerts)
            
            # Filter performance history by time range
            if start_time or end_time:
                filtered_performance = []
                for entry in self.performance_history:
                    if start_time and entry["timestamp"] < start_time:
                        continue
                    if end_time and entry["timestamp"] > end_time:
                        continue
                    filtered_performance.append(entry)
                export_data["performance_history"] = filtered_performance
            else:
                export_data["performance_history"] = list(self.performance_history)
            
            # Write to file
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            self.logger.info(f"Monitoring data exported to {filename}")
            
        except Exception as e:
            self.logger.error(f"Error exporting monitoring data: {e}")

# Global monitoring system instance
monitoring_system = None