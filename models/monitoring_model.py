import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import math
from datetime import datetime, timedelta

class MetricType(Enum):
    PERFORMANCE = "performance"
    POWER = "power"
    THERMAL = "thermal"
    NETWORK = "network"
    STORAGE = "storage"
    SECURITY = "security"
    ENVIRONMENTAL = "environmental"
    FINANCIAL = "financial"

class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class MonitoringFrequency(Enum):
    REAL_TIME = "real_time"  # < 1 second
    HIGH = "high"  # 1-10 seconds
    MEDIUM = "medium"  # 10-60 seconds
    LOW = "low"  # 1-5 minutes
    PERIODIC = "periodic"  # 5+ minutes

class DataRetention(Enum):
    SHORT_TERM = "short_term"  # 1-7 days
    MEDIUM_TERM = "medium_term"  # 1-4 weeks
    LONG_TERM = "long_term"  # 1-12 months
    ARCHIVAL = "archival"  # 1+ years

@dataclass
class MetricDefinition:
    """Definition of a monitoring metric"""
    name: str
    metric_type: MetricType
    unit: str
    description: str
    frequency: MonitoringFrequency
    retention: DataRetention
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None
    aggregation_method: str = "average"  # average, sum, max, min

@dataclass
class AlertRule:
    """Alert rule configuration"""
    name: str
    metric_name: str
    condition: str  # ">", "<", "==", "!="
    threshold: float
    severity: AlertSeverity
    duration_seconds: int  # How long condition must persist
    notification_channels: List[str]
    auto_resolution: bool = True

@dataclass
class MonitoringAgent:
    """Monitoring agent specification"""
    agent_id: str
    agent_type: str  # "hardware", "software", "network"
    location: str
    metrics_collected: List[str]
    collection_frequency: MonitoringFrequency
    power_consumption_w: float
    cost: float

@dataclass
class DataPoint:
    """Individual data point"""
    timestamp: datetime
    metric_name: str
    value: float
    unit: str
    source: str
    tags: Dict[str, str] = None

class PerformanceAnalyzer:
    """Analyze system performance metrics"""
    
    def __init__(self):
        self.performance_baselines = {}
        self.anomaly_thresholds = {}
        
    def establish_baseline(self, metric_name: str, historical_data: List[float], 
                          confidence_interval: float = 0.95):
        """Establish performance baseline for a metric"""
        if not historical_data:
            return
        
        mean = np.mean(historical_data)
        std = np.std(historical_data)
        
        # Calculate confidence interval
        z_score = 1.96 if confidence_interval == 0.95 else 2.58  # 95% or 99%
        margin_of_error = z_score * (std / math.sqrt(len(historical_data)))
        
        self.performance_baselines[metric_name] = {
            'mean': mean,
            'std': std,
            'min': min(historical_data),
            'max': max(historical_data),
            'confidence_lower': mean - margin_of_error,
            'confidence_upper': mean + margin_of_error,
            'sample_size': len(historical_data)
        }
    
    def detect_anomalies(self, metric_name: str, current_value: float) -> Dict[str, Any]:
        """Detect anomalies in current metric value"""
        if metric_name not in self.performance_baselines:
            return {'anomaly_detected': False, 'reason': 'No baseline established'}
        
        baseline = self.performance_baselines[metric_name]
        
        # Z-score based anomaly detection
        z_score = abs(current_value - baseline['mean']) / baseline['std'] if baseline['std'] > 0 else 0
        
        # Consider anomaly if z-score > 3 (99.7% confidence)
        is_anomaly = z_score > 3
        
        # Additional checks
        outside_confidence = (current_value < baseline['confidence_lower'] or 
                            current_value > baseline['confidence_upper'])
        
        return {
            'anomaly_detected': is_anomaly,
            'z_score': z_score,
            'outside_confidence_interval': outside_confidence,
            'current_value': current_value,
            'baseline_mean': baseline['mean'],
            'deviation_percent': ((current_value - baseline['mean']) / baseline['mean'] * 100) if baseline['mean'] != 0 else 0
        }
    
    def calculate_performance_score(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Calculate overall performance score"""
        scores = {}
        weights = {
            'gpu_utilization': 0.3,
            'memory_utilization': 0.2,
            'network_throughput': 0.2,
            'storage_iops': 0.15,
            'power_efficiency': 0.15
        }
        
        total_score = 0
        total_weight = 0
        
        for metric, value in metrics.items():
            if metric in weights:
                # Normalize to 0-100 scale
                if metric == 'gpu_utilization':
                    score = min(value, 100)  # Higher is better
                elif metric == 'memory_utilization':
                    score = min(value, 90)  # Up to 90% is good
                elif metric == 'power_efficiency':
                    score = min(value * 50, 100)  # Efficiency ratio * 50
                else:
                    score = min(value / 1000 * 100, 100)  # Normalize throughput/IOPS
                
                scores[metric] = score
                total_score += score * weights[metric]
                total_weight += weights[metric]
        
        overall_score = total_score / total_weight if total_weight > 0 else 0
        
        return {
            'overall_score': overall_score,
            'individual_scores': scores,
            'performance_grade': self._get_performance_grade(overall_score)
        }
    
    def _get_performance_grade(self, score: float) -> str:
        """Convert performance score to grade"""
        if score >= 90:
            return 'A'
        elif score >= 80:
            return 'B'
        elif score >= 70:
            return 'C'
        elif score >= 60:
            return 'D'
        else:
            return 'F'

class AlertManager:
    """Manage alerts and notifications"""
    
    def __init__(self):
        self.alert_rules: List[AlertRule] = []
        self.active_alerts: List[Dict] = []
        self.alert_history: List[Dict] = []
        
    def add_alert_rule(self, rule: AlertRule):
        """Add new alert rule"""
        self.alert_rules.append(rule)
    
    def evaluate_alerts(self, current_metrics: Dict[str, float]) -> List[Dict]:
        """Evaluate all alert rules against current metrics"""
        triggered_alerts = []
        
        for rule in self.alert_rules:
            if rule.metric_name in current_metrics:
                value = current_metrics[rule.metric_name]
                
                # Evaluate condition
                triggered = False
                if rule.condition == ">" and value > rule.threshold:
                    triggered = True
                elif rule.condition == "<" and value < rule.threshold:
                    triggered = True
                elif rule.condition == "==" and value == rule.threshold:
                    triggered = True
                elif rule.condition == "!=" and value != rule.threshold:
                    triggered = True
                
                if triggered:
                    alert = {
                        'rule_name': rule.name,
                        'metric_name': rule.metric_name,
                        'current_value': value,
                        'threshold': rule.threshold,
                        'severity': rule.severity.value,
                        'timestamp': datetime.now(),
                        'notification_channels': rule.notification_channels
                    }
                    triggered_alerts.append(alert)
                    
                    # Add to active alerts if not already present
                    if not any(a['rule_name'] == rule.name for a in self.active_alerts):
                        self.active_alerts.append(alert)
        
        return triggered_alerts
    
    def resolve_alert(self, rule_name: str):
        """Resolve an active alert"""
        self.active_alerts = [a for a in self.active_alerts if a['rule_name'] != rule_name]
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of alert status"""
        severity_counts = {}
        for alert in self.active_alerts:
            severity = alert['severity']
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        return {
            'total_active_alerts': len(self.active_alerts),
            'alerts_by_severity': severity_counts,
            'most_recent_alert': max(self.active_alerts, key=lambda x: x['timestamp']) if self.active_alerts else None
        }

class DataCollector:
    """Collect and store monitoring data"""
    
    def __init__(self):
        self.data_storage: Dict[str, List[DataPoint]] = {}
        self.collection_agents: List[MonitoringAgent] = []
        
    def add_monitoring_agent(self, agent: MonitoringAgent):
        """Add monitoring agent"""
        self.collection_agents.append(agent)
    
    def collect_data_point(self, data_point: DataPoint):
        """Collect a single data point"""
        metric_name = data_point.metric_name
        if metric_name not in self.data_storage:
            self.data_storage[metric_name] = []
        
        self.data_storage[metric_name].append(data_point)
        
        # Implement data retention policy
        self._apply_retention_policy(metric_name)
    
    def _apply_retention_policy(self, metric_name: str):
        """Apply data retention policy to limit storage"""
        if metric_name not in self.data_storage:
            return
        
        # Keep only last 10000 points per metric (simplified)
        if len(self.data_storage[metric_name]) > 10000:
            self.data_storage[metric_name] = self.data_storage[metric_name][-10000:]
    
    def get_historical_data(self, metric_name: str, hours: int = 24) -> List[DataPoint]:
        """Get historical data for a metric"""
        if metric_name not in self.data_storage:
            return []
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [dp for dp in self.data_storage[metric_name] if dp.timestamp >= cutoff_time]
    
    def calculate_aggregated_metrics(self, metric_name: str, 
                                   aggregation: str = "average", 
                                   hours: int = 1) -> float:
        """Calculate aggregated metric value"""
        data_points = self.get_historical_data(metric_name, hours)
        
        if not data_points:
            return 0.0
        
        values = [dp.value for dp in data_points]
        
        if aggregation == "average":
            return np.mean(values)
        elif aggregation == "sum":
            return np.sum(values)
        elif aggregation == "max":
            return np.max(values)
        elif aggregation == "min":
            return np.min(values)
        elif aggregation == "median":
            return np.median(values)
        else:
            return np.mean(values)

class MonitoringSystem:
    """Comprehensive monitoring system for AI data center"""
    
    def __init__(self, name: str):
        self.name = name
        self.metrics: Dict[str, MetricDefinition] = {}
        self.data_collector = DataCollector()
        self.performance_analyzer = PerformanceAnalyzer()
        self.alert_manager = AlertManager()
        
    def define_gpu_metrics(self):
        """Define GPU performance metrics"""
        gpu_metrics = [
            MetricDefinition(
                name="gpu_utilization",
                metric_type=MetricType.PERFORMANCE,
                unit="percent",
                description="GPU compute utilization percentage",
                frequency=MonitoringFrequency.HIGH,
                retention=DataRetention.MEDIUM_TERM,
                threshold_warning=90.0,
                threshold_critical=95.0
            ),
            MetricDefinition(
                name="gpu_memory_utilization",
                metric_type=MetricType.PERFORMANCE,
                unit="percent",
                description="GPU memory utilization percentage",
                frequency=MonitoringFrequency.HIGH,
                retention=DataRetention.MEDIUM_TERM,
                threshold_warning=85.0,
                threshold_critical=95.0
            ),
            MetricDefinition(
                name="gpu_temperature",
                metric_type=MetricType.THERMAL,
                unit="celsius",
                description="GPU temperature",
                frequency=MonitoringFrequency.MEDIUM,
                retention=DataRetention.LONG_TERM,
                threshold_warning=80.0,
                threshold_critical=90.0
            ),
            MetricDefinition(
                name="gpu_power_consumption",
                metric_type=MetricType.POWER,
                unit="watts",
                description="GPU power consumption",
                frequency=MonitoringFrequency.MEDIUM,
                retention=DataRetention.LONG_TERM,
                threshold_warning=450.0,
                threshold_critical=500.0
            )
        ]
        
        for metric in gpu_metrics:
            self.metrics[metric.name] = metric
    
    def define_system_metrics(self):
        """Define system-level metrics"""
        system_metrics = [
            MetricDefinition(
                name="total_power_consumption",
                metric_type=MetricType.POWER,
                unit="megawatts",
                description="Total facility power consumption",
                frequency=MonitoringFrequency.MEDIUM,
                retention=DataRetention.ARCHIVAL,
                threshold_warning=4500.0,
                threshold_critical=5000.0
            ),
            MetricDefinition(
                name="pue",
                metric_type=MetricType.PERFORMANCE,
                unit="ratio",
                description="Power Usage Effectiveness",
                frequency=MonitoringFrequency.LOW,
                retention=DataRetention.ARCHIVAL,
                threshold_warning=1.2,
                threshold_critical=1.3
            ),
            MetricDefinition(
                name="cooling_efficiency",
                metric_type=MetricType.THERMAL,
                unit="cop",
                description="Cooling system coefficient of performance",
                frequency=MonitoringFrequency.LOW,
                retention=DataRetention.LONG_TERM,
                threshold_warning=3.0,
                threshold_critical=2.5
            ),
            MetricDefinition(
                name="network_throughput",
                metric_type=MetricType.NETWORK,
                unit="gbps",
                description="Total network throughput",
                frequency=MonitoringFrequency.HIGH,
                retention=DataRetention.MEDIUM_TERM,
                threshold_warning=8000.0,
                threshold_critical=9000.0
            )
        ]
        
        for metric in system_metrics:
            self.metrics[metric.name] = metric
    
    def setup_default_alerts(self):
        """Setup default alert rules"""
        default_alerts = [
            AlertRule(
                name="High GPU Temperature",
                metric_name="gpu_temperature",
                condition=">",
                threshold=85.0,
                severity=AlertSeverity.WARNING,
                duration_seconds=300,
                notification_channels=["email", "slack"]
            ),
            AlertRule(
                name="Critical GPU Temperature",
                metric_name="gpu_temperature",
                condition=">",
                threshold=90.0,
                severity=AlertSeverity.CRITICAL,
                duration_seconds=60,
                notification_channels=["email", "slack", "sms"]
            ),
            AlertRule(
                name="High Power Consumption",
                metric_name="total_power_consumption",
                condition=">",
                threshold=4800.0,
                severity=AlertSeverity.WARNING,
                duration_seconds=600,
                notification_channels=["email"]
            ),
            AlertRule(
                name="Poor PUE",
                metric_name="pue",
                condition=">",
                threshold=1.25,
                severity=AlertSeverity.WARNING,
                duration_seconds=1800,
                notification_channels=["email"]
            )
        ]
        
        for alert in default_alerts:
            self.alert_manager.add_alert_rule(alert)
    
    def add_monitoring_agents(self, gpu_count: int, server_count: int):
        """Add monitoring agents for infrastructure"""
        # GPU monitoring agents
        for i in range(gpu_count // 8):  # One agent per 8 GPUs
            agent = MonitoringAgent(
                agent_id=f"gpu_agent_{i+1}",
                agent_type="hardware",
                location=f"rack_{i//4 + 1}",
                metrics_collected=["gpu_utilization", "gpu_memory_utilization", 
                                 "gpu_temperature", "gpu_power_consumption"],
                collection_frequency=MonitoringFrequency.HIGH,
                power_consumption_w=5.0,
                cost=200
            )
            self.data_collector.add_monitoring_agent(agent)
        
        # Server monitoring agents
        for i in range(server_count):
            agent = MonitoringAgent(
                agent_id=f"server_agent_{i+1}",
                agent_type="software",
                location=f"server_{i+1}",
                metrics_collected=["cpu_utilization", "memory_utilization", 
                                 "disk_utilization", "network_utilization"],
                collection_frequency=MonitoringFrequency.MEDIUM,
                power_consumption_w=2.0,
                cost=50
            )
            self.data_collector.add_monitoring_agent(agent)
        
        # Facility monitoring agents
        facility_agent = MonitoringAgent(
            agent_id="facility_agent",
            agent_type="hardware",
            location="facility",
            metrics_collected=["total_power_consumption", "pue", "cooling_efficiency"],
            collection_frequency=MonitoringFrequency.LOW,
            power_consumption_w=50.0,
            cost=5000
        )
        self.data_collector.add_monitoring_agent(facility_agent)
    
    def simulate_monitoring_data(self, hours: int = 24) -> Dict[str, List[float]]:
        """Simulate monitoring data for testing"""
        simulated_data = {}
        
        # Generate hourly data points
        for hour in range(hours):
            timestamp = datetime.now() - timedelta(hours=hours-hour)
            
            # Simulate GPU metrics with some variability
            gpu_util = 75 + np.random.normal(0, 10)  # Average 75% with variation
            gpu_temp = 70 + np.random.normal(0, 5)   # Average 70°C
            gpu_power = 400 + np.random.normal(0, 20) # Average 400W
            
            # Simulate system metrics
            total_power = 5000 + np.random.normal(0, 200)  # 5GW with variation
            pue = 1.1 + np.random.normal(0, 0.05)          # PUE around 1.1
            
            # Store data points
            metrics_data = {
                'gpu_utilization': gpu_util,
                'gpu_temperature': gpu_temp,
                'gpu_power_consumption': gpu_power,
                'total_power_consumption': total_power,
                'pue': pue
            }
            
            for metric_name, value in metrics_data.items():
                if metric_name not in simulated_data:
                    simulated_data[metric_name] = []
                simulated_data[metric_name].append(value)
                
                # Create data point
                data_point = DataPoint(
                    timestamp=timestamp,
                    metric_name=metric_name,
                    value=value,
                    unit=self.metrics[metric_name].unit if metric_name in self.metrics else "unknown",
                    source="simulation"
                )
                self.data_collector.collect_data_point(data_point)
        
        return simulated_data
    
    def calculate_monitoring_costs(self) -> Dict[str, float]:
        """Calculate monitoring system costs"""
        agent_costs = sum(agent.cost for agent in self.data_collector.collection_agents)
        agent_power = sum(agent.power_consumption_w for agent in self.data_collector.collection_agents)
        
        # Estimate infrastructure costs (storage, networking, processing)
        data_storage_cost = len(self.metrics) * 1000  # $1000 per metric for storage infrastructure
        processing_cost = len(self.data_collector.collection_agents) * 500  # $500 per agent for processing
        
        total_cost = agent_costs + data_storage_cost + processing_cost
        
        return {
            'monitoring_agents_cost': agent_costs,
            'data_storage_cost': data_storage_cost,
            'processing_infrastructure_cost': processing_cost,
            'total_monitoring_cost': total_cost,
            'annual_power_cost': agent_power * 8760 * 0.10 / 1000,  # Assume $0.10/kWh
            'agents_count': len(self.data_collector.collection_agents)
        }
    
    def generate_monitoring_dashboard(self) -> Dict[str, Any]:
        """Generate monitoring dashboard data"""
        current_metrics = {}
        
        # Get latest values for each metric
        for metric_name in self.metrics.keys():
            latest_data = self.data_collector.get_historical_data(metric_name, hours=1)
            if latest_data:
                current_metrics[metric_name] = latest_data[-1].value
            else:
                current_metrics[metric_name] = 0
        
        # Calculate performance score
        performance_score = self.performance_analyzer.calculate_performance_score(current_metrics)
        
        # Get alert summary
        alert_summary = self.alert_manager.get_alert_summary()
        
        # Calculate system health
        health_indicators = {
            'gpu_health': 'Good' if current_metrics.get('gpu_temperature', 0) < 80 else 'Warning',
            'power_health': 'Good' if current_metrics.get('total_power_consumption', 0) < 4800 else 'Warning',
            'efficiency_health': 'Good' if current_metrics.get('pue', 2.0) < 1.2 else 'Warning'
        }
        
        return {
            'current_metrics': current_metrics,
            'performance_score': performance_score,
            'alert_summary': alert_summary,
            'health_indicators': health_indicators,
            'monitoring_agents': len(self.data_collector.collection_agents),
            'metrics_tracked': len(self.metrics)
        }
    
    def generate_monitoring_report(self) -> Dict[str, Any]:
        """Generate comprehensive monitoring system report"""
        return {
            'system_overview': {
                'name': self.name,
                'metrics_defined': len(self.metrics),
                'monitoring_agents': len(self.data_collector.collection_agents),
                'alert_rules': len(self.alert_manager.alert_rules)
            },
            'metrics_catalog': {name: {
                'type': metric.metric_type.value,
                'unit': metric.unit,
                'frequency': metric.frequency.value,
                'retention': metric.retention.value
            } for name, metric in self.metrics.items()},
            'cost_analysis': self.calculate_monitoring_costs(),
            'dashboard_data': self.generate_monitoring_dashboard()
        }