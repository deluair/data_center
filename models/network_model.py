import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set
from enum import Enum
import math

class NetworkProtocol(Enum):
    ETHERNET = "ethernet"
    INFINIBAND = "infiniband"
    ROCE = "roce"  # RDMA over Converged Ethernet
    NVLINK = "nvlink"
    PCIE = "pcie"
    CXL = "cxl"  # Compute Express Link

class NetworkTopology(Enum):
    LEAF_SPINE = "leaf_spine"
    FAT_TREE = "fat_tree"
    DRAGONFLY = "dragonfly"
    TORUS = "torus"
    MESH = "mesh"
    RING = "ring"

class NetworkSpeed(Enum):
    GIGABIT_1 = "1gbps"
    GIGABIT_10 = "10gbps"
    GIGABIT_25 = "25gbps"
    GIGABIT_40 = "40gbps"
    GIGABIT_50 = "50gbps"
    GIGABIT_100 = "100gbps"
    GIGABIT_200 = "200gbps"
    GIGABIT_400 = "400gbps"
    GIGABIT_800 = "800gbps"
    TERABIT_1_6 = "1.6tbps"

class TrafficType(Enum):
    COMPUTE = "compute"  # GPU-to-GPU communication
    STORAGE = "storage"  # Storage access
    MANAGEMENT = "management"  # Control plane
    EXTERNAL = "external"  # Internet/WAN
    BACKUP = "backup"  # Backup traffic

@dataclass
class NetworkInterface:
    """Network interface specification"""
    name: str
    protocol: NetworkProtocol
    speed_gbps: float
    port_count: int
    power_consumption_w: float
    latency_ns: float
    bandwidth_efficiency: float  # Actual vs theoretical bandwidth
    cost_per_port: float
    form_factor: str

@dataclass
class NetworkSwitch:
    """Network switch specification"""
    model: str
    switch_type: str  # ToR, Leaf, Spine, Core
    port_count: int
    port_speed_gbps: float
    switching_capacity_tbps: float
    forwarding_rate_mpps: float  # Million packets per second
    buffer_size_mb: float
    power_consumption_w: float
    latency_ns: float
    rack_units: int
    cost: float
    protocols_supported: List[NetworkProtocol]

@dataclass
class NetworkCable:
    """Network cable specification"""
    type: str  # Copper, Fiber, DAC
    length_m: float
    speed_gbps: float
    connector_type: str
    power_consumption_w: float
    cost_per_meter: float
    max_distance_m: float

@dataclass
class NetworkSegment:
    """Network segment configuration"""
    name: str
    topology: NetworkTopology
    protocol: NetworkProtocol
    bandwidth_gbps: float
    node_count: int
    switch_count: int
    cable_length_total_m: float
    latency_avg_ns: float
    redundancy_level: int

class NetworkTrafficAnalyzer:
    """Analyze and model network traffic patterns"""
    
    def __init__(self):
        self.traffic_patterns = {}
        self.bandwidth_utilization = {}
        self.latency_requirements = {}
    
    def add_traffic_pattern(self, name: str, traffic_type: TrafficType, 
                          bandwidth_gbps: float, burst_factor: float = 1.5,
                          latency_requirement_ns: float = 1000):
        """Add traffic pattern"""
        self.traffic_patterns[name] = {
            'type': traffic_type,
            'bandwidth_gbps': bandwidth_gbps,
            'burst_bandwidth_gbps': bandwidth_gbps * burst_factor,
            'burst_factor': burst_factor,
            'latency_requirement_ns': latency_requirement_ns,
            'priority': self._get_traffic_priority(traffic_type)
        }
    
    def _get_traffic_priority(self, traffic_type: TrafficType) -> int:
        """Get traffic priority (1=highest, 5=lowest)"""
        priority_map = {
            TrafficType.COMPUTE: 1,
            TrafficType.STORAGE: 2,
            TrafficType.MANAGEMENT: 3,
            TrafficType.EXTERNAL: 4,
            TrafficType.BACKUP: 5
        }
        return priority_map.get(traffic_type, 3)
    
    def calculate_bandwidth_requirements(self, gpu_count: int, 
                                       storage_capacity_tb: float) -> Dict[str, float]:
        """Calculate bandwidth requirements based on system scale"""
        # GPU-to-GPU communication (AI training)
        gpu_interconnect_bw = gpu_count * 0.8  # 800 Gbps per GPU for high-end training
        
        # Storage bandwidth (assume 10% of storage accessed per hour)
        storage_bw = storage_capacity_tb * 0.1 * 8 / 3600  # Convert to Gbps
        
        # Management and monitoring
        management_bw = max(10, gpu_count * 0.001)  # 1 Mbps per GPU minimum
        
        # External connectivity (model serving, data ingestion)
        external_bw = gpu_count * 0.1  # 100 Mbps per GPU for inference
        
        return {
            'gpu_interconnect_gbps': gpu_interconnect_bw,
            'storage_access_gbps': storage_bw,
            'management_gbps': management_bw,
            'external_gbps': external_bw,
            'total_internal_gbps': gpu_interconnect_bw + storage_bw + management_bw,
            'total_external_gbps': external_bw
        }
    
    def analyze_traffic_flow(self, topology: NetworkTopology, 
                           node_count: int) -> Dict[str, any]:
        """Analyze traffic flow patterns for given topology"""
        if topology == NetworkTopology.LEAF_SPINE:
            # Leaf-spine typically has 3:1 oversubscription
            oversubscription_ratio = 3.0
            hop_count_avg = 3  # Leaf -> Spine -> Leaf
            bisection_bandwidth_ratio = 1.0
        elif topology == NetworkTopology.FAT_TREE:
            oversubscription_ratio = 1.0  # Non-blocking
            hop_count_avg = 6  # Worst case in fat-tree
            bisection_bandwidth_ratio = 1.0
        elif topology == NetworkTopology.DRAGONFLY:
            oversubscription_ratio = 2.0
            hop_count_avg = 5
            bisection_bandwidth_ratio = 0.5
        else:
            oversubscription_ratio = 2.0
            hop_count_avg = 4
            bisection_bandwidth_ratio = 0.7
        
        return {
            'topology': topology.value,
            'oversubscription_ratio': oversubscription_ratio,
            'average_hop_count': hop_count_avg,
            'bisection_bandwidth_ratio': bisection_bandwidth_ratio,
            'estimated_latency_ns': hop_count_avg * 100,  # 100ns per hop
            'scalability_factor': math.log2(node_count) / 10  # Logarithmic scaling
        }

class NetworkInfrastructure:
    """Comprehensive network infrastructure management"""
    
    def __init__(self, name: str):
        self.name = name
        self.switches: List[NetworkSwitch] = []
        self.interfaces: List[NetworkInterface] = []
        self.cables: List[NetworkCable] = []
        self.segments: List[NetworkSegment] = []
        self.traffic_analyzer = NetworkTrafficAnalyzer()
        self.topology = NetworkTopology.LEAF_SPINE
        
    def add_spine_switches(self, count: int, model: str = "Spine_Switch_400G"):
        """Add spine switches for leaf-spine topology"""
        for i in range(count):
            switch = NetworkSwitch(
                model=f"{model}_{i+1}",
                switch_type="Spine",
                port_count=64,
                port_speed_gbps=400,
                switching_capacity_tbps=25.6,  # 64 * 400 Gbps
                forwarding_rate_mpps=19200,
                buffer_size_mb=256,
                power_consumption_w=2500,
                latency_ns=300,
                rack_units=2,
                cost=150000,
                protocols_supported=[NetworkProtocol.ETHERNET, NetworkProtocol.ROCE]
            )
            self.switches.append(switch)
    
    def add_leaf_switches(self, count: int, model: str = "Leaf_Switch_100G"):
        """Add leaf switches (Top-of-Rack)"""
        for i in range(count):
            switch = NetworkSwitch(
                model=f"{model}_{i+1}",
                switch_type="Leaf",
                port_count=48,
                port_speed_gbps=100,
                switching_capacity_tbps=4.8,  # 48 * 100 Gbps
                forwarding_rate_mpps=3600,
                buffer_size_mb=128,
                power_consumption_w=800,
                latency_ns=200,
                rack_units=1,
                cost=50000,
                protocols_supported=[NetworkProtocol.ETHERNET, NetworkProtocol.ROCE]
            )
            self.switches.append(switch)
    
    def add_gpu_interconnect(self, gpu_count: int, interconnect_type: str = "NVLink"):
        """Add GPU-to-GPU interconnect network"""
        if interconnect_type == "NVLink":
            # NVLink 4.0 specifications
            links_per_gpu = 18  # GB300 has 18 NVLink connections
            bandwidth_per_link_gbps = 900  # 900 GB/s bidirectional
            
            interface = NetworkInterface(
                name=f"NVLink_4_0_Network",
                protocol=NetworkProtocol.NVLINK,
                speed_gbps=bandwidth_per_link_gbps,
                port_count=gpu_count * links_per_gpu,
                power_consumption_w=gpu_count * links_per_gpu * 5,  # 5W per link
                latency_ns=25,  # Very low latency
                bandwidth_efficiency=0.95,
                cost_per_port=100,
                form_factor="On-chip"
            )
            self.interfaces.append(interface)
        
        elif interconnect_type == "InfiniBand":
            # InfiniBand HDR specifications
            interface = NetworkInterface(
                name=f"InfiniBand_HDR_Network",
                protocol=NetworkProtocol.INFINIBAND,
                speed_gbps=200,  # HDR 200 Gbps
                port_count=gpu_count * 2,  # 2 IB ports per GPU
                power_consumption_w=gpu_count * 2 * 15,  # 15W per port
                latency_ns=100,
                bandwidth_efficiency=0.90,
                cost_per_port=500,
                form_factor="PCIe Card"
            )
            self.interfaces.append(interface)
    
    def add_storage_network(self, storage_nodes: int, bandwidth_per_node_gbps: float = 100):
        """Add storage network infrastructure"""
        interface = NetworkInterface(
            name="Storage_Network",
            protocol=NetworkProtocol.ETHERNET,
            speed_gbps=bandwidth_per_node_gbps,
            port_count=storage_nodes,
            power_consumption_w=storage_nodes * 25,  # 25W per 100G port
            latency_ns=500,
            bandwidth_efficiency=0.85,
            cost_per_port=1000,
            form_factor="QSFP28"
        )
        self.interfaces.append(interface)
    
    def add_management_network(self, node_count: int):
        """Add management and monitoring network"""
        interface = NetworkInterface(
            name="Management_Network",
            protocol=NetworkProtocol.ETHERNET,
            speed_gbps=1,  # 1 Gbps management
            port_count=node_count,
            power_consumption_w=node_count * 2,  # 2W per 1G port
            latency_ns=1000,
            bandwidth_efficiency=0.80,
            cost_per_port=50,
            form_factor="RJ45"
        )
        self.interfaces.append(interface)
    
    def calculate_network_capacity(self) -> Dict[str, float]:
        """Calculate total network capacity"""
        total_bandwidth_gbps = 0
        total_ports = 0
        capacity_by_protocol = {}
        
        # Switch capacity
        switch_capacity_tbps = sum(switch.switching_capacity_tbps for switch in self.switches)
        
        # Interface capacity
        for interface in self.interfaces:
            interface_bandwidth = interface.speed_gbps * interface.port_count * interface.bandwidth_efficiency
            total_bandwidth_gbps += interface_bandwidth
            total_ports += interface.port_count
            
            protocol = interface.protocol.value
            if protocol not in capacity_by_protocol:
                capacity_by_protocol[protocol] = 0
            capacity_by_protocol[protocol] += interface_bandwidth
        
        return {
            'total_bandwidth_gbps': total_bandwidth_gbps,
            'total_bandwidth_tbps': total_bandwidth_gbps / 1000,
            'switch_capacity_tbps': switch_capacity_tbps,
            'total_ports': total_ports,
            'capacity_by_protocol': capacity_by_protocol
        }
    
    def calculate_network_latency(self, topology: NetworkTopology = None) -> Dict[str, float]:
        """Calculate network latency characteristics"""
        if topology is None:
            topology = self.topology
        
        # Base latencies by component
        switch_latency_ns = np.mean([switch.latency_ns for switch in self.switches]) if self.switches else 200
        interface_latency_ns = np.mean([interface.latency_ns for interface in self.interfaces]) if self.interfaces else 100
        
        # Topology-specific calculations
        traffic_analysis = self.traffic_analyzer.analyze_traffic_flow(topology, len(self.switches))
        
        avg_latency_ns = (traffic_analysis['average_hop_count'] * switch_latency_ns + 
                         interface_latency_ns)
        
        return {
            'average_latency_ns': avg_latency_ns,
            'average_latency_us': avg_latency_ns / 1000,
            'switch_latency_ns': switch_latency_ns,
            'interface_latency_ns': interface_latency_ns,
            'topology_hops': traffic_analysis['average_hop_count'],
            'topology': topology.value
        }
    
    def calculate_power_consumption(self) -> Dict[str, float]:
        """Calculate network power consumption"""
        switch_power_w = sum(switch.power_consumption_w for switch in self.switches)
        interface_power_w = sum(interface.power_consumption_w for interface in self.interfaces)
        
        # Add cooling overhead for network equipment (typically 30%)
        cooling_overhead_w = (switch_power_w + interface_power_w) * 0.3
        
        total_power_w = switch_power_w + interface_power_w + cooling_overhead_w
        
        return {
            'switch_power_w': switch_power_w,
            'interface_power_w': interface_power_w,
            'cooling_overhead_w': cooling_overhead_w,
            'total_power_w': total_power_w,
            'total_power_kw': total_power_w / 1000,
            'total_power_mw': total_power_w / 1e6
        }
    
    def calculate_network_costs(self) -> Dict[str, float]:
        """Calculate network infrastructure costs"""
        switch_costs = sum(switch.cost for switch in self.switches)
        
        interface_costs = 0
        for interface in self.interfaces:
            interface_costs += interface.port_count * interface.cost_per_port
        
        # Estimate cabling costs (assume average 10m per connection)
        cable_costs = sum(interface.port_count for interface in self.interfaces) * 10 * 50  # $50/m for high-speed cables
        
        # Infrastructure costs (racks, power, cooling)
        infrastructure_costs = (switch_costs + interface_costs) * 0.2
        
        total_costs = switch_costs + interface_costs + cable_costs + infrastructure_costs
        
        return {
            'switch_costs': switch_costs,
            'interface_costs': interface_costs,
            'cable_costs': cable_costs,
            'infrastructure_costs': infrastructure_costs,
            'total_costs': total_costs
        }
    
    def optimize_network_topology(self, gpu_count: int, 
                                storage_nodes: int) -> Dict[str, any]:
        """Optimize network topology for given workload"""
        bandwidth_req = self.traffic_analyzer.calculate_bandwidth_requirements(
            gpu_count, storage_nodes * 100  # Assume 100TB per storage node
        )
        
        # Determine optimal topology based on scale
        if gpu_count < 1000:
            recommended_topology = NetworkTopology.LEAF_SPINE
            oversubscription = 3.0
        elif gpu_count < 10000:
            recommended_topology = NetworkTopology.FAT_TREE
            oversubscription = 2.0
        else:
            recommended_topology = NetworkTopology.DRAGONFLY
            oversubscription = 1.5
        
        # Calculate required switches
        gpus_per_leaf = 32  # Typical GPU density per leaf switch
        leaf_switches_needed = math.ceil(gpu_count / gpus_per_leaf)
        spine_switches_needed = math.ceil(leaf_switches_needed / 16)  # 16 leafs per spine
        
        return {
            'recommended_topology': recommended_topology.value,
            'bandwidth_requirements': bandwidth_req,
            'switch_requirements': {
                'leaf_switches': leaf_switches_needed,
                'spine_switches': spine_switches_needed,
                'total_switches': leaf_switches_needed + spine_switches_needed
            },
            'oversubscription_ratio': oversubscription,
            'estimated_cost': (leaf_switches_needed * 50000 + spine_switches_needed * 150000),
            'estimated_power_kw': (leaf_switches_needed * 0.8 + spine_switches_needed * 2.5)
        }
    
    def calculate_network_reliability(self) -> Dict[str, float]:
        """Calculate network reliability metrics"""
        # Assume switch MTBF of 200,000 hours
        switch_mtbf_hours = 200000
        total_switches = len(self.switches)
        
        # Calculate system MTBF (series reliability)
        if total_switches > 0:
            system_mtbf_hours = switch_mtbf_hours / total_switches
            availability = system_mtbf_hours / (system_mtbf_hours + 8)  # 8 hours MTTR
        else:
            system_mtbf_hours = switch_mtbf_hours
            availability = 0.999
        
        # Calculate redundancy factor
        spine_switches = len([s for s in self.switches if s.switch_type == "Spine"])
        leaf_switches = len([s for s in self.switches if s.switch_type == "Leaf"])
        
        redundancy_factor = min(spine_switches / max(1, leaf_switches / 16), 2.0)
        
        return {
            'switch_mtbf_hours': switch_mtbf_hours,
            'system_mtbf_hours': system_mtbf_hours,
            'availability_percentage': availability * 100,
            'redundancy_factor': redundancy_factor,
            'fault_tolerance': 'High' if redundancy_factor > 1.5 else 'Medium' if redundancy_factor > 1.0 else 'Low'
        }
    
    def simulate_network_performance(self, workload_type: str = "AI_Training") -> Dict[str, any]:
        """Simulate network performance under different workloads"""
        capacity = self.calculate_network_capacity()
        latency = self.calculate_network_latency()
        
        if workload_type == "AI_Training":
            # AI training is bandwidth-intensive with burst patterns
            utilization_factor = 0.8  # High utilization
            burst_factor = 2.0
            latency_sensitivity = "High"
        elif workload_type == "AI_Inference":
            # AI inference has lower bandwidth but latency-sensitive
            utilization_factor = 0.4
            burst_factor = 1.2
            latency_sensitivity = "Very High"
        elif workload_type == "Data_Processing":
            # Data processing is steady-state bandwidth
            utilization_factor = 0.6
            burst_factor = 1.5
            latency_sensitivity = "Medium"
        else:
            utilization_factor = 0.5
            burst_factor = 1.3
            latency_sensitivity = "Medium"
        
        effective_bandwidth_gbps = capacity['total_bandwidth_gbps'] * utilization_factor
        peak_bandwidth_gbps = capacity['total_bandwidth_gbps'] * utilization_factor * burst_factor
        
        return {
            'workload_type': workload_type,
            'effective_bandwidth_gbps': effective_bandwidth_gbps,
            'peak_bandwidth_gbps': peak_bandwidth_gbps,
            'utilization_factor': utilization_factor,
            'burst_factor': burst_factor,
            'latency_sensitivity': latency_sensitivity,
            'average_latency_us': latency['average_latency_us'],
            'performance_rating': self._calculate_performance_rating(effective_bandwidth_gbps, latency['average_latency_us'])
        }
    
    def _calculate_performance_rating(self, bandwidth_gbps: float, latency_us: float) -> str:
        """Calculate overall network performance rating"""
        # Simple scoring based on bandwidth and latency
        bandwidth_score = min(bandwidth_gbps / 10000, 1.0)  # Normalize to 10 Tbps
        latency_score = max(0, 1.0 - latency_us / 100)  # Penalize latency > 100us
        
        overall_score = (bandwidth_score * 0.6 + latency_score * 0.4)
        
        if overall_score > 0.9:
            return "Excellent"
        elif overall_score > 0.7:
            return "Good"
        elif overall_score > 0.5:
            return "Fair"
        else:
            return "Poor"
    
    def generate_network_report(self) -> Dict[str, any]:
        """Generate comprehensive network infrastructure report"""
        return {
            'infrastructure_overview': {
                'name': self.name,
                'switches': len(self.switches),
                'interfaces': len(self.interfaces),
                'topology': self.topology.value
            },
            'capacity_analysis': self.calculate_network_capacity(),
            'latency_analysis': self.calculate_network_latency(),
            'power_consumption': self.calculate_power_consumption(),
            'cost_analysis': self.calculate_network_costs(),
            'reliability_metrics': self.calculate_network_reliability(),
            'performance_simulation': {
                'ai_training': self.simulate_network_performance("AI_Training"),
                'ai_inference': self.simulate_network_performance("AI_Inference"),
                'data_processing': self.simulate_network_performance("Data_Processing")
            }
        }