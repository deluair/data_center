import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum
import math

class GPUArchitecture(Enum):
    HOPPER = "hopper"
    BLACKWELL = "blackwell"
    BLACKWELL_ULTRA = "blackwell_ultra"
    ADA_LOVELACE = "ada_lovelace"

class PrecisionType(Enum):
    FP64 = "fp64"
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    FP8 = "fp8"
    FP4 = "fp4"
    INT8 = "int8"
    INT4 = "int4"

class WorkloadType(Enum):
    TRAINING = "training"
    INFERENCE = "inference"
    MIXED = "mixed"

@dataclass
class GPUSpecification:
    """GPU hardware specifications"""
    model: str
    architecture: GPUArchitecture
    memory_gb: int
    memory_bandwidth_gbps: float
    memory_type: str
    tdp_w: int
    base_clock_mhz: int
    boost_clock_mhz: int
    cuda_cores: int
    tensor_cores: int
    rt_cores: int
    nvlink_version: str
    nvlink_bandwidth_gbps: float
    pcie_version: str
    form_factor: str
    
@dataclass
class PerformanceMetrics:
    """GPU performance metrics by precision"""
    fp64_tflops: float
    fp32_tflops: float
    fp16_tflops: float
    bf16_tflops: float
    fp8_tflops: float
    fp4_tflops: float
    int8_tops: float
    int4_tops: float
    sparsity_factor: float = 1.0

@dataclass
class ServerConfiguration:
    """Server hardware configuration"""
    name: str
    gpu_count: int
    gpu_model: str
    cpu_model: str
    cpu_cores: int
    cpu_tdp_w: int
    system_memory_gb: int
    storage_type: str
    storage_capacity_tb: float
    network_interfaces: List[str]
    power_consumption_w: int
    rack_units: int

class GPUCluster:
    """GPU cluster management and performance calculation"""
    
    def __init__(self, name: str, workload_type: WorkloadType):
        self.name = name
        self.workload_type = workload_type
        self.gpu_specs: Dict[str, GPUSpecification] = {}
        self.performance_metrics: Dict[str, PerformanceMetrics] = {}
        self.servers: List[ServerConfiguration] = []
        self.gpu_counts: Dict[str, int] = {}
        
    def add_gpu_specification(self, gpu_spec: GPUSpecification, performance: PerformanceMetrics):
        """Add GPU specification and performance metrics"""
        self.gpu_specs[gpu_spec.model] = gpu_spec
        self.performance_metrics[gpu_spec.model] = performance
        
    def add_servers(self, server_config: ServerConfiguration, count: int):
        """Add servers to the cluster"""
        for i in range(count):
            server = ServerConfiguration(
                name=f"{server_config.name}_{i+1}",
                gpu_count=server_config.gpu_count,
                gpu_model=server_config.gpu_model,
                cpu_model=server_config.cpu_model,
                cpu_cores=server_config.cpu_cores,
                cpu_tdp_w=server_config.cpu_tdp_w,
                system_memory_gb=server_config.system_memory_gb,
                storage_type=server_config.storage_type,
                storage_capacity_tb=server_config.storage_capacity_tb,
                network_interfaces=server_config.network_interfaces,
                power_consumption_w=server_config.power_consumption_w,
                rack_units=server_config.rack_units
            )
            self.servers.append(server)
            
        # Update GPU counts
        gpu_model = server_config.gpu_model
        if gpu_model not in self.gpu_counts:
            self.gpu_counts[gpu_model] = 0
        self.gpu_counts[gpu_model] += count * server_config.gpu_count
        
    def calculate_cluster_performance(self, precision: PrecisionType, 
                                    utilization: float = 0.8) -> Dict[str, float]:
        """Calculate cluster performance for given precision"""
        total_performance = 0
        performance_breakdown = {}
        
        for gpu_model, count in self.gpu_counts.items():
            if gpu_model in self.performance_metrics:
                perf = self.performance_metrics[gpu_model]
                
                # Get performance for specified precision
                if precision == PrecisionType.FP64:
                    gpu_perf = perf.fp64_tflops
                elif precision == PrecisionType.FP32:
                    gpu_perf = perf.fp32_tflops
                elif precision == PrecisionType.FP16:
                    gpu_perf = perf.fp16_tflops
                elif precision == PrecisionType.BF16:
                    gpu_perf = perf.bf16_tflops
                elif precision == PrecisionType.FP8:
                    gpu_perf = perf.fp8_tflops
                elif precision == PrecisionType.FP4:
                    gpu_perf = perf.fp4_tflops
                elif precision == PrecisionType.INT8:
                    gpu_perf = perf.int8_tops
                elif precision == PrecisionType.INT4:
                    gpu_perf = perf.int4_tops
                else:
                    gpu_perf = perf.fp16_tflops  # Default to FP16
                
                cluster_perf = count * gpu_perf * utilization * perf.sparsity_factor
                performance_breakdown[gpu_model] = cluster_perf
                total_performance += cluster_perf
        
        return {
            'total_performance': total_performance,
            'precision': precision.value,
            'utilization': utilization,
            'breakdown': performance_breakdown,
            'units': 'TFLOPS' if 'fp' in precision.value else 'TOPS'
        }
    
    def calculate_memory_capacity(self) -> Dict[str, float]:
        """Calculate total memory capacity"""
        total_memory_gb = 0
        memory_breakdown = {}
        
        for gpu_model, count in self.gpu_counts.items():
            if gpu_model in self.gpu_specs:
                gpu_memory = self.gpu_specs[gpu_model].memory_gb
                cluster_memory = count * gpu_memory
                memory_breakdown[gpu_model] = cluster_memory
                total_memory_gb += cluster_memory
        
        return {
            'total_memory_gb': total_memory_gb,
            'total_memory_tb': total_memory_gb / 1024,
            'total_memory_pb': total_memory_gb / (1024 * 1024),
            'breakdown': memory_breakdown
        }
    
    def calculate_memory_bandwidth(self) -> Dict[str, float]:
        """Calculate total memory bandwidth"""
        total_bandwidth_gbps = 0
        bandwidth_breakdown = {}
        
        for gpu_model, count in self.gpu_counts.items():
            if gpu_model in self.gpu_specs:
                gpu_bandwidth = self.gpu_specs[gpu_model].memory_bandwidth_gbps
                cluster_bandwidth = count * gpu_bandwidth
                bandwidth_breakdown[gpu_model] = cluster_bandwidth
                total_bandwidth_gbps += cluster_bandwidth
        
        return {
            'total_bandwidth_gbps': total_bandwidth_gbps,
            'total_bandwidth_tbps': total_bandwidth_gbps / 1000,
            'breakdown': bandwidth_breakdown
        }
    
    def calculate_power_consumption(self) -> Dict[str, float]:
        """Calculate cluster power consumption"""
        gpu_power_w = 0
        cpu_power_w = 0
        system_power_w = 0
        
        for server in self.servers:
            # GPU power
            if server.gpu_model in self.gpu_specs:
                gpu_tdp = self.gpu_specs[server.gpu_model].tdp_w
                gpu_power_w += server.gpu_count * gpu_tdp
            
            # CPU and system power
            cpu_power_w += server.cpu_tdp_w
            system_power_w += server.power_consumption_w - (server.gpu_count * 
                            self.gpu_specs.get(server.gpu_model, type('', (), {'tdp_w': 0})).tdp_w) - server.cpu_tdp_w
        
        total_power_w = gpu_power_w + cpu_power_w + system_power_w
        
        return {
            'gpu_power_w': gpu_power_w,
            'gpu_power_mw': gpu_power_w / 1e6,
            'cpu_power_w': cpu_power_w,
            'cpu_power_mw': cpu_power_w / 1e6,
            'system_power_w': system_power_w,
            'system_power_mw': system_power_w / 1e6,
            'total_power_w': total_power_w,
            'total_power_mw': total_power_w / 1e6
        }
    
    def calculate_rack_requirements(self) -> Dict[str, int]:
        """Calculate rack space requirements"""
        total_rack_units = sum(server.rack_units for server in self.servers)
        total_racks = math.ceil(total_rack_units / 42)  # Assuming 42U racks
        
        return {
            'total_servers': len(self.servers),
            'total_rack_units': total_rack_units,
            'total_racks': total_racks,
            'rack_utilization': total_rack_units / (total_racks * 42) if total_racks > 0 else 0
        }

class ComputeInfrastructure:
    """Overall compute infrastructure management"""
    
    def __init__(self, name: str):
        self.name = name
        self.training_clusters: List[GPUCluster] = []
        self.inference_clusters: List[GPUCluster] = []
        self.mixed_clusters: List[GPUCluster] = []
        
    def add_cluster(self, cluster: GPUCluster):
        """Add cluster to infrastructure"""
        if cluster.workload_type == WorkloadType.TRAINING:
            self.training_clusters.append(cluster)
        elif cluster.workload_type == WorkloadType.INFERENCE:
            self.inference_clusters.append(cluster)
        else:
            self.mixed_clusters.append(cluster)
    
    def get_all_clusters(self) -> List[GPUCluster]:
        """Get all clusters"""
        return self.training_clusters + self.inference_clusters + self.mixed_clusters
    
    def calculate_total_performance(self, precision: PrecisionType, 
                                  utilization: float = 0.8) -> Dict[str, any]:
        """Calculate total infrastructure performance"""
        training_perf = 0
        inference_perf = 0
        mixed_perf = 0
        
        for cluster in self.training_clusters:
            perf = cluster.calculate_cluster_performance(precision, utilization)
            training_perf += perf['total_performance']
        
        for cluster in self.inference_clusters:
            perf = cluster.calculate_cluster_performance(precision, utilization)
            inference_perf += perf['total_performance']
        
        for cluster in self.mixed_clusters:
            perf = cluster.calculate_cluster_performance(precision, utilization)
            mixed_perf += perf['total_performance']
        
        total_perf = training_perf + inference_perf + mixed_perf
        units = 'TFLOPS' if 'fp' in precision.value else 'TOPS'
        
        # Convert to appropriate scale
        if total_perf >= 1e6:
            scale_factor = 1e6
            scale_unit = 'Exa' + units
        elif total_perf >= 1e3:
            scale_factor = 1e3
            scale_unit = 'Peta' + units
        else:
            scale_factor = 1
            scale_unit = units
        
        return {
            'training_performance': training_perf,
            'inference_performance': inference_perf,
            'mixed_performance': mixed_perf,
            'total_performance': total_perf,
            'scaled_performance': total_perf / scale_factor,
            'units': scale_unit,
            'precision': precision.value,
            'utilization': utilization
        }
    
    def calculate_total_resources(self) -> Dict[str, any]:
        """Calculate total infrastructure resources"""
        total_gpus = 0
        total_servers = 0
        total_memory_gb = 0
        total_bandwidth_gbps = 0
        total_power_mw = 0
        total_racks = 0
        
        gpu_breakdown = {}
        
        for cluster in self.get_all_clusters():
            # GPU counts
            for gpu_model, count in cluster.gpu_counts.items():
                if gpu_model not in gpu_breakdown:
                    gpu_breakdown[gpu_model] = 0
                gpu_breakdown[gpu_model] += count
                total_gpus += count
            
            # Other resources
            total_servers += len(cluster.servers)
            
            memory_info = cluster.calculate_memory_capacity()
            total_memory_gb += memory_info['total_memory_gb']
            
            bandwidth_info = cluster.calculate_memory_bandwidth()
            total_bandwidth_gbps += bandwidth_info['total_bandwidth_gbps']
            
            power_info = cluster.calculate_power_consumption()
            total_power_mw += power_info['total_power_mw']
            
            rack_info = cluster.calculate_rack_requirements()
            total_racks += rack_info['total_racks']
        
        return {
            'total_gpus': total_gpus,
            'gpu_breakdown': gpu_breakdown,
            'total_servers': total_servers,
            'total_memory_gb': total_memory_gb,
            'total_memory_tb': total_memory_gb / 1024,
            'total_memory_pb': total_memory_gb / (1024 * 1024),
            'total_bandwidth_gbps': total_bandwidth_gbps,
            'total_bandwidth_tbps': total_bandwidth_gbps / 1000,
            'total_power_mw': total_power_mw,
            'total_racks': total_racks,
            'training_clusters': len(self.training_clusters),
            'inference_clusters': len(self.inference_clusters),
            'mixed_clusters': len(self.mixed_clusters)
        }
    
    def optimize_workload_distribution(self, training_ratio: float = 0.7) -> Dict[str, any]:
        """Optimize workload distribution across clusters"""
        total_resources = self.calculate_total_resources()
        total_gpus = total_resources['total_gpus']
        
        # Calculate optimal distribution
        training_gpus = int(total_gpus * training_ratio)
        inference_gpus = total_gpus - training_gpus
        
        # Performance estimates
        fp16_training = self.calculate_total_performance(PrecisionType.FP16, 0.8)
        fp4_training = self.calculate_total_performance(PrecisionType.FP4, 0.8)
        fp16_inference = self.calculate_total_performance(PrecisionType.FP16, 0.9)
        
        return {
            'optimal_distribution': {
                'training_gpus': training_gpus,
                'inference_gpus': inference_gpus,
                'training_ratio': training_ratio,
                'inference_ratio': 1 - training_ratio
            },
            'performance_estimates': {
                'fp16_training_exaflops': fp16_training['scaled_performance'],
                'fp4_training_pflops': fp4_training['scaled_performance'],
                'fp16_inference_exaflops': fp16_inference['scaled_performance']
            },
            'resource_utilization': {
                'total_gpus': total_gpus,
                'total_power_mw': total_resources['total_power_mw'],
                'total_memory_pb': total_resources['total_memory_pb']
            }
        }
    
    def generate_compute_report(self) -> Dict[str, any]:
        """Generate comprehensive compute infrastructure report"""
        return {
            'infrastructure_overview': {
                'name': self.name,
                'training_clusters': len(self.training_clusters),
                'inference_clusters': len(self.inference_clusters),
                'mixed_clusters': len(self.mixed_clusters)
            },
            'total_resources': self.calculate_total_resources(),
            'performance_fp16': self.calculate_total_performance(PrecisionType.FP16),
            'performance_fp4': self.calculate_total_performance(PrecisionType.FP4),
            'performance_int8': self.calculate_total_performance(PrecisionType.INT8),
            'workload_optimization': self.optimize_workload_distribution()
        }