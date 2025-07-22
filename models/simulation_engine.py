import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import math
from datetime import datetime, timedelta
import json

# Import all model components
from .power_model import PowerInfrastructure, PowerConsumption
from .cooling_model import CoolingSystem
from .compute_model import ComputeInfrastructure, GPUSpecification, WorkloadType
from .storage_model import StorageSystem
from .network_model import NetworkInfrastructure
from .facility_model import FacilityInfrastructure
from .financial_model import FinancialModel, FinancialAssumptions
from .monitoring_model import MonitoringSystem

class SimulationMode(Enum):
    STEADY_STATE = "steady_state"
    DYNAMIC = "dynamic"
    STRESS_TEST = "stress_test"
    OPTIMIZATION = "optimization"
    SCENARIO_ANALYSIS = "scenario_analysis"

class OptimizationObjective(Enum):
    MINIMIZE_COST = "minimize_cost"
    MAXIMIZE_PERFORMANCE = "maximize_performance"
    MINIMIZE_POWER = "minimize_power"
    MAXIMIZE_EFFICIENCY = "maximize_efficiency"
    MINIMIZE_CARBON = "minimize_carbon"

@dataclass
class SimulationParameters:
    """Simulation configuration parameters"""
    mode: SimulationMode
    duration_hours: int
    time_step_minutes: int
    optimization_objective: Optional[OptimizationObjective] = None
    workload_profile: str = "mixed"  # "training", "inference", "mixed"
    utilization_target: float = 0.8  # Target utilization percentage
    power_budget_mw: float = 5000.0  # Power budget in MW
    cost_budget_millions: float = 1000.0  # Cost budget in millions
    
@dataclass
class WorkloadScenario:
    """Workload scenario definition"""
    name: str
    description: str
    training_ratio: float  # 0.0 to 1.0
    inference_ratio: float  # 0.0 to 1.0
    peak_utilization: float
    duration_hours: int
    power_scaling_factor: float = 1.0
    performance_scaling_factor: float = 1.0

@dataclass
class SimulationResults:
    """Comprehensive simulation results"""
    timestamp: datetime
    simulation_id: str
    parameters: SimulationParameters
    
    # Performance metrics
    total_performance_fp16: float  # ExaFLOPS
    total_performance_fp4: float   # PFLOPS
    average_utilization: float
    peak_utilization: float
    
    # Power metrics
    total_power_consumption_mw: float
    average_power_consumption_mw: float
    peak_power_consumption_mw: float
    pue: float
    power_efficiency: float
    
    # Thermal metrics
    average_temperature: float
    peak_temperature: float
    cooling_efficiency: float
    thermal_design_margin: float
    
    # Financial metrics
    total_capex_millions: float
    annual_opex_millions: float
    annual_revenue_millions: float
    roi_percentage: float
    payback_period_years: float
    
    # Infrastructure metrics
    total_gpus: int
    total_servers: int
    total_racks: int
    floor_space_sqft: int
    
    # Environmental metrics
    annual_carbon_emissions_tons: float
    water_usage_gallons_per_year: float
    waste_heat_recovery_mw: float
    
    # Reliability metrics
    system_availability: float
    mtbf_hours: float
    redundancy_level: float
    
    # Optimization metrics
    optimization_score: float
    constraints_satisfied: bool
    optimization_iterations: int

class WorkloadGenerator:
    """Generate realistic workload patterns"""
    
    def __init__(self):
        self.workload_templates = {
            'training_heavy': {
                'training_ratio': 0.8,
                'inference_ratio': 0.2,
                'peak_utilization': 0.95,
                'power_scaling': 1.0
            },
            'inference_heavy': {
                'training_ratio': 0.2,
                'inference_ratio': 0.8,
                'peak_utilization': 0.85,
                'power_scaling': 0.9
            },
            'mixed_workload': {
                'training_ratio': 0.5,
                'inference_ratio': 0.5,
                'peak_utilization': 0.9,
                'power_scaling': 0.95
            },
            'research_burst': {
                'training_ratio': 0.9,
                'inference_ratio': 0.1,
                'peak_utilization': 0.98,
                'power_scaling': 1.05
            }
        }
    
    def generate_daily_pattern(self, scenario: WorkloadScenario, hours: int = 24) -> List[Dict[str, float]]:
        """Generate daily workload pattern"""
        pattern = []
        
        for hour in range(hours):
            # Create realistic daily variation
            base_utilization = scenario.peak_utilization
            
            # Business hours have higher utilization
            if 8 <= hour <= 18:
                utilization_factor = 0.9 + 0.1 * np.sin((hour - 8) * np.pi / 10)
            else:
                utilization_factor = 0.6 + 0.2 * np.random.random()
            
            current_utilization = base_utilization * utilization_factor
            
            # Add some randomness
            current_utilization += np.random.normal(0, 0.05)
            current_utilization = max(0.1, min(1.0, current_utilization))
            
            pattern.append({
                'hour': hour,
                'utilization': current_utilization,
                'training_ratio': scenario.training_ratio,
                'inference_ratio': scenario.inference_ratio,
                'power_scaling': scenario.power_scaling_factor
            })
        
        return pattern
    
    def generate_stress_test_pattern(self, duration_hours: int = 8) -> List[Dict[str, float]]:
        """Generate stress test workload pattern"""
        pattern = []
        
        for hour in range(duration_hours):
            # Gradually ramp up to maximum load
            if hour < 2:
                utilization = 0.5 + (hour / 2) * 0.5  # Ramp up
            elif hour < duration_hours - 2:
                utilization = 1.0  # Sustained maximum
            else:
                utilization = 1.0 - ((hour - (duration_hours - 2)) / 2) * 0.3  # Ramp down
            
            pattern.append({
                'hour': hour,
                'utilization': utilization,
                'training_ratio': 0.8,  # Training-heavy for stress test
                'inference_ratio': 0.2,
                'power_scaling': 1.1  # Higher power consumption
            })
        
        return pattern

class PerformanceOptimizer:
    """Optimize data center performance"""
    
    def __init__(self):
        self.optimization_history = []
        
    def optimize_gpu_allocation(self, total_gpus: int, workload_mix: Dict[str, float], 
                              performance_targets: Dict[str, float]) -> Dict[str, Any]:
        """Optimize GPU allocation across workloads"""
        
        # Define workload characteristics
        workload_efficiency = {
            'training_fp16': 1.0,
            'training_fp4': 0.9,  # Slightly lower efficiency for FP4
            'inference_fp16': 0.8,  # Lower utilization for inference
            'inference_fp4': 0.85
        }
        
        # Calculate optimal allocation
        training_gpus = int(total_gpus * workload_mix.get('training_ratio', 0.5))
        inference_gpus = total_gpus - training_gpus
        
        # Optimize precision allocation within each workload type
        training_fp16_gpus = int(training_gpus * 0.7)  # 70% FP16 for training
        training_fp4_gpus = training_gpus - training_fp16_gpus
        
        inference_fp16_gpus = int(inference_gpus * 0.3)  # 30% FP16 for inference
        inference_fp4_gpus = inference_gpus - inference_fp16_gpus
        
        allocation = {
            'training_fp16_gpus': training_fp16_gpus,
            'training_fp4_gpus': training_fp4_gpus,
            'inference_fp16_gpus': inference_fp16_gpus,
            'inference_fp4_gpus': inference_fp4_gpus,
            'total_training_gpus': training_gpus,
            'total_inference_gpus': inference_gpus
        }
        
        # Calculate expected performance
        gpu_spec = GPUSpecification(
            name="GB300",
            architecture="Blackwell",
            memory_gb=192,
            memory_bandwidth_gbps=8000,
            fp16_tflops=2500,
            fp4_tflops=10000,
            power_consumption_w=1000,
            cost=40000
        )
        
        expected_performance = {
            'training_fp16_exaflops': (training_fp16_gpus * gpu_spec.fp16_tflops * 
                                     workload_efficiency['training_fp16']) / 1e6,
            'training_fp4_pflops': (training_fp4_gpus * gpu_spec.fp4_tflops * 
                                  workload_efficiency['training_fp4']) / 1e3,
            'inference_fp16_exaflops': (inference_fp16_gpus * gpu_spec.fp16_tflops * 
                                      workload_efficiency['inference_fp16']) / 1e6,
            'inference_fp4_pflops': (inference_fp4_gpus * gpu_spec.fp4_tflops * 
                                   workload_efficiency['inference_fp4']) / 1e3
        }
        
        return {
            'allocation': allocation,
            'expected_performance': expected_performance,
            'optimization_score': self._calculate_allocation_score(allocation, expected_performance)
        }
    
    def _calculate_allocation_score(self, allocation: Dict, performance: Dict) -> float:
        """Calculate optimization score for GPU allocation"""
        # Score based on balanced utilization and performance
        training_ratio = allocation['total_training_gpus'] / (allocation['total_training_gpus'] + allocation['total_inference_gpus'])
        balance_score = 1.0 - abs(training_ratio - 0.5) * 2  # Prefer balanced allocation
        
        # Performance score based on total compute capability
        total_performance = (performance['training_fp16_exaflops'] + 
                           performance['inference_fp16_exaflops'] + 
                           performance['training_fp4_pflops'] / 1000 + 
                           performance['inference_fp4_pflops'] / 1000)
        
        performance_score = min(total_performance / 10000, 1.0)  # Normalize to 0-1
        
        return (balance_score * 0.3 + performance_score * 0.7) * 100
    
    def optimize_power_distribution(self, total_power_mw: float, 
                                  infrastructure_components: Dict[str, float]) -> Dict[str, Any]:
        """Optimize power distribution across infrastructure"""
        
        # Define optimal power ratios
        optimal_ratios = {
            'compute': 0.70,  # 70% for compute (GPUs, CPUs)
            'cooling': 0.15,  # 15% for cooling
            'storage': 0.05,  # 5% for storage
            'network': 0.03,  # 3% for networking
            'facility': 0.07  # 7% for facility (lighting, UPS, etc.)
        }
        
        # Calculate optimal power allocation
        optimal_allocation = {}
        for component, ratio in optimal_ratios.items():
            optimal_allocation[component] = total_power_mw * ratio
        
        # Calculate efficiency metrics
        compute_power = optimal_allocation['compute']
        total_infrastructure_power = sum(optimal_allocation.values()) - compute_power
        pue = total_power_mw / compute_power if compute_power > 0 else 2.0
        
        return {
            'optimal_allocation_mw': optimal_allocation,
            'pue': pue,
            'power_efficiency': compute_power / total_power_mw,
            'optimization_recommendations': self._generate_power_recommendations(optimal_allocation)
        }
    
    def _generate_power_recommendations(self, allocation: Dict[str, float]) -> List[str]:
        """Generate power optimization recommendations"""
        recommendations = []
        
        if allocation['cooling'] / sum(allocation.values()) > 0.20:
            recommendations.append("Consider more efficient cooling systems to reduce cooling power")
        
        if allocation['facility'] / sum(allocation.values()) > 0.10:
            recommendations.append("Optimize facility systems (lighting, UPS) for better efficiency")
        
        if allocation['compute'] / sum(allocation.values()) < 0.65:
            recommendations.append("Increase compute power ratio by optimizing infrastructure efficiency")
        
        return recommendations

class SimulationEngine:
    """Main simulation engine for AI data center"""
    
    def __init__(self, name: str):
        self.name = name
        self.simulation_id = f"sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize all subsystems
        self.power_infrastructure = None
        self.cooling_system = None
        self.compute_infrastructure = None
        self.storage_system = None
        self.network_infrastructure = None
        self.facility_infrastructure = None
        self.financial_model = None
        self.monitoring_system = None
        
        # Simulation components
        self.workload_generator = WorkloadGenerator()
        self.performance_optimizer = PerformanceOptimizer()
        
        # Results storage
        self.simulation_results: List[SimulationResults] = []
        
    def initialize_infrastructure(self, gpu_count: int, power_budget_mw: float):
        """Initialize all infrastructure components"""
        
        # Initialize compute infrastructure
        self.compute_infrastructure = ComputeInfrastructure("5GW_Compute")
        
        # Add GPU clusters
        from .compute_model import GPUArchitecture, GPUCluster, PerformanceMetrics, ServerConfiguration
        gpu_spec = GPUSpecification(
            model="GB300",
            architecture=GPUArchitecture.BLACKWELL,
            memory_gb=192,
            memory_bandwidth_gbps=8000,
            memory_type="HBM3e",
            tdp_w=1000,
            base_clock_mhz=1400,
            boost_clock_mhz=1900,
            cuda_cores=20480,
            tensor_cores=512,
            rt_cores=128,
            nvlink_version="5.0",
            nvlink_bandwidth_gbps=900,
            pcie_version="5.0",
            form_factor="SXM5"
        )
        
        # Create performance metrics for the GPU
        performance = PerformanceMetrics(
            fp64_tflops=80.0,
            fp32_tflops=165.0,
            fp16_tflops=2500.0,
            bf16_tflops=2500.0,
            fp8_tflops=5000.0,
            fp4_tflops=10000.0,
            int8_tops=5000.0,
            int4_tops=10000.0,
            sparsity_factor=2.0
        )
        
        # Create server configuration
        server_config = ServerConfiguration(
            name="GB300_Server",
            gpu_count=8,
            gpu_model="GB300",
            cpu_model="Intel Xeon Platinum 8480+",
            cpu_cores=112,
            cpu_tdp_w=350,
            system_memory_gb=2048,
            storage_type="NVMe SSD",
            storage_capacity_tb=30.72,
            network_interfaces=["400GbE", "InfiniBand HDR"],
            power_consumption_w=10000,
            rack_units=4
        )
        
        cluster_size = 8  # 8 GPUs per server
        servers_per_cluster = 100  # 100 servers per cluster
        gpus_per_cluster = cluster_size * servers_per_cluster
        num_clusters = gpu_count // gpus_per_cluster
        
        for i in range(num_clusters):
            cluster = GPUCluster(f"cluster_{i+1}", WorkloadType.MIXED)
            cluster.add_gpu_specification(gpu_spec, performance)
            cluster.add_servers(server_config, servers_per_cluster)
            self.compute_infrastructure.add_cluster(cluster)
        
        # Initialize power infrastructure
        self.power_infrastructure = PowerInfrastructure(power_budget_mw)
        self.power_infrastructure.add_grid_connection(voltage_kv=500.0, capacity_mw=power_budget_mw, cost_per_mwh=50.0)
        self.power_infrastructure.add_backup_generators(count=50, capacity_mw_each=100.0, fuel_type="natural_gas")
        self.power_infrastructure.add_transformers(count=20, capacity_mva_each=250.0, primary_kv=500.0, secondary_kv=13.8)
        self.power_infrastructure.add_ups_systems(count=100, capacity_mw_each=50.0, backup_time_minutes=15.0)
        
        # Initialize cooling system
        # Calculate cooling requirements
        total_heat_load_mw = power_budget_mw * 0.7  # 70% of power becomes heat
        self.cooling_system = CoolingSystem(total_heat_load_mw)
        
        # Add cooling systems
        cooling_capacity_kw = total_heat_load_mw * 1000 * 1.2  # 20% overhead, convert to kW
        chiller_count = max(1, int(cooling_capacity_kw / 50000))  # 50MW chillers
        capacity_per_chiller = cooling_capacity_kw / chiller_count
        
        self.cooling_system.add_chilled_water_system(count=chiller_count, capacity_kw_each=capacity_per_chiller, cop=6.0)
        self.cooling_system.add_liquid_cooling_loops(count=chiller_count * 10, capacity_kw_each=capacity_per_chiller / 10)
        self.cooling_system.add_cooling_towers(count=chiller_count, capacity_kw_each=capacity_per_chiller * 1.1)
        
        # Initialize storage system
        self.storage_system = StorageSystem("5GW_Storage")
        
        # Add high-performance storage for AI workloads
        storage_capacity_tb = gpu_count * 0.5  # 0.5 TB per GPU
        nvme_count = max(1, int(storage_capacity_tb / 100))  # 100TB NVMe drives
        capacity_per_drive = storage_capacity_tb / nvme_count
        
        self.storage_system.add_nvme_storage(count=nvme_count, capacity_tb_each=capacity_per_drive, model="Enterprise_NVMe_100TB")
        
        # Add additional SSD storage for warm data
        ssd_capacity_tb = storage_capacity_tb * 2  # 2x for warm storage
        ssd_count = max(1, int(ssd_capacity_tb / 50))  # 50TB SSD drives
        ssd_capacity_per_drive = ssd_capacity_tb / ssd_count
        
        self.storage_system.add_ssd_storage(count=ssd_count, capacity_tb_each=ssd_capacity_per_drive, model="Enterprise_SSD_50TB")
        
        # Initialize network infrastructure
        self.network_infrastructure = NetworkInfrastructure("5GW_Network")
        
        # Add high-speed networking
        # Calculate switch requirements based on GPU count
        spine_switch_count = max(1, gpu_count // 1000)  # 1 spine switch per 1000 GPUs
        leaf_switch_count = max(1, gpu_count // 100)    # 1 leaf switch per 100 GPUs
        
        self.network_infrastructure.add_spine_switches(
            count=spine_switch_count,
            model="Spine_Switch_400G"
        )
        
        self.network_infrastructure.add_leaf_switches(
            count=leaf_switch_count,
            model="Leaf_Switch_100G"
        )
        
        # Add GPU interconnect for high-performance computing
        self.network_infrastructure.add_gpu_interconnect(
            gpu_count=gpu_count,
            interconnect_type="NVLink"
        )
        
        # Initialize facility infrastructure
        self.facility_infrastructure = FacilityInfrastructure("5GW_Facility")
        
        # Calculate facility requirements
        server_count = gpu_count // 8
        rack_count = server_count // 4  # 4 servers per rack
        floor_space_sqm = rack_count * 2.8  # ~30 sqft = 2.8 sqm per rack
        
        # Add compute floors for GPU infrastructure
        compute_racks = int(rack_count * 0.8)  # 80% for compute
        storage_racks = int(rack_count * 0.15)  # 15% for storage
        network_racks = int(rack_count * 0.05)  # 5% for network
        
        self.facility_infrastructure.add_compute_floor(
            area_sqm=floor_space_sqm * 0.8,
            rack_count=compute_racks,
            power_density_kw_per_sqm=15.0
        )
        
        self.facility_infrastructure.add_storage_floor(
            area_sqm=floor_space_sqm * 0.15,
            rack_count=storage_racks
        )
        
        self.facility_infrastructure.add_network_floor(
            area_sqm=floor_space_sqm * 0.05,
            rack_count=network_racks
        )
        
        # Add high-density racks for GPU servers
        self.facility_infrastructure.add_high_density_racks(
            count=compute_racks,
            power_capacity_kw=50.0
        )
        
        # Initialize financial model with assumptions
        financial_assumptions = FinancialAssumptions(
            discount_rate=0.12,  # 12% WACC
            tax_rate=0.25,  # 25% corporate tax rate
            inflation_rate=0.03,  # 3% inflation
            electricity_cost_per_kwh=0.10,  # $0.10 per kWh
            land_cost_per_sqm=500.0,  # $500 per sqm
            construction_cost_per_sqm=2000.0,  # $2000 per sqm
            financing_rate=0.06,  # 6% debt financing
            debt_to_equity_ratio=0.6  # 60% debt, 40% equity
        )
        
        self.financial_model = FinancialModel("5GW_Financial", financial_assumptions)
        
        # Add AI training revenue
        self.financial_model.add_ai_training_revenue(
            gpu_hours_per_year=gpu_count * 8760 * 0.8,  # 80% utilization
            price_per_gpu_hour=5.0  # $5 per GPU hour
        )
        
        # Add AI inference revenue
        self.financial_model.add_ai_inference_revenue(
            inference_requests_per_year=gpu_count * 1000000,  # 1M requests per GPU per year
            price_per_request=0.001  # $0.001 per request
        )
        
        # Add hardware CAPEX
        self.financial_model.add_hardware_capex(
            gpu_cost=gpu_count * 40000,  # $40k per GPU
            server_cost=gpu_count * 10000 / 8,  # $10k per server (8 GPUs per server)
            network_cost=power_budget_mw * 5000,  # $5k per MW for networking
            storage_cost=gpu_count * 2000  # $2k storage per GPU
        )
        
        # Add infrastructure CAPEX
        self.financial_model.add_infrastructure_capex(
            facility_cost=power_budget_mw * 20000,  # $20k per MW for facility
            power_cost=power_budget_mw * 15000,  # $15k per MW for power infrastructure
            cooling_cost=power_budget_mw * 10000  # $10k per MW for cooling
        )
        
        # Add power OPEX
        self.financial_model.add_power_opex(
            annual_power_consumption_mwh=power_budget_mw * 8760 * 0.8  # 80% average utilization
        )
        
        # Add personnel OPEX
        self.financial_model.add_personnel_opex(
            staff_count=max(100, gpu_count // 10000),  # 1 staff per 10k GPUs, minimum 100
            average_salary=150000  # $150k average salary
        )
        
        # Initialize monitoring system
        self.monitoring_system = MonitoringSystem("5GW_Monitoring")
        self.monitoring_system.define_gpu_metrics()
        self.monitoring_system.define_system_metrics()
        self.monitoring_system.setup_default_alerts()
        self.monitoring_system.add_monitoring_agents(gpu_count, server_count)
    
    def run_simulation(self, parameters: SimulationParameters) -> SimulationResults:
        """Run comprehensive data center simulation"""
        
        print(f"Starting simulation: {self.simulation_id}")
        print(f"Mode: {parameters.mode.value}")
        print(f"Duration: {parameters.duration_hours} hours")
        
        # Generate workload pattern based on simulation mode
        if parameters.mode == SimulationMode.STRESS_TEST:
            workload_pattern = self.workload_generator.generate_stress_test_pattern(parameters.duration_hours)
        else:
            scenario = WorkloadScenario(
                name="default",
                description="Default workload scenario",
                training_ratio=0.6,
                inference_ratio=0.4,
                peak_utilization=parameters.utilization_target,
                duration_hours=parameters.duration_hours
            )
            workload_pattern = self.workload_generator.generate_daily_pattern(scenario, parameters.duration_hours)
        
        # Run simulation steps
        simulation_data = []
        
        for step, workload in enumerate(workload_pattern):
            step_data = self._simulate_time_step(workload, step)
            simulation_data.append(step_data)
        
        # Analyze results
        results = self._analyze_simulation_results(simulation_data, parameters)
        
        # Store results
        self.simulation_results.append(results)
        
        # Only print completion message for non-optimization runs to reduce verbosity
        if parameters.optimization_objective is None:
            print(f"Simulation completed: {self.simulation_id}")
        
        return results
    
    def _simulate_time_step(self, workload: Dict[str, float], step: int) -> Dict[str, Any]:
        """Simulate a single time step"""
        
        # Calculate current performance
        all_clusters = self.compute_infrastructure.get_all_clusters()
        total_gpus = sum(sum(cluster.gpu_counts.values()) for cluster in all_clusters)
        current_utilization = workload['utilization']
        
        # Performance calculations
        if all_clusters and all_clusters[0].gpu_specs:
            # Get the first GPU specification from the first cluster
            first_gpu_model = list(all_clusters[0].gpu_specs.keys())[0]
            gpu_spec = all_clusters[0].gpu_specs[first_gpu_model]
            performance_metrics = all_clusters[0].performance_metrics[first_gpu_model]
        else:
            # Fallback values if no clusters
            return {
                'step': step,
                'utilization': current_utilization,
                'performance_fp16': 0.0,
                'performance_fp4': 0.0,
                'power_consumption_mw': 0.0,
                'temperature_c': 25.0
            }
        
        training_gpus = int(total_gpus * workload['training_ratio'] * current_utilization)
        inference_gpus = int(total_gpus * workload['inference_ratio'] * current_utilization)
        
        training_performance_fp16 = training_gpus * performance_metrics.fp16_tflops / 1e6  # ExaFLOPS
        training_performance_fp4 = training_gpus * performance_metrics.fp4_tflops / 1e3   # PFLOPS
        inference_performance_fp16 = inference_gpus * performance_metrics.fp16_tflops * 0.8 / 1e6  # ExaFLOPS (80% efficiency)
        inference_performance_fp4 = inference_gpus * performance_metrics.fp4_tflops * 0.85 / 1e3   # PFLOPS (85% efficiency)
        
        # Power consumption calculations
        active_gpus = training_gpus + inference_gpus
        gpu_power_mw = active_gpus * gpu_spec.tdp_w * workload.get('power_scaling', 1.0) / 1e6
        
        # Infrastructure power (cooling, networking, etc.)
        cooling_power_mw = gpu_power_mw * 0.15  # 15% for cooling
        other_power_mw = gpu_power_mw * 0.10    # 10% for other infrastructure
        total_power_mw = gpu_power_mw + cooling_power_mw + other_power_mw
        
        # Thermal calculations
        heat_load_mw = gpu_power_mw * 0.95  # 95% of GPU power becomes heat
        average_temp = 65 + (current_utilization * 20)  # Temperature rises with utilization
        
        return {
            'step': step,
            'utilization': current_utilization,
            'active_gpus': active_gpus,
            'training_performance_fp16': training_performance_fp16,
            'training_performance_fp4': training_performance_fp4,
            'inference_performance_fp16': inference_performance_fp16,
            'inference_performance_fp4': inference_performance_fp4,
            'total_power_mw': total_power_mw,
            'gpu_power_mw': gpu_power_mw,
            'cooling_power_mw': cooling_power_mw,
            'heat_load_mw': heat_load_mw,
            'average_temp': average_temp
        }
    
    def _analyze_simulation_results(self, simulation_data: List[Dict], 
                                  parameters: SimulationParameters) -> SimulationResults:
        """Analyze simulation results and generate comprehensive report"""
        
        # Calculate aggregate metrics
        total_steps = len(simulation_data)
        
        # Performance metrics
        avg_training_fp16 = np.mean([step['training_performance_fp16'] for step in simulation_data])
        avg_training_fp4 = np.mean([step['training_performance_fp4'] for step in simulation_data])
        avg_inference_fp16 = np.mean([step['inference_performance_fp16'] for step in simulation_data])
        avg_inference_fp4 = np.mean([step['inference_performance_fp4'] for step in simulation_data])
        
        total_performance_fp16 = avg_training_fp16 + avg_inference_fp16
        total_performance_fp4 = avg_training_fp4 + avg_inference_fp4
        
        avg_utilization = np.mean([step['utilization'] for step in simulation_data])
        peak_utilization = np.max([step['utilization'] for step in simulation_data])
        
        # Power metrics
        avg_power_mw = np.mean([step['total_power_mw'] for step in simulation_data])
        peak_power_mw = np.max([step['total_power_mw'] for step in simulation_data])
        avg_gpu_power_mw = np.mean([step['gpu_power_mw'] for step in simulation_data])
        
        pue = avg_power_mw / avg_gpu_power_mw if avg_gpu_power_mw > 0 else 2.0
        power_efficiency = avg_gpu_power_mw / avg_power_mw if avg_power_mw > 0 else 0.5
        
        # Thermal metrics
        avg_temperature = np.mean([step['average_temp'] for step in simulation_data])
        peak_temperature = np.max([step['average_temp'] for step in simulation_data])
        cooling_efficiency = 4.5  # Assumed COP
        thermal_design_margin = max(0, (90 - peak_temperature) / 90 * 100)  # Margin to 90°C limit
        
        # Infrastructure metrics
        all_clusters = self.compute_infrastructure.get_all_clusters()
        total_gpus = sum(sum(cluster.gpu_counts.values()) for cluster in all_clusters)
        total_servers = total_gpus // 8
        total_racks = total_servers // 4
        floor_space_sqft = 1000000  # Default floor space in sqft
        
        # Financial metrics (get from financial model)
        revenue_projection = self.financial_model.calculate_revenue_projection(5)
        cost_projection = self.financial_model.calculate_cost_projection(5)
        cash_flow = self.financial_model.calculate_cash_flow(5)
        
        # Calculate CAPEX from initial costs in cost models
        total_capex = sum(cost.initial_cost for cost in self.financial_model.cost_models) / 1e6
        annual_opex = cost_projection['average_annual_costs'] / 1e6
        annual_revenue = revenue_projection['total_revenue'] / 1e6
        
        roi_metrics = self.financial_model.calculate_roi_metrics(5)
        roi_percentage = roi_metrics['roi_percent']
        payback_period = roi_metrics['payback_period_years']
        
        # Environmental metrics
        annual_power_gwh = avg_power_mw * 8760 / 1000  # GWh per year
        carbon_intensity_kg_per_mwh = 400  # Assumed grid carbon intensity
        annual_carbon_tons = annual_power_gwh * 1000 * carbon_intensity_kg_per_mwh / 1000
        
        water_usage_gallons = annual_power_gwh * 1000 * 300  # 300 gallons per MWh for cooling
        waste_heat_recovery_mw = avg_power_mw * 0.6  # 60% heat recovery potential
        
        # Reliability metrics
        system_availability = 99.9  # Assumed high availability
        mtbf_hours = 50000  # Mean time between failures
        redundancy_level = 1.2  # 20% redundancy
        
        # Optimization score
        optimization_score = self._calculate_optimization_score({
            'performance': total_performance_fp16,
            'efficiency': power_efficiency,
            'utilization': avg_utilization,
            'thermal_margin': thermal_design_margin,
            'roi': roi_percentage
        })
        
        return SimulationResults(
            timestamp=datetime.now(),
            simulation_id=self.simulation_id,
            parameters=parameters,
            
            # Performance metrics
            total_performance_fp16=total_performance_fp16,
            total_performance_fp4=total_performance_fp4,
            average_utilization=avg_utilization,
            peak_utilization=peak_utilization,
            
            # Power metrics
            total_power_consumption_mw=avg_power_mw,
            average_power_consumption_mw=avg_power_mw,
            peak_power_consumption_mw=peak_power_mw,
            pue=pue,
            power_efficiency=power_efficiency,
            
            # Thermal metrics
            average_temperature=avg_temperature,
            peak_temperature=peak_temperature,
            cooling_efficiency=cooling_efficiency,
            thermal_design_margin=thermal_design_margin,
            
            # Financial metrics
            total_capex_millions=total_capex,
            annual_opex_millions=annual_opex,
            annual_revenue_millions=annual_revenue,
            roi_percentage=roi_percentage,
            payback_period_years=payback_period,
            
            # Infrastructure metrics
            total_gpus=total_gpus,
            total_servers=total_servers,
            total_racks=total_racks,
            floor_space_sqft=floor_space_sqft,
            
            # Environmental metrics
            annual_carbon_emissions_tons=annual_carbon_tons,
            water_usage_gallons_per_year=water_usage_gallons,
            waste_heat_recovery_mw=waste_heat_recovery_mw,
            
            # Reliability metrics
            system_availability=system_availability,
            mtbf_hours=mtbf_hours,
            redundancy_level=redundancy_level,
            
            # Optimization metrics
            optimization_score=optimization_score,
            constraints_satisfied=self._check_constraints(avg_power_mw, parameters),
            optimization_iterations=1
        )
    
    def _calculate_optimization_score(self, metrics: Dict[str, float]) -> float:
        """Calculate overall optimization score"""
        weights = {
            'performance': 0.25,
            'efficiency': 0.25,
            'utilization': 0.20,
            'thermal_margin': 0.15,
            'roi': 0.15
        }
        
        # Normalize metrics to 0-100 scale
        normalized = {
            'performance': min(metrics['performance'] / 10000 * 100, 100),
            'efficiency': metrics['efficiency'] * 100,
            'utilization': metrics['utilization'] * 100,
            'thermal_margin': metrics['thermal_margin'],
            'roi': min(metrics['roi'], 100)
        }
        
        score = sum(normalized[metric] * weights[metric] for metric in weights.keys())
        return score
    
    def _check_constraints(self, avg_power_mw: float, parameters: SimulationParameters) -> bool:
        """Check if simulation results satisfy constraints"""
        power_constraint = avg_power_mw <= parameters.power_budget_mw
        return power_constraint
    
    def run_optimization(self, objective: OptimizationObjective, 
                        constraints: Dict[str, float]) -> Dict[str, Any]:
        """Run optimization to find optimal configuration"""
        
        print(f"Running optimization with objective: {objective.value}")
        
        # Define optimization parameters (reduced for faster execution)
        optimization_params = {
            'gpu_counts': [2000000, 3570000, 4000000],  # Reduced from 5 to 3 options
            'utilization_targets': [0.8, 0.85, 0.9],     # Reduced from 5 to 3 options
            'power_budgets': [4000, 5000, 6000]          # Reduced from 5 to 3 options
        }
        
        best_config = None
        best_score = float('-inf') if objective != OptimizationObjective.MINIMIZE_COST else float('inf')
        optimization_results = []
        
        # Calculate total configurations for progress tracking
        total_configs = len(optimization_params['gpu_counts']) * len(optimization_params['utilization_targets']) * len(optimization_params['power_budgets'])
        current_config = 0
        
        print(f"Testing {total_configs} configurations...")
        
        # Grid search optimization
        for gpu_count in optimization_params['gpu_counts']:
            for utilization in optimization_params['utilization_targets']:
                for power_budget in optimization_params['power_budgets']:
                    
                    current_config += 1
                    
                    # Check if configuration meets constraints
                    if gpu_count * 1000 / 1e6 > power_budget:  # Rough power check
                        continue
                    
                    # Show progress every 10 configurations
                    if current_config % 10 == 0 or current_config == total_configs:
                        progress = (current_config / total_configs) * 100
                        print(f"   Progress: {current_config}/{total_configs} ({progress:.1f}%) - Testing GPU:{gpu_count:,}, Util:{utilization:.0%}, Power:{power_budget}MW")
                    
                    # Initialize infrastructure for this configuration
                    self.initialize_infrastructure(gpu_count, power_budget)
                    
                    # Run simulation (shorter duration for optimization speed)
                    sim_params = SimulationParameters(
                        mode=SimulationMode.STEADY_STATE,
                        duration_hours=8,  # Reduced from 24 to 8 hours for faster optimization
                        time_step_minutes=120,  # Increased from 60 to 120 minutes for fewer steps
                        optimization_objective=objective,
                        utilization_target=utilization,
                        power_budget_mw=power_budget
                    )
                    
                    results = self.run_simulation(sim_params)
                    
                    # Calculate objective score
                    if objective == OptimizationObjective.MAXIMIZE_PERFORMANCE:
                        score = results.total_performance_fp16
                    elif objective == OptimizationObjective.MINIMIZE_COST:
                        score = results.total_capex_millions + results.annual_opex_millions * 5
                    elif objective == OptimizationObjective.MAXIMIZE_EFFICIENCY:
                        score = results.power_efficiency
                    elif objective == OptimizationObjective.MINIMIZE_POWER:
                        score = -results.average_power_consumption_mw
                    else:
                        score = results.optimization_score
                    
                    optimization_results.append({
                        'gpu_count': gpu_count,
                        'utilization': utilization,
                        'power_budget': power_budget,
                        'score': score,
                        'results': results
                    })
                    
                    # Update best configuration
                    if ((objective != OptimizationObjective.MINIMIZE_COST and score > best_score) or
                        (objective == OptimizationObjective.MINIMIZE_COST and score < best_score)):
                        best_score = score
                        best_config = {
                            'gpu_count': gpu_count,
                            'utilization': utilization,
                            'power_budget': power_budget,
                            'score': score,
                            'results': results
                        }
        
        return {
            'best_configuration': best_config,
            'optimization_objective': objective.value,
            'all_results': optimization_results,
            'optimization_summary': self._generate_optimization_summary(best_config, optimization_results)
        }
    
    def _generate_optimization_summary(self, best_config: Dict, all_results: List[Dict]) -> Dict[str, Any]:
        """Generate optimization summary"""
        if not best_config:
            return {'error': 'No valid configuration found'}
        
        return {
            'best_gpu_count': best_config['gpu_count'],
            'best_utilization': best_config['utilization'],
            'best_power_budget': best_config['power_budget'],
            'best_score': best_config['score'],
            'configurations_tested': len(all_results),
            'performance_range': {
                'min': min(r['results'].total_performance_fp16 for r in all_results),
                'max': max(r['results'].total_performance_fp16 for r in all_results)
            },
            'power_range': {
                'min': min(r['results'].average_power_consumption_mw for r in all_results),
                'max': max(r['results'].average_power_consumption_mw for r in all_results)
            },
            'cost_range': {
                'min': min(r['results'].total_capex_millions for r in all_results),
                'max': max(r['results'].total_capex_millions for r in all_results)
            }
        }
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive simulation report"""
        
        if not self.simulation_results:
            return {'error': 'No simulation results available'}
        
        latest_result = self.simulation_results[-1]
        
        # Infrastructure summary
        infrastructure_summary = {
            'compute': self.compute_infrastructure.generate_compute_report() if self.compute_infrastructure else {},
            'power': {
                'total_capacity_mw': self.power_infrastructure.calculate_total_capacity() if self.power_infrastructure else {},
                'redundancy_factor': self.power_infrastructure.calculate_redundancy_factor() if self.power_infrastructure else 0.0
            },
            'cooling': self.cooling_system.generate_cooling_report() if self.cooling_system else {},
            'storage': self.storage_system.generate_storage_report() if self.storage_system else {},
            'network': self.network_infrastructure.generate_network_report() if self.network_infrastructure else {},
            'facility': self.facility_infrastructure.generate_facility_report() if self.facility_infrastructure else {},
            'financial': self.financial_model.generate_financial_report() if self.financial_model else {},
            'monitoring': self.monitoring_system.generate_monitoring_report() if self.monitoring_system else {}
        }
        
        return {
            'simulation_overview': {
                'simulation_id': self.simulation_id,
                'name': self.name,
                'timestamp': latest_result.timestamp.isoformat(),
                'total_simulations': len(self.simulation_results)
            },
            'latest_results': {
                'performance': {
                    'total_performance_fp16_exaflops': latest_result.total_performance_fp16,
                    'total_performance_fp4_pflops': latest_result.total_performance_fp4,
                    'average_utilization': latest_result.average_utilization,
                    'peak_utilization': latest_result.peak_utilization
                },
                'power': {
                    'average_power_mw': latest_result.average_power_consumption_mw,
                    'peak_power_mw': latest_result.peak_power_consumption_mw,
                    'pue': latest_result.pue,
                    'power_efficiency': latest_result.power_efficiency
                },
                'thermal': {
                    'average_temperature': latest_result.average_temperature,
                    'peak_temperature': latest_result.peak_temperature,
                    'cooling_efficiency': latest_result.cooling_efficiency,
                    'thermal_design_margin': latest_result.thermal_design_margin
                },
                'financial': {
                    'total_capex_millions': latest_result.total_capex_millions,
                    'annual_opex_millions': latest_result.annual_opex_millions,
                    'annual_revenue_millions': latest_result.annual_revenue_millions,
                    'roi_percentage': latest_result.roi_percentage,
                    'payback_period_years': latest_result.payback_period_years
                },
                'infrastructure': {
                    'total_gpus': latest_result.total_gpus,
                    'total_servers': latest_result.total_servers,
                    'total_racks': latest_result.total_racks,
                    'floor_space_sqft': latest_result.floor_space_sqft
                },
                'environmental': {
                    'annual_carbon_emissions_tons': latest_result.annual_carbon_emissions_tons,
                    'water_usage_gallons_per_year': latest_result.water_usage_gallons_per_year,
                    'waste_heat_recovery_mw': latest_result.waste_heat_recovery_mw
                },
                'optimization': {
                    'optimization_score': latest_result.optimization_score,
                    'constraints_satisfied': latest_result.constraints_satisfied
                }
            },
            'infrastructure_summary': infrastructure_summary,
            'recommendations': self._generate_recommendations(latest_result)
        }
    
    def _generate_recommendations(self, results: SimulationResults) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []
        
        if results.pue > 1.2:
            recommendations.append("Consider improving cooling efficiency to reduce PUE")
        
        if results.average_utilization < 0.8:
            recommendations.append("Increase workload utilization to improve ROI")
        
        if results.thermal_design_margin < 20:
            recommendations.append("Enhance cooling capacity to improve thermal margins")
        
        if results.roi_percentage < 20:
            recommendations.append("Optimize cost structure or increase revenue to improve ROI")
        
        if results.power_efficiency < 0.7:
            recommendations.append("Improve infrastructure efficiency to increase compute power ratio")
        
        return recommendations
    
    def export_results(self, filename: str, format: str = "json"):
        """Export simulation results to file"""
        report = self.generate_comprehensive_report()
        
        if format == "json":
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2, default=str)
        
        print(f"Results exported to {filename}")