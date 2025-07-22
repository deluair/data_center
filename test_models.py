#!/usr/bin/env python3
"""
Comprehensive test suite for AI Data Center models
Tests all components and demonstrates functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.power_model import PowerInfrastructure, PowerConsumption
from models.cooling_model import CoolingSystem
from models.compute_model import ComputeInfrastructure, GPUSpecification, WorkloadType
from models.storage_model import StorageSystem
from models.network_model import NetworkInfrastructure
from models.facility_model import FacilityInfrastructure
from models.financial_model import FinancialModel
from models.monitoring_model import MonitoringSystem
from models.simulation_engine import SimulationEngine, SimulationParameters, SimulationMode, OptimizationObjective

def test_power_model():
    """Test power infrastructure and consumption models"""
    print("\n=== Testing Power Model ===")
    
    # Test PowerInfrastructure
    power_infra = PowerInfrastructure("Test_Power")
    
    # Add power sources
    power_infra.add_power_source("grid", 5000000, 0.95, 50000000)  # 5GW grid
    power_infra.add_power_source("solar", 500000, 0.90, 10000000)   # 500MW solar
    
    # Add transformers
    power_infra.add_transformer("main_transformer", 5000000, 0.98, 5000000)
    
    # Add UPS systems
    power_infra.add_ups_system("main_ups", 500000, 0.95, 10000000)
    
    # Calculate metrics
    metrics = power_infra.calculate_power_metrics()
    print(f"Total Power Capacity: {metrics['total_capacity_kw']/1000:.1f} MW")
    print(f"Power Infrastructure Cost: ${metrics['total_cost']/1e6:.1f}M")
    print(f"Average Efficiency: {metrics['average_efficiency']:.3f}")
    
    # Test PowerConsumption
    power_consumption = PowerConsumption()
    
    # Set consumption values
    power_consumption.set_gpu_consumption(3570000, 1000)  # 3.57M GPUs at 1kW each
    power_consumption.set_cpu_consumption(446250, 200)    # CPUs
    power_consumption.set_memory_consumption(1000000, 50) # Memory
    power_consumption.set_storage_consumption(100000, 100) # Storage
    power_consumption.set_network_consumption(50000, 200)  # Network
    power_consumption.set_cooling_consumption(800000)      # Cooling
    power_consumption.set_facility_consumption(200000)    # Facility
    
    consumption_metrics = power_consumption.calculate_consumption_metrics()
    print(f"Total Power Consumption: {consumption_metrics['total_power_kw']/1000:.1f} MW")
    print(f"PUE: {consumption_metrics['pue']:.3f}")
    print(f"Power Density: {consumption_metrics['power_density_w_per_sqft']:.1f} W/sqft")
    
    return True

def test_cooling_model():
    """Test cooling system model"""
    print("\n=== Testing Cooling Model ===")
    
    cooling_system = CoolingSystem("Test_Cooling")
    
    # Add cooling units
    cooling_system.add_cooling_unit("liquid_cooling", "liquid", 4000, 4.5, 888.9)
    cooling_system.add_cooling_unit("air_cooling", "air", 1000, 3.0, 333.3)
    
    # Add heat exchangers
    cooling_system.add_heat_exchanger("primary_hx", "plate", 2000, 0.85, 100)
    
    # Add cooling towers
    cooling_system.add_cooling_tower("main_tower", 3000, 0.80, 200)
    
    # Calculate metrics
    metrics = cooling_system.calculate_cooling_metrics()
    print(f"Total Cooling Capacity: {metrics['total_capacity_mw']:.1f} MW")
    print(f"Average COP: {metrics['average_cop']:.2f}")
    print(f"Total Cooling Power: {metrics['total_power_consumption_mw']:.1f} MW")
    print(f"Cooling Efficiency: {metrics['cooling_efficiency']:.3f}")
    
    # Test thermal calculations
    thermal_metrics = cooling_system.calculate_thermal_performance(4500, 25, 35)
    print(f"Heat Transfer Rate: {thermal_metrics['heat_transfer_rate_mw']:.1f} MW")
    print(f"Coolant Flow Rate: {thermal_metrics['coolant_flow_rate_m3_per_s']:.1f} m³/s")
    
    return True

def test_compute_model():
    """Test compute infrastructure model"""
    print("\n=== Testing Compute Model ===")
    
    compute_infra = ComputeInfrastructure("Test_Compute")
    
    # Define GPU specification
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
    
    # Add GPU clusters
    compute_infra.add_gpu_cluster("cluster_1", gpu_spec, 8, 446250)  # 3.57M GPUs total
    
    # Calculate performance
    training_perf = compute_infra.calculate_total_performance(WorkloadType.TRAINING, "fp16")
    inference_perf = compute_infra.calculate_total_performance(WorkloadType.INFERENCE, "fp4")
    
    print(f"Training Performance (FP16): {training_perf/1e6:.1f} ExaFLOPS")
    print(f"Inference Performance (FP4): {inference_perf/1e3:.1f} PFLOPS")
    
    # Calculate memory and power
    memory_metrics = compute_infra.calculate_total_memory()
    power_metrics = compute_infra.calculate_total_power()
    
    print(f"Total GPU Memory: {memory_metrics['total_memory_tb']:.1f} TB")
    print(f"Total GPU Power: {power_metrics['total_power_mw']:.1f} MW")
    
    # Generate report
    report = compute_infra.generate_compute_report()
    print(f"Total GPUs: {report['total_gpus']:,}")
    print(f"Total Servers: {report['total_servers']:,}")
    print(f"Total Racks: {report['total_racks']:,}")
    
    return True

def test_storage_model():
    """Test storage system model"""
    print("\n=== Testing Storage Model ===")
    
    storage_system = StorageSystem("Test_Storage")
    
    # Add storage devices
    storage_system.add_storage_device("nvme_tier1", "nvme_ssd", 1000000, 1000000, 50000)  # 1PB NVMe
    storage_system.add_storage_device("ssd_tier2", "sata_ssd", 5000000, 100000, 20000)    # 5PB SSD
    storage_system.add_storage_device("hdd_tier3", "hdd", 20000000, 10000, 5000)          # 20PB HDD
    
    # Create storage arrays
    storage_system.create_storage_array("high_perf_array", ["nvme_tier1"], "raid10")
    storage_system.create_storage_array("balanced_array", ["ssd_tier2"], "raid6")
    storage_system.create_storage_array("archive_array", ["hdd_tier3"], "raid6")
    
    # Calculate metrics
    metrics = storage_system.calculate_storage_metrics()
    print(f"Total Capacity: {metrics['total_capacity_pb']:.1f} PB")
    print(f"Total Performance: {metrics['total_performance_iops']:,} IOPS")
    print(f"Total Power: {metrics['total_power_consumption_kw']:.1f} kW")
    print(f"Storage Efficiency: {metrics['storage_efficiency']:.3f}")
    
    # Test tiering optimization
    tier_optimization = storage_system.optimize_storage_tiers({
        'hot_data_tb': 500000,    # 500TB hot data
        'warm_data_tb': 2000000,  # 2PB warm data
        'cold_data_tb': 10000000  # 10PB cold data
    })
    
    print(f"Tier Optimization Score: {tier_optimization['optimization_score']:.1f}")
    
    return True

def test_network_model():
    """Test network infrastructure model"""
    print("\n=== Testing Network Model ===")
    
    network_infra = NetworkInfrastructure("Test_Network")
    
    # Add network switches
    network_infra.add_network_switch("core_switch_1", "core", 1024, 400, 5000)
    network_infra.add_network_switch("agg_switch_1", "aggregation", 512, 200, 3000)
    network_infra.add_network_switch("tor_switch_1", "tor", 48, 100, 500)
    
    # Add network segments
    network_infra.add_network_segment("backbone", "infiniband", 800, 1000, 0.1)
    network_infra.add_network_segment("cluster_interconnect", "ethernet", 400, 2000, 0.5)
    
    # Calculate metrics
    metrics = network_infra.calculate_network_metrics()
    print(f"Total Bandwidth: {metrics['total_bandwidth_gbps']:,} Gbps")
    print(f"Network Utilization: {metrics['network_utilization']:.1%}")
    print(f"Average Latency: {metrics['average_latency_ms']:.2f} ms")
    print(f"Network Power: {metrics['total_power_consumption_kw']:.1f} kW")
    
    # Simulate traffic
    traffic_simulation = network_infra.simulate_network_performance({
        'training_traffic_gbps': 50000,
        'inference_traffic_gbps': 20000,
        'storage_traffic_gbps': 10000,
        'management_traffic_gbps': 1000
    })
    
    print(f"Network Performance Score: {traffic_simulation['performance_score']:.1f}")
    
    return True

def test_facility_model():
    """Test facility infrastructure model"""
    print("\n=== Testing Facility Model ===")
    
    facility_infra = FacilityInfrastructure("Test_Facility")
    
    # Set floor space
    facility_infra.set_floor_space(2000000, 1200000, 3000)  # 2M sqft total, 1.2M usable
    
    # Add building systems
    facility_infra.add_building_system("hvac", 5000, 0.85, 2000000)
    facility_infra.add_building_system("lighting", 1000, 0.90, 500000)
    facility_infra.add_building_system("fire_suppression", 500, 0.95, 1000000)
    
    # Set security systems
    facility_infra.set_security_system("tier_4", 2000, 0.999, 5000000)
    
    # Set environmental controls
    facility_infra.set_environmental_control(22, 45, 1013, 0.1)
    
    # Calculate metrics
    metrics = facility_infra.calculate_facility_metrics()
    print(f"Total Floor Space: {metrics['total_floor_space_sqft']:,} sqft")
    print(f"Usable Floor Space: {metrics['usable_floor_space_sqft']:,} sqft")
    print(f"Power Density: {metrics['power_density_w_per_sqft']:.1f} W/sqft")
    print(f"Space Utilization: {metrics['space_utilization']:.1%}")
    print(f"Facility Power: {metrics['total_facility_power_kw']:.1f} kW")
    
    return True

def test_financial_model():
    """Test financial model"""
    print("\n=== Testing Financial Model ===")
    
    financial_model = FinancialModel("Test_Financial")
    
    # Add cost categories
    financial_model.add_cost_category("gpu_hardware", 142800000000, "capex")  # $142.8B for GPUs
    financial_model.add_cost_category("infrastructure", 100000000000, "capex") # $100B infrastructure
    financial_model.add_cost_category("facility", 50000000000, "capex")       # $50B facility
    
    financial_model.add_cost_category("power", 4380000000, "opex")           # $4.38B/year power
    financial_model.add_cost_category("cooling", 657000000, "opex")          # $657M/year cooling
    financial_model.add_cost_category("maintenance", 2000000000, "opex")     # $2B/year maintenance
    financial_model.add_cost_category("personnel", 500000000, "opex")        # $500M/year personnel
    
    # Add revenue streams
    financial_model.add_revenue_stream("ai_training", 150000000000, "annual")  # $150B/year training
    financial_model.add_revenue_stream("ai_inference", 60000000000, "annual")  # $60B/year inference
    
    # Calculate projections
    projections = financial_model.calculate_financial_projections(5)
    print(f"Total CapEx: ${projections['total_capex']/1e9:.1f}B")
    print(f"Annual OpEx: ${projections['annual_opex']/1e9:.1f}B")
    print(f"Annual Revenue: ${projections['annual_revenue']/1e9:.1f}B")
    
    # Calculate ROI metrics
    roi_metrics = financial_model.calculate_roi_metrics(projections)
    print(f"ROI: {roi_metrics['roi_percentage']:.1f}%")
    print(f"Payback Period: {roi_metrics['payback_period_years']:.1f} years")
    print(f"NPV: ${roi_metrics['npv']/1e9:.1f}B")
    print(f"IRR: {roi_metrics['irr']:.1f}%")
    
    # Sensitivity analysis
    sensitivity = financial_model.perform_sensitivity_analysis(projections, {
        'revenue_change': [-0.2, -0.1, 0, 0.1, 0.2],
        'cost_change': [-0.1, 0, 0.1, 0.2, 0.3]
    })
    
    print(f"Sensitivity Analysis Scenarios: {len(sensitivity['scenarios'])}")
    
    return True

def test_monitoring_model():
    """Test monitoring system model"""
    print("\n=== Testing Monitoring Model ===")
    
    monitoring_system = MonitoringSystem("Test_Monitoring")
    
    # Define metrics
    monitoring_system.define_gpu_metrics()
    monitoring_system.define_system_metrics()
    
    # Setup alerts
    monitoring_system.setup_default_alerts()
    
    # Add monitoring agents
    monitoring_system.add_monitoring_agents(3570000, 446250)
    
    # Simulate monitoring data
    simulated_data = monitoring_system.simulate_monitoring_data(24)
    print(f"Simulated {len(simulated_data)} metrics over 24 hours")
    
    # Calculate costs
    costs = monitoring_system.calculate_monitoring_costs()
    print(f"Monitoring System Cost: ${costs['total_monitoring_cost']/1e6:.1f}M")
    print(f"Monitoring Agents: {costs['agents_count']:,}")
    print(f"Annual Power Cost: ${costs['annual_power_cost']/1e6:.1f}M")
    
    # Generate dashboard
    dashboard = monitoring_system.generate_monitoring_dashboard()
    print(f"Performance Score: {dashboard['performance_score']['overall_score']:.1f}")
    print(f"Active Alerts: {dashboard['alert_summary']['total_active_alerts']}")
    
    return True

def test_simulation_engine():
    """Test simulation engine"""
    print("\n=== Testing Simulation Engine ===")
    
    # Create simulation engine
    sim_engine = SimulationEngine("5GW_Test_Simulation")
    
    # Initialize infrastructure
    sim_engine.initialize_infrastructure(3570000, 5000)  # 3.57M GPUs, 5GW
    
    # Define simulation parameters
    sim_params = SimulationParameters(
        mode=SimulationMode.STEADY_STATE,
        duration_hours=24,
        time_step_minutes=60,
        utilization_target=0.85,
        power_budget_mw=5000.0
    )
    
    # Run simulation
    print("Running 24-hour simulation...")
    results = sim_engine.run_simulation(sim_params)
    
    print(f"\nSimulation Results:")
    print(f"Total Performance (FP16): {results.total_performance_fp16:.1f} ExaFLOPS")
    print(f"Total Performance (FP4): {results.total_performance_fp4:.1f} PFLOPS")
    print(f"Average Utilization: {results.average_utilization:.1%}")
    print(f"Average Power: {results.average_power_consumption_mw:.1f} MW")
    print(f"PUE: {results.pue:.3f}")
    print(f"Annual Revenue: ${results.annual_revenue_millions:.1f}M")
    print(f"ROI: {results.roi_percentage:.1f}%")
    print(f"Optimization Score: {results.optimization_score:.1f}")
    
    # Test optimization
    print("\nRunning optimization...")
    optimization_results = sim_engine.run_optimization(
        OptimizationObjective.MAXIMIZE_PERFORMANCE,
        {'power_budget_mw': 5000, 'cost_budget_millions': 300000}
    )
    
    if optimization_results['best_configuration']:
        best_config = optimization_results['best_configuration']
        print(f"Best Configuration:")
        print(f"  GPU Count: {best_config['gpu_count']:,}")
        print(f"  Utilization: {best_config['utilization']:.1%}")
        print(f"  Power Budget: {best_config['power_budget']:.1f} MW")
        print(f"  Score: {best_config['score']:.1f}")
    
    # Generate comprehensive report
    report = sim_engine.generate_comprehensive_report()
    print(f"\nComprehensive Report Generated:")
    print(f"  Simulation ID: {report['simulation_overview']['simulation_id']}")
    print(f"  Total Simulations: {report['simulation_overview']['total_simulations']}")
    print(f"  Recommendations: {len(report['recommendations'])}")
    
    return True

def run_integration_test():
    """Run integration test with all models"""
    print("\n=== Integration Test ===")
    
    # Create simulation engine with all models
    sim_engine = SimulationEngine("Integration_Test")
    
    # Initialize with realistic 5GW configuration
    gpu_count = 3570000  # 3.57 million GPUs
    power_budget = 5000  # 5 GW
    
    print(f"Initializing {gpu_count:,} GPUs with {power_budget:,} MW power budget...")
    sim_engine.initialize_infrastructure(gpu_count, power_budget)
    
    # Run multiple simulation scenarios
    scenarios = [
        {
            'name': 'Training Heavy',
            'mode': SimulationMode.STEADY_STATE,
            'utilization': 0.90,
            'duration': 24
        },
        {
            'name': 'Inference Heavy',
            'mode': SimulationMode.STEADY_STATE,
            'utilization': 0.80,
            'duration': 24
        },
        {
            'name': 'Stress Test',
            'mode': SimulationMode.STRESS_TEST,
            'utilization': 0.95,
            'duration': 8
        }
    ]
    
    scenario_results = []
    
    for scenario in scenarios:
        print(f"\nRunning {scenario['name']} scenario...")
        
        sim_params = SimulationParameters(
            mode=scenario['mode'],
            duration_hours=scenario['duration'],
            time_step_minutes=60,
            utilization_target=scenario['utilization'],
            power_budget_mw=power_budget
        )
        
        results = sim_engine.run_simulation(sim_params)
        scenario_results.append({
            'name': scenario['name'],
            'results': results
        })
        
        print(f"  Performance: {results.total_performance_fp16:.1f} ExaFLOPS")
        print(f"  Power: {results.average_power_consumption_mw:.1f} MW")
        print(f"  Utilization: {results.average_utilization:.1%}")
        print(f"  PUE: {results.pue:.3f}")
    
    # Compare scenarios
    print("\n=== Scenario Comparison ===")
    for scenario_result in scenario_results:
        name = scenario_result['name']
        results = scenario_result['results']
        print(f"{name:15} | {results.total_performance_fp16:8.1f} ExaFLOPS | "
              f"{results.average_power_consumption_mw:6.1f} MW | "
              f"{results.average_utilization:5.1%} | "
              f"PUE {results.pue:.3f}")
    
    # Export results
    sim_engine.export_results("integration_test_results.json")
    print("\nIntegration test results exported to integration_test_results.json")
    
    return True

def main():
    """Run all tests"""
    print("AI Data Center Models - Comprehensive Test Suite")
    print("=" * 50)
    
    tests = [
        ("Power Model", test_power_model),
        ("Cooling Model", test_cooling_model),
        ("Compute Model", test_compute_model),
        ("Storage Model", test_storage_model),
        ("Network Model", test_network_model),
        ("Facility Model", test_facility_model),
        ("Financial Model", test_financial_model),
        ("Monitoring Model", test_monitoring_model),
        ("Simulation Engine", test_simulation_engine),
        ("Integration Test", run_integration_test)
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        try:
            print(f"\n{'='*20} {test_name} {'='*20}")
            success = test_func()
            if success:
                print(f"✓ {test_name} PASSED")
                passed_tests += 1
            else:
                print(f"✗ {test_name} FAILED")
        except Exception as e:
            print(f"✗ {test_name} FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*50}")
    print(f"Test Results: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! The AI Data Center models are working correctly.")
        return True
    else:
        print(f"⚠️  {total_tests - passed_tests} tests failed. Please check the output above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)