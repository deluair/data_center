#!/usr/bin/env python3
"""
Main 5GW AI Data Center Simulation Application
Demonstrates the complete simulation using all models
"""

import sys
import os
import json
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.simulation_engine import (
    SimulationEngine, SimulationParameters, SimulationMode, 
    OptimizationObjective, WorkloadScenario
)
from models.compute_model import GPUSpecification

def print_banner():
    """Print application banner"""
    print("\n" + "="*80)
    print("                    5GW AI DATA CENTER SIMULATION")
    print("                     Advanced Modeling Suite")
    print("="*80)
    print(f"Simulation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Target Scale: 5 GW Power Capacity")
    print("GPU Count: 3.57 Million GB300/B300 GPUs")
    print("Expected Performance: ~9,000 ExaFLOPS (FP16)")
    print("="*80)

def create_5gw_simulation():
    """Create and configure the 5GW simulation"""
    print("\n🔧 Initializing 5GW AI Data Center Simulation...")
    
    # Create simulation engine
    sim_engine = SimulationEngine("5GW_AI_Factory")
    
    # 5GW Configuration Parameters
    gpu_count = 3570000      # 3.57 million GPUs
    power_budget_mw = 5000   # 5 GW total power
    
    print(f"   • GPU Count: {gpu_count:,} GB300/B300 GPUs")
    print(f"   • Power Budget: {power_budget_mw:,} MW")
    print(f"   • Expected Servers: {gpu_count//8:,}")
    print(f"   • Expected Racks: {gpu_count//32:,}")
    
    # Initialize all infrastructure
    print("   • Initializing compute infrastructure...")
    print("   • Initializing power systems...")
    print("   • Initializing cooling systems...")
    print("   • Initializing storage systems...")
    print("   • Initializing network infrastructure...")
    print("   • Initializing facility systems...")
    print("   • Initializing financial models...")
    print("   • Initializing monitoring systems...")
    
    sim_engine.initialize_infrastructure(gpu_count, power_budget_mw)
    
    print("✅ 5GW AI Data Center initialized successfully!")
    return sim_engine

def run_baseline_simulation(sim_engine):
    """Run baseline steady-state simulation"""
    print("\n📊 Running Baseline Simulation (24 hours)...")
    
    # Baseline simulation parameters
    baseline_params = SimulationParameters(
        mode=SimulationMode.STEADY_STATE,
        duration_hours=24,
        time_step_minutes=60,
        utilization_target=0.85,  # 85% target utilization
        power_budget_mw=5000.0,
        workload_profile="mixed"   # Mixed training/inference
    )
    
    # Run simulation
    results = sim_engine.run_simulation(baseline_params)
    
    # Display key results
    print("\n📈 Baseline Simulation Results:")
    print(f"   • Total Performance (FP16): {results.total_performance_fp16:,.1f} ExaFLOPS")
    print(f"   • Total Performance (FP4): {results.total_performance_fp4:,.1f} PFLOPS")
    print(f"   • Average GPU Utilization: {results.average_utilization:.1%}")
    print(f"   • Peak GPU Utilization: {results.peak_utilization:.1%}")
    print(f"   • Average Power Consumption: {results.average_power_consumption_mw:,.1f} MW")
    print(f"   • Peak Power Consumption: {results.peak_power_consumption_mw:,.1f} MW")
    print(f"   • Power Usage Effectiveness (PUE): {results.pue:.3f}")
    print(f"   • Power Efficiency: {results.power_efficiency:.1%}")
    print(f"   • Average Temperature: {results.average_temperature:.1f}°C")
    print(f"   • Peak Temperature: {results.peak_temperature:.1f}°C")
    print(f"   • Thermal Design Margin: {results.thermal_design_margin:.1f}%")
    
    print("\n💰 Financial Metrics:")
    print(f"   • Total CapEx: ${results.total_capex_millions:,.1f} Million")
    print(f"   • Annual OpEx: ${results.annual_opex_millions:,.1f} Million")
    print(f"   • Annual Revenue: ${results.annual_revenue_millions:,.1f} Million")
    print(f"   • Return on Investment: {results.roi_percentage:.1f}%")
    print(f"   • Payback Period: {results.payback_period_years:.1f} years")
    
    print("\n🏗️ Infrastructure Metrics:")
    print(f"   • Total GPUs: {results.total_gpus:,}")
    print(f"   • Total Servers: {results.total_servers:,}")
    print(f"   • Total Racks: {results.total_racks:,}")
    print(f"   • Floor Space: {results.floor_space_sqft:,} sqft")
    
    print("\n🌍 Environmental Impact:")
    print(f"   • Annual Carbon Emissions: {results.annual_carbon_emissions_tons:,.0f} tons CO2")
    print(f"   • Annual Water Usage: {results.water_usage_gallons_per_year:,.0f} gallons")
    print(f"   • Waste Heat Recovery Potential: {results.waste_heat_recovery_mw:.1f} MW")
    
    print(f"\n🎯 Optimization Score: {results.optimization_score:.1f}/100")
    print(f"   Constraints Satisfied: {'✅ Yes' if results.constraints_satisfied else '❌ No'}")
    
    return results

def run_workload_scenarios(sim_engine):
    """Run different workload scenarios"""
    print("\n🔄 Running Workload Scenarios...")
    
    scenarios = [
        {
            'name': 'Training Heavy',
            'description': 'AI model training focused workload',
            'params': SimulationParameters(
                mode=SimulationMode.STEADY_STATE,
                duration_hours=12,  # Reduced from 24 to 12 hours
                time_step_minutes=120,  # Increased from 60 to 120 minutes
                utilization_target=0.90,
                power_budget_mw=5000.0,
                workload_profile="training"
            )
        },
        {
            'name': 'Peak Load Stress Test',
            'description': 'Maximum capacity stress test',
            'params': SimulationParameters(
                mode=SimulationMode.STRESS_TEST,
                duration_hours=4,  # Reduced from 8 to 4 hours
                time_step_minutes=60,  # Reduced from 30 to 60 minutes
                utilization_target=0.98,
                power_budget_mw=5000.0,
                workload_profile="mixed"
            )
        }
    ]
    
    scenario_results = []
    
    for scenario in scenarios:
        print(f"\n   Running '{scenario['name']}' scenario...")
        print(f"   Description: {scenario['description']}")
        
        results = sim_engine.run_simulation(scenario['params'])
        scenario_results.append({
            'name': scenario['name'],
            'results': results
        })
        
        print(f"   ✅ Performance: {results.total_performance_fp16:.1f} ExaFLOPS")
        print(f"      Power: {results.average_power_consumption_mw:.1f} MW")
        print(f"      Utilization: {results.average_utilization:.1%}")
        print(f"      PUE: {results.pue:.3f}")
    
    # Compare scenarios
    print("\n📊 Scenario Comparison Summary:")
    print(f"{'Scenario':<20} | {'Performance':<12} | {'Power':<8} | {'Util':<6} | {'PUE':<6} | {'ROI':<6}")
    print("-" * 70)
    
    for scenario_result in scenario_results:
        name = scenario_result['name']
        results = scenario_result['results']
        print(f"{name:<20} | {results.total_performance_fp16:8.1f} EF | "
              f"{results.average_power_consumption_mw:6.1f} MW | "
              f"{results.average_utilization:5.1%} | "
              f"{results.pue:5.3f} | "
              f"{results.roi_percentage:5.1f}%")
    
    return scenario_results

def run_optimization_analysis(sim_engine):
    """Run optimization analysis"""
    print("\n🎯 Running Optimization Analysis...")
    
    optimization_objectives = [
        (OptimizationObjective.MAXIMIZE_PERFORMANCE, "Maximum Performance"),
        (OptimizationObjective.MAXIMIZE_EFFICIENCY, "Maximum Efficiency")
        # Removed MINIMIZE_COST to reduce optimization time
    ]
    
    optimization_results = []
    
    for objective, description in optimization_objectives:
        print(f"\n   Optimizing for: {description}")
        
        # Define constraints
        constraints = {
            'power_budget_mw': 5000,
            'cost_budget_millions': 350000,  # $350B budget
            'utilization_min': 0.7,
            'pue_max': 1.3
        }
        
        opt_result = sim_engine.run_optimization(objective, constraints)
        optimization_results.append({
            'objective': description,
            'result': opt_result
        })
        
        if opt_result['best_configuration']:
            best_config = opt_result['best_configuration']
            print(f"   ✅ Best Configuration Found:")
            print(f"      GPU Count: {best_config['gpu_count']:,}")
            print(f"      Utilization: {best_config['utilization']:.1%}")
            print(f"      Power Budget: {best_config['power_budget']:.1f} MW")
            print(f"      Optimization Score: {best_config['score']:.1f}")
            
            # Show key metrics from best configuration
            best_results = best_config['results']
            print(f"      Performance: {best_results.total_performance_fp16:.1f} ExaFLOPS")
            print(f"      ROI: {best_results.roi_percentage:.1f}%")
            print(f"      PUE: {best_results.pue:.3f}")
        else:
            print(f"   ❌ No valid configuration found within constraints")
    
    return optimization_results

def generate_comprehensive_report(sim_engine, baseline_results, scenario_results, optimization_results):
    """Generate comprehensive analysis report"""
    print("\n📋 Generating Comprehensive Report...")
    
    # Generate full report from simulation engine
    full_report = sim_engine.generate_comprehensive_report()
    
    # Create summary report
    summary_report = {
        'simulation_overview': {
            'timestamp': datetime.now().isoformat(),
            'simulation_name': '5GW AI Data Center Analysis',
            'total_gpu_count': 3570000,
            'power_capacity_mw': 5000,
            'target_performance_exaflops': 9000
        },
        'baseline_results': {
            'performance_fp16_exaflops': baseline_results.total_performance_fp16,
            'performance_fp4_pflops': baseline_results.total_performance_fp4,
            'average_utilization': baseline_results.average_utilization,
            'average_power_mw': baseline_results.average_power_consumption_mw,
            'pue': baseline_results.pue,
            'annual_revenue_millions': baseline_results.annual_revenue_millions,
            'roi_percentage': baseline_results.roi_percentage,
            'optimization_score': baseline_results.optimization_score
        },
        'scenario_analysis': [
            {
                'name': scenario['name'],
                'performance_exaflops': scenario['results'].total_performance_fp16,
                'power_mw': scenario['results'].average_power_consumption_mw,
                'utilization': scenario['results'].average_utilization,
                'pue': scenario['results'].pue,
                'roi_percentage': scenario['results'].roi_percentage
            }
            for scenario in scenario_results
        ],
        'optimization_analysis': [
            {
                'objective': opt['objective'],
                'best_score': opt['result']['best_configuration']['score'] if opt['result']['best_configuration'] else None,
                'configurations_tested': len(opt['result']['all_results']) if 'all_results' in opt['result'] else 0
            }
            for opt in optimization_results
        ],
        'infrastructure_summary': full_report['infrastructure_summary'],
        'recommendations': full_report['recommendations']
    }
    
    # Save reports
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save comprehensive report
    comprehensive_filename = f"5GW_comprehensive_report_{timestamp}.json"
    with open(comprehensive_filename, 'w') as f:
        json.dump(full_report, f, indent=2, default=str)
    
    # Save summary report
    summary_filename = f"5GW_summary_report_{timestamp}.json"
    with open(summary_filename, 'w') as f:
        json.dump(summary_report, f, indent=2, default=str)
    
    print(f"   ✅ Comprehensive report saved: {comprehensive_filename}")
    print(f"   ✅ Summary report saved: {summary_filename}")
    
    return summary_report

def print_executive_summary(baseline_results, scenario_results):
    """Print executive summary"""
    print("\n" + "="*80)
    print("                        EXECUTIVE SUMMARY")
    print("                     5GW AI Data Center Analysis")
    print("="*80)
    
    print("\n🎯 KEY PERFORMANCE INDICATORS:")
    print(f"   • Peak AI Performance: {baseline_results.total_performance_fp16:,.1f} ExaFLOPS (FP16)")
    print(f"   • Peak AI Performance: {baseline_results.total_performance_fp4:,.1f} PFLOPS (FP4)")
    print(f"   • GPU Utilization: {baseline_results.average_utilization:.1%} average")
    print(f"   • Power Consumption: {baseline_results.average_power_consumption_mw:,.1f} MW average")
    print(f"   • Power Efficiency (PUE): {baseline_results.pue:.3f}")
    
    print("\n💰 FINANCIAL PERFORMANCE:")
    print(f"   • Total Investment (CapEx): ${baseline_results.total_capex_millions:,.0f} Million")
    print(f"   • Annual Operating Cost: ${baseline_results.annual_opex_millions:,.0f} Million")
    print(f"   • Annual Revenue: ${baseline_results.annual_revenue_millions:,.0f} Million")
    print(f"   • Return on Investment: {baseline_results.roi_percentage:.1f}%")
    print(f"   • Payback Period: {baseline_results.payback_period_years:.1f} years")
    
    print("\n🏗️ INFRASTRUCTURE SCALE:")
    print(f"   • Total GPUs: {baseline_results.total_gpus:,} GB300/B300 units")
    print(f"   • Total Servers: {baseline_results.total_servers:,} units")
    print(f"   • Total Racks: {baseline_results.total_racks:,} units")
    print(f"   • Facility Size: {baseline_results.floor_space_sqft:,} square feet")
    
    print("\n🌍 ENVIRONMENTAL IMPACT:")
    print(f"   • Annual Carbon Footprint: {baseline_results.annual_carbon_emissions_tons:,.0f} tons CO2")
    print(f"   • Annual Water Usage: {baseline_results.water_usage_gallons_per_year:,.0f} gallons")
    print(f"   • Waste Heat Recovery: {baseline_results.waste_heat_recovery_mw:.1f} MW potential")
    
    print("\n📊 SCENARIO PERFORMANCE RANGE:")
    min_perf = min(s['results'].total_performance_fp16 for s in scenario_results)
    max_perf = max(s['results'].total_performance_fp16 for s in scenario_results)
    min_power = min(s['results'].average_power_consumption_mw for s in scenario_results)
    max_power = max(s['results'].average_power_consumption_mw for s in scenario_results)
    
    print(f"   • Performance Range: {min_perf:.1f} - {max_perf:.1f} ExaFLOPS")
    print(f"   • Power Range: {min_power:.1f} - {max_power:.1f} MW")
    
    print(f"\n🎯 OVERALL OPTIMIZATION SCORE: {baseline_results.optimization_score:.1f}/100")
    
    # Status assessment
    if baseline_results.optimization_score >= 80:
        status = "🟢 EXCELLENT - Highly optimized configuration"
    elif baseline_results.optimization_score >= 70:
        status = "🟡 GOOD - Well-balanced configuration with room for improvement"
    elif baseline_results.optimization_score >= 60:
        status = "🟠 FAIR - Acceptable performance but needs optimization"
    else:
        status = "🔴 POOR - Significant optimization required"
    
    print(f"   Status: {status}")
    
    print("\n" + "="*80)

def main():
    """Main application entry point"""
    try:
        # Print banner
        print_banner()
        
        # Create simulation
        sim_engine = create_5gw_simulation()
        
        # Run baseline simulation
        baseline_results = run_baseline_simulation(sim_engine)
        
        # Run workload scenarios
        scenario_results = run_workload_scenarios(sim_engine)
        
        # Run optimization analysis
        optimization_results = run_optimization_analysis(sim_engine)
        
        # Generate comprehensive report
        summary_report = generate_comprehensive_report(
            sim_engine, baseline_results, scenario_results, optimization_results
        )
        
        # Print executive summary
        print_executive_summary(baseline_results, scenario_results)
        
        print("\n🎉 5GW AI Data Center Simulation Completed Successfully!")
        print("\n📁 Generated Files:")
        print("   • Comprehensive analysis report (JSON)")
        print("   • Executive summary report (JSON)")
        print("   • Detailed infrastructure specifications")
        print("   • Optimization recommendations")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Simulation failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)