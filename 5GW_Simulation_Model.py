#!/usr/bin/env python3
"""
5 GW AI Factory Data Center - Simulation Model

This module provides a comprehensive simulation model for a 5 GW AI factory data center,
including power consumption, cooling requirements, performance metrics, and cost analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Tuple
import json
from datetime import datetime, timedelta

@dataclass
class DataCenterConfig:
    """Configuration parameters for the 5 GW data center with GB300/B300 GPUs"""
    
    # Power Infrastructure
    total_power_capacity_mw: float = 5000.0
    grid_connections: int = 5  # 3x 500kV + 2x 345kV
    backup_generator_capacity_mw: float = 5500.0
    ups_capacity_mw: float = 500.0
    battery_storage_gwh: float = 2.0
    
    # Compute Infrastructure (Scaled to 5 GW)
    total_gpus: int = 3571428  # Scaled to utilize full 5 GW capacity
    training_gpus: int = 2380952  # GB300 for training (2/3 of total)
    inference_gpus: int = 1190476  # B300 for inference (1/3 of total)
    training_gpu_power_w: float = 1400.0  # GB300 TDP
    inference_gpu_power_w: float = 1300.0  # B300 TDP
    gpu_memory_gb: float = 288.0  # 288GB HBM3e per GPU
    cpu_power_w: float = 400.0  # Per server
    memory_power_w: float = 200.0  # Per server
    
    # Cooling Infrastructure
    cooling_efficiency: float = 0.9  # 90% of IT load needs cooling
    pue_target: float = 1.08
    chilled_water_plants: int = 3
    cooling_towers: int = 60
    
    # Networking
    backend_bandwidth_tbps: float = 60.0
    frontend_bandwidth_tbps: float = 80.0
    network_latency_us: float = 0.5
    
    # Storage
    high_perf_storage_pb: float = 500.0
    capacity_storage_eb: float = 1.0
    
    # Financial Parameters (Scaled for 5 GW)
    electricity_cost_kwh: float = 0.04
    capex_billion: float = 300.0  # $300B for 3.57M GPUs at 5 GW scale
    annual_opex_billion: float = 70.0  # $70B/year operational costs
    
    # Environmental
    water_consumption_mgd: float = 1.5  # Million gallons per day
    carbon_intensity_kg_kwh: float = 0.4

class DataCenterSimulator:
    """Main simulation class for the 5 GW AI factory data center"""
    
    def __init__(self, config: DataCenterConfig):
        self.config = config
        self.simulation_results = {}
        
    def calculate_power_consumption(self) -> Dict[str, float]:
        """Calculate detailed power consumption breakdown"""
        
        # IT Load Calculation
        training_gpu_power_mw = (self.config.training_gpus * self.config.training_gpu_power_w) / 1e6
        inference_gpu_power_mw = (self.config.inference_gpus * self.config.inference_gpu_power_w) / 1e6
        gpu_power_mw = training_gpu_power_mw + inference_gpu_power_mw
        cpu_power_mw = (self.config.total_gpus / 4 * self.config.cpu_power_w) / 1e6  # 4 GPUs per server
        memory_power_mw = (self.config.total_gpus / 4 * self.config.memory_power_w) / 1e6
        storage_power_mw = 200.0  # Scaled storage power for 5 GW facility
        network_power_mw = 400.0  # Scaled network power for 5 GW facility
        
        it_load_mw = gpu_power_mw + cpu_power_mw + memory_power_mw + storage_power_mw + network_power_mw
        
        # Infrastructure Load
        cooling_power_mw = it_load_mw * (self.config.pue_target - 1)
        lighting_power_mw = 50.0  # Scaled for 5 GW facility
        facility_power_mw = 100.0  # Scaled for 5 GW facility
        
        total_power_mw = it_load_mw + cooling_power_mw + lighting_power_mw + facility_power_mw
        
        power_breakdown = {
            'gpu_power_mw': gpu_power_mw,
            'cpu_power_mw': cpu_power_mw,
            'memory_power_mw': memory_power_mw,
            'storage_power_mw': storage_power_mw,
            'network_power_mw': network_power_mw,
            'it_load_mw': it_load_mw,
            'cooling_power_mw': cooling_power_mw,
            'lighting_power_mw': lighting_power_mw,
            'facility_power_mw': facility_power_mw,
            'total_power_mw': total_power_mw,
            'actual_pue': total_power_mw / it_load_mw
        }
        
        return power_breakdown
    
    def calculate_cooling_requirements(self) -> Dict[str, float]:
        """Calculate cooling system requirements"""
        
        power_data = self.calculate_power_consumption()
        heat_load_mw = power_data['it_load_mw'] * self.config.cooling_efficiency
        
        # Cooling capacity (with 20% overhead)
        cooling_capacity_mw = heat_load_mw * 1.2
        
        # Chilled water system
        chiller_capacity_mw = cooling_capacity_mw / self.config.chilled_water_plants
        cooling_tower_capacity_mw = cooling_capacity_mw / self.config.cooling_towers
        
        # Water consumption
        water_consumption_gpm = heat_load_mw * 3.0  # 3 GPM per MW of heat
        
        cooling_data = {
            'heat_load_mw': heat_load_mw,
            'cooling_capacity_mw': cooling_capacity_mw,
            'chiller_capacity_mw': chiller_capacity_mw,
            'cooling_tower_capacity_mw': cooling_tower_capacity_mw,
            'water_consumption_gpm': water_consumption_gpm,
            'water_consumption_mgd': water_consumption_gpm * 1440 / 1e6
        }
        
        return cooling_data
    
    def calculate_performance_metrics(self) -> Dict[str, float]:
        """Calculate compute performance metrics"""
        
        # GPU Performance (GB300/B300 specifications)
        gb300_fp16_tflops = 3750.0  # GB300 FP16 performance
        gb300_fp8_tflops = 7500.0   # GB300 FP8 performance
        gb300_fp4_pflops = 15.0     # GB300 FP4 performance in PFLOPS
        b300_fp16_tflops = 3750.0   # B300 FP16 performance
        b300_fp8_tflops = 7500.0    # B300 FP8 performance
        
        # Training Performance (GB300)
        training_fp16_tflops = self.config.training_gpus * gb300_fp16_tflops
        training_exaflops = training_fp16_tflops / 1e6
        training_fp4_pflops = self.config.training_gpus * gb300_fp4_pflops
        
        # Inference Performance (B300)
        inference_fp16_tflops = self.config.inference_gpus * b300_fp16_tflops
        inference_exaflops = inference_fp16_tflops / 1e6
        inference_fp8_tflops = self.config.inference_gpus * b300_fp8_tflops
        inference_exaops = inference_fp8_tflops / 1e6
        
        # Memory Bandwidth (HBM3e)
        memory_bandwidth_per_gpu_gbps = 8000.0  # GB300/B300 memory bandwidth estimate
        total_memory_bandwidth_tbps = (self.config.total_gpus * memory_bandwidth_per_gpu_gbps) / 1000.0
        
        # Storage Performance
        storage_bandwidth_tbps = 1.0  # High-performance storage
        storage_iops_millions = 100.0  # Million IOPS
        
        performance_data = {
            'training_exaflops_fp16': training_exaflops,
            'training_fp4_pflops': training_fp4_pflops,
            'inference_exaflops_fp16': inference_exaflops,
            'inference_exaops_fp8': inference_exaops,
            'total_memory_bandwidth_tbps': total_memory_bandwidth_tbps,
            'storage_bandwidth_tbps': storage_bandwidth_tbps,
            'storage_iops_millions': storage_iops_millions,
            'network_backend_tbps': self.config.backend_bandwidth_tbps,
            'network_frontend_tbps': self.config.frontend_bandwidth_tbps
        }
        
        return performance_data
    
    def calculate_financial_metrics(self) -> Dict[str, float]:
        """Calculate financial performance metrics"""
        
        power_data = self.calculate_power_consumption()
        
        # Annual electricity cost
        annual_kwh = power_data['total_power_mw'] * 1000 * 8760  # kWh per year
        annual_electricity_cost = annual_kwh * self.config.electricity_cost_kwh
        
        # Revenue projections (scaled for 5 GW with 3.57M GPUs)
        ai_training_revenue = 80.0e9  # $80B/year (massive scale with GB300)
        ai_inference_revenue = 95.0e9  # $95B/year (massive scale with B300)
        cloud_services_revenue = 25.0e9  # $25B/year
        data_services_revenue = 10.0e9  # $10B/year
        total_annual_revenue = ai_training_revenue + ai_inference_revenue + cloud_services_revenue + data_services_revenue
        
        # Financial metrics
        annual_profit = total_annual_revenue - self.config.annual_opex_billion * 1e9
        roi_5_year = (annual_profit * 5 - self.config.capex_billion * 1e9) / (self.config.capex_billion * 1e9)
        payback_period_years = (self.config.capex_billion * 1e9) / annual_profit
        
        financial_data = {
            'annual_kwh': annual_kwh,
            'annual_electricity_cost': annual_electricity_cost,
            'total_annual_revenue': total_annual_revenue,
            'annual_profit': annual_profit,
            'roi_5_year': roi_5_year,
            'payback_period_years': payback_period_years,
            'electricity_cost_percentage': (annual_electricity_cost / total_annual_revenue) * 100
        }
        
        return financial_data
    
    def calculate_environmental_impact(self) -> Dict[str, float]:
        """Calculate environmental impact metrics"""
        
        power_data = self.calculate_power_consumption()
        cooling_data = self.calculate_cooling_requirements()
        
        # Carbon emissions
        annual_kwh = power_data['total_power_mw'] * 1000 * 8760
        annual_carbon_emissions_kg = annual_kwh * self.config.carbon_intensity_kg_kwh
        annual_carbon_emissions_tons = annual_carbon_emissions_kg / 1000
        
        # Water usage
        annual_water_consumption_mg = cooling_data['water_consumption_mgd'] * 365
        
        # Waste heat recovery potential
        waste_heat_mw = power_data['it_load_mw'] * 0.9  # 90% of IT load becomes waste heat
        district_heating_homes = waste_heat_mw * 1000 / 60  # 60 kW per home average
        
        environmental_data = {
            'annual_carbon_emissions_tons': annual_carbon_emissions_tons,
            'annual_water_consumption_mg': annual_water_consumption_mg,
            'waste_heat_mw': waste_heat_mw,
            'district_heating_homes': district_heating_homes,
            'pue': power_data['actual_pue'],
            'wue_l_per_kwh': (annual_water_consumption_mg * 3785.41) / annual_kwh  # Convert to L/kWh
        }
        
        return environmental_data
    
    def simulate_workload_patterns(self, days: int = 365) -> pd.DataFrame:
        """Simulate daily workload patterns over specified period"""
        
        dates = pd.date_range(start='2024-01-01', periods=days, freq='D')
        
        # Generate realistic workload patterns
        base_utilization = 0.75  # 75% base utilization
        seasonal_variation = 0.1 * np.sin(2 * np.pi * np.arange(days) / 365)  # Seasonal pattern
        weekly_variation = 0.05 * np.sin(2 * np.pi * np.arange(days) / 7)     # Weekly pattern
        random_variation = np.random.normal(0, 0.02, days)                    # Random variation
        
        gpu_utilization = base_utilization + seasonal_variation + weekly_variation + random_variation
        gpu_utilization = np.clip(gpu_utilization, 0.4, 0.95)  # Realistic bounds
        
        # Calculate corresponding metrics
        power_consumption_mw = gpu_utilization * self.calculate_power_consumption()['total_power_mw']
        cooling_load_mw = power_consumption_mw * 0.9
        
        # Performance metrics
        training_performance = gpu_utilization * self.calculate_performance_metrics()['training_exaflops_fp16']
        inference_performance = gpu_utilization * self.calculate_performance_metrics()['inference_exaops_fp8']
        
        workload_df = pd.DataFrame({
            'date': dates,
            'gpu_utilization': gpu_utilization,
            'power_consumption_mw': power_consumption_mw,
            'cooling_load_mw': cooling_load_mw,
            'training_exaflops': training_performance,
            'inference_exaops': inference_performance
        })
        
        return workload_df
    
    def run_full_simulation(self) -> Dict[str, any]:
        """Run complete data center simulation"""
        
        print("Running 5 GW AI Factory Data Center Simulation...")
        
        # Calculate all metrics
        power_metrics = self.calculate_power_consumption()
        cooling_metrics = self.calculate_cooling_requirements()
        performance_metrics = self.calculate_performance_metrics()
        financial_metrics = self.calculate_financial_metrics()
        environmental_metrics = self.calculate_environmental_impact()
        
        # Generate workload simulation
        workload_simulation = self.simulate_workload_patterns()
        
        # Compile results
        simulation_results = {
            'timestamp': datetime.now().isoformat(),
            'config': self.config.__dict__,
            'power_metrics': power_metrics,
            'cooling_metrics': cooling_metrics,
            'performance_metrics': performance_metrics,
            'financial_metrics': financial_metrics,
            'environmental_metrics': environmental_metrics,
            'workload_statistics': {
                'avg_gpu_utilization': workload_simulation['gpu_utilization'].mean(),
                'peak_power_mw': workload_simulation['power_consumption_mw'].max(),
                'avg_power_mw': workload_simulation['power_consumption_mw'].mean(),
                'annual_energy_gwh': workload_simulation['power_consumption_mw'].sum() * 24 / 1000
            }
        }
        
        self.simulation_results = simulation_results
        return simulation_results
    
    def generate_report(self) -> str:
        """Generate a comprehensive simulation report"""
        
        if not self.simulation_results:
            self.run_full_simulation()
        
        results = self.simulation_results
        
        report = f"""
# 5 GW AI Factory Data Center - Simulation Report

Generated: {results['timestamp']}

## Executive Summary

This simulation models a {self.config.total_power_capacity_mw/1000:.1f} GW AI factory data center with {self.config.total_gpus:,} GPUs.

### Key Performance Indicators

- **Total Power Capacity**: {self.config.total_power_capacity_mw:,.0f} MW
- **Actual Power Consumption**: {results['power_metrics']['total_power_mw']:,.0f} MW
- **Power Utilization**: {(results['power_metrics']['total_power_mw']/self.config.total_power_capacity_mw)*100:.1f}%
- **PUE**: {results['power_metrics']['actual_pue']:.3f}
- **AI Training Performance**: {results['performance_metrics']['training_exaflops_fp16']:.1f} exaFLOPS (FP16), {results['performance_metrics']['training_fp4_pflops']:.0f} PFLOPS (FP4)
- **AI Inference Performance**: {results['performance_metrics']['inference_exaflops_fp16']:.1f} exaFLOPS (FP16), {results['performance_metrics']['inference_exaops_fp8']:.1f} exaOPS (FP8)

## Power Infrastructure

- **GPU Power**: {results['power_metrics']['gpu_power_mw']:,.0f} MW ({(results['power_metrics']['gpu_power_mw']/results['power_metrics']['total_power_mw'])*100:.1f}%)
- **CPU Power**: {results['power_metrics']['cpu_power_mw']:,.0f} MW
- **Cooling Power**: {results['power_metrics']['cooling_power_mw']:,.0f} MW
- **Total IT Load**: {results['power_metrics']['it_load_mw']:,.0f} MW
- **Infrastructure Load**: {results['power_metrics']['total_power_mw'] - results['power_metrics']['it_load_mw']:,.0f} MW

## Cooling Infrastructure

- **Heat Load**: {results['cooling_metrics']['heat_load_mw']:,.0f} MW
- **Cooling Capacity**: {results['cooling_metrics']['cooling_capacity_mw']:,.0f} MW
- **Water Consumption**: {results['cooling_metrics']['water_consumption_mgd']:.1f} million gallons/day
- **Chiller Capacity per Plant**: {results['cooling_metrics']['chiller_capacity_mw']:,.0f} MW

## Performance Metrics

- **Training Performance (GB300)**: {results['performance_metrics']['training_exaflops_fp16']:.1f} exaFLOPS (FP16), {results['performance_metrics']['training_fp4_pflops']:.0f} PFLOPS (FP4)
- **Inference Performance (B300)**: {results['performance_metrics']['inference_exaflops_fp16']:.1f} exaFLOPS (FP16), {results['performance_metrics']['inference_exaops_fp8']:.1f} exaOPS (FP8)
- **Memory Bandwidth**: {results['performance_metrics']['total_memory_bandwidth_tbps']:,.0f} TB/s
- **Storage Bandwidth**: {results['performance_metrics']['storage_bandwidth_tbps']:.1f} TB/s
- **Network Backend**: {results['performance_metrics']['network_backend_tbps']:.0f} TB/s
- **Network Frontend**: {results['performance_metrics']['network_frontend_tbps']:.0f} TB/s

## Financial Analysis

- **Annual Revenue**: ${results['financial_metrics']['total_annual_revenue']/1e9:.1f}B
- **Annual Profit**: ${results['financial_metrics']['annual_profit']/1e9:.1f}B
- **5-Year ROI**: {results['financial_metrics']['roi_5_year']*100:.1f}%
- **Payback Period**: {results['financial_metrics']['payback_period_years']:.1f} years
- **Electricity Cost**: ${results['financial_metrics']['annual_electricity_cost']/1e9:.1f}B/year ({results['financial_metrics']['electricity_cost_percentage']:.1f}% of revenue)

## Environmental Impact

- **Annual Carbon Emissions**: {results['environmental_metrics']['annual_carbon_emissions_tons']:,.0f} tons CO2
- **Water Usage Efficiency**: {results['environmental_metrics']['wue_l_per_kwh']:.2f} L/kWh
- **Waste Heat Recovery**: {results['environmental_metrics']['waste_heat_mw']:,.0f} MW thermal
- **District Heating Potential**: {results['environmental_metrics']['district_heating_homes']:,.0f} homes

## Workload Analysis

- **Average GPU Utilization**: {results['workload_statistics']['avg_gpu_utilization']*100:.1f}%
- **Peak Power Demand**: {results['workload_statistics']['peak_power_mw']:,.0f} MW
- **Average Power Demand**: {results['workload_statistics']['avg_power_mw']:,.0f} MW
- **Annual Energy Consumption**: {results['workload_statistics']['annual_energy_gwh']:,.0f} GWh

## Recommendations

1. **Power Infrastructure**: Current design provides {((self.config.total_power_capacity_mw - results['power_metrics']['total_power_mw'])/self.config.total_power_capacity_mw)*100:.1f}% headroom for growth
2. **Cooling Optimization**: Consider advanced liquid cooling to improve PUE below {results['power_metrics']['actual_pue']:.3f}
3. **Renewable Energy**: Implement on-site solar/wind to reduce carbon footprint
4. **Waste Heat Recovery**: Capture {results['environmental_metrics']['waste_heat_mw']:,.0f} MW for district heating
5. **Performance Scaling**: Current configuration supports {results['performance_metrics']['training_exaflops_fp16']:.1f} exaFLOPS training capacity

---
*Simulation completed at {results['timestamp']}*
        """
        
        return report
    
    def save_results(self, filename: str = "5gw_simulation_results.json"):
        """Save simulation results to JSON file"""
        
        if not self.simulation_results:
            self.run_full_simulation()
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # Deep convert all values
        json_results = json.loads(json.dumps(self.simulation_results, default=convert_numpy))
        
        with open(filename, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"Simulation results saved to {filename}")

def main():
    """Main simulation execution"""
    
    # Initialize configuration
    config = DataCenterConfig()
    
    # Create simulator
    simulator = DataCenterSimulator(config)
    
    # Run simulation
    results = simulator.run_full_simulation()
    
    # Generate and print report
    report = simulator.generate_report()
    print(report)
    
    # Save results
    simulator.save_results("c:\\Users\\mhossen\\OneDrive - University of Tennessee\\AI\\data_center\\5gw_simulation_results.json")
    
    # Generate summary statistics
    print("\n" + "="*80)
    print("SIMULATION SUMMARY")
    print("="*80)
    print(f"Total GPUs: {config.total_gpus:,} (GB300: {config.training_gpus:,}, B300: {config.inference_gpus:,})")
    print(f"Peak AI Training Performance: {results['performance_metrics']['training_exaflops_fp16']:.1f} exaFLOPS (FP16), {results['performance_metrics']['training_fp4_pflops']:.0f} PFLOPS (FP4)")
    print(f"Peak AI Inference Performance: {results['performance_metrics']['inference_exaflops_fp16']:.1f} exaFLOPS (FP16)")
    print(f"Power Consumption: {results['power_metrics']['total_power_mw']:,.0f} MW")
    print(f"PUE: {results['power_metrics']['actual_pue']:.3f}")
    print(f"Annual Revenue: ${results['financial_metrics']['total_annual_revenue']/1e9:.1f}B")
    print(f"ROI (5-year): {results['financial_metrics']['roi_5_year']*100:.1f}%")
    print(f"Carbon Emissions: {results['environmental_metrics']['annual_carbon_emissions_tons']:,.0f} tons CO2/year")
    print("="*80)

if __name__ == "__main__":
    main()