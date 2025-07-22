import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum

class PowerSourceType(Enum):
    GRID = "grid"
    GENERATOR = "generator"
    UPS = "ups"
    BATTERY = "battery"
    RENEWABLE = "renewable"

@dataclass
class PowerSource:
    """Individual power source configuration"""
    name: str
    type: PowerSourceType
    capacity_mw: float
    voltage_kv: float
    efficiency: float
    redundancy_level: str
    maintenance_schedule: str
    cost_per_mwh: float

@dataclass
class Transformer:
    """Electrical transformer specifications"""
    name: str
    primary_voltage_kv: float
    secondary_voltage_kv: float
    capacity_mva: float
    efficiency: float
    load_factor: float
    cooling_type: str

@dataclass
class UPSSystem:
    """Uninterruptible Power Supply configuration"""
    name: str
    capacity_mw: float
    efficiency: float
    backup_time_minutes: float
    battery_capacity_mwh: float
    redundancy: str
    maintenance_bypass: bool

class PowerInfrastructure:
    """Complete power infrastructure management"""
    
    def __init__(self, total_capacity_mw: float):
        self.total_capacity_mw = total_capacity_mw
        self.power_sources: List[PowerSource] = []
        self.transformers: List[Transformer] = []
        self.ups_systems: List[UPSSystem] = []
        self.distribution_losses = 0.03  # 3% distribution losses
        
    def add_grid_connection(self, voltage_kv: float, capacity_mw: float, cost_per_mwh: float = 50.0):
        """Add grid power connection"""
        source = PowerSource(
            name=f"Grid_{voltage_kv}kV",
            type=PowerSourceType.GRID,
            capacity_mw=capacity_mw,
            voltage_kv=voltage_kv,
            efficiency=0.98,
            redundancy_level="N+1",
            maintenance_schedule="Annual",
            cost_per_mwh=cost_per_mwh
        )
        self.power_sources.append(source)
        
    def add_backup_generators(self, count: int, capacity_mw_each: float, fuel_type: str = "natural_gas"):
        """Add backup generator systems"""
        for i in range(count):
            cost_per_mwh = 150.0 if fuel_type == "natural_gas" else 200.0
            source = PowerSource(
                name=f"Generator_{i+1}_{fuel_type}",
                type=PowerSourceType.GENERATOR,
                capacity_mw=capacity_mw_each,
                voltage_kv=13.8,
                efficiency=0.42 if fuel_type == "natural_gas" else 0.38,
                redundancy_level="N+1",
                maintenance_schedule="Monthly",
                cost_per_mwh=cost_per_mwh
            )
            self.power_sources.append(source)
            
    def add_transformers(self, count: int, capacity_mva_each: float, 
                        primary_kv: float, secondary_kv: float):
        """Add main transformers"""
        for i in range(count):
            transformer = Transformer(
                name=f"Main_Transformer_{i+1}",
                primary_voltage_kv=primary_kv,
                secondary_voltage_kv=secondary_kv,
                capacity_mva=capacity_mva_each,
                efficiency=0.99,
                load_factor=0.8,
                cooling_type="ONAN"
            )
            self.transformers.append(transformer)
            
    def add_ups_systems(self, count: int, capacity_mw_each: float, 
                       backup_time_minutes: float = 15.0):
        """Add UPS systems"""
        for i in range(count):
            ups = UPSSystem(
                name=f"UPS_{i+1}",
                capacity_mw=capacity_mw_each,
                efficiency=0.96,
                backup_time_minutes=backup_time_minutes,
                battery_capacity_mwh=capacity_mw_each * backup_time_minutes / 60,
                redundancy="N+1",
                maintenance_bypass=True
            )
            self.ups_systems.append(ups)
            
    def calculate_total_capacity(self) -> Dict[str, float]:
        """Calculate total capacity by source type"""
        capacity_by_type = {}
        for source in self.power_sources:
            source_type = source.type.value
            if source_type not in capacity_by_type:
                capacity_by_type[source_type] = 0
            capacity_by_type[source_type] += source.capacity_mw
            
        return capacity_by_type
        
    def calculate_redundancy_factor(self) -> float:
        """Calculate overall redundancy factor"""
        total_capacity = sum(source.capacity_mw for source in self.power_sources)
        return total_capacity / self.total_capacity_mw
        
    def estimate_power_costs(self, annual_consumption_mwh: float) -> Dict[str, float]:
        """Estimate annual power costs by source"""
        costs = {}
        total_capacity = sum(source.capacity_mw for source in self.power_sources)
        
        for source in self.power_sources:
            # Proportional usage based on capacity and efficiency
            usage_factor = (source.capacity_mw / total_capacity) * source.efficiency
            annual_usage = annual_consumption_mwh * usage_factor
            costs[source.name] = annual_usage * source.cost_per_mwh
            
        return costs

class PowerConsumption:
    """Power consumption modeling and analysis"""
    
    def __init__(self):
        self.gpu_power_w = {}
        self.cpu_power_w = {}
        self.memory_power_w = {}
        self.storage_power_w = {}
        self.network_power_w = {}
        self.cooling_power_w = {}
        self.facility_power_w = {}
        
    def set_gpu_power(self, gpu_type: str, count: int, tdp_w: float, utilization: float = 0.8):
        """Set GPU power consumption"""
        self.gpu_power_w[gpu_type] = {
            'count': count,
            'tdp_w': tdp_w,
            'utilization': utilization,
            'total_w': count * tdp_w * utilization
        }
        
    def set_cpu_power(self, cpu_type: str, count: int, tdp_w: float, utilization: float = 0.6):
        """Set CPU power consumption"""
        self.cpu_power_w[cpu_type] = {
            'count': count,
            'tdp_w': tdp_w,
            'utilization': utilization,
            'total_w': count * tdp_w * utilization
        }
        
    def set_infrastructure_power(self, memory_mw: float, storage_mw: float, 
                               network_mw: float, cooling_mw: float, facility_mw: float):
        """Set infrastructure power consumption"""
        self.memory_power_w = memory_mw * 1e6
        self.storage_power_w = storage_mw * 1e6
        self.network_power_w = network_mw * 1e6
        self.cooling_power_w = cooling_mw * 1e6
        self.facility_power_w = facility_mw * 1e6
        
    def calculate_total_power_mw(self) -> Dict[str, float]:
        """Calculate total power consumption by category"""
        gpu_total = sum(gpu['total_w'] for gpu in self.gpu_power_w.values())
        cpu_total = sum(cpu['total_w'] for cpu in self.cpu_power_w.values())
        
        power_breakdown = {
            'gpu_mw': gpu_total / 1e6,
            'cpu_mw': cpu_total / 1e6,
            'memory_mw': self.memory_power_w / 1e6,
            'storage_mw': self.storage_power_w / 1e6,
            'network_mw': self.network_power_w / 1e6,
            'cooling_mw': self.cooling_power_w / 1e6,
            'facility_mw': self.facility_power_w / 1e6
        }
        
        power_breakdown['total_mw'] = sum(power_breakdown.values())
        return power_breakdown
        
    def calculate_pue(self, it_power_mw: float, total_power_mw: float) -> float:
        """Calculate Power Usage Effectiveness"""
        return total_power_mw / it_power_mw if it_power_mw > 0 else 0
        
    def estimate_annual_consumption(self, total_power_mw: float, 
                                  utilization_factor: float = 0.8) -> float:
        """Estimate annual power consumption in MWh"""
        hours_per_year = 8760
        return total_power_mw * utilization_factor * hours_per_year
        
    def calculate_power_density(self, total_power_mw: float, floor_area_sqm: float) -> float:
        """Calculate power density in kW/sqm"""
        return (total_power_mw * 1000) / floor_area_sqm
        
    def optimize_power_distribution(self, target_pue: float = 1.1) -> Dict[str, float]:
        """Optimize power distribution to achieve target PUE"""
        power_breakdown = self.calculate_total_power_mw()
        it_power = power_breakdown['gpu_mw'] + power_breakdown['cpu_mw'] + \
                  power_breakdown['memory_mw'] + power_breakdown['storage_mw'] + \
                  power_breakdown['network_mw']
        
        # Calculate required infrastructure power for target PUE
        target_total_power = it_power * target_pue
        target_infrastructure_power = target_total_power - it_power
        
        # Distribute infrastructure power
        cooling_ratio = 0.65  # 65% for cooling
        facility_ratio = 0.35  # 35% for facility
        
        optimized = {
            'it_power_mw': it_power,
            'cooling_power_mw': target_infrastructure_power * cooling_ratio,
            'facility_power_mw': target_infrastructure_power * facility_ratio,
            'total_power_mw': target_total_power,
            'achieved_pue': target_pue
        }
        
        return optimized