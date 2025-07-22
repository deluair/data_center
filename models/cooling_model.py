import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum
import math

class CoolingType(Enum):
    AIR_COOLING = "air_cooling"
    LIQUID_COOLING = "liquid_cooling"
    IMMERSION_COOLING = "immersion_cooling"
    HYBRID_COOLING = "hybrid_cooling"

class CoolantType(Enum):
    WATER = "water"
    DIELECTRIC_FLUID = "dielectric_fluid"
    MINERAL_OIL = "mineral_oil"
    SYNTHETIC_FLUID = "synthetic_fluid"

@dataclass
class CoolingUnit:
    """Individual cooling unit specification"""
    name: str
    type: CoolingType
    capacity_kw: float
    efficiency_cop: float  # Coefficient of Performance
    flow_rate_lpm: float  # Liters per minute
    inlet_temp_c: float
    outlet_temp_c: float
    power_consumption_kw: float
    maintenance_interval_hours: int

@dataclass
class HeatExchanger:
    """Heat exchanger specifications"""
    name: str
    type: str  # plate, shell-tube, etc.
    capacity_kw: float
    effectiveness: float
    pressure_drop_kpa: float
    fouling_factor: float
    material: str

@dataclass
class CoolingTower:
    """Cooling tower specifications"""
    name: str
    capacity_kw: float
    approach_temp_c: float
    range_temp_c: float
    water_flow_rate_lpm: float
    fan_power_kw: float
    pump_power_kw: float
    evaporation_rate_lph: float  # Liters per hour

class ThermalManagement:
    """Advanced thermal management calculations"""
    
    @staticmethod
    def calculate_heat_load(power_w: float, efficiency: float = 1.0) -> float:
        """Calculate heat load from power consumption"""
        return power_w * (1 - efficiency) + power_w * efficiency
    
    @staticmethod
    def calculate_cooling_capacity_required(heat_load_w: float, safety_factor: float = 1.2) -> float:
        """Calculate required cooling capacity with safety factor"""
        return heat_load_w * safety_factor
    
    @staticmethod
    def calculate_fluid_flow_rate(heat_load_w: float, temp_delta_c: float, 
                                specific_heat: float = 4186) -> float:
        """Calculate required fluid flow rate (kg/s)"""
        return heat_load_w / (specific_heat * temp_delta_c)
    
    @staticmethod
    def calculate_pump_power(flow_rate_lps: float, pressure_drop_pa: float, 
                           pump_efficiency: float = 0.8) -> float:
        """Calculate pump power requirements (W)"""
        return (flow_rate_lps * pressure_drop_pa) / pump_efficiency
    
    @staticmethod
    def calculate_heat_transfer_coefficient(reynolds: float, prandtl: float, 
                                          geometry_factor: float = 0.023) -> float:
        """Calculate heat transfer coefficient using Dittus-Boelter equation"""
        return geometry_factor * (reynolds ** 0.8) * (prandtl ** 0.4)

class CoolingSystem:
    """Comprehensive cooling system management"""
    
    def __init__(self, total_heat_load_mw: float):
        self.total_heat_load_mw = total_heat_load_mw
        self.cooling_units: List[CoolingUnit] = []
        self.heat_exchangers: List[HeatExchanger] = []
        self.cooling_towers: List[CoolingTower] = []
        self.ambient_temp_c = 25.0
        self.target_temp_c = 27.0
        self.humidity_percent = 45.0
        
    def add_chilled_water_system(self, count: int, capacity_kw_each: float, cop: float = 6.0):
        """Add chilled water cooling systems"""
        for i in range(count):
            unit = CoolingUnit(
                name=f"Chiller_{i+1}",
                type=CoolingType.LIQUID_COOLING,
                capacity_kw=capacity_kw_each,
                efficiency_cop=cop,
                flow_rate_lpm=capacity_kw_each * 2.86,  # Typical flow rate
                inlet_temp_c=12.0,
                outlet_temp_c=7.0,
                power_consumption_kw=capacity_kw_each / cop,
                maintenance_interval_hours=8760
            )
            self.cooling_units.append(unit)
    
    def add_liquid_cooling_loops(self, count: int, capacity_kw_each: float):
        """Add direct liquid cooling loops"""
        for i in range(count):
            unit = CoolingUnit(
                name=f"Liquid_Loop_{i+1}",
                type=CoolingType.LIQUID_COOLING,
                capacity_kw=capacity_kw_each,
                efficiency_cop=15.0,  # Higher efficiency for direct cooling
                flow_rate_lpm=capacity_kw_each * 1.43,
                inlet_temp_c=45.0,
                outlet_temp_c=55.0,
                power_consumption_kw=capacity_kw_each / 15.0,
                maintenance_interval_hours=4380
            )
            self.cooling_units.append(unit)
    
    def add_immersion_cooling_systems(self, count: int, capacity_kw_each: float, 
                                    coolant_type: CoolantType = CoolantType.DIELECTRIC_FLUID):
        """Add immersion cooling systems"""
        cop_map = {
            CoolantType.DIELECTRIC_FLUID: 25.0,
            CoolantType.MINERAL_OIL: 20.0,
            CoolantType.SYNTHETIC_FLUID: 30.0
        }
        
        for i in range(count):
            unit = CoolingUnit(
                name=f"Immersion_{coolant_type.value}_{i+1}",
                type=CoolingType.IMMERSION_COOLING,
                capacity_kw=capacity_kw_each,
                efficiency_cop=cop_map[coolant_type],
                flow_rate_lpm=capacity_kw_each * 0.5,  # Lower flow rate for immersion
                inlet_temp_c=50.0,
                outlet_temp_c=65.0,
                power_consumption_kw=capacity_kw_each / cop_map[coolant_type],
                maintenance_interval_hours=17520  # Less frequent maintenance
            )
            self.cooling_units.append(unit)
    
    def add_cooling_towers(self, count: int, capacity_kw_each: float):
        """Add cooling towers for heat rejection"""
        for i in range(count):
            tower = CoolingTower(
                name=f"Cooling_Tower_{i+1}",
                capacity_kw=capacity_kw_each,
                approach_temp_c=5.0,
                range_temp_c=10.0,
                water_flow_rate_lpm=capacity_kw_each * 4.3,
                fan_power_kw=capacity_kw_each * 0.02,
                pump_power_kw=capacity_kw_each * 0.01,
                evaporation_rate_lph=capacity_kw_each * 1.5
            )
            self.cooling_towers.append(tower)
    
    def add_heat_exchangers(self, count: int, capacity_kw_each: float, 
                          exchanger_type: str = "plate"):
        """Add heat exchangers"""
        effectiveness_map = {
            "plate": 0.85,
            "shell_tube": 0.75,
            "brazed_plate": 0.90
        }
        
        for i in range(count):
            exchanger = HeatExchanger(
                name=f"Heat_Exchanger_{exchanger_type}_{i+1}",
                type=exchanger_type,
                capacity_kw=capacity_kw_each,
                effectiveness=effectiveness_map.get(exchanger_type, 0.8),
                pressure_drop_kpa=50.0,
                fouling_factor=0.0001,
                material="Stainless Steel 316L"
            )
            self.heat_exchangers.append(exchanger)
    
    def calculate_total_cooling_capacity(self) -> Dict[str, float]:
        """Calculate total cooling capacity by type"""
        capacity_by_type = {}
        for unit in self.cooling_units:
            cooling_type = unit.type.value
            if cooling_type not in capacity_by_type:
                capacity_by_type[cooling_type] = 0
            capacity_by_type[cooling_type] += unit.capacity_kw
        
        capacity_by_type['total_kw'] = sum(capacity_by_type.values())
        capacity_by_type['total_mw'] = capacity_by_type['total_kw'] / 1000
        return capacity_by_type
    
    def calculate_cooling_power_consumption(self) -> Dict[str, float]:
        """Calculate power consumption of cooling systems"""
        power_consumption = {
            'chillers_kw': sum(unit.power_consumption_kw for unit in self.cooling_units),
            'cooling_towers_kw': sum(tower.fan_power_kw + tower.pump_power_kw for tower in self.cooling_towers),
            'pumps_kw': sum(unit.capacity_kw * 0.02 for unit in self.cooling_units),  # Estimated pump power
            'controls_kw': len(self.cooling_units) * 5.0  # Control system power
        }
        
        power_consumption['total_kw'] = sum(power_consumption.values())
        power_consumption['total_mw'] = power_consumption['total_kw'] / 1000
        return power_consumption
    
    def calculate_system_cop(self) -> float:
        """Calculate overall system Coefficient of Performance"""
        total_cooling_capacity = sum(unit.capacity_kw for unit in self.cooling_units)
        total_power_consumption = sum(unit.power_consumption_kw for unit in self.cooling_units)
        
        return total_cooling_capacity / total_power_consumption if total_power_consumption > 0 else 0
    
    def calculate_water_consumption(self) -> Dict[str, float]:
        """Calculate water consumption for cooling systems"""
        evaporation = sum(tower.evaporation_rate_lph for tower in self.cooling_towers)
        blowdown = evaporation * 0.3  # Typical blowdown rate
        makeup = evaporation + blowdown
        
        return {
            'evaporation_lph': evaporation,
            'blowdown_lph': blowdown,
            'makeup_lph': makeup,
            'annual_consumption_m3': makeup * 8760 / 1000
        }
    
    def optimize_cooling_distribution(self, target_temp_c: float = 27.0) -> Dict[str, float]:
        """Optimize cooling distribution for target temperature"""
        total_capacity = sum(unit.capacity_kw for unit in self.cooling_units)
        required_capacity = self.total_heat_load_mw * 1000 * 1.2  # 20% safety factor
        
        # Calculate optimal distribution
        liquid_cooling_ratio = 0.7  # 70% liquid cooling
        immersion_cooling_ratio = 0.25  # 25% immersion cooling
        air_cooling_ratio = 0.05  # 5% air cooling
        
        optimization = {
            'required_capacity_kw': required_capacity,
            'available_capacity_kw': total_capacity,
            'capacity_utilization': required_capacity / total_capacity if total_capacity > 0 else 0,
            'liquid_cooling_kw': required_capacity * liquid_cooling_ratio,
            'immersion_cooling_kw': required_capacity * immersion_cooling_ratio,
            'air_cooling_kw': required_capacity * air_cooling_ratio,
            'target_temperature_c': target_temp_c,
            'estimated_pue_cooling': 1.0 + (required_capacity * 0.0002)  # Cooling contribution to PUE
        }
        
        return optimization
    
    def calculate_heat_recovery_potential(self) -> Dict[str, float]:
        """Calculate waste heat recovery potential"""
        total_heat_mw = self.total_heat_load_mw
        
        # Heat recovery scenarios
        recovery_scenarios = {
            'district_heating': {
                'recoverable_percentage': 0.6,
                'temperature_c': 65.0,
                'efficiency': 0.85
            },
            'hot_water_generation': {
                'recoverable_percentage': 0.4,
                'temperature_c': 45.0,
                'efficiency': 0.90
            },
            'space_heating': {
                'recoverable_percentage': 0.3,
                'temperature_c': 35.0,
                'efficiency': 0.95
            }
        }
        
        recovery_potential = {}
        for scenario, params in recovery_scenarios.items():
            recoverable_mw = total_heat_mw * params['recoverable_percentage'] * params['efficiency']
            recovery_potential[scenario] = {
                'recoverable_heat_mw': recoverable_mw,
                'annual_energy_mwh': recoverable_mw * 8760 * 0.6,  # 60% utilization
                'temperature_c': params['temperature_c'],
                'estimated_value_usd': recoverable_mw * 8760 * 0.6 * 30  # $30/MWh value
            }
        
        return recovery_potential
    
    def generate_cooling_report(self) -> Dict[str, any]:
        """Generate comprehensive cooling system report"""
        return {
            'system_overview': {
                'total_heat_load_mw': self.total_heat_load_mw,
                'cooling_units_count': len(self.cooling_units),
                'heat_exchangers_count': len(self.heat_exchangers),
                'cooling_towers_count': len(self.cooling_towers)
            },
            'capacity_analysis': self.calculate_total_cooling_capacity(),
            'power_consumption': self.calculate_cooling_power_consumption(),
            'system_efficiency': {
                'overall_cop': self.calculate_system_cop(),
                'target_temperature_c': self.target_temp_c
            },
            'water_usage': self.calculate_water_consumption(),
            'optimization': self.optimize_cooling_distribution(),
            'heat_recovery': self.calculate_heat_recovery_potential()
        }