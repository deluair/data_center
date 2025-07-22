import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum
import math

class FacilityTier(Enum):
    TIER_I = "tier_1"  # Basic capacity
    TIER_II = "tier_2"  # Redundant capacity components
    TIER_III = "tier_3"  # Concurrently maintainable
    TIER_IV = "tier_4"  # Fault tolerant

class SecurityLevel(Enum):
    BASIC = "basic"
    ENHANCED = "enhanced"
    HIGH = "high"
    MAXIMUM = "maximum"

class EnvironmentalZone(Enum):
    COMPUTE = "compute"  # Server areas
    STORAGE = "storage"  # Storage areas
    NETWORK = "network"  # Network equipment
    POWER = "power"  # Electrical equipment
    COOLING = "cooling"  # Cooling equipment
    OFFICE = "office"  # Administrative areas
    LOADING = "loading"  # Loading docks

class BuildingSystem(Enum):
    HVAC = "hvac"
    ELECTRICAL = "electrical"
    FIRE_SUPPRESSION = "fire_suppression"
    SECURITY = "security"
    LIGHTING = "lighting"
    MONITORING = "monitoring"
    ACCESS_CONTROL = "access_control"

@dataclass
class RackSpecification:
    """Server rack specification"""
    model: str
    height_units: int  # Standard 42U
    width_mm: int
    depth_mm: int
    max_weight_kg: int
    power_capacity_kw: float
    cooling_capacity_kw: float
    cable_management: bool
    cost: float

@dataclass
class FloorSpace:
    """Data center floor space specification"""
    zone_type: EnvironmentalZone
    area_sqm: float
    height_m: float
    rack_count: int
    power_density_kw_per_sqm: float
    cooling_density_kw_per_sqm: float
    access_level: SecurityLevel

@dataclass
class BuildingInfrastructure:
    """Building infrastructure component"""
    system_type: BuildingSystem
    capacity: float
    redundancy_level: str  # N, N+1, N+2, 2N
    efficiency: float
    power_consumption_kw: float
    maintenance_interval_hours: int
    cost: float
    lifespan_years: int

@dataclass
class SecuritySystem:
    """Security system specification"""
    system_name: str
    security_level: SecurityLevel
    coverage_area_sqm: float
    power_consumption_w: float
    monitoring_capability: bool
    integration_level: str
    cost: float

class EnvironmentalControl:
    """Environmental monitoring and control system"""
    
    def __init__(self):
        self.temperature_zones = {}
        self.humidity_zones = {}
        self.air_quality_zones = {}
        self.monitoring_points = []
    
    def add_temperature_zone(self, zone: EnvironmentalZone, 
                           target_temp_c: float, tolerance_c: float = 2.0):
        """Add temperature control zone"""
        self.temperature_zones[zone] = {
            'target_temperature_c': target_temp_c,
            'tolerance_c': tolerance_c,
            'min_temperature_c': target_temp_c - tolerance_c,
            'max_temperature_c': target_temp_c + tolerance_c
        }
    
    def add_humidity_zone(self, zone: EnvironmentalZone, 
                         target_humidity_percent: float, tolerance_percent: float = 5.0):
        """Add humidity control zone"""
        self.humidity_zones[zone] = {
            'target_humidity_percent': target_humidity_percent,
            'tolerance_percent': tolerance_percent,
            'min_humidity_percent': target_humidity_percent - tolerance_percent,
            'max_humidity_percent': target_humidity_percent + tolerance_percent
        }
    
    def calculate_environmental_load(self, zone: EnvironmentalZone, 
                                   equipment_power_kw: float) -> Dict[str, float]:
        """Calculate environmental control load for a zone"""
        # Heat load from equipment (assume 95% of power becomes heat)
        heat_load_kw = equipment_power_kw * 0.95
        
        # Cooling load (includes heat removal + overcooling for humidity control)
        cooling_load_kw = heat_load_kw * 1.3  # 30% overhead for humidity control
        
        # Ventilation requirements (air changes per hour)
        if zone == EnvironmentalZone.COMPUTE:
            air_changes_per_hour = 15
        elif zone == EnvironmentalZone.STORAGE:
            air_changes_per_hour = 10
        elif zone == EnvironmentalZone.NETWORK:
            air_changes_per_hour = 12
        else:
            air_changes_per_hour = 6
        
        return {
            'heat_load_kw': heat_load_kw,
            'cooling_load_kw': cooling_load_kw,
            'air_changes_per_hour': air_changes_per_hour,
            'ventilation_load_kw': cooling_load_kw * 0.1  # 10% for ventilation
        }

class FacilityInfrastructure:
    """Comprehensive facility infrastructure management"""
    
    def __init__(self, name: str, facility_tier: FacilityTier = FacilityTier.TIER_III):
        self.name = name
        self.facility_tier = facility_tier
        self.floor_spaces: List[FloorSpace] = []
        self.racks: List[RackSpecification] = []
        self.building_systems: List[BuildingInfrastructure] = []
        self.security_systems: List[SecuritySystem] = []
        self.environmental_control = EnvironmentalControl()
        
    def add_compute_floor(self, area_sqm: float, rack_count: int, 
                         power_density_kw_per_sqm: float = 15.0):
        """Add compute floor space"""
        floor = FloorSpace(
            zone_type=EnvironmentalZone.COMPUTE,
            area_sqm=area_sqm,
            height_m=4.0,  # Standard data center height
            rack_count=rack_count,
            power_density_kw_per_sqm=power_density_kw_per_sqm,
            cooling_density_kw_per_sqm=power_density_kw_per_sqm * 1.3,  # PUE factor
            access_level=SecurityLevel.HIGH
        )
        self.floor_spaces.append(floor)
        
        # Add environmental controls for compute zone
        self.environmental_control.add_temperature_zone(
            EnvironmentalZone.COMPUTE, target_temp_c=22.0, tolerance_c=2.0
        )
        self.environmental_control.add_humidity_zone(
            EnvironmentalZone.COMPUTE, target_humidity_percent=45.0, tolerance_percent=5.0
        )
    
    def add_storage_floor(self, area_sqm: float, rack_count: int):
        """Add storage floor space"""
        floor = FloorSpace(
            zone_type=EnvironmentalZone.STORAGE,
            area_sqm=area_sqm,
            height_m=3.5,
            rack_count=rack_count,
            power_density_kw_per_sqm=8.0,  # Lower power density for storage
            cooling_density_kw_per_sqm=10.0,
            access_level=SecurityLevel.ENHANCED
        )
        self.floor_spaces.append(floor)
        
        # Add environmental controls for storage zone
        self.environmental_control.add_temperature_zone(
            EnvironmentalZone.STORAGE, target_temp_c=20.0, tolerance_c=3.0
        )
        self.environmental_control.add_humidity_zone(
            EnvironmentalZone.STORAGE, target_humidity_percent=40.0, tolerance_percent=10.0
        )
    
    def add_network_floor(self, area_sqm: float, rack_count: int):
        """Add network equipment floor space"""
        floor = FloorSpace(
            zone_type=EnvironmentalZone.NETWORK,
            area_sqm=area_sqm,
            height_m=3.0,
            rack_count=rack_count,
            power_density_kw_per_sqm=12.0,
            cooling_density_kw_per_sqm=15.0,
            access_level=SecurityLevel.MAXIMUM
        )
        self.floor_spaces.append(floor)
        
        # Add environmental controls for network zone
        self.environmental_control.add_temperature_zone(
            EnvironmentalZone.NETWORK, target_temp_c=21.0, tolerance_c=1.5
        )
        self.environmental_control.add_humidity_zone(
            EnvironmentalZone.NETWORK, target_humidity_percent=45.0, tolerance_percent=5.0
        )
    
    def add_standard_racks(self, count: int, power_capacity_kw: float = 20.0):
        """Add standard server racks"""
        for i in range(count):
            rack = RackSpecification(
                model=f"Standard_42U_Rack_{i+1}",
                height_units=42,
                width_mm=600,
                depth_mm=1200,
                max_weight_kg=1500,
                power_capacity_kw=power_capacity_kw,
                cooling_capacity_kw=power_capacity_kw * 1.3,
                cable_management=True,
                cost=5000
            )
            self.racks.append(rack)
    
    def add_high_density_racks(self, count: int, power_capacity_kw: float = 50.0):
        """Add high-density server racks for AI workloads"""
        for i in range(count):
            rack = RackSpecification(
                model=f"High_Density_42U_Rack_{i+1}",
                height_units=42,
                width_mm=800,
                depth_mm=1400,
                max_weight_kg=2500,
                power_capacity_kw=power_capacity_kw,
                cooling_capacity_kw=power_capacity_kw * 1.5,  # Higher cooling for AI
                cable_management=True,
                cost=12000
            )
            self.racks.append(rack)
    
    def add_hvac_system(self, capacity_kw: float, redundancy: str = "N+1"):
        """Add HVAC system"""
        efficiency = 0.85 if redundancy == "N+1" else 0.90
        power_consumption = capacity_kw * 0.3  # HVAC typically uses 30% of cooling capacity
        
        hvac = BuildingInfrastructure(
            system_type=BuildingSystem.HVAC,
            capacity=capacity_kw,
            redundancy_level=redundancy,
            efficiency=efficiency,
            power_consumption_kw=power_consumption,
            maintenance_interval_hours=2160,  # 3 months
            cost=capacity_kw * 1500,  # $1500 per kW
            lifespan_years=15
        )
        self.building_systems.append(hvac)
    
    def add_electrical_system(self, capacity_kw: float, redundancy: str = "2N"):
        """Add electrical distribution system"""
        efficiency = 0.95 if redundancy == "2N" else 0.93
        power_consumption = capacity_kw * 0.02  # 2% losses in distribution
        
        electrical = BuildingInfrastructure(
            system_type=BuildingSystem.ELECTRICAL,
            capacity=capacity_kw,
            redundancy_level=redundancy,
            efficiency=efficiency,
            power_consumption_kw=power_consumption,
            maintenance_interval_hours=4380,  # 6 months
            cost=capacity_kw * 800,  # $800 per kW
            lifespan_years=25
        )
        self.building_systems.append(electrical)
    
    def add_fire_suppression_system(self, coverage_area_sqm: float):
        """Add fire suppression system"""
        fire_suppression = BuildingInfrastructure(
            system_type=BuildingSystem.FIRE_SUPPRESSION,
            capacity=coverage_area_sqm,
            redundancy_level="N+1",
            efficiency=0.99,
            power_consumption_kw=coverage_area_sqm * 0.01,  # 10W per sqm
            maintenance_interval_hours=4380,  # 6 months
            cost=coverage_area_sqm * 200,  # $200 per sqm
            lifespan_years=20
        )
        self.building_systems.append(fire_suppression)
    
    def add_lighting_system(self, area_sqm: float, led_efficiency: bool = True):
        """Add lighting system"""
        power_density = 8 if led_efficiency else 15  # W per sqm
        
        lighting = BuildingInfrastructure(
            system_type=BuildingSystem.LIGHTING,
            capacity=area_sqm,
            redundancy_level="N",
            efficiency=0.90 if led_efficiency else 0.70,
            power_consumption_kw=area_sqm * power_density / 1000,
            maintenance_interval_hours=8760,  # 1 year
            cost=area_sqm * 100,  # $100 per sqm
            lifespan_years=10 if led_efficiency else 5
        )
        self.building_systems.append(lighting)
    
    def add_security_system(self, system_name: str, security_level: SecurityLevel, 
                          coverage_area_sqm: float):
        """Add security system"""
        # Power consumption varies by security level
        power_multiplier = {
            SecurityLevel.BASIC: 0.5,
            SecurityLevel.ENHANCED: 1.0,
            SecurityLevel.HIGH: 2.0,
            SecurityLevel.MAXIMUM: 4.0
        }
        
        cost_multiplier = {
            SecurityLevel.BASIC: 50,
            SecurityLevel.ENHANCED: 150,
            SecurityLevel.HIGH: 300,
            SecurityLevel.MAXIMUM: 600
        }
        
        security = SecuritySystem(
            system_name=system_name,
            security_level=security_level,
            coverage_area_sqm=coverage_area_sqm,
            power_consumption_w=coverage_area_sqm * power_multiplier[security_level],
            monitoring_capability=security_level in [SecurityLevel.HIGH, SecurityLevel.MAXIMUM],
            integration_level="Full" if security_level == SecurityLevel.MAXIMUM else "Partial",
            cost=coverage_area_sqm * cost_multiplier[security_level]
        )
        self.security_systems.append(security)
    
    def calculate_total_floor_space(self) -> Dict[str, float]:
        """Calculate total facility floor space"""
        total_area_sqm = sum(floor.area_sqm for floor in self.floor_spaces)
        total_volume_m3 = sum(floor.area_sqm * floor.height_m for floor in self.floor_spaces)
        total_racks = sum(floor.rack_count for floor in self.floor_spaces)
        
        area_by_zone = {}
        for floor in self.floor_spaces:
            zone = floor.zone_type.value
            if zone not in area_by_zone:
                area_by_zone[zone] = 0
            area_by_zone[zone] += floor.area_sqm
        
        return {
            'total_area_sqm': total_area_sqm,
            'total_area_sqft': total_area_sqm * 10.764,  # Convert to sq ft
            'total_volume_m3': total_volume_m3,
            'total_racks': total_racks,
            'rack_density_per_sqm': total_racks / total_area_sqm if total_area_sqm > 0 else 0,
            'area_by_zone': area_by_zone
        }
    
    def calculate_power_requirements(self) -> Dict[str, float]:
        """Calculate facility power requirements"""
        # IT equipment power
        it_power_kw = sum(floor.area_sqm * floor.power_density_kw_per_sqm for floor in self.floor_spaces)
        
        # Building systems power
        building_power_kw = sum(system.power_consumption_kw for system in self.building_systems)
        
        # Security systems power
        security_power_kw = sum(system.power_consumption_w / 1000 for system in self.security_systems)
        
        # Total facility power
        total_power_kw = it_power_kw + building_power_kw + security_power_kw
        
        return {
            'it_power_kw': it_power_kw,
            'it_power_mw': it_power_kw / 1000,
            'building_systems_power_kw': building_power_kw,
            'security_systems_power_kw': security_power_kw,
            'total_facility_power_kw': total_power_kw,
            'total_facility_power_mw': total_power_kw / 1000
        }
    
    def calculate_cooling_requirements(self) -> Dict[str, float]:
        """Calculate facility cooling requirements"""
        total_cooling_kw = 0
        cooling_by_zone = {}
        
        for floor in self.floor_spaces:
            zone_cooling = floor.area_sqm * floor.cooling_density_kw_per_sqm
            total_cooling_kw += zone_cooling
            
            zone = floor.zone_type.value
            if zone not in cooling_by_zone:
                cooling_by_zone[zone] = 0
            cooling_by_zone[zone] += zone_cooling
        
        # Add cooling for building systems (they generate heat too)
        building_heat_kw = sum(system.power_consumption_kw * 0.8 for system in self.building_systems)
        total_cooling_kw += building_heat_kw
        
        return {
            'total_cooling_kw': total_cooling_kw,
            'total_cooling_mw': total_cooling_kw / 1000,
            'total_cooling_tons': total_cooling_kw * 0.284,  # Convert kW to tons
            'cooling_by_zone': cooling_by_zone,
            'building_systems_heat_kw': building_heat_kw
        }
    
    def calculate_facility_costs(self) -> Dict[str, float]:
        """Calculate facility infrastructure costs"""
        # Rack costs
        rack_costs = sum(rack.cost for rack in self.racks)
        
        # Building systems costs
        building_systems_costs = sum(system.cost for system in self.building_systems)
        
        # Security systems costs
        security_costs = sum(system.cost for system in self.security_systems)
        
        # Construction costs (estimated based on area and complexity)
        total_area = self.calculate_total_floor_space()['total_area_sqm']
        construction_cost_per_sqm = {
            FacilityTier.TIER_I: 2000,
            FacilityTier.TIER_II: 3000,
            FacilityTier.TIER_III: 4500,
            FacilityTier.TIER_IV: 6500
        }
        construction_costs = total_area * construction_cost_per_sqm[self.facility_tier]
        
        total_costs = rack_costs + building_systems_costs + security_costs + construction_costs
        
        return {
            'rack_costs': rack_costs,
            'building_systems_costs': building_systems_costs,
            'security_costs': security_costs,
            'construction_costs': construction_costs,
            'total_facility_costs': total_costs,
            'cost_per_sqm': total_costs / total_area if total_area > 0 else 0
        }
    
    def calculate_facility_efficiency(self) -> Dict[str, float]:
        """Calculate facility efficiency metrics"""
        power_req = self.calculate_power_requirements()
        cooling_req = self.calculate_cooling_requirements()
        
        # Power Usage Effectiveness (PUE)
        it_power = power_req['it_power_kw']
        total_power = power_req['total_facility_power_kw']
        pue = total_power / it_power if it_power > 0 else 1.0
        
        # Cooling efficiency
        cooling_power = sum(s.power_consumption_kw for s in self.building_systems 
                          if s.system_type == BuildingSystem.HVAC)
        cooling_efficiency = cooling_req['total_cooling_kw'] / cooling_power if cooling_power > 0 else 0
        
        # Space utilization
        floor_space = self.calculate_total_floor_space()
        compute_area = floor_space['area_by_zone'].get('compute', 0)
        space_utilization = compute_area / floor_space['total_area_sqm'] if floor_space['total_area_sqm'] > 0 else 0
        
        return {
            'pue': pue,
            'cooling_efficiency_cop': cooling_efficiency,
            'space_utilization': space_utilization,
            'power_density_kw_per_sqm': total_power / floor_space['total_area_sqm'] if floor_space['total_area_sqm'] > 0 else 0,
            'rack_density_per_sqm': floor_space['rack_density_per_sqm']
        }
    
    def assess_facility_tier_compliance(self) -> Dict[str, any]:
        """Assess compliance with facility tier requirements"""
        redundancy_requirements = {
            FacilityTier.TIER_I: {"power": "N", "cooling": "N", "uptime": 99.671},
            FacilityTier.TIER_II: {"power": "N+1", "cooling": "N+1", "uptime": 99.741},
            FacilityTier.TIER_III: {"power": "N+1", "cooling": "N+1", "uptime": 99.982},
            FacilityTier.TIER_IV: {"power": "2N", "cooling": "2N", "uptime": 99.995}
        }
        
        requirements = redundancy_requirements[self.facility_tier]
        
        # Check power redundancy
        power_systems = [s for s in self.building_systems if s.system_type == BuildingSystem.ELECTRICAL]
        power_compliance = any(s.redundancy_level == requirements["power"] for s in power_systems)
        
        # Check cooling redundancy
        cooling_systems = [s for s in self.building_systems if s.system_type == BuildingSystem.HVAC]
        cooling_compliance = any(s.redundancy_level == requirements["cooling"] for s in cooling_systems)
        
        # Overall compliance
        overall_compliance = power_compliance and cooling_compliance
        
        return {
            'target_tier': self.facility_tier.value,
            'required_uptime_percent': requirements["uptime"],
            'power_redundancy_required': requirements["power"],
            'cooling_redundancy_required': requirements["cooling"],
            'power_compliance': power_compliance,
            'cooling_compliance': cooling_compliance,
            'overall_compliance': overall_compliance,
            'compliance_status': 'Compliant' if overall_compliance else 'Non-Compliant'
        }
    
    def optimize_facility_layout(self) -> Dict[str, any]:
        """Optimize facility layout for efficiency"""
        floor_space = self.calculate_total_floor_space()
        power_req = self.calculate_power_requirements()
        
        # Calculate optimal zone allocation
        total_area = floor_space['total_area_sqm']
        
        optimal_allocation = {
            'compute_zone': {
                'percentage': 60,
                'area_sqm': total_area * 0.6,
                'purpose': 'Primary compute infrastructure'
            },
            'storage_zone': {
                'percentage': 15,
                'area_sqm': total_area * 0.15,
                'purpose': 'Storage systems'
            },
            'network_zone': {
                'percentage': 10,
                'area_sqm': total_area * 0.10,
                'purpose': 'Network infrastructure'
            },
            'power_zone': {
                'percentage': 8,
                'area_sqm': total_area * 0.08,
                'purpose': 'Electrical distribution'
            },
            'cooling_zone': {
                'percentage': 5,
                'area_sqm': total_area * 0.05,
                'purpose': 'Cooling equipment'
            },
            'support_zone': {
                'percentage': 2,
                'area_sqm': total_area * 0.02,
                'purpose': 'Office and support areas'
            }
        }
        
        return {
            'current_layout': floor_space['area_by_zone'],
            'optimal_layout': optimal_allocation,
            'optimization_recommendations': [
                'Maximize compute zone allocation for AI workloads',
                'Ensure adequate cooling zone proximity to compute areas',
                'Implement hot/cold aisle containment',
                'Position network equipment for minimal cable runs',
                'Separate power distribution for redundancy'
            ]
        }
    
    def generate_facility_report(self) -> Dict[str, any]:
        """Generate comprehensive facility infrastructure report"""
        return {
            'facility_overview': {
                'name': self.name,
                'tier': self.facility_tier.value,
                'floor_spaces': len(self.floor_spaces),
                'total_racks': len(self.racks),
                'building_systems': len(self.building_systems),
                'security_systems': len(self.security_systems)
            },
            'space_analysis': self.calculate_total_floor_space(),
            'power_requirements': self.calculate_power_requirements(),
            'cooling_requirements': self.calculate_cooling_requirements(),
            'cost_analysis': self.calculate_facility_costs(),
            'efficiency_metrics': self.calculate_facility_efficiency(),
            'tier_compliance': self.assess_facility_tier_compliance(),
            'layout_optimization': self.optimize_facility_layout()
        }