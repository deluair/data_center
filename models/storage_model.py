import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum
import math

class StorageType(Enum):
    NVME_SSD = "nvme_ssd"
    SATA_SSD = "sata_ssd"
    HDD = "hdd"
    OPTANE = "optane"
    OBJECT_STORAGE = "object_storage"
    TAPE = "tape"

class StorageTier(Enum):
    HOT = "hot"  # Frequently accessed
    WARM = "warm"  # Occasionally accessed
    COLD = "cold"  # Rarely accessed
    ARCHIVE = "archive"  # Long-term storage

class RAIDLevel(Enum):
    RAID0 = "raid0"
    RAID1 = "raid1"
    RAID5 = "raid5"
    RAID6 = "raid6"
    RAID10 = "raid10"
    RAID50 = "raid50"
    RAID60 = "raid60"

@dataclass
class StorageDevice:
    """Individual storage device specification"""
    model: str
    type: StorageType
    capacity_tb: float
    sequential_read_mbps: int
    sequential_write_mbps: int
    random_read_iops: int
    random_write_iops: int
    power_consumption_w: float
    mtbf_hours: int
    interface: str
    form_factor: str
    cost_per_tb: float

@dataclass
class StorageArray:
    """Storage array configuration"""
    name: str
    device_type: StorageType
    device_count: int
    raid_level: RAIDLevel
    hot_spares: int
    controller_model: str
    cache_size_gb: int
    total_capacity_tb: float
    usable_capacity_tb: float
    performance_iops: int
    power_consumption_w: float

@dataclass
class FileSystem:
    """File system configuration"""
    name: str
    type: str  # ext4, xfs, zfs, lustre, etc.
    mount_point: str
    capacity_tb: float
    block_size_kb: int
    compression_enabled: bool
    deduplication_enabled: bool
    encryption_enabled: bool
    backup_enabled: bool

class DataManagement:
    """Data lifecycle and management policies"""
    
    def __init__(self):
        self.tiering_policies = {}
        self.backup_policies = {}
        self.retention_policies = {}
        
    def add_tiering_policy(self, name: str, hot_days: int, warm_days: int, cold_days: int):
        """Add data tiering policy"""
        self.tiering_policies[name] = {
            'hot_tier_days': hot_days,
            'warm_tier_days': warm_days,
            'cold_tier_days': cold_days,
            'archive_after_days': hot_days + warm_days + cold_days
        }
    
    def add_backup_policy(self, name: str, frequency_hours: int, retention_days: int, 
                         compression: bool = True, encryption: bool = True):
        """Add backup policy"""
        self.backup_policies[name] = {
            'frequency_hours': frequency_hours,
            'retention_days': retention_days,
            'compression_enabled': compression,
            'encryption_enabled': encryption,
            'backup_window_hours': 8
        }
    
    def calculate_storage_requirements(self, data_growth_rate: float, 
                                     years: int = 5) -> Dict[str, float]:
        """Calculate storage requirements with growth projection"""
        base_capacity = 1000  # TB
        projected_capacity = base_capacity * ((1 + data_growth_rate) ** years)
        
        return {
            'current_capacity_tb': base_capacity,
            'projected_capacity_tb': projected_capacity,
            'growth_factor': projected_capacity / base_capacity,
            'annual_growth_rate': data_growth_rate,
            'years_projected': years
        }

class StorageSystem:
    """Comprehensive storage system management"""
    
    def __init__(self, name: str):
        self.name = name
        self.storage_devices: List[StorageDevice] = []
        self.storage_arrays: List[StorageArray] = []
        self.file_systems: List[FileSystem] = []
        self.data_management = DataManagement()
        
    def add_nvme_storage(self, count: int, capacity_tb_each: float, model: str = "Enterprise_NVMe"):
        """Add NVMe SSD storage devices"""
        for i in range(count):
            device = StorageDevice(
                model=f"{model}_{i+1}",
                type=StorageType.NVME_SSD,
                capacity_tb=capacity_tb_each,
                sequential_read_mbps=7000,
                sequential_write_mbps=6000,
                random_read_iops=1000000,
                random_write_iops=200000,
                power_consumption_w=25.0,
                mtbf_hours=2000000,
                interface="PCIe 4.0 x4",
                form_factor="U.2",
                cost_per_tb=200.0
            )
            self.storage_devices.append(device)
    
    def add_ssd_storage(self, count: int, capacity_tb_each: float, model: str = "Enterprise_SSD"):
        """Add SATA SSD storage devices"""
        for i in range(count):
            device = StorageDevice(
                model=f"{model}_{i+1}",
                type=StorageType.SATA_SSD,
                capacity_tb=capacity_tb_each,
                sequential_read_mbps=550,
                sequential_write_mbps=520,
                random_read_iops=100000,
                random_write_iops=90000,
                power_consumption_w=7.0,
                mtbf_hours=2500000,
                interface="SATA 3.0",
                form_factor="2.5\"",
                cost_per_tb=100.0
            )
            self.storage_devices.append(device)
    
    def add_hdd_storage(self, count: int, capacity_tb_each: float, model: str = "Enterprise_HDD"):
        """Add HDD storage devices"""
        for i in range(count):
            device = StorageDevice(
                model=f"{model}_{i+1}",
                type=StorageType.HDD,
                capacity_tb=capacity_tb_each,
                sequential_read_mbps=250,
                sequential_write_mbps=250,
                random_read_iops=200,
                random_write_iops=200,
                power_consumption_w=12.0,
                mtbf_hours=2000000,
                interface="SATA 3.0",
                form_factor="3.5\"",
                cost_per_tb=25.0
            )
            self.storage_devices.append(device)
    
    def create_storage_array(self, name: str, device_type: StorageType, device_count: int, 
                           raid_level: RAIDLevel, hot_spares: int = 2):
        """Create storage array from devices"""
        # Find devices of specified type
        devices = [d for d in self.storage_devices if d.type == device_type]
        
        if len(devices) < device_count:
            raise ValueError(f"Not enough {device_type.value} devices available")
        
        # Calculate RAID capacity and performance
        device = devices[0]  # Use first device as reference
        
        # RAID capacity calculations
        if raid_level == RAIDLevel.RAID0:
            usable_capacity = device_count * device.capacity_tb
            performance_multiplier = device_count
        elif raid_level == RAIDLevel.RAID1:
            usable_capacity = (device_count / 2) * device.capacity_tb
            performance_multiplier = device_count / 2
        elif raid_level == RAIDLevel.RAID5:
            usable_capacity = (device_count - 1) * device.capacity_tb
            performance_multiplier = device_count - 1
        elif raid_level == RAIDLevel.RAID6:
            usable_capacity = (device_count - 2) * device.capacity_tb
            performance_multiplier = device_count - 2
        elif raid_level == RAIDLevel.RAID10:
            usable_capacity = (device_count / 2) * device.capacity_tb
            performance_multiplier = device_count
        else:
            usable_capacity = device_count * device.capacity_tb * 0.8  # Conservative estimate
            performance_multiplier = device_count * 0.8
        
        array = StorageArray(
            name=name,
            device_type=device_type,
            device_count=device_count,
            raid_level=raid_level,
            hot_spares=hot_spares,
            controller_model="Enterprise_RAID_Controller",
            cache_size_gb=32,
            total_capacity_tb=device_count * device.capacity_tb,
            usable_capacity_tb=usable_capacity,
            performance_iops=int(device.random_read_iops * performance_multiplier * 0.7),
            power_consumption_w=device_count * device.power_consumption_w + 200  # Controller power
        )
        
        self.storage_arrays.append(array)
        return array
    
    def create_file_system(self, name: str, fs_type: str, mount_point: str, 
                          capacity_tb: float, enable_features: Dict[str, bool] = None):
        """Create file system"""
        if enable_features is None:
            enable_features = {
                'compression': True,
                'deduplication': False,
                'encryption': True,
                'backup': True
            }
        
        fs = FileSystem(
            name=name,
            type=fs_type,
            mount_point=mount_point,
            capacity_tb=capacity_tb,
            block_size_kb=64,
            compression_enabled=enable_features.get('compression', False),
            deduplication_enabled=enable_features.get('deduplication', False),
            encryption_enabled=enable_features.get('encryption', False),
            backup_enabled=enable_features.get('backup', False)
        )
        
        self.file_systems.append(fs)
        return fs
    
    def calculate_total_capacity(self) -> Dict[str, float]:
        """Calculate total storage capacity"""
        capacity_by_type = {}
        total_raw_capacity = 0
        total_usable_capacity = 0
        
        for array in self.storage_arrays:
            storage_type = array.device_type.value
            if storage_type not in capacity_by_type:
                capacity_by_type[storage_type] = {
                    'raw_capacity_tb': 0,
                    'usable_capacity_tb': 0,
                    'array_count': 0
                }
            
            capacity_by_type[storage_type]['raw_capacity_tb'] += array.total_capacity_tb
            capacity_by_type[storage_type]['usable_capacity_tb'] += array.usable_capacity_tb
            capacity_by_type[storage_type]['array_count'] += 1
            
            total_raw_capacity += array.total_capacity_tb
            total_usable_capacity += array.usable_capacity_tb
        
        return {
            'total_raw_capacity_tb': total_raw_capacity,
            'total_raw_capacity_pb': total_raw_capacity / 1024,
            'total_usable_capacity_tb': total_usable_capacity,
            'total_usable_capacity_pb': total_usable_capacity / 1024,
            'capacity_efficiency': total_usable_capacity / total_raw_capacity if total_raw_capacity > 0 else 0,
            'breakdown_by_type': capacity_by_type
        }
    
    def calculate_performance_metrics(self) -> Dict[str, float]:
        """Calculate storage performance metrics"""
        total_iops = sum(array.performance_iops for array in self.storage_arrays)
        total_bandwidth_mbps = 0
        
        # Estimate bandwidth based on device types
        for array in self.storage_arrays:
            if array.device_type == StorageType.NVME_SSD:
                array_bandwidth = array.device_count * 6500  # Average of read/write
            elif array.device_type == StorageType.SATA_SSD:
                array_bandwidth = array.device_count * 535
            elif array.device_type == StorageType.HDD:
                array_bandwidth = array.device_count * 250
            else:
                array_bandwidth = array.device_count * 1000  # Conservative estimate
            
            total_bandwidth_mbps += array_bandwidth
        
        return {
            'total_iops': total_iops,
            'total_bandwidth_mbps': total_bandwidth_mbps,
            'total_bandwidth_gbps': total_bandwidth_mbps / 1000,
            'arrays_count': len(self.storage_arrays)
        }
    
    def calculate_power_consumption(self) -> Dict[str, float]:
        """Calculate storage power consumption"""
        total_power_w = sum(array.power_consumption_w for array in self.storage_arrays)
        
        power_by_type = {}
        for array in self.storage_arrays:
            storage_type = array.device_type.value
            if storage_type not in power_by_type:
                power_by_type[storage_type] = 0
            power_by_type[storage_type] += array.power_consumption_w
        
        return {
            'total_power_w': total_power_w,
            'total_power_kw': total_power_w / 1000,
            'total_power_mw': total_power_w / 1e6,
            'breakdown_by_type': power_by_type
        }
    
    def calculate_storage_costs(self) -> Dict[str, float]:
        """Calculate storage costs"""
        total_cost = 0
        cost_by_type = {}
        
        for device in self.storage_devices:
            device_cost = device.capacity_tb * device.cost_per_tb
            total_cost += device_cost
            
            storage_type = device.type.value
            if storage_type not in cost_by_type:
                cost_by_type[storage_type] = 0
            cost_by_type[storage_type] += device_cost
        
        # Add infrastructure costs (controllers, cables, racks)
        infrastructure_cost = total_cost * 0.3  # 30% of device cost
        total_cost += infrastructure_cost
        
        total_usable_capacity = self.calculate_total_capacity()['total_usable_capacity_tb']
        cost_per_tb = total_cost / total_usable_capacity if total_usable_capacity > 0 else 0.0
        
        return {
            'device_costs': cost_by_type,
            'infrastructure_cost': infrastructure_cost,
            'total_cost': total_cost,
            'cost_per_tb_usable': cost_per_tb
        }
    
    def optimize_storage_tiers(self, hot_ratio: float = 0.1, warm_ratio: float = 0.2, 
                              cold_ratio: float = 0.4, archive_ratio: float = 0.3) -> Dict[str, any]:
        """Optimize storage across different tiers"""
        total_capacity = self.calculate_total_capacity()['total_usable_capacity_tb']
        
        tier_allocation = {
            'hot_tier': {
                'capacity_tb': total_capacity * hot_ratio,
                'storage_type': 'nvme_ssd',
                'performance_tier': 'highest',
                'cost_multiplier': 8.0
            },
            'warm_tier': {
                'capacity_tb': total_capacity * warm_ratio,
                'storage_type': 'sata_ssd',
                'performance_tier': 'high',
                'cost_multiplier': 4.0
            },
            'cold_tier': {
                'capacity_tb': total_capacity * cold_ratio,
                'storage_type': 'hdd',
                'performance_tier': 'medium',
                'cost_multiplier': 1.0
            },
            'archive_tier': {
                'capacity_tb': total_capacity * archive_ratio,
                'storage_type': 'tape',
                'performance_tier': 'low',
                'cost_multiplier': 0.1
            }
        }
        
        # Calculate cost optimization
        total_optimized_cost = 0
        for tier, config in tier_allocation.items():
            tier_cost = config['capacity_tb'] * 25 * config['cost_multiplier']  # Base $25/TB
            total_optimized_cost += tier_cost
            config['estimated_cost'] = tier_cost
        
        cost_per_tb_avg = 0.0
        if total_capacity > 0:
            cost_per_tb_avg = total_optimized_cost / total_capacity
            
        return {
            'tier_allocation': tier_allocation,
            'total_optimized_cost': total_optimized_cost,
            'cost_per_tb_average': cost_per_tb_avg,
            'performance_optimization': 'Balanced across tiers'
        }
    
    def calculate_backup_requirements(self, backup_ratio: float = 3.0) -> Dict[str, any]:
        """Calculate backup storage requirements"""
        primary_capacity = self.calculate_total_capacity()['total_usable_capacity_tb']
        backup_capacity = primary_capacity * backup_ratio
        
        # Backup strategy
        backup_strategy = {
            'local_backup': {
                'capacity_tb': backup_capacity * 0.4,
                'retention_days': 30,
                'storage_type': 'hdd',
                'rpo_hours': 4,
                'rto_hours': 8
            },
            'offsite_backup': {
                'capacity_tb': backup_capacity * 0.3,
                'retention_days': 365,
                'storage_type': 'tape',
                'rpo_hours': 24,
                'rto_hours': 72
            },
            'cloud_backup': {
                'capacity_tb': backup_capacity * 0.3,
                'retention_days': 2555,  # 7 years
                'storage_type': 'object_storage',
                'rpo_hours': 12,
                'rto_hours': 24
            }
        }
        
        return {
            'primary_capacity_tb': primary_capacity,
            'total_backup_capacity_tb': backup_capacity,
            'backup_ratio': backup_ratio,
            'backup_strategy': backup_strategy
        }
    
    def generate_storage_report(self) -> Dict[str, any]:
        """Generate comprehensive storage system report"""
        return {
            'system_overview': {
                'name': self.name,
                'storage_devices': len(self.storage_devices),
                'storage_arrays': len(self.storage_arrays),
                'file_systems': len(self.file_systems)
            },
            'capacity_analysis': self.calculate_total_capacity(),
            'performance_metrics': self.calculate_performance_metrics(),
            'power_consumption': self.calculate_power_consumption(),
            'cost_analysis': self.calculate_storage_costs(),
            'tier_optimization': self.optimize_storage_tiers(),
            'backup_requirements': self.calculate_backup_requirements()
        }