import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum
import math

class RevenueStream(Enum):
    AI_TRAINING = "ai_training"
    AI_INFERENCE = "ai_inference"
    CLOUD_COMPUTE = "cloud_compute"
    DATA_STORAGE = "data_storage"
    COLOCATION = "colocation"
    MANAGED_SERVICES = "managed_services"
    RESEARCH_PARTNERSHIPS = "research_partnerships"

class CostCategory(Enum):
    CAPEX_HARDWARE = "capex_hardware"
    CAPEX_INFRASTRUCTURE = "capex_infrastructure"
    OPEX_POWER = "opex_power"
    OPEX_COOLING = "opex_cooling"
    OPEX_PERSONNEL = "opex_personnel"
    OPEX_MAINTENANCE = "opex_maintenance"
    OPEX_INSURANCE = "opex_insurance"
    OPEX_CONNECTIVITY = "opex_connectivity"

class DepreciationMethod(Enum):
    STRAIGHT_LINE = "straight_line"
    ACCELERATED = "accelerated"
    DOUBLE_DECLINING = "double_declining"

@dataclass
class RevenueModel:
    """Revenue stream specification"""
    stream_type: RevenueStream
    unit_price: float  # Price per unit (hour, GB, etc.)
    units_per_year: float
    growth_rate: float  # Annual growth rate
    margin_percent: float  # Profit margin
    contract_length_years: int
    pricing_model: str  # "usage", "subscription", "reserved"

@dataclass
class CostModel:
    """Cost component specification"""
    category: CostCategory
    initial_cost: float
    annual_cost: float
    escalation_rate: float  # Annual cost increase
    depreciation_years: int
    depreciation_method: DepreciationMethod
    tax_deductible: bool

@dataclass
class FinancialAssumptions:
    """Financial modeling assumptions"""
    discount_rate: float  # WACC or required return
    tax_rate: float
    inflation_rate: float
    electricity_cost_per_kwh: float
    land_cost_per_sqm: float
    construction_cost_per_sqm: float
    financing_rate: float
    debt_to_equity_ratio: float

class TaxOptimization:
    """Tax optimization and incentive management"""
    
    def __init__(self):
        self.incentives = {}
        self.deductions = {}
        self.credits = {}
    
    def add_tax_incentive(self, name: str, incentive_type: str, 
                         amount: float, duration_years: int):
        """Add tax incentive (credit, deduction, exemption)"""
        self.incentives[name] = {
            'type': incentive_type,
            'amount': amount,
            'duration_years': duration_years,
            'annual_benefit': amount / duration_years if duration_years > 0 else amount
        }
    
    def calculate_depreciation(self, asset_cost: float, useful_life: int, 
                             method: DepreciationMethod) -> List[float]:
        """Calculate depreciation schedule"""
        if method == DepreciationMethod.STRAIGHT_LINE:
            annual_depreciation = asset_cost / useful_life
            return [annual_depreciation] * useful_life
        
        elif method == DepreciationMethod.DOUBLE_DECLINING:
            depreciation_schedule = []
            remaining_value = asset_cost
            rate = 2 / useful_life
            
            for year in range(useful_life):
                if year == useful_life - 1:  # Last year
                    depreciation = remaining_value
                else:
                    depreciation = remaining_value * rate
                    depreciation = min(depreciation, remaining_value)
                
                depreciation_schedule.append(depreciation)
                remaining_value -= depreciation
                
                if remaining_value <= 0:
                    break
            
            return depreciation_schedule
        
        elif method == DepreciationMethod.ACCELERATED:
            # MACRS 7-year property (typical for data center equipment)
            macrs_rates = [0.1429, 0.2449, 0.1749, 0.1249, 0.0893, 0.0892, 0.0893, 0.0446]
            return [asset_cost * rate for rate in macrs_rates[:useful_life]]
        
        else:
            return [asset_cost / useful_life] * useful_life
    
    def calculate_total_tax_benefits(self, years: int) -> Dict[str, float]:
        """Calculate total tax benefits over specified period"""
        total_benefits = 0
        benefits_by_type = {}
        
        for name, incentive in self.incentives.items():
            applicable_years = min(years, incentive['duration_years'])
            benefit = incentive['annual_benefit'] * applicable_years
            total_benefits += benefit
            
            incentive_type = incentive['type']
            if incentive_type not in benefits_by_type:
                benefits_by_type[incentive_type] = 0
            benefits_by_type[incentive_type] += benefit
        
        return {
            'total_benefits': total_benefits,
            'benefits_by_type': benefits_by_type,
            'annual_average': total_benefits / years if years > 0 else 0
        }

class FinancialModel:
    """Comprehensive financial modeling for AI data center"""
    
    def __init__(self, name: str, assumptions: FinancialAssumptions):
        self.name = name
        self.assumptions = assumptions
        self.revenue_streams: List[RevenueModel] = []
        self.cost_models: List[CostModel] = []
        self.tax_optimization = TaxOptimization()
        
    def add_ai_training_revenue(self, gpu_hours_per_year: float, price_per_gpu_hour: float):
        """Add AI training revenue stream"""
        revenue = RevenueModel(
            stream_type=RevenueStream.AI_TRAINING,
            unit_price=price_per_gpu_hour,
            units_per_year=gpu_hours_per_year,
            growth_rate=0.25,  # 25% annual growth
            margin_percent=60,  # High margin for specialized AI training
            contract_length_years=2,
            pricing_model="usage"
        )
        self.revenue_streams.append(revenue)
    
    def add_ai_inference_revenue(self, inference_requests_per_year: float, price_per_request: float):
        """Add AI inference revenue stream"""
        revenue = RevenueModel(
            stream_type=RevenueStream.AI_INFERENCE,
            unit_price=price_per_request,
            units_per_year=inference_requests_per_year,
            growth_rate=0.40,  # 40% annual growth for inference
            margin_percent=70,  # Higher margin for inference
            contract_length_years=1,
            pricing_model="usage"
        )
        self.revenue_streams.append(revenue)
    
    def add_cloud_compute_revenue(self, compute_hours_per_year: float, price_per_hour: float):
        """Add cloud compute revenue stream"""
        revenue = RevenueModel(
            stream_type=RevenueStream.CLOUD_COMPUTE,
            unit_price=price_per_hour,
            units_per_year=compute_hours_per_year,
            growth_rate=0.15,  # 15% annual growth
            margin_percent=40,  # Standard cloud margins
            contract_length_years=1,
            pricing_model="subscription"
        )
        self.revenue_streams.append(revenue)
    
    def add_data_storage_revenue(self, storage_tb_per_year: float, price_per_tb_month: float):
        """Add data storage revenue stream"""
        revenue = RevenueModel(
            stream_type=RevenueStream.DATA_STORAGE,
            unit_price=price_per_tb_month,
            units_per_year=storage_tb_per_year * 12,  # Convert to monthly billing
            growth_rate=0.20,  # 20% annual growth
            margin_percent=50,
            contract_length_years=3,
            pricing_model="subscription"
        )
        self.revenue_streams.append(revenue)
    
    def add_hardware_capex(self, gpu_cost: float, server_cost: float, 
                          network_cost: float, storage_cost: float):
        """Add hardware capital expenditure"""
        total_hardware_cost = gpu_cost + server_cost + network_cost + storage_cost
        
        capex = CostModel(
            category=CostCategory.CAPEX_HARDWARE,
            initial_cost=total_hardware_cost,
            annual_cost=0,
            escalation_rate=0.0,
            depreciation_years=5,  # Typical for IT equipment
            depreciation_method=DepreciationMethod.ACCELERATED,
            tax_deductible=True
        )
        self.cost_models.append(capex)
    
    def add_infrastructure_capex(self, facility_cost: float, power_cost: float, cooling_cost: float):
        """Add infrastructure capital expenditure"""
        total_infrastructure_cost = facility_cost + power_cost + cooling_cost
        
        capex = CostModel(
            category=CostCategory.CAPEX_INFRASTRUCTURE,
            initial_cost=total_infrastructure_cost,
            annual_cost=0,
            escalation_rate=0.0,
            depreciation_years=20,  # Longer for infrastructure
            depreciation_method=DepreciationMethod.STRAIGHT_LINE,
            tax_deductible=True
        )
        self.cost_models.append(capex)
    
    def add_power_opex(self, annual_power_consumption_mwh: float):
        """Add power operational expenditure"""
        annual_power_cost = annual_power_consumption_mwh * 1000 * self.assumptions.electricity_cost_per_kwh
        
        opex = CostModel(
            category=CostCategory.OPEX_POWER,
            initial_cost=0,
            annual_cost=annual_power_cost,
            escalation_rate=0.03,  # 3% annual increase
            depreciation_years=0,
            depreciation_method=DepreciationMethod.STRAIGHT_LINE,
            tax_deductible=True
        )
        self.cost_models.append(opex)
    
    def add_personnel_opex(self, staff_count: int, average_salary: float):
        """Add personnel operational expenditure"""
        annual_personnel_cost = staff_count * average_salary * 1.4  # Include benefits
        
        opex = CostModel(
            category=CostCategory.OPEX_PERSONNEL,
            initial_cost=0,
            annual_cost=annual_personnel_cost,
            escalation_rate=0.04,  # 4% annual salary increases
            depreciation_years=0,
            depreciation_method=DepreciationMethod.STRAIGHT_LINE,
            tax_deductible=True
        )
        self.cost_models.append(opex)
    
    def add_maintenance_opex(self, hardware_value: float, maintenance_rate: float = 0.08):
        """Add maintenance operational expenditure"""
        annual_maintenance_cost = hardware_value * maintenance_rate
        
        opex = CostModel(
            category=CostCategory.OPEX_MAINTENANCE,
            initial_cost=0,
            annual_cost=annual_maintenance_cost,
            escalation_rate=0.05,  # 5% annual increase
            depreciation_years=0,
            depreciation_method=DepreciationMethod.STRAIGHT_LINE,
            tax_deductible=True
        )
        self.cost_models.append(opex)
    
    def calculate_revenue_projection(self, years: int) -> Dict[str, any]:
        """Calculate revenue projections over specified years"""
        revenue_by_year = []
        revenue_by_stream = {}
        
        for year in range(years):
            year_revenue = 0
            year_breakdown = {}
            
            for stream in self.revenue_streams:
                # Calculate revenue with growth
                base_revenue = stream.unit_price * stream.units_per_year
                grown_revenue = base_revenue * ((1 + stream.growth_rate) ** year)
                year_revenue += grown_revenue
                
                stream_name = stream.stream_type.value
                year_breakdown[stream_name] = grown_revenue
                
                if stream_name not in revenue_by_stream:
                    revenue_by_stream[stream_name] = []
                revenue_by_stream[stream_name].append(grown_revenue)
            
            revenue_by_year.append(year_revenue)
        
        total_revenue = sum(revenue_by_year)
        
        return {
            'revenue_by_year': revenue_by_year,
            'revenue_by_stream': revenue_by_stream,
            'total_revenue': total_revenue,
            'average_annual_revenue': total_revenue / years if years > 0 else 0,
            'cagr': ((revenue_by_year[-1] / revenue_by_year[0]) ** (1/years) - 1) if years > 1 and revenue_by_year[0] > 0 else 0
        }
    
    def calculate_cost_projection(self, years: int) -> Dict[str, any]:
        """Calculate cost projections over specified years"""
        costs_by_year = []
        costs_by_category = {}
        depreciation_by_year = []
        
        for year in range(years):
            year_costs = 0
            year_depreciation = 0
            year_breakdown = {}
            
            for cost in self.cost_models:
                # Calculate annual operational costs with escalation
                if cost.annual_cost > 0:
                    escalated_cost = cost.annual_cost * ((1 + cost.escalation_rate) ** year)
                    year_costs += escalated_cost
                
                # Calculate depreciation for capital expenditures
                if cost.initial_cost > 0 and year < cost.depreciation_years:
                    depreciation_schedule = self.tax_optimization.calculate_depreciation(
                        cost.initial_cost, cost.depreciation_years, cost.depreciation_method
                    )
                    if year < len(depreciation_schedule):
                        year_depreciation += depreciation_schedule[year]
                
                category_name = cost.category.value
                category_cost = escalated_cost if cost.annual_cost > 0 else 0
                year_breakdown[category_name] = category_cost
                
                if category_name not in costs_by_category:
                    costs_by_category[category_name] = []
                costs_by_category[category_name].append(category_cost)
            
            costs_by_year.append(year_costs)
            depreciation_by_year.append(year_depreciation)
        
        total_costs = sum(costs_by_year)
        total_depreciation = sum(depreciation_by_year)
        
        return {
            'costs_by_year': costs_by_year,
            'costs_by_category': costs_by_category,
            'depreciation_by_year': depreciation_by_year,
            'total_costs': total_costs,
            'total_depreciation': total_depreciation,
            'average_annual_costs': total_costs / years if years > 0 else 0
        }
    
    def calculate_cash_flow(self, years: int) -> Dict[str, any]:
        """Calculate cash flow projections"""
        revenue_proj = self.calculate_revenue_projection(years)
        cost_proj = self.calculate_cost_projection(years)
        tax_benefits = self.tax_optimization.calculate_total_tax_benefits(years)
        
        cash_flows = []
        cumulative_cash_flow = []
        ebitda_by_year = []
        net_income_by_year = []
        
        # Calculate initial capital investment
        initial_capex = sum(cost.initial_cost for cost in self.cost_models)
        
        cumulative = -initial_capex  # Start with negative initial investment
        
        for year in range(years):
            revenue = revenue_proj['revenue_by_year'][year]
            operating_costs = cost_proj['costs_by_year'][year]
            depreciation = cost_proj['depreciation_by_year'][year]
            
            # EBITDA (Earnings Before Interest, Taxes, Depreciation, Amortization)
            ebitda = revenue - operating_costs
            ebitda_by_year.append(ebitda)
            
            # Taxable income
            taxable_income = ebitda - depreciation
            
            # Taxes (with benefits)
            taxes = max(0, taxable_income * self.assumptions.tax_rate)
            annual_tax_benefits = tax_benefits['annual_average']
            net_taxes = max(0, taxes - annual_tax_benefits)
            
            # Net income
            net_income = taxable_income - net_taxes
            net_income_by_year.append(net_income)
            
            # Cash flow (add back depreciation as it's non-cash)
            cash_flow = net_income + depreciation
            cash_flows.append(cash_flow)
            
            cumulative += cash_flow
            cumulative_cash_flow.append(cumulative)
        
        # Calculate payback period
        payback_period = None
        for i, cum_cf in enumerate(cumulative_cash_flow):
            if cum_cf >= 0:
                payback_period = i + 1
                break
        
        return {
            'initial_investment': initial_capex,
            'annual_cash_flows': cash_flows,
            'cumulative_cash_flows': cumulative_cash_flow,
            'ebitda_by_year': ebitda_by_year,
            'net_income_by_year': net_income_by_year,
            'payback_period_years': payback_period,
            'total_cash_flow': sum(cash_flows),
            'average_annual_cash_flow': sum(cash_flows) / years if years > 0 else 0
        }
    
    def calculate_roi_metrics(self, years: int) -> Dict[str, float]:
        """Calculate return on investment metrics"""
        cash_flow = self.calculate_cash_flow(years)
        initial_investment = cash_flow['initial_investment']
        annual_cash_flows = cash_flow['annual_cash_flows']
        
        # Net Present Value (NPV)
        npv = -initial_investment
        for i, cf in enumerate(annual_cash_flows):
            npv += cf / ((1 + self.assumptions.discount_rate) ** (i + 1))
        
        # Internal Rate of Return (IRR) - simplified calculation
        irr = self._calculate_irr([-initial_investment] + annual_cash_flows)
        
        # Return on Investment (ROI)
        total_return = sum(annual_cash_flows)
        roi = (total_return - initial_investment) / initial_investment if initial_investment > 0 else 0
        
        # Return on Assets (ROA)
        average_net_income = sum(cash_flow['net_income_by_year']) / years if years > 0 else 0
        roa = average_net_income / initial_investment if initial_investment > 0 else 0
        
        # EBITDA margin
        total_revenue = self.calculate_revenue_projection(years)['total_revenue']
        total_ebitda = sum(cash_flow['ebitda_by_year'])
        ebitda_margin = total_ebitda / total_revenue if total_revenue > 0 else 0
        
        return {
            'npv': npv,
            'irr': irr,
            'roi_percent': roi * 100,
            'roa_percent': roa * 100,
            'ebitda_margin_percent': ebitda_margin * 100,
            'payback_period_years': cash_flow['payback_period_years']
        }
    
    def _calculate_irr(self, cash_flows: List[float], precision: float = 0.0001) -> float:
        """Calculate Internal Rate of Return using Newton-Raphson method"""
        def npv_func(rate):
            return sum(cf / ((1 + rate) ** i) for i, cf in enumerate(cash_flows))
        
        def npv_derivative(rate):
            return sum(-i * cf / ((1 + rate) ** (i + 1)) for i, cf in enumerate(cash_flows) if i > 0)
        
        # Initial guess
        rate = 0.1
        
        for _ in range(100):  # Maximum iterations
            npv = npv_func(rate)
            if abs(npv) < precision:
                return rate
            
            derivative = npv_derivative(rate)
            if abs(derivative) < precision:
                break
            
            rate = rate - npv / derivative
            
            # Bounds checking
            if rate < -0.99:
                rate = -0.99
            elif rate > 10:
                rate = 10
        
        return rate
    
    def optimize_pricing(self, target_roi: float, years: int) -> Dict[str, any]:
        """Optimize pricing to achieve target ROI"""
        current_roi = self.calculate_roi_metrics(years)['roi_percent'] / 100
        
        if current_roi >= target_roi:
            return {
                'optimization_needed': False,
                'current_roi': current_roi,
                'target_roi': target_roi,
                'message': 'Current pricing already meets target ROI'
            }
        
        # Calculate required revenue increase
        revenue_multiplier = target_roi / current_roi if current_roi > 0 else 2.0
        
        # Suggest pricing adjustments
        pricing_adjustments = {}
        for stream in self.revenue_streams:
            current_price = stream.unit_price
            suggested_price = current_price * revenue_multiplier
            price_increase_percent = (suggested_price - current_price) / current_price * 100
            
            pricing_adjustments[stream.stream_type.value] = {
                'current_price': current_price,
                'suggested_price': suggested_price,
                'increase_percent': price_increase_percent
            }
        
        return {
            'optimization_needed': True,
            'current_roi': current_roi,
            'target_roi': target_roi,
            'revenue_multiplier': revenue_multiplier,
            'pricing_adjustments': pricing_adjustments
        }
    
    def sensitivity_analysis(self, years: int) -> Dict[str, any]:
        """Perform sensitivity analysis on key variables"""
        base_roi = self.calculate_roi_metrics(years)['roi_percent']
        
        # Variables to analyze
        variables = {
            'electricity_cost': [-20, -10, 0, 10, 20],  # Percentage changes
            'ai_training_price': [-20, -10, 0, 10, 20],
            'growth_rate': [-50, -25, 0, 25, 50],
            'discount_rate': [-20, -10, 0, 10, 20]
        }
        
        sensitivity_results = {}
        
        for variable, changes in variables.items():
            results = []
            
            for change_percent in changes:
                # Create temporary model with adjusted variable
                temp_model = self._create_temp_model_with_change(variable, change_percent)
                temp_roi = temp_model.calculate_roi_metrics(years)['roi_percent']
                
                results.append({
                    'change_percent': change_percent,
                    'roi_percent': temp_roi,
                    'roi_change': temp_roi - base_roi
                })
            
            sensitivity_results[variable] = results
        
        return {
            'base_roi_percent': base_roi,
            'sensitivity_analysis': sensitivity_results
        }
    
    def _create_temp_model_with_change(self, variable: str, change_percent: float):
        """Create temporary model with adjusted variable for sensitivity analysis"""
        # This is a simplified implementation
        # In practice, you would create a deep copy and modify the specific variable
        temp_assumptions = self.assumptions
        
        if variable == 'electricity_cost':
            temp_assumptions.electricity_cost_per_kwh *= (1 + change_percent / 100)
        elif variable == 'discount_rate':
            temp_assumptions.discount_rate *= (1 + change_percent / 100)
        
        temp_model = FinancialModel(self.name + "_temp", temp_assumptions)
        temp_model.revenue_streams = self.revenue_streams.copy()
        temp_model.cost_models = self.cost_models.copy()
        
        return temp_model
    
    def generate_financial_report(self, years: int = 10) -> Dict[str, any]:
        """Generate comprehensive financial analysis report"""
        return {
            'model_overview': {
                'name': self.name,
                'analysis_period_years': years,
                'revenue_streams': len(self.revenue_streams),
                'cost_categories': len(self.cost_models)
            },
            'revenue_projections': self.calculate_revenue_projection(years),
            'cost_projections': self.calculate_cost_projection(years),
            'cash_flow_analysis': self.calculate_cash_flow(years),
            'roi_metrics': self.calculate_roi_metrics(years),
            'tax_benefits': self.tax_optimization.calculate_total_tax_benefits(years),
            'sensitivity_analysis': self.sensitivity_analysis(years)
        }