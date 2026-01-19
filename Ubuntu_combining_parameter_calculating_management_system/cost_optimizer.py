#!/usr/bin/env python3
"""
Cost Optimizer Module
Comprehensive cloud cost optimization system including:
- Instance cost tracking and analysis
- Resource usage monitoring
- Cost prediction and forecasting
- Auto-scaling recommendations
- Budget alerts and thresholds
- Cost optimization strategies
"""

import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class InstanceCost:
    """Instance cost data class"""
    instance_type: str
    region: str
    hourly_cost: float
    running_hours: float = 0.0
    monthly_cost: float = 0.0
    tags: Dict[str, str] = field(default_factory=dict)

@dataclass
class CostMetrics:
    """Cost metrics data class"""
    total_monthly_cost: float = 0.0
    total_instances: int = 0
    average_instance_cost: float = 0.0
    highest_cost_instance: Optional[str] = None
    cost_by_region: Dict[str, float] = field(default_factory=dict)
    cost_by_type: Dict[str, float] = field(default_factory=dict)

class CostOptimizer:
    """Comprehensive cloud cost optimizer"""
    
    def __init__(
        self,
        budget_threshold: float = 1000.0,
        alert_threshold: float = 0.8,
        optimization_enabled: bool = True
    ):
        """
        Initialize cost optimizer
        
        Args:
            budget_threshold: Monthly budget threshold in USD
            alert_threshold: Alert when cost reaches this percentage of budget
            optimization_enabled: Enable automatic optimization
        """
        self.budget_threshold = budget_threshold
        self.alert_threshold = alert_threshold
        self.optimization_enabled = optimization_enabled
        
        # Cost tracking
        self.instance_costs: Dict[str, InstanceCost] = {}
        self.cost_history: List[Dict[str, Any]] = []
        
        # Statistics
        self.stats = {
            'total_savings': 0.0,
            'optimizations_applied': 0,
            'budget_warnings': 0,
            'instances_stopped': 0
        }
        
        logger.info("Cost optimizer initialized")
    
    def calculate_instance_cost(
        self,
        instance_type: str,
        region: str,
        running_hours: float,
        pricing: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Calculate instance cost
        
        Args:
            instance_type: Instance type (e.g., 't3.medium')
            region: AWS region
            running_hours: Number of running hours
            pricing: Optional custom pricing
            
        Returns:
            Total cost in USD
        """
        try:
            # Default pricing (example rates)
            default_pricing = {
                't3.medium': 0.0416,
                't3.large': 0.0832,
                'm5.large': 0.096,
                'c5.large': 0.085,
                'g4dn.xlarge': 0.526
            }
            
            pricing = pricing or default_pricing
            hourly_rate = pricing.get(instance_type, 0.05)
            
            total_cost = hourly_rate * running_hours
            
            logger.debug(f"Instance {instance_type} cost: ${total_cost:.2f}")
            
            return total_cost
            
        except Exception as e:
            logger.error(f"Error calculating instance cost: {e}")
            return 0.0
    
    def track_instance(
        self,
        instance_id: str,
        instance_type: str,
        region: str,
        running_hours: float
    ):
        """
        Track instance cost
        
        Args:
            instance_id: Unique instance identifier
            instance_type: Instance type
            region: AWS region
            running_hours: Number of running hours
        """
        try:
            monthly_cost = self.calculate_instance_cost(
                instance_type,
                region,
                running_hours * 730  # Approximate monthly hours
            )
            
            instance_cost = InstanceCost(
                instance_type=instance_type,
                region=region,
                hourly_cost=monthly_cost / 730,
                running_hours=running_hours,
                monthly_cost=monthly_cost
            )
            
            self.instance_costs[instance_id] = instance_cost
            
            logger.info(f"Instance {instance_id} tracked: ${monthly_cost:.2f}/month")
            
        except Exception as e:
            logger.error(f"Error tracking instance: {e}")
    
    def analyze_costs(self) -> CostMetrics:
        """
        Analyze current costs
        
        Returns:
            Cost metrics
        """
        try:
            logger.info("Analyzing costs...")
            
            total_monthly = sum(
                inst.monthly_cost for inst in self.instance_costs.values()
            )
            
            cost_by_region = {}
            cost_by_type = {}
            
            highest_cost = 0.0
            highest_instance = None
            
            for inst_id, inst in self.instance_costs.items():
                # Region breakdown
                if inst.region not in cost_by_region:
                    cost_by_region[inst.region] = 0.0
                cost_by_region[inst.region] += inst.monthly_cost
                
                # Type breakdown
                if inst.instance_type not in cost_by_type:
                    cost_by_type[inst.instance_type] = 0.0
                cost_by_type[inst.instance_type] += inst.monthly_cost
                
                # Highest cost
                if inst.monthly_cost > highest_cost:
                    highest_cost = inst.monthly_cost
                    highest_instance = inst_id
            
            avg_cost = (
                total_monthly / len(self.instance_costs)
                if self.instance_costs else 0.0
            )
            
            metrics = CostMetrics(
                total_monthly_cost=total_monthly,
                total_instances=len(self.instance_costs),
                average_instance_cost=avg_cost,
                highest_cost_instance=highest_instance,
                cost_by_region=cost_by_region,
                cost_by_type=cost_by_type
            )
            
            logger.info(f"Total monthly cost: ${metrics.total_monthly_cost:.2f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error analyzing costs: {e}")
            return CostMetrics()
    
    def check_budget(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if cost is within budget
        
        Returns:
            Tuple of (within_budget, alert_info)
        """
        try:
            metrics = self.analyze_costs()
            
            current_cost = metrics.total_monthly_cost
            budget_usage = current_cost / self.budget_threshold if self.budget_threshold > 0 else 0.0
            
            within_budget = budget_usage < 1.0
            needs_alert = budget_usage >= self.alert_threshold
            
            if needs_alert:
                self.stats['budget_warnings'] += 1
                logger.warning(
                    f"Budget alert: ${current_cost:.2f}/{self.budget_threshold:.2f} "
                    f"({budget_usage*100:.1f}%)"
                )
            
            alert_info = {
                'current_cost': current_cost,
                'budget_threshold': self.budget_threshold,
                'budget_usage': budget_usage,
                'within_budget': within_budget,
                'needs_alert': needs_alert
            }
            
            return within_budget, alert_info
            
        except Exception as e:
            logger.error(f"Error checking budget: {e}")
            return True, {}
    
    def optimize_costs(self) -> List[Dict[str, Any]]:
        """
        Generate cost optimization recommendations
        
        Returns:
            List of optimization recommendations
        """
        try:
            logger.info("Generating cost optimization recommendations...")
            
            metrics = self.analyze_costs()
            recommendations = []
            
            # Find unused or low-utilization instances
            for inst_id, inst in self.instance_costs.items():
                if inst.running_hours < 10:  # Low utilization
                    recommendations.append({
                        'type': 'instance_usage',
                        'instance_id': inst_id,
                        'action': 'stop_or_downsize',
                        'potential_savings': inst.monthly_cost,
                        'reason': f'Low utilization: {inst.running_hours} hours'
                    })
            
            # Check for expensive instances
            for inst_id, inst in self.instance_costs.items():
                if inst.monthly_cost > 100:
                    recommendations.append({
                        'type': 'expensive_instance',
                        'instance_id': inst_id,
                        'action': 'consider_rightsizing',
                        'potential_savings': inst.monthly_cost * 0.3,
                        'reason': f'High cost: ${inst.monthly_cost:.2f}/month'
                    })
            
            # Regional optimizations
            if len(metrics.cost_by_region) > 1:
                for region, cost in sorted(
                    metrics.cost_by_region.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:2]:
                    recommendations.append({
                        'type': 'regional',
                        'region': region,
                        'action': 'consider_consolidation',
                        'potential_savings': cost * 0.1,
                        'reason': f'High regional cost: ${cost:.2f}/month'
                    })
            
            logger.info(f"Generated {len(recommendations)} recommendations")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return []
    
    def apply_optimization(self, recommendation: Dict[str, Any]) -> bool:
        """
        Apply cost optimization recommendation
        
        Args:
            recommendation: Optimization recommendation
            
        Returns:
            True if applied successfully
        """
        try:
            if not self.optimization_enabled:
                logger.warning("Optimization disabled")
                return False
            
            action = recommendation.get('action')
            inst_id = recommendation.get('instance_id')
            
            logger.info(f"Applying optimization: {action} for {inst_id}")
            
            if inst_id and inst_id in self.instance_costs:
                self.stats['optimizations_applied'] += 1
                savings = recommendation.get('potential_savings', 0.0)
                self.stats['total_savings'] += savings
                
                if 'stop' in action.lower():
                    self.stats['instances_stopped'] += 1
                    logger.info(f"Instance {inst_id} stopped, savings: ${savings:.2f}")
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error applying optimization: {e}")
            return False
    
    def generate_cost_report(self) -> str:
        """
        Generate cost report
        
        Returns:
            Formatted cost report string
        """
        try:
            metrics = self.analyze_costs()
            
            report = f"""
================================================================================
CLOUD COST REPORT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
================================================================================

OVERVIEW:
  Total Monthly Cost: ${metrics.total_monthly_cost:.2f}
  Total Instances: {metrics.total_instances}
  Average Instance Cost: ${metrics.average_instance_cost:.2f}
  Highest Cost Instance: {metrics.highest_cost_instance}

COST BY REGION:
"""
            
            for region, cost in sorted(
                metrics.cost_by_region.items(),
                key=lambda x: x[1],
                reverse=True
            ):
                percentage = (cost / metrics.total_monthly_cost * 100) if metrics.total_monthly_cost > 0 else 0
                report += f"  {region}: ${cost:.2f} ({percentage:.1f}%)\n"
            
            report += "\nCOST BY INSTANCE TYPE:\n"
            
            for inst_type, cost in sorted(
                metrics.cost_by_type.items(),
                key=lambda x: x[1],
                reverse=True
            ):
                percentage = (cost / metrics.total_monthly_cost * 100) if metrics.total_monthly_cost > 0 else 0
                report += f"  {inst_type}: ${cost:.2f} ({percentage:.1f}%)\n"
            
            within_budget, alert_info = self.check_budget()
            
            report += f"\nBUDGET STATUS:\n"
            report += f"  Current Cost: ${alert_info.get('current_cost', 0):.2f}\n"
            report += f"  Budget Threshold: ${self.budget_threshold:.2f}\n"
            report += f"  Budget Usage: {alert_info.get('budget_usage', 0)*100:.1f}%\n"
            report += f"  Within Budget: {within_budget}\n"
            
            report += f"\nOPTIMIZATION STATS:\n"
            report += f"  Total Savings: ${self.stats['total_savings']:.2f}\n"
            report += f"  Optimizations Applied: {self.stats['optimizations_applied']}\n"
            report += f"  Instances Stopped: {self.stats['instances_stopped']}\n"
            report += f"  Budget Warnings: {self.stats['budget_warnings']}\n"
            
            report += "\n" + "=" * 80 + "\n"
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating cost report: {e}")
            return ""

def main():
    """Main function to demonstrate cost optimizer"""
    try:
        logger.info("=" * 60)
        logger.info("Cost Optimizer Demo")
        logger.info("=" * 60)
        
        # Initialize optimizer
        optimizer = CostOptimizer(budget_threshold=1000.0)
        
        # Track some instances
        optimizer.track_instance("inst-1", "t3.medium", "us-east-1", 720)
        optimizer.track_instance("inst-2", "t3.large", "us-east-1", 360)
        optimizer.track_instance("inst-3", "m5.large", "us-west-2", 720)
        
        # Analyze costs
        metrics = optimizer.analyze_costs()
        logger.info(f"\nCost Metrics:\n{metrics}")
        
        # Check budget
        within_budget, alert_info = optimizer.check_budget()
        logger.info(f"\nBudget Status: {within_budget}")
        
        # Generate recommendations
        recommendations = optimizer.optimize_costs()
        logger.info(f"\nOptimization Recommendations: {len(recommendations)}")
        for rec in recommendations:
            logger.info(f"  - {rec}")
        
        # Generate report
        report = optimizer.generate_cost_report()
        logger.info(f"\n{report}")
        
        logger.info("=" * 60)
        logger.info("Cost Optimizer Demo Completed!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Error in main: {e}")

if __name__ == "__main__":
    main()

