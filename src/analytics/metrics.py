"""Business metrics calculations: LTV, CAC, ROI."""

import pandas as pd
from typing import Dict, Any


def calculate_ltv(customers_df: pd.DataFrame, avg_lifespan_months: int = 24) -> Dict[str, Any]:
    """
    Calculate Customer Lifetime Value.
    
    LTV = Average Order Value × Purchase Frequency × Customer Lifespan
    
    Args:
        customers_df: DataFrame with total_orders, total_spent columns
        avg_lifespan_months: Average customer lifespan in months
        
    Returns:
        Dict with LTV metrics
    """
    total_customers = len(customers_df)
    total_revenue = customers_df['total_spent'].sum()
    total_orders = customers_df['total_orders'].sum()
    
    avg_order_value = total_revenue / total_orders if total_orders > 0 else 0
    avg_orders_per_customer = total_orders / total_customers if total_customers > 0 else 0
    
    # Simple LTV calculation
    ltv = avg_order_value * avg_orders_per_customer * (avg_lifespan_months / 12)
    
    return {
        "ltv": round(ltv, 2),
        "avg_order_value": round(avg_order_value, 2),
        "avg_orders_per_customer": round(avg_orders_per_customer, 2),
        "total_customers": total_customers,
        "total_revenue": round(total_revenue, 2)
    }


def calculate_cac(customers_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate Customer Acquisition Cost.
    
    CAC = Total Acquisition Costs / Number of Customers Acquired
    
    Args:
        customers_df: DataFrame with acquisition_cost column
        
    Returns:
        Dict with CAC metrics by source
    """
    total_cost = customers_df['acquisition_cost'].sum()
    total_customers = len(customers_df)
    
    avg_cac = total_cost / total_customers if total_customers > 0 else 0
    
    # CAC by acquisition source
    cac_by_source = customers_df.groupby('acquisition_source').agg({
        'acquisition_cost': 'sum',
        'customer_id': 'count',
        'total_spent': 'sum'
    }).reset_index()
    
    cac_by_source.columns = ['source', 'total_cost', 'customers', 'revenue']
    cac_by_source['cac'] = cac_by_source['total_cost'] / cac_by_source['customers']
    cac_by_source['roi'] = (cac_by_source['revenue'] - cac_by_source['total_cost']) / cac_by_source['total_cost']
    
    return {
        "avg_cac": round(avg_cac, 2),
        "total_acquisition_cost": round(total_cost, 2),
        "total_customers": total_customers,
        "by_source": cac_by_source.to_dict('records')
    }


def calculate_roi(revenue: float, cost: float) -> Dict[str, Any]:
    """
    Calculate Return on Investment.
    
    ROI = (Revenue - Cost) / Cost × 100%
    
    Args:
        revenue: Total revenue generated
        cost: Total cost incurred
        
    Returns:
        Dict with ROI metrics
    """
    if cost == 0:
        return {"roi_percent": 0, "profit": revenue, "revenue": revenue, "cost": cost}
    
    profit = revenue - cost
    roi_percent = (profit / cost) * 100
    
    return {
        "roi_percent": round(roi_percent, 2),
        "profit": round(profit, 2),
        "revenue": round(revenue, 2),
        "cost": round(cost, 2)
    }


def calculate_conversion_funnel(
    visitors: int,
    signups: int,
    trials: int,
    purchases: int
) -> Dict[str, Any]:
    """
    Calculate conversion funnel metrics.
    
    Args:
        visitors: Total website visitors
        signups: Number of signups
        trials: Number of trial starts
        purchases: Number of purchases
        
    Returns:
        Dict with funnel metrics
    """
    return {
        "stages": [
            {"stage": "Visitors", "count": visitors, "rate": 100.0},
            {"stage": "Signups", "count": signups, "rate": round(signups/visitors*100, 2) if visitors > 0 else 0},
            {"stage": "Trials", "count": trials, "rate": round(trials/visitors*100, 2) if visitors > 0 else 0},
            {"stage": "Purchases", "count": purchases, "rate": round(purchases/visitors*100, 2) if visitors > 0 else 0}
        ],
        "overall_conversion": round(purchases/visitors*100, 2) if visitors > 0 else 0
    }
