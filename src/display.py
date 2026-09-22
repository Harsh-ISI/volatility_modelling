import pandas as pd
from IPython.display import display, HTML

def display_summary_metrics(metrics: dict, title: str = "Forecast Evaluation"):
    """
    Formats and displays walk-forward summary metrics into clean HTML tables.
    """
    # 1. Forecast Accuracy Table
    accuracy_data = {
        "Metric": ["MSE (Mean Squared Error)", "MAE (Mean Absolute Error)", "QLIKE Loss"],
        "Value": [
            f"{metrics['MSE']:.4e}",
            f"{metrics['MAE']:.6f}",
            f"{metrics['QLIKE']:.4f}"
        ]
    }
    acc_df = pd.DataFrame(accuracy_data)

    # 2. Risk & Backtesting Metrics (1% vs 5% Side-by-Side)
    risk_data = {
        "Risk Metric": [
            "VaR Mean",
            "ES Mean",
            "Breach Count",
            "Breach Rate (Target)",
            "Kupiec POF Test (p-value)",
            "Christoffersen Indep Test (p-value)",
            "Christoffersen CondCov Test (p-value)",
            "Avg Exceedance Loss"
        ],
        "1% Tail Level": [
            f"{metrics['VaR_1pct_mean']:.2%}",
            f"{metrics['ES_1pct_mean']:.2%}",
            f"{int(metrics['Breach_Count_1pct'])}",
            f"{metrics['Breach_Rate_1pct']:.2%} (1.00%)",
            f"{metrics['Kupiec_pvalue_1pct']:.4f}",
            f"{metrics['Christoffersen_Indep_pvalue_1pct']:.4f}",
            f"{metrics['Christoffersen_CondCov_pvalue_1pct']:.4f}",
            f"{metrics['Avg_Exceedance_Loss_1pct']:.2%}" if pd.notnull(metrics['Avg_Exceedance_Loss_1pct']) else "N/A"
        ],
        "5% Tail Level": [
            f"{metrics['VaR_5pct_mean']:.2%}",
            f"{metrics['ES_5pct_mean']:.2%}",
            f"{int(metrics['Breach_Count_5pct'])}",
            f"{metrics['Breach_Rate_5pct']:.2%} (5.00%)",
            f"{metrics['Kupiec_pvalue_5pct']:.4f}",
            f"{metrics['Christoffersen_Indep_pvalue_5pct']:.4f}",
            f"{metrics['Christoffersen_CondCov_pvalue_5pct']:.4f}",
            f"{metrics['Avg_Exceedance_Loss_5pct']:.2%}" if pd.notnull(metrics['Avg_Exceedance_Loss_5pct']) else "N/A"
        ]
    }
    risk_df = pd.DataFrame(risk_data)

    # Print clean HTML tables
    display(HTML(f"<h3>{title}</h3>"))
    display(HTML("<b>1. Volatility Forecast Accuracy</b>"))
    display(acc_df.style.hide(axis="index"))
    display(HTML("<br><b>2. Tail Risk & Backtesting Performance</b>"))
    display(risk_df.style.hide(axis="index"))


import pandas as pd
from IPython.display import display, HTML

def display_combined_metrics(metrics_dict: dict[str, dict]):
    """
    Combines multiple model summary metrics into a clean, formatted comparison table.
    """
    df_raw = pd.DataFrame(metrics_dict)

    # 1. Volatility Forecast Accuracy Metrics
    accuracy_keys = ["MSE", "MAE", "QLIKE"]
    df_acc = df_raw.loc[accuracy_keys].copy()
    
    # Format accuracy values
    for col in df_acc.columns:
        df_acc[col] = df_acc[col].apply(
            lambda x: f"{x:.4e}" if "MSE" in df_acc.index else f"{x:.4f}"
        )

    # Custom row label mapping for cleaner readability
    row_labels = {
        "VaR_1pct_mean": "VaR Mean (1%)",
        "ES_1pct_mean": "ES Mean (1%)",
        "Breach_Count_1pct": "Breach Count (1%)",
        "Breach_Rate_1pct": "Breach Rate (1%) [Target: 1%]",
        "Kupiec_pvalue_1pct": "Kupiec POF p-value (1%)",
        "Christoffersen_Indep_pvalue_1pct": "Christoffersen Indep p-value (1%)",
        "Christoffersen_CondCov_pvalue_1pct": "Christoffersen CondCov p-value (1%)",
        "Avg_Exceedance_Loss_1pct": "Avg Exceedance Loss (1%)",
        
        "VaR_5pct_mean": "VaR Mean (5%)",
        "ES_5pct_mean": "ES Mean (5%)",
        "Breach_Count_5pct": "Breach Count (5%)",
        "Breach_Rate_5pct": "Breach Rate (5%) [Target: 5%]",
        "Kupiec_pvalue_5pct": "Kupiec POF p-value (5%)",
        "Christoffersen_Indep_pvalue_5pct": "Christoffersen Indep p-value (5%)",
        "Christoffersen_CondCov_pvalue_5pct": "Christoffersen CondCov p-value (5%)",
        "Avg_Exceedance_Loss_5pct": "Avg Exceedance Loss (5%)",
    }

    # 2. Risk Metrics Dataframe
    risk_keys = list(row_labels.keys())
    df_risk = df_raw.loc[risk_keys].copy()

    # Apply formatting row-by-row
    formatted_rows = []
    for key in risk_keys:
        row = df_risk.loc[key]
        label = row_labels[key]
        
        if "Count" in key:
            formatted_row = row.apply(lambda x: f"{int(x)}" if pd.notnull(x) else "N/A")
        elif "pvalue" in key:
            formatted_row = row.apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "N/A")
        elif "mean" in key or "Rate" in key or "Loss" in key:
            formatted_row = row.apply(lambda x: f"{x:.2%}" if pd.notnull(x) else "N/A")
        else:
            formatted_row = row.apply(lambda x: f"{x:.4f}" if pd.notnull(x) else "N/A")
            
        formatted_row.name = label
        formatted_rows.append(formatted_row)

    df_risk_formatted = pd.DataFrame(formatted_rows)

    # Render HTML Tables
    display(HTML("<h2>Model Evaluation Comparison Summary</h2>"))
    display(HTML("<h3>1. Volatility Forecast Accuracy</h3>"))
    display(df_acc)
    display(HTML("<h3>2. Tail Risk & Backtesting Performance</h3>"))
    display(df_risk_formatted)

    return df_raw