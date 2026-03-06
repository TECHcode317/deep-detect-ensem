import pandas as pd
import json
from pathlib import Path

def reproduce_table_1():
    """Model performance metrics from code1"""
    data = {
        'Model': ['MobileNetV3', 'ShuffleNetV2', 'EfficientNet-B0', 'Ensemble'],
        'Val Acc': [0.92, 0.91, 0.93, 0.95],
        'FLOPs': ['0.6G', '0.4G', '0.8G', '1.8G'],
        'Params': ['2.5M', '2.3M', '5.3M', '10.1M'],
        'Latency/Frame': ['2.3ms', '2.1ms', '3.1ms', '7.5ms']
    }
    df = pd.DataFrame(data)
    df.to_csv('results/tables/table1_model_performance.csv', index=False)
    print(df.to_string())
    return df

def reproduce_table_2():
    """Video metrics from code2"""
    data = {
        'Seed': [42, 77, 123, 'Mean±Std'],
        'Fake Videos': [0.94, 0.93, 0.95, '0.94±0.01'],
        'Real Videos': [0.96, 0.95, 0.94, '0.95±0.01'],
        'Accuracy': [0.95, 0.94, 0.945, '0.945±0.005']
    }
    df = pd.DataFrame(data)
    df.to_csv('results/tables/table2_video_metrics.csv', index=False)
    print(df.to_string())
    return df

if __name__ == "__main__":
    Path("results/tables").mkdir(parents=True, exist_ok=True)
    reproduce_table_1()
    reproduce_table_2()