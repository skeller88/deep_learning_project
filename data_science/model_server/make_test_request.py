import json
from collections import defaultdict

import numpy as np
import pandas as pd
import requests

if __name__ == "__main__":
    arr = np.load('./S2B_MSIL2A_20180421T100031_7_89.npy')
    stats = defaultdict(list)
    for x in range(10):
        req = json.dumps({'image': arr.tolist()})
        response = requests.post(url='http://localhost:8889/classify', json=req).json()
        stats['inference_time'].append(response['inference_time'])
        stats['processing_time'].append(response['processing_time'])
        stats['request_time'].append(response['request_time'])

    df = pd.DataFrame(stats)
    df['inference_pct_of_request'] = df['inference_time'] / df['request_time'] * 100
    df['processing_pct_of_request'] = df['processing_time'] / df['request_time'] * 100
    df.to_csv('model_server_profiling.csv', index=False)
