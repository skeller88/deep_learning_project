import json


def numpy_to_json(np_array):
    return json.dumps({
        'data': np_array.tolist(),
        'shape': [dim for dim in np_array.shape],
        'dtype': str(np_array.dtype)
    })


def sklearn_precision_recall_curve_to_dict(prc):
    return {
        'precision': numpy_to_json(prc[0]),
        'recall': numpy_to_json(prc[1]),
        'threshold': numpy_to_json(prc[2])
    }