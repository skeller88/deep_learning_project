import glob
import json
import os
import time
from hashlib import sha256

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np

from data_engineering.data_aggregator.parallelize import parallelize_task


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def metadata_files_from_json_to_csv(logger, num_workers, cloud_and_snow_csv_dir, json_dir, csv_files_path):
    if not os.path.exists(csv_files_path):
        os.mkdir(csv_files_path)

    # From BigEarth team: we used the same labels of the CORINE Land Cover​ program operated by the European Environment
    # Agency. You can check the label names from
    # https://land.copernicus.eu/user-corner/technical-library/corine-land-cover-nomenclature-guidelines/html/.
    replacements = {
        'Bare rocks': 'Bare rock',
        'Natural grasslands': 'Natural grassland',
        'Peat bogs': 'Peatbogs',
        'Transitional woodland-shrub': 'Transitional woodland/shrub'
    }

    def multi_replace(arr):
        return [replacements[el] if replacements.get(el) is not None else el for el in arr]

    def read_and_augment_metadata(json_metadata_file, mlb):
        with open(json_metadata_file) as fileobj:
            obj = json.load(fileobj)
            obj['labels'] = multi_replace(obj['labels'])
            obj['labels_sha256_hexdigest'] = sha256('-'.join(obj['labels']).encode('utf-8')).hexdigest()
            obj['binarized_labels'] = json.dumps(mlb.transform([obj['labels']]), cls=NumpyEncoder)
            obj['image_prefix'] = json_metadata_file.rsplit('/')[-2]
            return obj

    def json_metadata_from_files(json_metadata_files, mlb):
        return [read_and_augment_metadata(json_metadata_file, mlb) for json_metadata_file in json_metadata_files]

    start = time.time()
    paths = os.listdir(json_dir)
    paths = [f"{json_dir}/{path}/{path}_labels_metadata.json" for path in paths]
    logger.info(f"Fetched {len(paths)} paths. in {time.time() - start} seconds.")
    start = time.time()

    # 44 level 3 classes:
    # Currently using:
    # https://land.copernicus.eu/user-corner/technical-library/corine-land-cover-nomenclature-guidelines/html/
    classes = ["Continuous urban fabric", "Discontinuous urban fabric", "Industrial or commercial units",
           "Road and rail networks and associated land", "Port areas", "Airports", "Mineral extraction sites",
           "Dump sites",
           "Construction sites", "Green urban areas", "Sport and leisure facilities", "Non-irrigated arable land",
           "Permanently irrigated land", "Rice fields", "Vineyards", "Fruit trees and berry plantations",
           "Olive groves",
           "Pastures", "Annual crops associated with permanent crops", "Complex cultivation patterns",
           "Land principally occupied by agriculture, with significant areas of natural vegetation",
           "Agro-forestry areas",
           "Broad-leaved forest", "Coniferous forest", "Mixed forest", "Natural grassland", "Moors and heathland",
           "Sclerophyllous vegetation", "Transitional woodland/shrub", "Beaches, dunes, sands", "Bare rock",
           "Sparsely vegetated areas", "Burnt areas", "Glaciers and perpetual snow", "Inland marshes", "Peatbogs",
           "Salt marshes", "Salines", "Intertidal flats", "Water courses", "Water bodies", "Coastal lagoons",
           "Estuaries",
           "Sea and ocean"]

    mlb = MultiLabelBinarizer()
    mlb.fit([classes])
    # sanity check the output
    logger.info(f"Sea and ocean: {mlb.transform([['Sea and ocean']])}")

    json_object_lists = parallelize_task(num_workers=num_workers, iterator=paths, task=json_metadata_from_files, **dict(mlb=mlb))
    df = pd.concat([pd.DataFrame.from_records(json_object_list) for json_object_list in json_object_lists])
    # Check the dimensions
    logger.info(f"len(df): {len(df)}, len(paths): {len(paths)}")
    logger.info(f"Read files into dataframe in {time.time() - start} seconds.")

    # Denote if patch has snow and/or cloudsrandom_state
    snow = pd.read_csv(os.path.join(cloud_and_snow_csv_dir, 'patches_with_seasonal_snow.csv'), header=None, names=['image_prefix'])
    snow_col = 'has_snow'
    snow[snow_col] = 1
    snow[f'{snow_col}_target'] = json.dumps(np.array([1]), cls=NumpyEncoder)
    snow = snow.set_index('image_prefix')

    clouds = pd.read_csv(os.path.join(cloud_and_snow_csv_dir, 'patches_with_cloud_and_shadow.csv'), header=None, names=['image_prefix'])
    cloud_col = 'has_cloud_and_shadow'
    clouds[cloud_col] = 1
    clouds[f'{cloud_col}_target'] = json.dumps(np.array([1]), cls=NumpyEncoder)
    clouds = clouds.set_index('image_prefix')

    print(snow.head(3))
    len_snow = len(snow)
    print('\n')
    print(clouds.head(3))
    len_clouds = len(clouds)

    for column in [snow_col, cloud_col]:
        df[column] = 0

    for column in [f"{snow_col}_target", f"{cloud_col}_target"]:
        df[column] = json.dumps(np.array([0]), cls=NumpyEncoder)

    df = df.set_index('image_prefix', drop=False)
    df.update(snow)
    df.update(clouds)
    assert df[snow_col].sum() == len_snow
    assert df[cloud_col].sum() == len_clouds

    df.to_csv(csv_files_path + '/metadata.csv')

    return df
