import os
import zipfile
from os import path
from typing import List, Tuple, Any, Dict, Union

import numpy
import sklearn as sklearn
from PIL import Image
from google.cloud.storage import Client
from numpy import asarray, full, ndarray

_TRAIN_URL = "https://storage.googleapis.com/face-mask-recognition-dataset/train.zip"
_TEST_URL = "https://storage.googleapis.com/face-mask-recognition-dataset/test.zip"


class RetimoDatasetConfigRecord:
    def __init__(self, dataset_name: str, name: str, gcs_path: str, labels: dict):
        self.dataset_name = dataset_name
        self.name = name
        self.gcs_path = gcs_path
        self.labels = labels
        self.local_path = None
        self.is_downloaded = False
        self.is_unzipped = False

    def add_local_path(self, local_path):
        self.local_path = local_path
        self.is_downloaded = True
        return self

    def unzipped(self):
        self.is_unzipped = True
        return self

    @property
    def unzipped_path(self):
        if not self.is_unzipped:
            raise Exception("Unzipped Path is not ready. Firstly unzip to downloaded files.")
        return f"{self.local_path}/unziped"

    @property
    def raw_path(self):
        if not self.is_downloaded and self.local_path is None:
            raise Exception("Row Path is not ready. Firstly download to local files.")
        return f"{self.local_path}/raw"


class RetimoDataset:
    cache_path = '.cache'

    def __init__(self, config_records: List[RetimoDatasetConfigRecord]):
        self.config_records = config_records
        self.storage_client = Client()

    def load(self, to_shuffle=True):
        self.config_records = [self._download(config_record) for config_record in self.config_records]
        self.config_records = [self._unzip(config_record) for config_record in self.config_records]
        arrays = dict(self._load_to_nparray(config_record) for config_record in self.config_records)
        if to_shuffle:
            values = arrays.values()
            y = {x for x in values}
            values_2 = values.values()
            shuffeled = sklearn.utils.shuffle()
        return arrays

    def _download(self, config_record: RetimoDatasetConfigRecord) -> RetimoDatasetConfigRecord:
        local_path = f"{self.cache_path}/{config_record.dataset_name}/{config_record.name}"
        os.makedirs(local_path, exist_ok=True)
        raw_path = f"{local_path}/raw"
        if not path.exists(raw_path):
            with open(raw_path, "w") as f:
                f.write("")

            print(f"Downloading raw data from {config_record.gcs_path} to {raw_path}")
            with open(raw_path, 'wb') as file_obj:
                self.storage_client.download_blob_to_file(config_record.gcs_path, file_obj)
        else:
            print(f"Downloading raw data for '{config_record.name}' not needed. Using cache '{raw_path}'")
        return config_record.add_local_path(local_path)

    def _unzip(self, config_record: RetimoDatasetConfigRecord) -> RetimoDatasetConfigRecord:
        unzipped_path = f"{config_record.local_path}/unziped"
        if not path.exists(unzipped_path):
            print(f"Unzipping file {config_record.raw_path} to {unzipped_path}")
            with zipfile.ZipFile(config_record.raw_path, 'r') as zip_ref:
                zip_ref.extractall(unzipped_path)
        else:
            print(f"Unzipping file for '{config_record.name}' not needed. Using cache '{unzipped_path}'")
        return config_record.unzipped()

    def _load_to_nparray(self, config_record: RetimoDatasetConfigRecord) -> Tuple[str, Dict[str, Union[ndarray, Any]]]:
        collector = {}
        for label in config_record.labels.keys():
            directory = f"{os.getcwd()}/{config_record.unzipped_path}/{config_record.name}/{label}/"
            dataset = numpy.asarray(
                [asarray(Image.open(f"{directory}/{image}"), dtype=numpy.uint8) for image in os.listdir(directory) if
                 image.endswith('.jpg')])
            label_list = full((len(dataset)), fill_value=config_record.labels[label], dtype=numpy.uint8)
            collector['dataset'] = numpy.append(collector.get('dataset', numpy.empty(0)), dataset).reshape(
                dataset.shape[0] + collector.get('dataset', numpy.empty(0)).shape[0], *dataset.shape[1:])
            collector['label'] = numpy.append(collector.get('label', numpy.empty(0)), label_list)

        return config_record.name, collector


dataset_name = 'face_mask_recognition'
labels_map = {'correct': 0, 'incorrect': 1, 'no_mask': 2}
config = [RetimoDatasetConfigRecord(dataset_name, 'train', 'gs://face-mask-recognition-dataset/train.zip', labels_map),
          RetimoDatasetConfigRecord(dataset_name, 'test', 'gs://face-mask-recognition-dataset/test.zip', labels_map)]
RetimoDataset(config).load()
