import nmslib
import numpy as np
from rectools.dataset import Dataset
from rectools import Columns

import pandas as pd
import numpy as np
from service.utils import load_model


from settings import Settings


class ApproximateNearestNeighbors:
    def __init__(
        self,
        model,
        dataset: Dataset,
        M: int = Settings.M,
        efC: int = Settings.EFC,
        efS: int = Settings.EFS,
        num_threads: int = Settings.NUM_THREADS,
    ):
        self.model = model
        self.dataset = dataset
        self.num_threads = num_threads
        self.index = nmslib.init(
            method='hnsw',
            space=Settings.SPACE_NAME,
            data_type=nmslib.DataType.DENSE_VECTOR
        )
        self.index_time_params = {
            'M': M,
            'indexThreadQty': self.num_threads,
            'efConstruction': efC
        }
        self.neighbours = None
        self.query_time_params = {'efSearch': efS}

    @staticmethod
    def augment_inner_product(factors):
        normed_factors = np.linalg.norm(factors, axis=1)
        max_norm = normed_factors.max()

        extra_dim = np.sqrt(max_norm ** 2 - normed_factors ** 2).reshape(-1, 1)
        augmented_factors = np.append(factors, extra_dim, axis=1)
        return max_norm, augmented_factors

    def fit(self, k_reco: int = 10):
        user_embeddings, item_embeddings = self.model.get_vectors(self.dataset)
        max_norm, augmented_item_embeddings = self.augment_inner_product(
            item_embeddings
        )
        extra_zero = np.zeros((user_embeddings.shape[0], 1))
        augmented_user_embeddings = np.append(
            user_embeddings,
            extra_zero,
            axis=1
        )

        self.index.addDataPointBatch(augmented_item_embeddings)
        self.index.createIndex(self.index_time_params)
        self.index.setQueryTimeParams(self.query_time_params)

        self.neighbours = self.index.knnQueryBatch(
            augmented_user_embeddings,
            k=k_reco,
            num_threads=self.num_threads
        )

    def predict(self, user_id: int):
        int_id = self.dataset.user_id_map.convert_to_internal([user_id])[0]
        user_recs = self.neighbours[int_id][0]  # type: ignore
        return self.dataset.item_id_map.convert_to_external(user_recs)


def get_ann():
    interactions = pd.read_csv('data/kion_train/interactions.csv')
    Columns.Datetime = 'last_watch_dt'
    interactions.drop(interactions[interactions[Columns.Datetime].str.len() != 10].index, inplace=True)
    interactions[Columns.Datetime] = pd.to_datetime(interactions[Columns.Datetime], format='%Y-%m-%d')
    interactions[Columns.Weight] = np.where(interactions['watched_pct'] > 10, 3, 1)
    dataset = Dataset.construct(interactions_df=interactions)
    model = load_model('models/warp_12.pickle')
    ann = ApproximateNearestNeighbors(model=model, dataset=dataset)
    ann.fit(k_reco=10)
    return ann
