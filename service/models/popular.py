from rectools.models import PopularModel
from rectools.dataset import Dataset
from rectools import Columns
import pandas as pd
import numpy as np
from service.api.exceptions import ModelNotFoundError


def get_popular_items(model_type, k_recs):
    interactions = pd.read_csv('data/kion_train/interactions.csv')
    Columns.Datetime = 'last_watch_dt'
    interactions.drop(interactions[interactions[Columns.Datetime].str.len() != 10].index, inplace=True)
    interactions[Columns.Datetime] = pd.to_datetime(interactions[Columns.Datetime], format='%Y-%m-%d')
    interactions[Columns.Weight] = np.where(interactions['watched_pct'] > 10, 3, 1)

    if model_type == 'most_rated':
        model = PopularModel(popularity="sum_weight")
    elif model_type == 'default':
        model = PopularModel()
    else:
        raise ModelNotFoundError(error_message=f"Popular model {model_type} not found")
    dataset = Dataset.construct(interactions_df=interactions)
    model.fit(dataset)
    items = model.recommend(users=[1], dataset=dataset, k=k_recs, filter_viewed=True).item_id.tolist()[:k_recs]
    return items
