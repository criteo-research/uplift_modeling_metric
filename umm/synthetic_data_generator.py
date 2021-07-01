from typing import List, Tuple, Union, Optional
import itertools
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class SyntheticDataGenerator:
    def __init__(self,
                 size: int,
                 treatment_ratio: float,
                 compliances: Union[List[float], int, float],
                 ctrl_outcome_rate: float,
                 treatment_uplifts: List[float],
                 n_splits: int,
                 seed: int,
                 split_with_replacement: bool = False,
                 noisy_features: float = 0.):
        if type(compliances) in [int, float]:
            compliances = [compliances] * len(treatment_uplifts)
        else:
            assert len(treatment_uplifts) == len(compliances)
        self._size = size
        self._treatment_ratio = treatment_ratio

        # rates
        self._compliances = compliances
        self._ctrl_outcome_rate = ctrl_outcome_rate
        #  P(Y=1|T=1,M=1,X=xi)
        self._treated_outcome_rates = np.clip([u + ctrl_outcome_rate for u in treatment_uplifts], 0, 1)

        self._n_splits = n_splits
        self._seed = seed
        self._split_with_replacement = split_with_replacement
        self._noisy_features = noisy_features
        self._features = [f"f{i}" for i in range(len(treatment_uplifts))]
        self._unique_items_cols = ['item_cat', 'uplift', 'T'] + self._features

        self._df = None
        self._unique_df = None

    def get_feature_names(self):
        return self._features

    def _get_data(self):
        if self._df is None:
            self._df = self._generate_data()
            self._unique_df = self._to_unique_items(self._df)
        return self._df, self._unique_df

    def _generate_data(self):
        np.random.seed(self._seed)
        n_item_categories = len(self._treated_outcome_rates)

        df = pd.DataFrame()
        df['count'] = np.ones(self._size, dtype=int)
        df['item_cat'] = np.random.randint(0, n_item_categories, size=self._size)
        q_1s = df['item_cat'].replace(to_replace=dict(enumerate(self._treated_outcome_rates))).values
        compliance = df['item_cat'].replace(to_replace=dict(enumerate(self._compliances))).values
        df['uplift'] = compliance * (q_1s - self._ctrl_outcome_rate)  # P(Y=1|T=1,X=xi) - P(Y=1|T=0,X=xi)

        df['T'] = np.random.binomial(1, self._treatment_ratio, self._size)
        df['M'] = (np.random.rand(self._size) < compliance).astype(int) * df['T']
        outcome_if_not_treated = np.random.binomial(1, self._ctrl_outcome_rate, self._size)
        outcome_if_treated = (np.random.rand(len(q_1s)) < q_1s).astype(int)
        outcome_t = df['T'] * (df['M'] * outcome_if_treated + (1 - df['M']) * outcome_if_not_treated)
        outcome_c = (1 - df['T']) * outcome_if_not_treated
        df['Y'] = outcome_t + outcome_c
        df['th_Y'] = self._ctrl_outcome_rate + df['uplift'] * df['T']  # theoretical outcome (for the 'AUUC_thout' metric)
        rdm_noise_feature = np.random.normal(loc=0., scale=self._noisy_features, size=(self._size, n_item_categories))#.rand(self._size, n_item_categories)

        # create features
        for i in range(n_item_categories):
            df[f'f{i}'] = (df['item_cat'] == i).astype(int)+rdm_noise_feature[:,i]
        return df

    def _to_unique_items(self, df):
        # Convert set to unique items (i.e. item categories).
        # If we have 'c' item categories and 'g' compliance values, we have '2 * c * g' unique items.
        # ('2 *' because treatment/control).
        df = df.groupby(self._unique_items_cols)\
            .agg({'Y': 'mean', 'th_Y': 'mean', 'count': 'sum'})\
            .reset_index()
        ratio = np.where(df['T'] == 1, self._treatment_ratio, 1 - self._treatment_ratio)
        df['th_count'] = df['count'].sum() * ratio * 2 / len(df)  # theoretical count (for the 'AUUC_thout' metric)
        return df

    def __len__(self):
        return self._n_splits

    def get(self) -> pd.DataFrame:
        return self._get_data()[1]

    def __getitem__(self, split: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        assert 0 <= split < self._n_splits
        df = self._get_data()[0]
        if self._split_with_replacement:
            train = None
            test = df.sample(frac=0.5, replace=True, random_state=split)
        else:
            train, test = train_test_split(df, test_size=0.5, random_state=split)
        test = self._to_unique_items(test)
        test['emp_Y'] = self._unique_df['Y']
        test['emp_count'] = self._unique_df['count'] / 2  # divide by 2 to simulate the test dataset size
        return train, test

    def __iter__(self):
        for split in range(self._n_splits):
            yield self[split]


class SyntheticUniqueDataGenerator:
    def __init__(self,
                 size: int,
                 treatment_ratio: float,
                 compliances: Union[List[float], int, float],
                 ctrl_outcome_rate: float,
                 treatment_uplifts: List[float],
                 n_datasets: int,
                 seed: int):
        if type(compliances) in [int, float]:
            compliances = [compliances] * len(treatment_uplifts)
        else:
            assert len(treatment_uplifts) == len(compliances)
        self._size = size
        self._treatment_ratio = treatment_ratio

        # rates
        self._compliances = compliances
        self._ctrl_outcome_rate = ctrl_outcome_rate
        #  P(Y=1|T=1,M=1,X=xi)
        self._treated_outcome_rates = np.clip([u + ctrl_outcome_rate for u in treatment_uplifts], 0, 1)

        self._n_datasets = n_datasets
        self._seed = seed
        self._features = [f"f{i}" for i in range(len(treatment_uplifts))]
        self._unique_items_cols = ['item_cat', 'uplift', 'T'] + self._features

    def get_feature_names(self):
        return self._features

    def _generate_data(self, i):
        np.random.seed(self._seed + i)
        n_item_categories = len(self._treated_outcome_rates)
        df = pd.DataFrame(data=itertools.product(range(n_item_categories), [0, 1]), columns=['item_cat', 'T'])
        pop_ratios =  np.tile([1 - self._treatment_ratio, self._treatment_ratio], n_item_categories) / n_item_categories
        df['th_count'] = self._size * pop_ratios
        df['count'] = np.random.multinomial(self._size, pop_ratios)

        q_1s = df['item_cat'].replace(to_replace=dict(enumerate(self._treated_outcome_rates))).values
        compliance = df['item_cat'].replace(to_replace=dict(enumerate(self._compliances))).values
        df['uplift'] = compliance * (q_1s - self._ctrl_outcome_rate)  # P(Y=1|T=1,X=xi) - P(Y=1|T=0,X=xi)
        df['th_Y'] = self._ctrl_outcome_rate + df['uplift'] * df['T']  # theoretical outcome (for the 'AUUC_thout' metric)
        df['Y'] = df.apply(lambda row: np.random.binomial(row['count'], row['th_Y']), axis=1) / df['count']

        # create features
        for i in range(n_item_categories):
            df[f'f{i}'] = (df['item_cat'] == i).astype(int)
        return df

    def __len__(self):
        return self._n_datasets

    def __getitem__(self, i: int) -> pd.DataFrame:
        assert 0 <= i < self._n_datasets
        return self._generate_data(i)

    def __iter__(self):
        for i in range(self._n_datasets):
            yield self[i]
