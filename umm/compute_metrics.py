import argparse
import sys
import os
from typing import List, Union
import pandas as pd
from tqdm import tqdm
from umm.synthetic_data_generator import SyntheticDataGenerator, SyntheticUniqueDataGenerator
from umm.metrics import AUUC_ConstantRatio_V1, AUUC_ConstantRatio_V2, AUUC_ConstantRatio_V1V2, AUUC
from umm.uplift_predictor import noisy_oracle
from model.GaussianMixtureCEM import *
from model.TwoClassifiers import *
from model.OneClassifier import *
from sklearn.linear_model import LogisticRegression


def put_metric_results(res_df, method, split, auuc_ver_str, auuc, auuc_th, auuc_emp):#, auuc2, auuc2_th, auuc2_emp):
    res_df.loc[len(res_df)] = {'method': method,
                               'split': split,
                               'AUUC_version': auuc_ver_str,
                               'AUUC': auuc,
                               'AUUC_thout': auuc_th,
                               'AUUC_emp': auuc_emp
                               }

# def use_model_OneClassifier(test_df: pd.DataFrame):


def compute_model_metrics(test_df: pd.DataFrame,
                          split: int,
                          model_name: str,
                          m_auuc_list: List[AUUC],
                          uplift: Union[str, pd.Series],
                          res_df: pd.DataFrame):

    for m_auuc in m_auuc_list:
        auuc = m_auuc.compute(test_df, 'Y', 'T', uplift, count='count')
        auuc_th = m_auuc.compute(test_df, 'th_Y', 'T', uplift, count='th_count')
        if 'emp_Y' in test_df.columns:
            auuc_emp = m_auuc.compute(test_df, 'emp_Y', 'T', uplift, count='emp_count')
        else:
            auuc_emp = auuc
        put_metric_results(res_df, model_name, split, m_auuc.get_name(), auuc, auuc_th, auuc_emp)


def compute_metrics(dataset: SyntheticDataGenerator) -> pd.DataFrame:
    res_df = pd.DataFrame(columns=['method',
                                   'split',
                                   'AUUC_version',
                                   'AUUC',
                                   'AUUC_thout',
                                   'AUUC_emp'
                                   ])
    m_auuc_list = [AUUC_ConstantRatio_V1(),\
                   AUUC_ConstantRatio_V2()]+\
                   [AUUC_ConstantRatio_V1V2(ratio) for ratio in np.linspace(0,1.0,11)]

    for split, tup in tqdm(enumerate(dataset), total=len(dataset)):
        if type(tup) is tuple:
            train_df, test_df = tup
        else:
            train_df = None
            test_df = tup

        for m_auuc in m_auuc_list:
            auuc_random = m_auuc.compute_random(test_df, 'Y', 'T', count='count')
            auuc_random_th = m_auuc.compute_random(test_df, 'th_Y', 'T', count='th_count')
            if 'emp_Y' in test_df.columns:
                auuc_random_emp = m_auuc.compute_random(test_df, 'emp_Y', 'T', count='emp_count')
            else:
                auuc_random_emp = auuc_random
            put_metric_results(res_df, 'RAND', split, m_auuc.get_name(), auuc_random, auuc_random_th, auuc_random_emp)

        compute_model_metrics(test_df=test_df,
                              split=split,
                              model_name='Oracle',
                              m_auuc_list=m_auuc_list,
                              uplift='uplift',
                              res_df=res_df)

        for sigma in [0.1, 0.5, 1.0, 2.0]:
            uplift = noisy_oracle(test_df, sigma, 'uplift', seed=split)
            compute_model_metrics(test_df=test_df,
                                  split=split,
                                  model_name=f'NO{sigma}',
                                  m_auuc_list=m_auuc_list,
                                  uplift=uplift,
                                  res_df=res_df)

        if train_df is not None:
            predictors = dataset.get_feature_names()

            two_classif = TwoClassifiers(model_t=LogisticRegression(solver='lbfgs'),
                                         model_c=LogisticRegression(solver='lbfgs'))
            two_classif.fit(train_df, predictors)
            two_classif.predict(test_df, predictors)
            compute_model_metrics(test_df=test_df,
                                  split=split,
                                  model_name='TwoClassif',
                                  m_auuc_list=m_auuc_list,
                                  uplift='ITE',
                                  res_df=res_df)

            # one_classif = OneClassifier(LogisticRegression(solver='lbfgs'))
            # one_classif.fit(train_df, predictors)
            # one_classif.predict(test_df, predictors)
            # compute_model_metrics(test_df=test_df,
            #                       split=split,
            #                       model_name='OneClassif',
            #                       m_auuc_list=m_auuc_list,
            #                       uplift='ITE',
            #                       res_df=res_df)


            # gauss2 = GaussianMixtureCEM()
            # gauss2.fit(train_df, predictors)
            # gauss2.predict(test_df, predictors)
            # compute_model_metrics(test_df=test_df,
            #                       split=split,
            #                       model_name='GaussianMixtureCEM',
            #                       m_auuc_list=m_auuc_list,
            #                       uplift='ITE',
            #                       res_df=res_df)
    return res_df




def main(args: List[str]):
    compliances = [0.005, 0.005, 0.005, 0.005, 0.005, 0.01, 0.01, 0.01, 0.01, 0.01]
    treatment_uplifts = [-1e-1, 1e-1, 3e-1, 5e-1, 7e-1, -1e-1, 1e-1, 3e-1, 5e-1, 7e-1]
    treatment_uplifts2 = [2e-1]


    data_gens = {
        'multi': SyntheticDataGenerator(
            size=int(2e6),
            treatment_ratio=0.5,
            compliances=compliances,
            ctrl_outcome_rate=1e-1,
            treatment_uplifts=treatment_uplifts,
            n_splits=51,
            seed=338
        ),
        'remulti': SyntheticDataGenerator(
            size=int(2e6),
            treatment_ratio=0.5,
            compliances=compliances,
            ctrl_outcome_rate=1e-1,
            treatment_uplifts=treatment_uplifts,
            n_splits=51,
            seed=338,
            split_with_replacement=True
        ),
        'umulti': SyntheticUniqueDataGenerator(
            size=int(1e6),
            treatment_ratio=0.5,
            compliances=compliances,
            ctrl_outcome_rate=1e-1,
            treatment_uplifts=treatment_uplifts,
            n_datasets=101,
            seed=338
        ),
        'multi_80_20': SyntheticDataGenerator(
            size=int(2e6),
            treatment_ratio=0.8,
            compliances=compliances,
            ctrl_outcome_rate=1e-1,
            treatment_uplifts=treatment_uplifts,
            n_splits=51,
            seed=338
        ),
        'remulti_80_20': SyntheticDataGenerator(
            size=int(2e6),
            treatment_ratio=0.8,
            compliances=compliances,
            ctrl_outcome_rate=1e-1,
            treatment_uplifts=treatment_uplifts,
            n_splits=51,
            seed=338,
            split_with_replacement=True
        ),
        'umulti_80_20': SyntheticUniqueDataGenerator(
            size=int(1e6),
            treatment_ratio=0.8,
            compliances=compliances,
            ctrl_outcome_rate=1e-1,
            treatment_uplifts=treatment_uplifts,
            n_datasets=101,
            seed=338
        ),
        'multi_monoc': SyntheticDataGenerator(
            size=int(2e6),
            treatment_ratio=0.5,
            compliances=0.01,
            ctrl_outcome_rate=1e-1,
            treatment_uplifts=treatment_uplifts,
            n_splits=51,
            seed=338
        )
    }

    parser = argparse.ArgumentParser(prog=args[0])
    parser.add_argument("input", choices=sorted(data_gens.keys()), help="Id of the dataset config to be used")
    parser.add_argument("output", type=str, help="Path to the output directory")
    # parser.add_argument("-n", "--ncpus", type=int, default=1, help="Number of CPUs to be used")
    # args = parser.parse_args(args[1:])
    args = parser.parse_args()
    data_gen = data_gens[args.input]
    output_directory = os.path.abspath(args.output)
    # n_cpus = args.ncpus

    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)
    prefix_output_filename = os.path.join(output_directory, f"result_{args.input}")

    res_df = compute_metrics(data_gen)
    res_df.to_csv(path_or_buf=f"{prefix_output_filename}.csv", index=False)


if __name__ == "__main__":
    main(sys.argv)

