from typing import Union
import numpy as np
import pandas as pd


class AUUC:
    def get_name(self) -> str:
        raise NotImplementedError()

    def compute_random(self, df, outcome="outcome", treatment="treatment", count=None) -> float:
        raise NotImplementedError()

    def compute(self,
                df,
                outcome="outcome",
                treatment="treatment",
                u_pred: Union[str, pd.Series] = "u_pred",
                count=None,
                ) -> float:
        raise NotImplementedError()


class AUUC_ConstantRatio(AUUC):
    @staticmethod
    def _count_with_constant_ratio(df, treatment="treatment", count=None):
        if count is None:
            count = pd.Series(np.ones(len(df)), index=df.index)
        else:
            count = df[count]

        treat_tot_count = count[df[treatment] == 1].sum()
        ctrl_tot_count = count[df[treatment] == 0].sum()
        tot_count = treat_tot_count + ctrl_tot_count
        treat_factor = tot_count / (2 * treat_tot_count)
        ctrl_factor = tot_count / (2 * ctrl_tot_count)
        return count * np.where(df[treatment] == 1, treat_factor, ctrl_factor)

    def compute_random(self, df, outcome="outcome", treatment="treatment", count=None) -> float:
        count = AUUC_ConstantRatio._count_with_constant_ratio(df, treatment, count)
        delta_r = self._compute_rlabel(outcome=df[outcome], count=count, treatment=df[treatment]).sum()
        return delta_r / 2

    def compute(self,
                df,
                outcome="outcome",
                treatment="treatment",
                u_pred: Union[str, pd.Series] = "u_pred",
                count=None,
                ) -> float:
        count = AUUC_ConstantRatio._count_with_constant_ratio(df, treatment, count)
        sub_df = pd.DataFrame(index=df.index)
        sub_df['count'] = count

        sub_df['label'] = self._compute_rlabel(outcome=df[outcome], count=sub_df["count"], treatment=df[treatment])

        if type(u_pred) is str:
            sub_df['u_pred'] = df[u_pred]
        else:
            assert u_pred.index is sub_df.index
            sub_df['u_pred'] = u_pred

        group_df = sub_df.groupby('u_pred')[["count", 'label']].sum()
        group_df.sort_values(by='u_pred', ascending=False, inplace=True)

        return ((group_df['label'].cumsum() * group_df["count"]).sum()
                - (group_df['label'] * group_df["count"]).sum() / 2
                ) / count.sum()

    def _compute_rlabel(self, outcome, count, treatment) -> pd.Series:
        raise NotImplementedError()


class AUUC_ConstantRatio_V1(AUUC_ConstantRatio):
    def _compute_rlabel(self, outcome, count, treatment):
        return outcome * count * (2 * treatment - 1)

    def get_name(self) -> str:
        return "AUUC_cr_V1"


class AUUC_ConstantRatio_V2(AUUC_ConstantRatio):
    def _compute_rlabel(self, outcome, count, treatment):
        return (outcome - 1) * count * (2 * treatment - 1)

    def get_name(self) -> str:
        return "AUUC_cr_V2"


class AUUC_ConstantRatio_V1V2(AUUC_ConstantRatio):
    # (1-nu) * V1 + nu * V2 = V1+\nu (V1-V2)
    def __init__(self, nu: float):
        assert 0 <= nu <= 1
        self._nu = nu

    def _compute_rlabel(self, outcome, count, treatment):
        return (outcome - self._nu) * count * (2 * treatment - 1)

    def get_name(self):
        return f"AUUC_cr_V1V2({self._nu})"


# used for ICML experiments
def metric_auuc_random(df, outcome="outcome", treatment="treatment", weight=None):
    weight = df[weight] if weight is not None else 1
    r_t = (df[outcome] * weight * df[treatment]).sum()
    r_c = (df[outcome] * weight * (1 - df[treatment])).sum()
    n_t = (weight * df[treatment]).sum()
    n_c = (weight * (1 - df[treatment])).sum()
    return (r_t - r_c * n_t / n_c) / 2


# used for ICML experiments
def metric_auuc_cum_ratios(df, outcome="outcome", treatment="treatment", u_pred="u_pred", weight=None):
    if weight is None:
        weight = 'loc_weight'
        sub_df = pd.DataFrame(np.ones(len(df)), index=df.index, columns=[weight])
    else:
        sub_df = df[[weight]].copy()

    tot_weights = sub_df[weight].sum()

    sub_df['r_t'] = df[outcome] * sub_df[weight] * df[treatment]
    sub_df['r_c'] = df[outcome] * sub_df[weight] * (1 - df[treatment])
    sub_df['n_t'] = sub_df[weight] * df[treatment]
    sub_df['n_c'] = sub_df[weight] * (1 - df[treatment])
    sub_df['u_pred'] = df[u_pred]

    group_df = sub_df.groupby('u_pred')[[weight, 'r_t', 'r_c', 'n_t', 'n_c']].sum()
    group_df.sort_values(by='u_pred', ascending=False, inplace=True)

    ratios = (group_df['n_t'].cumsum() / group_df['n_c'].cumsum()).replace(np.inf, 0)
    bars = ((group_df['r_t'].cumsum() - group_df['r_c'].cumsum() * ratios) * group_df[weight]).sum()
    triangles = ((group_df['r_t'] - group_df['r_c'] * ratios) * group_df[weight]).sum() / 2
    return (bars - triangles) / tot_weights


def qini_np(df, outcome="outcome", treatment="treatment", u_pred="u_pred"):
    """
    Compute Qini curve for the uplift model
    """
    y = df[outcome].values
    t = df[treatment].values
    u = df[u_pred].values

    arr = np.vstack((y,t,u)).T
    arr = arr[arr[:,2].argsort()[::-1]]
    r_t = np.cumsum(arr[:,1] * arr[:,0])
    r_c = np.cumsum((1 - arr[:,1]) * arr[:,0])
    n_t = np.cumsum(arr[:,1])
    n_c = np.cumsum(1 - arr[:,1])
    q_curve = r_t - r_c * np.nan_to_num(n_t / n_c)
    area = np.trapz(q_curve, dx = 1./len(y))
    return area
