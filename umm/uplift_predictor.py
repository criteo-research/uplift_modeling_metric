import numpy as np
import pandas as pd


def noisy_oracle(df, sigma, uplift="uplift", seed=0):
    np.random.seed(seed)
    # TODO: cap in [0, 1]?
    cats = df['item_cat'].unique()
    series = pd.Series(data=np.random.lognormal(mean=0, sigma=sigma, size=len(cats)),
                       index=cats,
                       name="mul")
    df = df[["item_cat", "uplift"]].join(series, on="item_cat")
    return df[uplift] * df["mul"]


