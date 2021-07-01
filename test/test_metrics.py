import unittest
import pandas as pd
import random
from umm.metrics import metric_auuc_cum_ratios, qini_np, metric_auuc_random, \
    AUUC_ConstantRatio_V1, AUUC_ConstantRatio_V2, AUUC_ConstantRatio_V1V2


T1_Y1 = (1, 1)
T1_Y0 = (1, 0)
T0_Y1 = (0, 1)
T0_Y0 = (0, 0)


class TestMetrics(unittest.TestCase):
    def test_toy_data(self):
        """
        https://confluence.criteois.com/display/RSC/Contextual+uplift+modeling
        """

        def prep_data(data_):
            num = len(data_)
            data_ = [(t, o, (num - i) / num) for i, (t, o) in enumerate(data_)]
            random.shuffle(data_)
            return pd.DataFrame(data=data_, columns=["treatment", "outcome", "u_pred"])

        # (T, Y)
        df_50_50 = prep_data([T1_Y1] * 40 + [T1_Y0] * 10 + [T0_Y0] * 40 + [T0_Y1] * 10)
        df_80_20 = prep_data([T1_Y1] * 64 + [T1_Y0] * 16 + [T0_Y0] * 16 + [T0_Y1] * 4)
        df_80_20['weight'] =\
            df_80_20['treatment'] * (80 + 20) / (2 * 80) + (1 - df_80_20['treatment']) * (80 + 20) / (2 * 20)

        for df, n_t, n_c in [(df_50_50, 50, 50), (df_80_20, 80, 20)]:
            self.assertEqual(100, len(df))
            self.assertEqual(n_t, (df.treatment == 1).sum())
            self.assertEqual(n_c, (df.treatment == 0).sum())
            self.assertEqual(
                4, len(df[(df.treatment == 1) & (df.outcome == 1)]) / len(df[(df.treatment == 1) & (df.outcome == 0)]))
            self.assertEqual(
                4, len(df[(df.treatment == 0) & (df.outcome == 0)]) / len(df[(df.treatment == 0) & (df.outcome == 1)]))

        auuc_v1 = AUUC_ConstantRatio_V1()
        auuc_v2 = AUUC_ConstantRatio_V2()
        auuc_v1v2 = AUUC_ConstantRatio_V1V2(0.5)

        self.assertEqual(31.5, auuc_v1.compute(df_50_50))
        self.assertEqual(31.5, auuc_v1.compute(df_80_20))
        df_5_5 = prep_data([T1_Y1] * 4 + [T1_Y0] + [T0_Y0] * 4 + [T0_Y1])
        df_5_5['count'] = 10
        self.assertEqual(3.15, auuc_v1.compute(df_5_5))
        self.assertEqual(31.5, auuc_v1.compute(df_5_5, count="count"))

        self.assertEqual(6.5, auuc_v2.compute(df_50_50))
        self.assertEqual(6.5, auuc_v2.compute(df_80_20))
        self.assertEqual(0.65, auuc_v2.compute(df_5_5))
        self.assertEqual(6.5, auuc_v2.compute(df_5_5, count="count"))

        self.assertEqual((31.5 + 6.5) / 2, auuc_v1v2.compute(df_50_50))
        self.assertEqual((31.5 + 6.5) / 2, auuc_v1v2.compute(df_80_20))
        self.assertEqual((3.15 + 0.65) / 2, auuc_v1v2.compute(df_5_5))
        self.assertEqual((31.5 + 6.5) / 2, auuc_v1v2.compute(df_5_5, count="count"))

        self.assertAlmostEqual(31.46841156270924, metric_auuc_cum_ratios(df_50_50))
        self.assertAlmostEqual(26.990337977296186, metric_auuc_cum_ratios(df_80_20) * (80 + 20) / (2 * 80))
        self.assertAlmostEqual(31.475844943240453, metric_auuc_cum_ratios(df_80_20, weight='weight'))

        self.assertAlmostEqual(31.45824598786098, qini_np(df_50_50))
        self.assertAlmostEqual(26.98296031131751, qini_np(df_80_20) * (80 + 20) / (2 * 80))

        self.assertEqual(15, metric_auuc_random(df_50_50))
        self.assertEqual(24, metric_auuc_random(df_80_20))
        self.assertEqual(15, metric_auuc_random(df_80_20, weight="weight"))
        self.assertEqual(15, auuc_v1.compute_random(df_50_50))
        self.assertEqual(15, auuc_v1.compute_random(df_80_20))
        self.assertEqual(15, auuc_v2.compute_random(df_50_50))
        self.assertEqual(15, auuc_v2.compute_random(df_80_20))
        self.assertEqual(15, auuc_v1v2.compute_random(df_50_50))
        self.assertEqual(15, auuc_v1v2.compute_random(df_80_20))

        # if all items have the same predicted uplift, auuc should equals acuuc random
        df_50_50['u_pred'] = 0
        df_80_20['u_pred'] = 0
        self.assertEqual(15, auuc_v1.compute(df_50_50))
        self.assertEqual(15, auuc_v1.compute(df_80_20))


    def test_toy_data_2(self):
        def prep_data(groups):
            n_groups = len(groups)
            data_ = [(t, o, (n_groups - i) / n_groups) for i, group in enumerate(groups) for t, o in group]
            random.shuffle(data_)
            return pd.DataFrame(data=data_, columns=["treatment", "outcome", "u_pred"])

        groups = [[T1_Y1, T1_Y0, T0_Y0, T0_Y0],
                  [T1_Y1, T1_Y1],
                  [T1_Y1, T1_Y1, T1_Y0, T0_Y1],
                  [T1_Y1, T1_Y1, T1_Y0, T0_Y1, T0_Y1],
                  [T1_Y0, T1_Y0, T1_Y0, T1_Y0, T0_Y1, T0_Y1]]
        df = prep_data(groups)
        self.assertEqual(21, len(df))
        self.assertEqual(14, (df.treatment == 1).sum())
        self.assertEqual(7, (df.treatment == 0).sum())

        df_rebalanced = df.append(df[df.treatment == 0], ignore_index=False)
        self.assertEqual(28, len(df_rebalanced))
        self.assertEqual(14, (df_rebalanced.treatment == 1).sum())
        self.assertEqual(14, (df_rebalanced.treatment == 0).sum())

        auuc_v1 = AUUC_ConstantRatio_V1()

        self.assertEqual((3 + 4 + 15 + 14 - 8) / (28**2) * 21, auuc_v1.compute(df))
        df["count"] = 2 - df.treatment
        self.assertEqual((3 + 4 + 15 + 14 - 8) / 28, auuc_v1.compute(df, count="count"))
        self.assertEqual((3 + 4 + 15 + 14 - 8) / 28, auuc_v1.compute(df_rebalanced))

        self.assertEqual(-3 / (2 * 28) * 21, auuc_v1.compute_random(df))
        self.assertEqual(-3 / 2, auuc_v1.compute_random(df, count="count"))
        self.assertEqual(-3 / 2, auuc_v1.compute_random(df_rebalanced))

        # if all items have the same predicted uplift, auuc should equals acuuc random
        df['u_pred'] = 0
        df_rebalanced['u_pred'] = 0
        self.assertEqual(-3 / (2 * 28) * 21, auuc_v1.compute(df))
        self.assertEqual(-3 / 2, auuc_v1.compute(df, count="count"))
        self.assertEqual(-3 / 2, auuc_v1.compute(df_rebalanced))
