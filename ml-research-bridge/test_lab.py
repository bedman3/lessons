import unittest

from lab import chronological_split, make_synthetic_data, scale_with_train_only


class ChronologicalLabTests(unittest.TestCase):
    def test_label_windows_end_before_the_next_split(self):
        data = make_synthetic_data(seed=7, n_samples=240)
        split = chronological_split(data, train_end=100, validation_end=170, horizon=5)

        self.assertLess(split.train.label_end.max(), split.validation.decision_time.min())
        self.assertLess(split.validation.label_end.max(), split.test.decision_time.min())

    def test_scaler_is_fit_only_on_training_features(self):
        data = make_synthetic_data(seed=11, n_samples=240)
        split = chronological_split(data, train_end=100, validation_end=170, horizon=5)
        scaled = scale_with_train_only(split)

        self.assertTrue((abs(scaled.scaler_mean - split.train.features.mean(axis=0)) < 1e-12).all())
        self.assertGreater(abs(scaled.scaler_mean[0] - split.validation.features[:, 0].mean()), 1e-4)

    def test_run_is_reproducible_and_test_is_scored_once_after_selection(self):
        from lab import run_experiment

        first = run_experiment(seed=19, n_samples=360)
        second = run_experiment(seed=19, n_samples=360)

        self.assertEqual(first["selected_model"], second["selected_model"])
        self.assertEqual(first["test_metrics"], second["test_metrics"])
        self.assertEqual(first["test_evaluation_count"], 1)
        self.assertFalse(first["leakage_demo"]["used_for_selection"])


if __name__ == "__main__":
    unittest.main()
