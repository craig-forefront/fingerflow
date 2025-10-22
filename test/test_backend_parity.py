"""Parity and performance checks between TensorFlow and PyTorch backends."""

from __future__ import annotations

import time
import unittest
from typing import Iterator, Tuple

try:  # Optional heavy dependencies; tests will be skipped when unavailable.
    import numpy as np
except ImportError:  # pragma: no cover - handled gracefully in tests
    np = None  # type: ignore[assignment]

try:  # pragma: no cover - handled gracefully in tests
    import tensorflow as tf
except ImportError:  # pragma: no cover - handled gracefully in tests
    tf = None  # type: ignore[assignment]

try:  # pragma: no cover - handled gracefully in tests
    import torch
except ImportError:  # pragma: no cover - handled gracefully in tests
    torch = None  # type: ignore[assignment]

BACKEND_DEPS_AVAILABLE = all(module is not None for module in (np, tf, torch))

if BACKEND_DEPS_AVAILABLE:  # pragma: no branch - executed only when deps exist
    from fingerflow.matcher.VerifyNet import verify_net_model
    from fingerflow_torch.matcher.verify_net import SiameseMatcher

    class VerifyNetHarness:
        """Utility that mirrors VerifyNet weights across TensorFlow and PyTorch."""

        def __init__(
            self,
            precision: int = 8,
            features: int = 9,
            dataset_size: int = 4,
            seed: int = 2024,
            device: str | torch.device = "cpu",
        ) -> None:
            self.precision = precision
            self.features = features
            self.input_shape = (precision, features, 1)
            self.device = torch.device(device)

            np.random.seed(seed)
            tf.random.set_seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            self.tf_model = verify_net_model.get_verify_net_model(precision)
            rng = np.random.default_rng(seed)
            tf_weights = [
                rng.standard_normal(weight.shape).astype(np.float32)
                for weight in self.tf_model.get_weights()
            ]
            self.tf_model.set_weights(tf_weights)

            self.torch_model = SiameseMatcher(self.input_shape).eval()
            state_dict = self.torch_model.state_dict()
            state_dict["embedding.features.0.weight"] = torch.from_numpy(
                tf_weights[0].transpose(3, 2, 0, 1)
            )
            state_dict["embedding.features.0.bias"] = torch.from_numpy(tf_weights[1])
            state_dict["embedding.features.3.weight"] = torch.from_numpy(
                tf_weights[2].transpose(3, 2, 0, 1)
            )
            state_dict["embedding.features.3.bias"] = torch.from_numpy(tf_weights[3])
            state_dict["embedding.dense.0.weight"] = torch.from_numpy(tf_weights[4].T)
            state_dict["embedding.dense.0.bias"] = torch.from_numpy(tf_weights[5])
            state_dict["bn.weight"] = torch.from_numpy(tf_weights[6].reshape(-1))
            state_dict["bn.bias"] = torch.from_numpy(tf_weights[7].reshape(-1))
            state_dict["bn.running_mean"] = torch.from_numpy(tf_weights[8].reshape(-1))
            state_dict["bn.running_var"] = torch.from_numpy(tf_weights[9].reshape(-1))
            state_dict["fc.weight"] = torch.from_numpy(tf_weights[10].T)
            state_dict["fc.bias"] = torch.from_numpy(tf_weights[11])
            self.torch_model.load_state_dict(state_dict)
            self.torch_model.to(self.device)
            self.torch_model.eval()

            self.anchor_batch = np.stack(
                [
                    rng.standard_normal(self.input_shape).astype(np.float32)
                    for _ in range(dataset_size)
                ]
            )
            self.sample_batch = np.stack(
                [
                    rng.standard_normal(self.input_shape).astype(np.float32)
                    for _ in range(dataset_size)
                ]
            )
            self.torch_anchor_tensor = torch.from_numpy(self.anchor_batch).permute(0, 3, 1, 2)
            self.torch_sample_tensor = torch.from_numpy(self.sample_batch).permute(0, 3, 1, 2)
            self.torch_anchor_tensor = self.torch_anchor_tensor.to(self.device)
            self.torch_sample_tensor = self.torch_sample_tensor.to(self.device)

        def iter_pairs(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
            for anchor, sample in zip(self.anchor_batch, self.sample_batch):
                yield anchor, sample

        def tf_predict(self, anchor: np.ndarray, sample: np.ndarray) -> float:
            prediction = self.tf_model.predict(
                [anchor[np.newaxis, ...], sample[np.newaxis, ...]], verbose=0
            )
            return float(prediction.squeeze())

        def torch_predict(self, anchor: np.ndarray, sample: np.ndarray) -> float:
            anchor_tensor = torch.from_numpy(anchor).unsqueeze(0).permute(0, 3, 1, 2).to(self.device)
            sample_tensor = torch.from_numpy(sample).unsqueeze(0).permute(0, 3, 1, 2).to(self.device)
            with torch.no_grad():
                score = self.torch_model(anchor_tensor, sample_tensor)
            return float(score.squeeze().cpu().numpy())

        def tf_batch_predict(self) -> np.ndarray:
            batch_prediction = self.tf_model.predict(
                [self.anchor_batch, self.sample_batch], verbose=0
            )
            return batch_prediction.reshape(-1)

        def torch_batch_predict(self) -> np.ndarray:
            with torch.no_grad():
                scores = self.torch_model(self.torch_anchor_tensor, self.torch_sample_tensor)
            return scores.squeeze(-1).cpu().numpy()

        def measure_throughput(self, repeats: int = 5) -> Tuple[float, float]:
            batch_size = self.anchor_batch.shape[0]

            start = time.perf_counter()
            for _ in range(repeats):
                self.tf_model.predict([self.anchor_batch, self.sample_batch], verbose=0)
            tf_elapsed = max(time.perf_counter() - start, 1e-9)
            tf_throughput = repeats * batch_size / tf_elapsed

            start = time.perf_counter()
            with torch.no_grad():
                for _ in range(repeats):
                    self.torch_model(self.torch_anchor_tensor, self.torch_sample_tensor)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            torch_elapsed = max(time.perf_counter() - start, 1e-9)
            torch_throughput = repeats * batch_size / torch_elapsed
            return tf_throughput, torch_throughput

        def close(self) -> None:
            tf.keras.backend.clear_session()


GPU_AVAILABLE = BACKEND_DEPS_AVAILABLE and bool(tf and tf.config.list_physical_devices("GPU")) and bool(
    torch and torch.cuda.is_available()
)


@unittest.skipUnless(BACKEND_DEPS_AVAILABLE, "TensorFlow, PyTorch, and NumPy are required for backend parity tests")
class VerifyNetParityTests(unittest.TestCase):
    """Unit tests ensuring backend parity between TensorFlow and PyTorch."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.harness = VerifyNetHarness(dataset_size=6)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.harness.close()

    def test_pairwise_predictions_align(self) -> None:
        for anchor, sample in self.harness.iter_pairs():
            tf_score = self.harness.tf_predict(anchor, sample)
            torch_score = self.harness.torch_predict(anchor, sample)
            self.assertAlmostEqual(tf_score, torch_score, delta=1e-4)

    def test_batch_predictions_align(self) -> None:
        tf_scores = self.harness.tf_batch_predict()
        torch_scores = self.harness.torch_batch_predict()
        np.testing.assert_allclose(tf_scores, torch_scores, atol=1e-5)


@unittest.skipUnless(BACKEND_DEPS_AVAILABLE, "TensorFlow, PyTorch, and NumPy are required for performance benchmarks")
class VerifyNetBenchmarkTests(unittest.TestCase):
    """Micro-benchmarks comparing backend throughput."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.harness = VerifyNetHarness(dataset_size=8)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.harness.close()

    def test_cpu_throughput_is_positive(self) -> None:
        tf_throughput, torch_throughput = self.harness.measure_throughput(repeats=3)
        self.assertGreater(tf_throughput, 0.0)
        self.assertGreater(torch_throughput, 0.0)

    @unittest.skipUnless(GPU_AVAILABLE, "Skipping GPU benchmark: GPU devices are unavailable")
    def test_gpu_throughput_is_positive(self) -> None:
        gpu_harness = VerifyNetHarness(dataset_size=4, device="cuda")
        try:
            tf_throughput, torch_throughput = gpu_harness.measure_throughput(repeats=2)
        finally:
            gpu_harness.close()
        self.assertGreater(tf_throughput, 0.0)
        self.assertGreater(torch_throughput, 0.0)


if __name__ == "__main__":  # pragma: no cover - manual invocation helper
    unittest.main(verbosity=2)
