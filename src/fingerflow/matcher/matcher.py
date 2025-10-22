from typing import Any, Iterable, Sequence

from .verify_net import VerifyNet
from . import BaseMatcher


class Matcher(BaseMatcher):
    def __init__(self, precision, verify_net_path):
        self.__verification_module = VerifyNet(precision, verify_net_path)

    def verify(self, anchor: Any, sample: Any) -> Any:
        return self.__verification_module.verify_fingerprints(anchor, sample)

    def verify_batch(self, pairs: Iterable[Sequence[Any]]) -> Any:
        return self.__verification_module.verify_fingerprints_batch(pairs)

    def plot_model(self, file_path: str) -> None:
        self.__verification_module.plot_model(file_path)
