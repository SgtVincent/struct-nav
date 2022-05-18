from abc import ABC, abstractmethod


class BaseDetector:
    def __init__(self) -> None:
        pass

    @abstractmethod
    def detect(self):
        pass
