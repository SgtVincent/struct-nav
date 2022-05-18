from detection.base_detector import BaseDetector


class GroundTruthDetecor(BaseDetector):
    def __init__(self, args) -> None:
        super().__init__()  # dummy parent init function

    def detect(self, sem_img):
        pass
