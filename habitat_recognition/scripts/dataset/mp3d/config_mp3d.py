class ConfigMP3D:
    def __init__(self) -> None:

        self.scene_dir = (
            "/media/junting/SSD_data/habitat_data/scene_datasets/mp3d/v1/scans"
        )
        self.dataset = "mp3d"

        self.ap_calculators = [0.5]  # [0.25, 0.5]
