import numpy as np

from spreadnn.model import SpreadModel


class TestSpreadModel:
    def test_loads_bundled_model(self):
        model = SpreadModel()
        # Force load — should not raise
        model._load_model()
        assert model._net is not None

    def test_score_pair_different_heights(self):
        """Pages with different heights should be resized before scoring."""
        model = SpreadModel()
        img_e = np.zeros((400, 300, 3), dtype=np.uint8)
        img_o = np.zeros((500, 300, 3), dtype=np.uint8)
        prob = model.score_pair(img_e, img_o)
        assert 0.0 <= prob <= 1.0
