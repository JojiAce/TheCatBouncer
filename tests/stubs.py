"""Test stubs for legacy pipeline checks."""


class DummyEngine:
    """Minimal inference engine used for tests."""

    def __init__(self):
        self.predictions = []

    def predict(self, image):
        self.predictions.append(image)
        return [[0.5]]
