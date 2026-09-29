import numpy as np
from scipy.special import softmax
from .base import Criterion
from .activations import LogSoftmax


class MSELoss(Criterion):
    """
    Mean squared error criterion
    """
    def compute_output(self, input: np.ndarray, target: np.ndarray) -> float:
        """
        :param input: array of size (batch_size, *)
        :param target:  array of size (batch_size, *)
        :return: loss value
        """
        assert input.shape == target.shape, 'input and target shapes not matching'
        return np.mean((input - target) ** 2)

    def compute_grad_input(self, input: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        :param input: array of size (batch_size, *)
        :param target:  array of size (batch_size, *)
        :return: array of size (batch_size, *)
        """
        assert input.shape == target.shape, 'input and target shapes not matching'
        return 2.0 * (input - target) / input.size


class CrossEntropyLoss(Criterion):
    """
    Cross-entropy criterion over distribution logits
    """
    def __init__(self, label_smoothing: float = 0.0):
        super().__init__()
        self.log_softmax = LogSoftmax()
        self.label_smoothing = label_smoothing

    def compute_output(self, input: np.ndarray, target: np.ndarray) -> float:
        """
        :param input: logits array of size (batch_size, num_classes)
        :param target: labels array of size (batch_size, )
        :return: loss value
        """
        batch_size, num_classes = input.shape
        log_probs = self.log_softmax.compute_output(input)

        q = np.full_like(input, self.label_smoothing / num_classes)
        q[np.arange(batch_size), target] += 1.0 - self.label_smoothing

        return -np.sum(q * log_probs) / batch_size

    def compute_grad_input(self, input: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        :param input: logits array of size (batch_size, num_classes)
        :param target: labels array of size (batch_size, )
        :return: array of size (batch_size, num_classes)
        """
        batch_size, num_classes = input.shape
        log_probs = self.log_softmax.compute_output(input)
        p = np.exp(log_probs)

        q = np.full_like(input, self.label_smoothing / num_classes)
        q[np.arange(batch_size), target] += 1.0 - self.label_smoothing

        return (p - q) / batch_size
