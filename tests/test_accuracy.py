import unittest
import torch

from cyclegan_with_spectrogram.ai_version import calculate_accuracy


class TestCalculateAccuracy(unittest.TestCase):
    def test_binary_classification(self):
        # Logits for a binary classification problem
        logits = torch.tensor([2.0, -1.0, 0.5, -2.0])
        targets = torch.tensor([1, 0, 1, 0])  # True class labels
        accuracy = calculate_accuracy(logits, targets)
        self.assertEqual(accuracy, 1.0)

    def test_perfect_accuracy(self):
        outputs = torch.tensor([[0.1, 0.9, 0.8, 0.9]])
        targets = torch.tensor([1, 1, 1, 1])
        accuracy = calculate_accuracy(outputs, targets)
        self.assertEqual(accuracy, 1.0)

    def test_zero_accuracy(self):
        outputs = torch.tensor([[0.18, 0.2, 0.1, 0.19]])
        targets = torch.tensor([1, 1, 1, 1])
        accuracy = calculate_accuracy(outputs, targets)
        self.assertEqual(accuracy, 0.0)

    def test_half_accuracy(self):
        outputs = torch.tensor([[0.8, 0.2, 0.8, 0.2]])
        targets = torch.tensor([0, 1, 1, 0])
        accuracy = calculate_accuracy(outputs, targets)
        self.assertEqual(accuracy, 0.5)


if __name__ == '__main__':
    unittest.main()
