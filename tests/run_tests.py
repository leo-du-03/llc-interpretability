"""
Runs all tests
"""
from tests.test_palindrome import palindromesTests
from tests.test_fractok import fractokTests
from tests.test_peak import peakTests
from tests.test_haiku_to_pytorch import haikuToPytorchTests
from tests.test_pytorch_models import testLoadingGrokkingPeakModels

if __name__ == "__main__":
    print("Running Palindrome Tests")
    palindromesTests()
    print("\nRunning Peak Tests")
    peakTests()
    print("\nRunning Fractok Tests")
    fractokTests()
    print("\nTesting Haiku to PyTorch conversion")
    haikuToPytorchTests()
    print("\nTesting peak grokking models")
    testLoadingGrokkingPeakModels()
