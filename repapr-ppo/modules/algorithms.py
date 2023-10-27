from abc import ABC, abstractmethod
import numpy as np

class Algorithm(ABC):
    """ Algorithm super class """
    def __init__(self, tones: int, theta_k_values=None) -> None:
        self.tones: int = tones
        if theta_k_values is not None:
            self.theta_k_values = theta_k_values

    @abstractmethod
    def calc(self) -> np.ndarray[float]:
        """ calc theta_k abstract method """

class All0(Algorithm):
    """ All 0 Algorithm """
    def calc(self) -> np.ndarray[float]:
        theta_k_values: np.ndarray[float] = np.zeros(self.tones)
        return np.array(theta_k_values, dtype='float16')

class Narahashi(Algorithm):
    """ Narahashi Algorithm """
    def calc(self) -> np.ndarray[float]:
        indexes: np.ndarray = np.arange(self.tones)
        theta_k_values: np.ndarray[float] = ((indexes) * (indexes - 1) * np.pi) / (self.tones - 1)
        return np.array(theta_k_values, dtype='float16')

class Newman(Algorithm):
    """ Newman Algorithm """
    def calc(self) -> np.ndarray[float]:
        indexes: np.NDArray = np.arange(self.tones)
        theta_k_values: np.ndarray[float] = ((indexes - 1)**2 * np.pi) / (self.tones)
        return np.array(theta_k_values, dtype='float16')

class Random(Algorithm):
    """ Random theta_k_values """
    def calc(self) -> np.ndarray[float]:
        theta_k_values: np.ndarray[float] = 2*np.pi*np.random.rand(self.tones)
        return np.array(theta_k_values, dtype='float16')

class Manual(Algorithm):
    """ Manual theta_k_values """
    def calc(self) -> np.ndarray[float]:
        return np.array(self.theta_k_values, dtype='float16')


class AContext:
    """ Algorithm Context """
    def __init__(self, strategy: Algorithm) -> None:
        self._strategy = strategy
        self.theta_k_values: np.ndarray[float] = None

    def calc_algo(self) -> np.ndarray[float]:
        """ Calculation each algorithm """
        if self.theta_k_values is None:
            self.theta_k_values = self._strategy.calc()
        return self.theta_k_values
