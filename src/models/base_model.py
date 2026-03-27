from abc import ABC, abstractmethod

class BaseModel(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def fit():
        pass


    @abstractmethod
    def predict():
        pass


    @abstractmethod
    def evaluate():
        pass