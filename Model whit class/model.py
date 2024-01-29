# Model Base Class
from abc import ABC, abstractmethod
from pathlib import Path
from helper import logger
from sklearn.neighbors import KNeighborsClassifier


class Model(ABC):
    def __init__(self, name: str):
        self.name = name
        self.__is_trained = False
        self.model = None

    def __is_train(self):
        if self.__is_trained:
            return True
        else:
            logger.error(f'{self} is not trained yet')
            return False

    @abstractmethod
    def train(self, data, label, test_size=0.2):
        pass

    @abstractmethod
    def predict(self,):
        pass

    @abstractmethod
    def save(self, path=None):
        pass

    @abstractmethod
    def load(self, path=None):
        pass

    def __str__(self):
        return self.name


class KNN(Model):
    def __init__(self, name='KNN'):
        super().__init__(name)

    def train(self, data=None, label=None, test_size=0.2):
        if not self.__is_train():
            logger.info(f'{self} training')
            self.model = KNeighborsClassifier()
            try:
                self.model.fit(data, label)
                self.__is_trained = True
            except Exception as e:
                logger.error(f'{self} training failed: {e}')
                return False
        else:
            logger.info(f"{self} already trained")

        return self.model

