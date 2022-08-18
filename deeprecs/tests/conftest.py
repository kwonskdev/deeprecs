# pylint: disable=import-error, unused-variable
import pandas as pd
import pytest
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from deeprecs.data.data import AEDataset
from deeprecs.models.autorec import AutoEncoder


@pytest.fixture
def autorec():
    """
    pytest에서 사용하게 될 AutoRec
    """

    def wrapper(input_dim: int, hidden_dim: int) -> AutoEncoder:
        """
        AutoRec의 추천모델을 반환하는 함수

        Parameters
        ----------
        input_dim : int
            AutoRec의 input dimension
        hidden_dim : int
            AutoRec의 hidden dimension

        Returns
        -------
        AutoEncoder
            AutoRec의 추천모델

        Notes
        -----
        데이터마다 input/hidden dimension이 다르므로,
        각 test마다 input/hidden을 입력하기 위해 wrapper 사용
        """
        autoencoder = AutoEncoder(input_dim=input_dim, hidden_dim=hidden_dim)
        return autoencoder

    return wrapper


@pytest.fixture
def ml100k():
    """
    pytest에서 모델 최저 성능 보장을 위해 사용할 MovieLens 100k data
    """
    ml = pd.read_csv("./data/ml-100k/ml-100k_pivot.csv", index_col=0)
    ml = ml.fillna(0)
    train, test = map(
        AEDataset,
        train_test_split(
            torch.Tensor(ml.values), test_size=0.1, random_state=42
        ),
    )
    train_loader = DataLoader(train, batch_size=32)
    test_loader = DataLoader(test, batch_size=len(test))

    return (train_loader, test_loader)
