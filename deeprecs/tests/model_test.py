# pylint: disable=import-error
from typing import Tuple

import numpy as np
import pytest
import pytest_lazyfixture
from torch.utils.data import DataLoader

from deeprecs.models.base import BaseRecommender
from deeprecs.utils.loss import RMSELoss


@pytest.mark.parametrize(
    "model, data",
    [
        (
            pytest_lazyfixture.lazy_fixture("autorec"),
            pytest_lazyfixture.lazy_fixture("ml100k"),
        )
    ],
)
def test_ml100k(model: BaseRecommender, data: Tuple[DataLoader, DataLoader]):
    """
    movielens 100k 데이터에 대해서 모델의 예측값이 잘 나오는지 확인하는 테스트함수

    Parameters
    ----------
    model : BaseRecommender
        추천 모델
    data : Tuple[DataLoader, DataLoader]
        학습/실험 데이터셋
    """
    model = model(input_dim=1682, hidden_dim=128)
    train_loader, test_loader = data

    model.fit(train_loader, epochs=1)
    pred = model.predict(test_loader)
    pred = np.clip(pred, 1, 5)

    rmse = RMSELoss()(test_loader.dataset.y, pred)
    # https://paperswithcode.com/sota/collaborative-filtering-on-movielens-100k
    # 기준으로 가장 높은 rmse는 0.996
    assert rmse <= 1.15
