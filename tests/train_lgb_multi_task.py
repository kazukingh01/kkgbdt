import numpy as np
from kkgbdt.model import KkGBDT
from kkgbdt.loss import MultiTaskLoss, CrossEntropyLoss, CrossEntropyNDCGLoss, MultiTaskEvalLoss


if __name__ == "__main__":
    """
    Basicaly, it is not recommended to use multi task training.
    """
    n_class, n_data = 12, 1000
    train_x = np.random.rand(n_data, 100)
    train_y = np.random.rand(n_data, n_class)
    valid_x = np.random.rand(n_data, 100)
    valid_y = np.random.rand(n_data, n_class)
    model   = KkGBDT(n_class, mode="lgb", subsample=0.5)
    loss_func      = MultiTaskLoss([CrossEntropyLoss(n_classes=6), CrossEntropyNDCGLoss(n_classes=n_class-6),])
    loss_func_eval = [
        MultiTaskEvalLoss(CrossEntropyNDCGLoss(n_classes=6),         [i for i in range(0, 6)]),
        MultiTaskEvalLoss(CrossEntropyNDCGLoss(n_classes=n_class-6), [i for i in range(n_class-6,n_class)]),
    ]
    model.fit(
        train_x, train_y, loss_func=loss_func, num_iterations=20,
        x_valid=valid_x, y_valid=valid_y, loss_func_eval=loss_func_eval,
        early_stopping_rounds=None, early_stopping_idx=0, 
    )
    print(model.predict(valid_x, is_softmax=False))
