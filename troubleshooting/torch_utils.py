#!/usr/bin/env python
"""
A bunch of pytorch utils that are helpful for performing regression on data
"""
import numpy as np
import torch


def render_dataloaders(x, y, validation_fraction, train_batch_size, validate_batch_size):
    """
    aggregate all data into dataloaders

    arguments
        x : torch.tensor
            input data
        y : torch.tensor
            output data; same shape as x
        validation_fraction : float or None
            float fraction between 0 and 1 that extracts a subset of x, y for validation;
            if None, there is no validation
        train_batch_size : int
            batch size of training set
        validate_batch_size : int
            batch size of validation set;
            unused if validate_fraction is None
    returns
        train_dataloader : torch.utils.data.DataLoader
            training dataloader
        validation_dataloader : torch.utils.data.DataLoader or None
            validation dataloader
    """
    import math
    from torch.utils.data import TensorDataset, DataLoader
    assert len(x) == len(y)
    datapoints = len(x)
    if validation_fraction is not None:
        assert validation_fraction <= 1. and validation_fraction>0.
        validation_datapoints = math.ceil(datapoints*validation_fraction)
        if validation_datapoints == datapoints: validation_datapoints-=1

        #randomly split data
        validation_indices = np.random.choice(datapoints, size=validation_datapoints)
        x_validation = x[validation_indices]
        y_validation = y[validation_indices]
        train_indices = list(set(range(datapoints)).difference(set(validation_indices)))
        x_train, y_train = x[train_indices], y[train_indices]
    else:
        x_train, y_train = x, y
        x_validation, y_validation = None, None

    #make the dataloaders
    train_dataset = TensorDataset(x_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size = train_batch_size)
    if x_validation is not None:
        validate_dataset = TensorDataset(x_validation, y_validation)
        validation_dataloader = DataLoader(validate_dataset, batch_size = validate_batch_size)
    else:
        validation_dataloader = None

    return train_dataloader, validation_dataloader

def check_models(models, model_aggregate_function, optimizers):
    import inspect
    if type(models) == list:
        composite_model = True
        assert callable(model_aggregate_function)
        assert type(optimizers) == list
        assert len(optimizers) == len(models)
        assert all(callable(model) for model in models)
    else:
        assert callable(models)
        composite_model = False
    return composite_model



def loss_batch(models, loss_function, model_aggregate_function, x_batch, y_batch, optimizers, model_aggregate_function_kwargs):
    """
    perform a single loss update
    """
    composite_model = check_models(models, model_aggregate_function, optimizers)
    if composite_model:
        modeller = model_aggregate_function(models, x_batch, **model_aggregate_function_kwargs)
    else:
        modeller = models(x_batch)

    loss = loss_function(modeller, y_batch)
    loss.backward()
    if composite_model:
        for opt in optimizers:
            opt.step()
            opt.zero_grad()
    else:
        optimizers.step()
        optimizers.zero_grad()

    return loss.item(), len(x_batch)




def fit(epochs, models, loss_function, model_aggregate_function, optimizers, train_dataloader, validation_dataloader, model_aggregate_function_kwargs):
    """
    perform regression procedure
    """
    training_losses = []
    validation_losses = [] if validation_dataloader is not None else None
    composite_model = check_models(models, model_aggregate_function, optimizers)

    for epoch in range(epochs):
        if composite_model:
            for model in models: model.train()
        else:
            models.train()
        train_losses, nums = zip(*[loss_batch(models, loss_function, model_aggregate_function, xb, yb, optimizers, model_aggregate_function_kwargs) for xb, yb in train_dataloader])
        train_loss = np.sum(np.multiply(train_losses, nums)) / np.sum(nums)
        training_losses.append(train_loss)

        if composite_model:
            for model in models: model.eval()
        else:
            models.eval()

        if validation_dataloader is not None:
            with torch.no_grad():
                valid_losses, nums = zip(*[loss_batch(models, loss_function, model_aggregate_function, xb, yb, optimizers, model_aggregate_function_kwargs) for xb, yb in validation_dataloader])
            valid_loss = np.sum(np.multiply(valid_losses, nums)) / np.sum(nums)
            validation_losses.append(valid_loss)
    return training_losses, validation_losses

def batch_quadratic(v, M):
    """
    compute batch v.T * M * v
    """
    v_shape, M_shape = list(v.size()), list(M.size())
    assert v_shape[0] == M_shape[0]
    assert v_shape[1] == M_shape[1]
    assert M_shape[2] == M_shape[1]
    assert len(v_shape) == 2
    assert len(M_shape) == 3
    bs, dimension = v_shape[0], v_shape[1]
    return torch.matmul(v.view(bs, dimension, 1).transpose(1,2), torch.matmul(M, v.view(bs, dimension, 1))).squeeze()

def batch_dot(v1, v2):
    assert v1.size() == v2.size() and len(list(v1.size())) == 2
    return torch.sum(v1*v2, dim=1)
