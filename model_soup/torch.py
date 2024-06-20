import inspect
import os
import sys
import time
import numpy as np
import torch
from copy import deepcopy

def uniform_soup(model, path, device="cpu", by_name=False):
    if not isinstance(path, list):
        path = [path]
    model = model.to(device)
    model_dict = model.state_dict()
    soups = {key: [] for key in model_dict}

    for model_path in path:
        weight = torch.load(model_path, map_location=device)
        weight_dict = weight['state_dict']
   
        if by_name:
            weight_dict = {k: v for k, v in weight_dict.items() if k in model_dict}
        for k, v in weight_dict.items():
            if by_name:
                if k in model_dict:
                    soups[k].append(v)
            else:
                soups[k[6:]].append(v)

    if soups:
        soups = {
            k: (torch.sum(torch.stack(v), axis=0) / len(v)).type(v[0].dtype)
            for k, v in soups.items() if v  # Ensure there are values to process
        }
        model_dict.update(soups)
        model.load_state_dict(model_dict)
    return model




def greedy_souping(model, paths, metric, device="cpu", digits=4, verbose=True):
    # Ensure paths is a list
    if not isinstance(paths, list):
        paths = [paths]

    # Load the initial model to the specified device
    best_model = deepcopy(model)
    best_model.to(device)
    soup_size = 1

    # Move the model to the same device as the best model
    for model_path in paths[1:]:
        soup_model = deepcopy(best_model)
        soup_model.to(device)
        soup_state_dict = soup_model.state_dict()

        checkpoint = torch.load(model_path, map_location=device)
        checkpoint_state_dict = checkpoint["state_dict"]
        checkpoint_state_dict = {k.replace('model.', ''): v for k, v in checkpoint_state_dict.items()}

        # Ensure all tensors are on the same device
        for key in checkpoint_state_dict.keys():
            checkpoint_state_dict[key] = checkpoint_state_dict[key].to(device)

        # Try adding the model's weights to the soup
        for key in soup_state_dict.keys():
            soup_state_dict[key] *= soup_size
            soup_state_dict[key] += checkpoint_state_dict[key]

        # Average the weights
        for key in soup_state_dict.keys():
            soup_state_dict[key] /= (soup_size + 1)

        soup_model.load_state_dict(soup_state_dict)
        print("Metrics for the current soup:")
        metric_val = metric(soup_model)

        print("Metrics for the best model:")

        # If performance improves, keep the new soup
        if metric_val > metric(best_model):
            best_model = deepcopy(soup_model)
            soup_size += 1


    return best_model
