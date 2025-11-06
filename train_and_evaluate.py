import random
from typing import Tuple
import numpy as np
import configparser

import pandas as pd
import pyjuice as juice
from pyjuice.optim import CircuitOptimizer
from pyjuice.queries import conditional
import torch
from torch.utils.data import TensorDataset, DataLoader


def get_datasets_and_dataloaders(
    fn: str, train_set_proportion=0.8
) -> Tuple[torch.Tensor, torch.Tensor, DataLoader, DataLoader]:
    if not 0 < train_set_proportion < 1:
        raise ValueError("test_set_proportion has to be a in (0,1)")

    data = pd.read_csv(fn, dtype=int)

    # create indices for splitting train and test set
    data_np = data.to_numpy(dtype=int)
    indices = torch.randperm(data.shape[0])
    cut_idx = int(len(indices) * train_set_proportion)

    # create train set as TensorDataset
    train_idx = indices[:cut_idx]
    train_tensor = torch.tensor(data_np[train_idx, :], dtype=torch.long)
    train_data = TensorDataset(train_tensor)

    # create test set as TensorDataset
    test_idx = indices[cut_idx:]
    val_tensor = torch.tensor(data_np[test_idx, :], dtype=torch.long)
    val_data = TensorDataset(val_tensor)

    # create DataLoaders from the DataSets
    train_loader = DataLoader(train_data, shuffle=True, batch_size=512)
    val_loader = DataLoader(val_data, shuffle=True, batch_size=512)

    return train_tensor, val_tensor, train_loader, val_loader


def get_test_score(model, val_loader):
    """computes accuracy of the given model on the test set"""

    acc = 0.0
    n_samples = 0
    for batch in val_loader:
        # get samples of current batch and add them to the overall count
        n_samples += batch[0].shape[0]
        n_variables = batch[0].shape[1]
        batch = batch[0].to(device)
        truth = batch[:, -1].clone().to(device)
        batch[:, -1] = -100
        # print(batch.shape)
        # each known variable is set to False
        missing_mask = torch.zeros_like(batch, dtype=torch.bool).to(device)
        missing_mask[:, -1] = 1
        # print(missing_mask)
        # print(missing_mask.shape)
        # print(batch.shape)
        # print(batch)
        # compute P(class| x_1...x_10)
        outputs = conditional(
            model,
            data=batch,
            missing_mask=missing_mask,
            target_vars=[n_variables - 1],
        )

        # get class with maximum probability
        preds = outputs.argmax(dim=2).flatten()
        # truth = batch[:, -1]

        # compute correct count accuracy
        acc += (preds == truth).sum().item()

    # compute accuracy
    return acc / n_samples


# def get_test_score(model, val_loader):
#     """computes accuracy of the given model on the test set"""
#
#     acc = 0.0
#     n_samples = 0
#     for batch in val_loader:
#         # get samples of current batch and add them to the overall count
#         n_samples += batch[0].shape[0]
#         batch = batch[0].to(device)
#
#         # each known variable is set to False
#         missing_mask = torch.tensor(
#             [False, False, False, False, False, False, False, False, False, False, True]
#         ).to(device)
#
#         # compute P(class| x_1...x_10)
#         outputs = conditional(
#             model, data=batch, missing_mask=missing_mask, target_vars=[10]
#         )
#
#         # get class with maximum probability
#         preds = outputs.argmax(dim=2).flatten()
#         truth = batch[:, 10]
#
#         # compute correct count accuracy
#         acc += (preds == truth).sum().item()
#
#     # compute accuracy
#     return acc / n_samples
#     get samples of current batch and add them to the overall count


def train_model(
    model,
    train_loader: DataLoader,
    n_epochs=500,
    lr=1,
    pseudocount=0.1,
):
    optimizer = CircuitOptimizer(model, lr=lr, pseudocount=pseudocount, method="EM")

    for batch in train_loader:
        x = batch[0].to(device)

        lls = pc(x, record_cudagraph=True)
        lls.mean().backward()
        break

    for epoch in range(1, n_epochs + 1):
        # t0 = time.time()
        train_ll = 0.0
        for batch in train_loader:
            x = batch[0].to(device)

            # Similar to PyTorch optimizers zeroling out the gradients, we zero out the parameter flows
            optimizer.zero_grad()

            # Forward pass
            lls = pc(x)

            # Backward pass
            lls.mean().backward()

            train_ll += lls.mean().detach().cpu().numpy().item()

            # Perform a mini-batch EM step
            optimizer.step()

        train_ll /= len(train_loader)

        print(
            f"[Epoch {epoch}/{n_epochs}][train LL: {train_ll:.2f}]",
            end="\r",
        )

    return model


if __name__ == "__main__":
    # get options from ini file
    config_parser = configparser.ConfigParser()
    config_parser.read("config.ini")
    options = {
        "SEED": config_parser.getint("setup", "SEED"),
        "n_epochs": config_parser.getint("setup", "n_epochs"),
        "train_proportion": config_parser.getfloat("setup", "train_proportion"),
        "lr": config_parser.getfloat("setup", "lr"),
        "pseudocount": config_parser.getfloat("setup", "pseudocount"),
    }

    # get all artificial datasets from the data folder
    from glob import glob
    from re import match

    fns = glob("./data/*.csv")
    fns = [fn for fn in fns if match(r"./data/\d+.csv", fn) is not None]
    results = pd.DataFrame(
        data={
            "dataset": [],
            "test_accuracy": [],
        }
    )

    # loop over all datasets
    for fn in fns[:]:
        print(f"run training and testing for {fn}")

        # set random seed in loop so that the order does not affect the results
        torch.random.manual_seed(options["SEED"])
        random.seed(options["SEED"])
        np.random.seed(options["SEED"])

        # get dataloaders and sets
        train_set, val_set, train_dl, val_dl = get_datasets_and_dataloaders(
            fn, options["train_proportion"]
        )

        # define the model
        device = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )

        inputs = [
            juice.inputs(
                i, num_node_blocks=8, dist=juice.distributions.Categorical(num_cats=2)
            )
            for i in range(11)
        ]

        prods = [juice.multiply(*inputs) for _ in range(3)]

        ns = juice.summate(*prods, num_node_blocks=1)
        ns.init_parameters()

        pc = juice.compile(ns)
        pc.to(device)

        # train the model
        model = train_model(
            pc,
            train_dl,
            n_epochs=options["n_epochs"],
            lr=options["lr"],
            pseudocount=options["pseudocount"],
        )

        # compute score from test set
        acc = get_test_score(model, val_dl)

        # put results into result dataframe
        out_name = fn[fn.rfind("/") + 1 : -4]
        results.loc[len(results)] = [out_name, acc]  # , dist, p_val]

    # sort dataframe according to name and save it as csv
    results["dataset"] = results["dataset"].astype(int)
    results = results.sort_values(by="dataset")
    print(results)
    results.to_csv("results_for_own_model_on_artificial_datasets.csv", index=False)
