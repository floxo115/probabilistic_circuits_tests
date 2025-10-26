import random
from typing import Tuple
import numpy as np
import configparser

import pandas as pd
import pyjuice as juice
from pyjuice.structures import HCLT
from pyjuice.optim import CircuitOptimizer, CircuitScheduler
from pyjuice.queries import conditional, sample
import torch
from torch.utils.data import TensorDataset, DataLoader

import hyppo.ksample


def get_datasets_and_dataloaders(
    fn: str, test_set_proportion=0.8
) -> Tuple[torch.Tensor, torch.Tensor, DataLoader, DataLoader]:
    if not 0 < test_set_proportion < 1:
        raise ValueError("test_set_proportion has to be a in (0,1)")
    data = pd.read_csv(fn, dtype=int)

    data_np = data.to_numpy(dtype=int)
    indices = torch.randperm(data.shape[0])

    cut_idx = int(len(indices) * test_set_proportion)

    train_idx = indices[:cut_idx]

    train_tensor = torch.tensor(data_np[train_idx, :], dtype=torch.long)
    # y_tensor = torch.tensor(data_np[train_idx, -2:-1])
    train_data = TensorDataset(train_tensor)

    test_idx = indices[cut_idx:]
    val_tensor = torch.tensor(data_np[test_idx, :], dtype=torch.long)
    # y_tensor = torch.tensor(data_np[test_idx, -2:-1])
    val_data = TensorDataset(val_tensor)

    train_loader = DataLoader(train_data, shuffle=True, batch_size=512)
    val_loader = DataLoader(val_data, shuffle=True, batch_size=512)

    return train_tensor, val_tensor, train_loader, val_loader


def get_sample(pc, sample_size: int):
    return sample(pc, sample_size)


def get_test_score(model, val_loader):
    acc = 0.0
    n_samples = 0
    for batch in val_loader:
        n_samples += batch[0].shape[0]
        batch = batch[0].to(device)
        missing_mask = torch.tensor(
            [False, False, False, False, False, False, False, False, False, False, True]
        ).to(device)

        outputs = conditional(
            model, data=batch, missing_mask=missing_mask, target_vars=[10]
        )
        preds = outputs.argmax(dim=2).flatten()
        truth = batch[:, 10]
        acc += (preds == truth).sum().item()
    return acc / n_samples


def train_model(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_epochs=500,
    lr=1,
    pseudocount=0.1,
):
    optimizer = CircuitOptimizer(model, lr=lr, pseudocount=pseudocount, method="EM")
    scheduler = CircuitScheduler(
        optimizer,
        method="multi_linear",
        lrs=[0.9, 0.5, 0.01],
        milestone_steps=[0, len(train_loader) * 100, len(train_loader) * 350],
    )

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
            # scheduler.step()

        train_ll /= len(train_loader)

        # t1 = time.time()
        test_ll = 0.0
        for batch in val_loader:
            x = batch[0].to(pc.device)
            lls = pc(x)
            test_ll += lls.mean().detach().cpu().numpy().item()

        test_ll /= len(val_loader)
        # t2 = time.time()

        print(
            f"[Epoch {epoch}/{n_epochs}][train LL: {train_ll:.2f}; val LL: {test_ll:.2f}] ",
            end="\r",
        )

    return model


if __name__ == "__main__":
    config_parser = configparser.ConfigParser()
    config_parser.read("config.ini")

    options = {
        "SEED": config_parser.getint("setup", "SEED"),
        "n_epochs": config_parser.getint("setup", "n_epochs"),
        "train_proportion": config_parser.getfloat("setup", "train_proportion"),
        "lr": config_parser.getfloat("setup", "lr"),
        "pseudocount": config_parser.getfloat("setup", "pseudocount"),
    }

    from glob import glob
    from re import match

    fns = glob("./data/*.csv")
    fns = [fn for fn in fns if match(r"./data/\d+.csv", fn) is not None]
    results = pd.DataFrame(
        data={
            "dataset": [],
            "test_accuracy": [],
            "energy_dist_for_sample": [],
            "p_val_of_dist": [],
        }
    )
    for fn in fns[:]:
        print(f"run training and testing for {fn}")
        torch.random.manual_seed(options["SEED"])
        random.seed(options["SEED"])
        np.random.seed(options["SEED"])

        train_set, val_set, train_dl, val_dl = get_datasets_and_dataloaders(
            fn, options["train_proportion"]
        )

        device = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )

        ns = HCLT(
            train_set,
            num_latents=5,
            input_node_params={"num_cats": 2},
        )

        pc = juice.compile(ns)
        pc.to(device)

        model = train_model(
            pc,
            train_dl,
            val_dl,
            n_epochs=options["n_epochs"],
            lr=options["lr"],
            pseudocount=options["pseudocount"],
        )

        acc = get_test_score(model, val_dl)

        generated_sample = get_sample(model, 500).cpu().numpy()
        val_set = val_set.numpy()

        dist, p_val = hyppo.ksample.Energy().test(generated_sample, val_set)

        out_name = fn[fn.rfind("/") + 1 : -4]
        results.loc[len(results)] = [out_name, acc, dist, p_val]

    results["dataset"] = results["dataset"].astype(int)
    results = results.sort_values(by="dataset")
    print(results)
    results.to_csv("results_for_own_model.csv", index=False)
