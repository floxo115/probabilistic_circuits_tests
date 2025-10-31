import pandas as pd

import pyjuice as juice
from pyjuice.optim import CircuitOptimizer, CircuitScheduler

import torch
from torch.utils.data import DataLoader, TensorDataset

from fusinter_python_implementation import FUSINTERDiscretizer

import configparser

config_parser = configparser.ConfigParser()
config_parser.read("config.ini")

options = {
    "SEED": config_parser.getint("setup", "SEED"),
    "n_epochs": config_parser.getint("setup", "n_epochs"),
    "train_proportion": config_parser.getfloat("setup", "train_proportion"),
    "lr": config_parser.getfloat("setup", "lr"),
    "pseudocount": config_parser.getfloat("setup", "pseudocount"),
}

torch.random.manual_seed(options["SEED"])

data = pd.read_csv("./data/hepatitis.csv")

data = data.drop(columns=["protime", "alk_phosphate"]).dropna()

categorical = [
    #'age', 
    'sex', 
    'steroid', 
    'antivirals', 
    'fatigue', 
    'malaise', 
    'anorexia',
    'liver_big', 
    'liver_firm', 
    'spleen_palpable', 
    'spiders', 
    'ascites',
    'varices', 
    #'bilirubin', 
    #'sgot', 
    #'albumin', 
    'histology', 
    'class'
]

numerical = list(data.columns.difference(categorical))
data[categorical] = data[categorical].astype("category")
data = data.apply(lambda x: x.cat.codes if x.dtype == "category" else x)

indices = torch.randperm(data.shape[0])
cut_idx = int(len(indices) * 0.6)
train_idx = indices[:cut_idx]
x_nums = data[numerical].to_numpy()
y = data["class"].to_numpy()

for col_i, col_name in enumerate(numerical):
    discretizer = FUSINTERDiscretizer()
    discretizer.computed_splits = discretizer.fit(x_nums[train_idx, col_i], y[train_idx])
    data[col_name] = discretizer.transform(x_nums[:, col_i])
    

num_cats = data.max(axis=0)+1

data_np = data.to_numpy(dtype=float)

train_tensor = torch.tensor(data_np[train_idx, :], dtype=torch.long)
# y_tensor = torch.tensor(data_np[train_idx, -2:-1])
train_data = TensorDataset(train_tensor)

test_idx = indices[cut_idx:]
val_tensor = torch.tensor(data_np[test_idx, :], dtype=torch.long)
# y_tensor = torch.tensor(data_np[test_idx, -2:-1])
val_data = TensorDataset(val_tensor)

train_loader = DataLoader(train_data, shuffle=True, batch_size=512)
val_loader = DataLoader(val_data, shuffle=True, batch_size=512)

device = (
    torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
)

inputs = []
for i, is_num in enumerate(data.columns):
    dist = juice.distributions.Categorical(num_cats=num_cats[i])
    inputs.append(
        juice.inputs(
            i, num_node_blocks=8, dist=dist
        )
    )

prods = [juice.multiply(*inputs) for _ in range(3)]

ns = juice.summate(*prods, num_node_blocks=1)
ns.init_parameters()

pc = juice.compile(ns)
pc.to(device)



def train_model(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_epochs=500,
    lr=0.01,
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

model = train_model(pc, train_loader,val_loader)

def get_test_score(model, val_loader):
    acc = 0.0
    n_samples = 0
    for batch in val_loader:
        n_samples += batch[0].shape[0]
        batch = batch[0].to(device)
        missing_mask = torch.tensor(
            [False] * 17 + [True]
        ).to(device)

        outputs = conditional(
            model, data=batch, missing_mask=missing_mask, target_vars=[17]
        )
        preds = outputs.argmax(dim=2).flatten()
        truth = batch[:, 17]
        acc += (preds == truth).sum().item()
    return acc / n_samples

from pyjuice.queries import conditional, sample

acc = get_test_score(model, val_loader)

print("accuracy for hepatitis test set:", acc)



