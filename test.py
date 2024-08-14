import os
os.chdir("/clifford-group-equivariant-neural-networks")

import torch
from torch import nn, optim
import torch.nn.functional as F
import engineer
import yaml
import argparse
import ast
import inspect
import os
import sys
import typing
from engineer.utils.load_module import load_module
import os
import ipdb
import socket
import tempfile
from typing import Any, Callable
from dotenv import load_dotenv

import torch
import torch.distributed as dist

from engineer.utils.seed import set_seed


argv = "o3.py -C configs/engineer/trainer.yaml -C configs/optimizer/adam.yaml -C configs/dataset/o3.yaml -C configs/model/o3_cgmlp.yaml --trainer.max_steps=131072 --trainer.val_check_interval=1024 --dataset.batch_size=128 --dataset.num_samples=65536 --model.hidden_features=96 --model.num_layers=4 --optimizer.lr=0.001"
argv = argv.split()
for i in range(len(argv)):
        if argv[i].startswith("--_"):
            argv[i] = argv[i].split("=", maxsplit=1)[1]
argv = [v for v_ in argv for v in v_.replace("'", "").split(" ")]

yamls = []
i = 0
while i < len(argv):
    if argv[i] == "-C":
        yamls.append(argv[i + 1])
        del argv[i]
        del argv[i]
    else:
        i += 1

print(yamls)

config_dict = {}
for y in yamls:
    with open(y, "r") as f:
        config = yaml.load(f, yaml.SafeLoader)
        if config is not None:
            config_dict = config_dict | config

print(config_dict)

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)

def string_to_list(s):
    # Try to convert the string directly
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        # If that fails, assume the string represents a list of strings without quotes
        s = s.strip("[]").split(",")
        s = "[{}]".format(", ".join("'{}'".format(i.strip()) for i in s))
        return ast.literal_eval(s)


def try_literal_eval(v):
    if v.startswith("[") and v.endswith("]"):
        return string_to_list(v)

    try:
        return ast.literal_eval(v)
    except ValueError:
        return v
    except SyntaxError:
        return v


def get_type(v):
    if isinstance(v, bool):
        t = lambda x: (str(x).lower() == "true")
    else:
        t = type(v)
    return t


def pretty(d, indent=0):
    for k, v in d.items():
        if isinstance(v, dict):
            print("  " * indent + k)
            pretty(v, indent + 1)
        else:
            print("  " * indent + f"{k}: {v}")


def unflatten(dictionary, sep="."):
    result = dict()
    for k, v in dictionary.items():
        parts = k.split(sep)
        d = result
        for part in parts[:-1]:
            if part not in d:
                d[part] = dict()
            d = d[part]
        d[parts[-1]] = v
    return result


def check_optional(obj: typing.Any) -> bool:
    return (
        typing.get_origin(obj) is typing.Union
        and len(typing.get_args(obj)) == 2
        and typing.get_args(obj)[1] == type(None)
    )


def get_default_args(func):
    signature = inspect.signature(func)
    types = typing.get_type_hints(func)
    arguments = {}

    for k, v in signature.parameters.items():
        if v.default is inspect.Parameter.empty:
            continue

        if k in types and check_optional(types[k]):
            arguments[k] = None
        else:
            arguments[k] = v.default

    return arguments


def get_run_name(argv: list[str]):
    name_parts: list[str] = []
    for v in argv:
        if v.startswith("-C"):
            v = v[3:]
        if v.startswith("--"):
            name_parts.append(v[2:])
        elif os.path.exists(v):
            name_parts.append(os.path.splitext(os.path.basename(v))[0])

    return "_".join(name_parts)

for k in config_dict:
    if "module" in config_dict[k].keys():
        module = load_module(config_dict[k]["module"])
        argspec = get_default_args(module.__init__)
        group = parser.add_argument_group(k)
        v = config_dict[k].pop("module")

        group.add_argument(
            f"--{k}.module", default=v, type=str, help=f"Default: {v}"
        )
        for k_, v in argspec.items():
            if k_ in config_dict[k]:
                v = config_dict[k].pop(k_)

            if v.__class__.__module__ == "builtins":
                if v is None or isinstance(v, list):
                    t = try_literal_eval
                else:
                    t = get_type(v)

                group.add_argument(
                    f"--{k}.{k_}", default=v, type=t, help=f"Default: {v}"
                )

        if len(config_dict[k]) > 0:
            raise KeyError(
                f"Got unknown keys for {k} config: {tuple(config_dict[k].keys())}."
            )
    else:
        raise KeyError(f"Got unknown key: {k}.")
args = parser.parse_args(argv[1:])

config = unflatten(vars(args))

print("\nConfiguration\n---")
pretty(config)

name = get_run_name(sys.argv[1:])
experiment = os.path.splitext(os.path.basename(sys.argv[0]))[0]

USE_DISTRIBUTED = "NCCL_SYNC_FILE" in os.environ or "TORCHELASTIC_RUN_ID" in os.environ

def _ddp_setup():
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        raise ValueError("Cannot initialize NCCL without visible CUDA devices.")

    hostname = socket.gethostname()
    print(f"Setting up DDP on {hostname}.")
    if "TORCHELASTIC_RUN_ID" in os.environ:
        print("TorchElastic detected.")
        _setup = _setup_torchelastic
    elif "NCCL_SYNC_FILE" in os.environ:
        print("Detected NCCL_SYNC_FILE. Assuming SLURM cluster.")
        _setup = _setup_slurm
    else:
        raise ValueError("Unable to detect DDP setup.")

    rank, local_rank, world_size = _setup()

    print(
        f"{hostname} ready! Rank: {rank}. Local rank: {local_rank}. World size: {world_size}."
    )
    devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    device = f"cuda:{int(devices[local_rank])}"
    torch.cuda.set_device(device)

    assert dist.is_initialized()

    return {
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "device": device,
    }

def main(config):
    dataset_config = config["dataset"]
    dataset = engineer.load_module(dataset_config.pop("module"))(**dataset_config)

    train_loader = dataset.train_loader()
    val_loader = dataset.val_loader()
    test_loader = dataset.test_loader()

    model_config = config["model"]
    model_module = engineer.load_module(model_config.pop("module"))
    model = model_module(**model_config)

    model = model.cuda()

    optimizer_config = config["optimizer"]
    optimizer = engineer.load_module(optimizer_config.pop("module"))(
        model.parameters(), **optimizer_config
    )

    trainer_module = engineer.load_module(config["trainer"].pop("module"))
    trainer_config = config["trainer"]
    scheduler = None
    trainer_config["scheduler"] = scheduler
    trainer = trainer_module(
        **trainer_config,
    )

    trainer.fit(model, optimizer, train_loader, val_loader, test_loader)

# Load environment variables from .env file
load_dotenv()

# Access the DATAROOT variable
dataroot = os.getenv("DATAROOT")


seed = config["seed"]

assert isinstance(seed, int)
seed = set_seed(seed)
tempdir = tempfile.TemporaryDirectory()

dist_cfg = None
if USE_DISTRIBUTED:
    dist_cfg = _ddp_setup()
config["dist"] = dist_cfg

main(config)

tempdir.cleanup()
if dist.is_initialized():
    dist.destroy_process_group()