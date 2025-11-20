
names = {
    "linear": "Linear",
    "population": "Population Readout",
    "population_encoding": "Population Encoding",
    "preferred_value": "Preferred Value",
    "softmax_gaussian": "Softmax Gaussian",
    "mnist": "MNIST",
    "cifar10": "CIFAR10",
    "mexican_hat": "Mexican Hat",
    "avg_loss": "Loss (Normalized)",
    "noi.fgsm.mean": "FGSM Mean (Normalized)",
    "noi.fgsm.std": "FGSM Std (Normalized)",
    "sha.layers.0.weight": "Weight Sharpness Normalized",
    "sha.layers.0.bias": "Bias Sharpness Normalized",
    "sha.layers.2.weight": "Weight Sharpness Normalized",
    "sha.layers.2.bias": "Bias Sharpness Normalized",
    "rub.fsa_inf.mean": rf"FSA $\infty$ Mean",
    "rub.fsa_inf.std": rf"FSA $\infty$ Std",
    "rub.fsa_2.mean": rf"FSA $2$ Mean",
    "rub.fsa_2.std": rf"FSA $2$ Std",
    "rub.fsd_inf.mean": rf"FSD $\infty$ Mean",
    "rub.fsd_inf.std": rf"FSD $\infty$ Std",
    "rub.fsd_2.mean": rf"FSD $2$ Mean",
    "rub.fsd_2.std": rf"FSD $2$ Std",
    "loss_norm": "Loss Normalized",
    "fsa_inf_mean_norm": "FSA $\infty$ Mean Normalized",
    "fsa_inf_mean_diff": "FSA $\infty$ Mean Difference",
}

def get_display_name(name: str):
    if "_" in name and name not in names:
        " ".join(map(lambda x: x.capitalize(), name.split("_"), ))

    if name not in names:
        return name.capitalize()
    
    return names[name]

def get_evaluation_identifier(dataset: str, stack: str):
    return f"{dataset}_evaluation_{stack}"
    

def get_evaluation_folder(identifier: str):
    return f"./experiments/{identifier}/tuning/"

def get_target_image(file: str):
    split_symbol = "/"
    if "\\" in file:
        split_symbol = "\\"

    return f"./images/{file.split(split_symbol)[-1][:-3]}.png"

def normalize_columns(df, columns: list[str]):
    global_val = df[columns].values.flatten()
    global_min = global_val.min()
    global_max = global_val.max()

    for col in columns:
        df[col]=(df[col] - global_min)/(global_max - global_min)

    return df

import torch

def to_grayscale(x):
    """
    Convert an RGB tensor to grayscale using the ITU-R BT.601 formula.
    Supports shapes (C, H, W) or (N, C, H, W).
    
    Gray = 0.299*R + 0.587*G + 0.114*B
    """
    if x.ndim == 3:  # (C, H, W)
        if x.size(0) != 3:
            raise ValueError("Expected 3 channels (RGB). Got {}".format(x.size(0)))
        r, g, b = x[0], x[1], x[2]
        gray = 0.299*r + 0.587*g + 0.114*b
        return gray.unsqueeze(0)  # return shape (1, H, W)

    elif x.ndim == 4:  # (N, C, H, W)
        if x.size(1) != 3:
            raise ValueError("Expected 3 channels (RGB). Got {}".format(x.size(1)))
        r = x[:, 0, :, :]
        g = x[:, 1, :, :]
        b = x[:, 2, :, :]
        gray = 0.299*r + 0.587*g + 0.114*b
        return gray.unsqueeze(1)  # return shape (N, 1, H, W)

    else:
        raise ValueError("Input must have 3 or 4 dimensions.")
    
def to_grayscale_flat(x):
    gray = to_grayscale(x)
    return gray.view(gray.size(0), -1) if gray.ndim == 4 else gray.view(-1)

