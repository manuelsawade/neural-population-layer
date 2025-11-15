
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
    "sha.layers.0.weight": "Hidden Weight Sharpness (Norm)",
    "sha.layers.0.bias": "Hidden Bias Sharpness (Norm)",
    "sha.layers.2.weight": "Output Weight Sharpness (Norm)",
    "sha.layers.2.bias": "Output Bias Sharpness (Norm)",
    "rub.fsa_inf.mean": rf"FSA $\infty$ Mean",
    "rub.fsa_inf.std": rf"FSA $\infty$ Std",
    "rub.fsa_2.mean": rf"FSA $2$ Mean",
    "rub.fsa_2.std": rf"FSA $2$ Std",
    "rub.fsd_inf.mean": rf"FSD $\infty$ Mean",
    "rub.fsd_inf.std": rf"FSD $\infty$ Std",
    "rub.fsd_2.mean": rf"FSD $2$ Mean",
    "rub.fsd_2.std": rf"FSD $2$ Std",
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