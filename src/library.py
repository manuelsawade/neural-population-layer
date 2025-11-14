
names = {
    "linear": "Linear",
    "population": "Population Readout",
    "population_encoding": "Population Encoding",
    "preferred_value": "Preferred Value",
    "softmax_gaussian": "Softmax Gaussian",
    "mnist": "MNIST",
    "cifar10": "CIFAR10",
    "mexican_hat": "Mexican Hat"
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