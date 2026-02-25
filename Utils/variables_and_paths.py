from pathlib import Path

TQDM_BAR_FORMAT = "{l_bar}{bar:10}{r_bar}{bar:-10b}"
MODELS = ["ViT-B-32", "ViT-B-16", "ViT-L-14"]
OPENCLIP_CACHEDIR = Path(Path.home(), "openclip-cachedir", "open_clip").as_posix()
CACHEDIR = None

ALL_DATASETS = [
    "SUN397",
    "Cars",
    "RESISC45",
    "EuroSAT",
    "SVHN",
    "GTSRB",
    "MNIST",
    "DTD", # 8
    "CIFAR100",
    "STL10",
    "Flowers102",
    "OxfordIIITPet",
    "PCAM",
    "FER2013", # 14
    "Food101",
    "FashionMNIST",
    "RenderedSST2",
    "CIFAR10",
    "EMNIST",
    "KMNIST", # 20
    "Weather",
    "Vegetables",
    "MangoLeafBD",
    "Landscapes",
    "Beans",
    "IntelImages",
    "Garbage",
    "Kvasir",
    "KenyanFood13",
    "Dogs" # 30
]

ALL_IA3_DATASETS = [
    "rte",
    "cb",
    "winogrande",
    "wic",
    "wsc",
    "copa",
    "h-swag",
    "story_cloze",
    "anli-r1",
    "anli-r2",
    "anli-r3",
]

ALL_NLPFFT_DATASETS = [
    "paws",
    "qasc",
    "quartz",
    "story_cloze",
    "wiki_qa",
    "winogrande",
    "wsc"
]

def cleanup_dataset_name(dataset_name: str):
    return dataset_name.replace("Val", "") + "Val"

def get_finetuned_path(root, dataset, model):
    return Path(root, model, cleanup_dataset_name(dataset), f"nonlinear_finetuned.pt").as_posix()

def get_zeroshot_path(root, dataset, model):
    return Path(root, model, cleanup_dataset_name(dataset), f"nonlinear_zeroshot.pt").as_posix()