from Utils.args_utils import get_args
from model.mtl_models import ImageEncoder
from client import MTL_Client

if __name__ == "__main__":
    args = get_args()

    epochs = {
        "Cars": 35,
        "DTD": 76,
        "EuroSAT": 12,
        "GTSRB": 11,
        "MNIST": 5,
        "RESISC45": 15,
        "SUN397": 14,
        "SVHN": 4, ##### 8
        "CIFAR10": 6,
        "CIFAR100": 6,
        "STL10": 5,
        "Food101": 4,
        "Flowers102": 147,
        "FER2013": 10,
        "PCAM": 1,
        "OxfordIIITPet": 82,
        "RenderedSST2": 39,
        "EMNIST": 2,
        "FashionMNIST": 5,
        "KMNIST": 5, ##### 20
        "Weather": 5,
        "Vegetables": 2,
        "MangoLeafBD": 3,
        "Landscapes": 1,
        "Beans": 13,
        "IntelImages": 2,
        "Garbage": 10,
        "Kvasir": 10,
        "KenyanFood13": 10,
        "Dogs": 16 ##### 30
    }

    args = get_args()
    dataset = args.DATATYPE
    args.LEARNING_RATE = 1e-5
    args.LOCAL_EPOCH = epochs[dataset]
    args_dict = dict(vars(args))

    # We use gradient accumulation to simulate larger batch sizes if the model does not fit in memory.
    args.BATCH_SIZE = 64 if args.MODEL == "ViT-L-14" else 128
    args.num_grad_accumulation = 2 if args.MODEL == "ViT-L-14" else 1

    for k, v in args_dict.items():
        print(f"{k}: {v}")

    model = ImageEncoder(args.MODEL)
    client = MTL_Client(args, 0, model.train_preprocess, model.val_preprocess, dataset)
    client.train()
