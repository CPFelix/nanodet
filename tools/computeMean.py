# https://zhuanlan.zhihu.com/p/275742390
import torch
import torchvision
import torchvision.datasets as Datasat
from tqdm import tqdm

def getStat(train_data):
    '''
    Compute mean and variance for training data
    :param train_data: 自定义类Dataset(或ImageFolder即可)
    :return: (mean, std)
    '''
    print('Compute mean and variance for training data.')
    print(len(train_data))
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=1, shuffle=False, num_workers=0,
        pin_memory=True)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for X, _ in tqdm(train_loader):
        for d in range(3):
            mean[d] += X[:, d, :, :].mean()
            std[d] += X[:, d, :, :].std()
    mean.div_(len(train_data))
    std.div_(len(train_data))
    return list(mean.numpy() * 255.0), list(std.numpy() * 255.0)

if __name__ == "__main__":
    # config_file = "config/nanodet-plus-m_320X192_frame-16.yml"
    # load_config(cfg, config_file)
    # train_dataset = build_dataset(cfg.data.train, "train")
    train_dataset = Datasat.ImageFolder(root="/home/chenpengfei/dataset/DSMhand_smoke3/yolov5/images", transform=torchvision.transforms.ToTensor())
    print(getStat(train_dataset))