from torch_py.FaceRec import Recognition
from torch_py.MobileNetV1 import MobileNetV1
from torchvision.models.mobilenetv2 import MobileNetV2
from torchvision.models.mobilenetv3 import MobileNetV3
from torch_py.MTCNN.detector import FaceDetector
from torch_py.Utils import plot_image
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
import torch.optim as optim
import torch.nn as nn
import torch
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import copy
import numpy as np
from PIL import Image
import cv2
import warnings
# 忽视警告
warnings.filterwarnings('ignore')


def letterbox_image(image, size):
    """
    调整图片尺寸
    :param image: 用于训练的图片
    :param size: 需要调整到网络输入的图片尺寸
    :return: 返回经过调整的图片
    """
    new_image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    return new_image


def processing_data(data_path, height=224, width=224, batch_size=32,
                    test_split=0.1):
    """
    数据处理部分
    :param data_path: 数据路径
    :param height:高度
    :param width: 宽度
    :param batch_size: 每次读取图片的数量
    :param test_split: 测试集划分比例
    :return: 
    """
    transforms = T.Compose([
        T.Resize((height, width)),
        T.RandomHorizontalFlip(0.1),  # 进行随机水平翻转
        # T.RandomVerticalFlip(0.1),  # 进行随机竖直翻转
        T.ToTensor(),  # 转化为张量
        T.Normalize([0], [1]),  # 归一化
    ])

    dataset = ImageFolder(data_path, transform=transforms)
    # 划分数据集
    train_size = int((1-test_split)*len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size])
    # 创建一个 DataLoader 对象
    train_data_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    valid_data_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True)
    # print(dataset.shape, train_data_loader.shape)
    return train_data_loader, valid_data_loader


def show_tensor_img(img_tensor):
    img = img_tensor[0].data.numpy()
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 0, 1)
    img = np.array(img)
    plot_image(img)


if __name__ == "__main__":

    data_path = './datasets/5f680a696ec9b83bb0037081-momodel/data/image'
    pnet_path = "./torch_py/MTCNN/weights/pnet.npy"
    rnet_path = "./torch_py/MTCNN/weights/rnet.npy"
    onet_path = "./torch_py/MTCNN/weights/onet.npy"

    torch.set_num_threads(1)

    # 加载 MobileNet 的预训练模型权
    train_height = 40
    train_width = 40
    train_batch_size = 32

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    train_data_loader, valid_data_loader = processing_data(
        data_path=data_path, 
        height=train_height, 
        width=train_width, 
        batch_size=train_batch_size
    )
    modify_x, modify_y = torch.ones((train_batch_size, 3, train_height, train_width)), torch.ones(train_batch_size)

    epochs = 1000
    model = MobileNetV1(num_classes=2).to(device)
    # model = MobileNetV2(num_classes=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)  # 优化器
    # optimizer = optim.SGD(model.parameters(), lr=1e-3)  # 优化器

    print('加载完成...')

    # 学习率下降的方式，acc三次不下降就下降学习率继续训练，衰减学习率
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                     'max',
                                                     factor=0.5,
                                                     patience=3)
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    # train = True
    train = False
    if train:
        # 训练模型
        best_loss = 1e9
        best_model_weights = copy.deepcopy(model.state_dict())
        loss_list = []  # 存储损失函数值
        for epoch in range(epochs):
            model.train()

            loss = 0

            for batch_idx, (x, y) in tqdm(enumerate(train_data_loader, 1)):
                x = x.to(device)
                y = y.to(device)
                pred_y = model(x)

                # print(pred_y.shape)
                # print(y.shape)
                print(pred_y, y)

                loss = criterion(pred_y, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if loss < best_loss:
                    best_model_weights = copy.deepcopy(model.state_dict())
                    best_loss = loss

                # loss_list.append(loss.cpu().detach().numpy())

            print('step:' + str(epoch + 1) + '/' +
                  str(epochs) + ' || Total Loss: %.4f' % loss)

        torch.save(model.state_dict(), './results/temp.pth', _use_new_zipfile_serialization=False)
        print('Finish Training.')

    # print(loss_list)
    # plt.plot(loss_list, label="loss")
    # plt.legend()
    # plt.show()

    # 检测图片中人数及戴口罩的人数
    img = Image.open("test1.jpg")
    detector = FaceDetector()
    recognize = Recognition(model_path='results/001.pth')

    draw, all_num, mask_nums = recognize.mask_recognize(img)
    plt.figure(figsize=(15, 15))
    plt.imshow(draw)
    # plt.show()
    plt.savefig("result.png")
    print("all_num:", all_num, "mask_num", mask_nums)

    # mask_num = 16
    # fig = plt.figure(figsize=(15, 15))
    # for i in range(mask_num):
    #     sub_img = Image.open(data_path + "/mask/mask_" + str(i + 101) + ".jpg")
    #     draw, all_num, mask_nums = recognize.mask_recognize(sub_img)
    #     ax = fig.add_subplot(4, 4, (i + 1))
    #     ax.set_xticks([])
    #     ax.set_yticks([])
    #     ax.set_title("mask_" + str(i + 1))
    #     ax.imshow(draw)
    # plt.show()