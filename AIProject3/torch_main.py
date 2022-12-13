# from MobileNetV1 import MobileNetV1
from MobileNetV2 import MobileNetV2
from ResNet import resnet50
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
import os
import warnings
import sys
from MTCNN.detector import FaceDetector

# 忽视警告
warnings.filterwarnings('ignore')

def processing_data(data_path, height=224, width=224, batch_size=32,
                    test_split=0.2):
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
        T.RandomHorizontalFlip(0.5),  # 进行随机水平翻转
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

    return train_data_loader, valid_data_loader

def face_recognize(image):
    detector = FaceDetector()
    b_boxes, landmarks = detector.detect(image)
    if len(b_boxes) > 0:
        box = b_boxes[0]
        w, h = image.size
        if box[0] < 0: box[0] = 0
        if box[1] < 0: box[1] = 0
        if box[2] > w: box[2] = w
        if box[3] > h: box[3] = h
        face = image.crop(tuple(box[:4]))
    else:
        face = image
    face.save('face.jpg')
    return face    


if __name__ == "__main__":
    data_path = './dataset'

    torch.set_num_threads(1)

    train_height = 160
    train_width = 160
    train_batch_size = 16

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    
    train_data_loader, valid_data_loader = processing_data(
        data_path=data_path, 
        height=train_height, 
        width=train_width, 
        batch_size=train_batch_size,
        test_split=0.2
    )
    modify_x, modify_y = torch.ones((train_batch_size, 3, train_height, train_width)), torch.ones(train_batch_size)

    model = resnet50(num_classes=10)
    # model = MobileNetV1(num_classes=10)
    # model = MobileNetV2(num_classes=10)
    model_path = './results/res.pth'
    model.load_state_dict(torch.load(model_path, map_location=device))

    epochs = 100
    optimizer = optim.Adam(model.parameters(), lr=1e-5)  # 优化器
    # optimizer = optim.Adam(model.parameters(), lr=1e-3)  # 优化器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        'max',
        factor=0.5,
        patience=4
    )
    # 损失函数
    criterion = nn.CrossEntropyLoss()

    train = eval(sys.argv[1])

    best_correct = 0

    if train:
        # 训练模型
        best_loss = 1e9
        best_model_weights = copy.deepcopy(model.state_dict())
        loss_list = []  # 存储损失函数值
        for epoch in range(epochs):
            # 模型训练
            model.to(device)
            model.train()
            loss = 0
            for batch_idx, (x, y) in tqdm(enumerate(train_data_loader, 1)):
                x = x.to(device)
                y = y.to(device)
                pred_y = model(x)

                loss = criterion(pred_y, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # loss_list.append(loss.cpu().detach().numpy())

            print('step:' + str(epoch + 1) + '/' +
                  str(epochs) + ' || Total Loss: %.4f' % loss)
            
            # if (epoch + 1) % 10 == 0:
            if True:
                # 模型验证
                # model.to("cpu")
                model.eval()
                all_n = 0
                yes_n = 0
                for batch_idx, (x, y) in tqdm(enumerate(valid_data_loader, 1)):
                    x = x.to(device)
                    y = y.to(device)
                    pred_y = model(x)
                    for i, j in zip(pred_y.cpu().detach().numpy(), y.cpu().detach().numpy()):
                        if np.argmax(i) == j:
                            yes_n += 1
                        all_n += 1
                correct = round(100 * float(yes_n) / float(all_n), 2)
                print('Valid sum:', all_n, ', accuracy:', correct, "%")

                if correct > best_correct:
                    best_correct = correct
                    print("find best correct rate: ", best_correct, "%")
                    torch.save(model.state_dict(), './results/res.pth', _use_new_zipfile_serialization=False)

            print()

        print('Finish Training. Best correct rate: ', best_correct, "%")

    labels = {0: 'CL', 1: 'FBB', 2: 'HG', 3: 'HJ', 4: 'LHR', 5: 'LSS', 6: 'LYF', 7: 'PYY', 8: 'TY', 9: 'YM'}
    labels_cn = ['成龙', '范冰冰', '胡歌', '何炅', '刘昊然', '刘诗诗', '刘亦菲', '彭于晏', '唐嫣', '杨幂']

    transforms = T.Compose([
        T.Resize((train_height, train_width)),
        # T.RandomHorizontalFlip(0.1),  # 进行随机水平翻转
        # T.RandomVerticalFlip(0.1),  # 进行随机竖直翻转
        T.ToTensor(),  # 转化为张量
        T.Normalize([0], [1]),  # 归一化
    ])

    model.eval()

    if eval(sys.argv[2]): # test
        for i in labels:
            test_person = labels[i]
            test_box = []

            for filename in os.listdir(f'dataset/{test_person}'):
                img = Image.open(f'dataset/{test_person}/' + filename)
                # img = cv2.imread(f'dataset/{test_person}/' + filename)
                # img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) 

                img = face_recognize(img)
                x = transforms(img).unsqueeze(0)
                
                y = model(x).cpu().data.numpy()

                # 获取输入图片的类别
                y_predict = labels[np.argmax(y)]

                if y_predict == test_person:
                    test_box.append(1)
                else:
                    test_box.append(0)

            print(test_person, round(sum(test_box) / len(test_box), 2) * 100, '%')

    if eval(sys.argv[3]):
        img = Image.open("test.jpg")

        face = face_recognize(img)

        x = transforms(face).unsqueeze(0)
        y = model(x).cpu().data.numpy()

        print('图里的明星是:', labels_cn[np.argmax(y)])



        