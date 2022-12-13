import torch
import numpy as np
from train_model import LSTMNet, evaluate_accuracy, MyDataset


def predict(test_x):
    """
    对于给定的 x 预测未来的 y 。
    :param test_x: 给定的数据集合 x ，对于其中的每一个元素需要预测对应的 y
    :return: test_y 对于每一个 test_x 中的元素，给出一个对应的预测值
    """
    # test 的数目
    n_test = test_x.shape[0]

    test_y = None
    # --------------------------- 此处下方加入读入模型和预测相关代码 -------------------------------
    # 此处为 Notebook 模型示范，你可以根据自己数据处理方式进行改动
    # scaler = MinMaxScaler().fit(np.array([0, 300]).reshape(-1, 1))
    # test_x = scaler.transform(test_x.reshape(-1, 1)).reshape(-1, 14)
    test_x = torch.tensor(test_x, dtype=torch.float32)

    test_y = model(test_x)

    # 如果使用 MinMaxScaler 进行数据处理，预测后应使用下一句将预测值放缩到原范围内
    # test_y = scaler.inverse_transform(test_y.detach().cpu())
    test_y = test_y.detach().cpu().numpy()
    # --------------------------- 此处上方加入读入模型和预测相关代码 -------------------------------

    # 保证输出的是一个 numpy 数组
    assert (type(test_y) == np.ndarray)

    # 保证 test_y 的 shape 正确
    assert (test_y.shape == (n_test, 1))

    return test_y


if __name__ == '__main__':
    # 指定目录
    model_path = 'results/mymodel.pt'
    # 选用使用的模型类
    model = LSTMNet(14, 128, 1)
    # 读入对应的参数
    model.load_state_dict(torch.load(model_path))
    #
    model.eval()
    # 测试用例
    model_test_x = np.array([[6.69, 6.72, 6.52, 6.66, 6.74, 6.55, 6.35, 6.14, 6.18, 6.17, 5.72, 5.78, 5.69, 5.67],
                             [6.69, 6.72, 6.52, 6.66, 6.74, 6.55, 6.35, 6.14, 6.18, 6.17, 5.72, 5.78, 5.69, 5.67]])

    print(predict(test_x=model_test_x))