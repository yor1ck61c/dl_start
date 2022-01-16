import numpy as np
import h5py
 
def load_dataset():
    train_dataset = h5py.File('dl_homework/lesson_1/week_2/datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels
    test_dataset = h5py.File('dl_homework/lesson_1/week_2/datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels
    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes
if __name__ == '__main__':
    train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes=load_dataset()
    print('训练样本数={}'.format(train_set_x_orig.shape))
    print('训练样本对应的标签={}'.format(train_set_y_orig.shape))
    print('前10张训练样本标签={}'.format(train_set_y_orig[:,:10]))
    print('测试样本数={}'.format(test_set_x_orig.shape))
    print('测试样本对应的标签={}'.format(test_set_y_orig.shape))
    print('{}'.format(classes))

    