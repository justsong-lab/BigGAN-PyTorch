import torch
from torch.autograd import Variable
import os, errno
import numpy as np
from scipy import linalg


import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
from itertools import repeat, cycle

def get_face_loaders(batch_size=64, workers=2, dataset='facescrub100', labels_per_class=112):
    class_num = 100 
    n = class_num * labels_per_class 
    data_dir = 'facescrub100/'
    loaders = []
    from collections import defaultdict
    import pickle
    def load_face_files(filenames):
        if isinstance(filenames, str):
            filenames = [filenames]
        images = []
        labels = []
        for fn in filenames:
            f = os.path.join(data_dir, fn)
            dataset = np.load(f)
            print(dataset)
        
        trainx = dataset['trainx']
        trainy = dataset['trainy']
        testx = dataset['testx']
        testy = dataset['testy']

        return trainx, trainy, testx, testy

    train_data, train_labels, test_data, test_labels = load_face_files('facescrub100_64.npz')
    
    print(train_data.shape)
    criteria = n // class_num
    print(criteria)
    input_dict, labelled_x, labelled_y = defaultdict(int), list(), list()

    for image, label in zip(train_data, train_labels) :
        if input_dict[int(label)] != labels_per_class :      
            labelled_x.append(image)
            labelled_y.append(label)
            input_dict[int(label)] += 1
        else:
            continue
        if len(labelled_x) == 11199:
            labelled_x.append(image)
            labelled_y.append(label)
            break

        # unlabelled_x.append(image)
        # unlabelled_y.append(label)

    # print(len(labelled_x))
    # print(len(labelled_y))
    # print(len(unlabelled_x))
    # print(len(unlabelled_y))
    labelled_x = np.asarray(labelled_x)
    labelled_y = np.asarray(labelled_y)
    # unlabelled_x = np.asarray(unlabelled_x)
    # unlabelled_y = np.asarray(unlabelled_y)
    
    indices = np.random.permutation(len(labelled_x))
    labelled_x = labelled_x[indices]
    labelled_y = labelled_y[indices]

    # indices = np.random.permutation(len(unlabelled_x))
    # unlabelled_x = unlabelled_x[indices]
    # unlabelled_y = unlabelled_y[indices]

    labelled_y_vec = np.zeros((len(labelled_y), class_num), dtype=np.float)
    for i, label in enumerate(labelled_y) :
        labelled_y_vec[i, labelled_y[i]] = 1.0

    # unlabelled_y_vec = np.zeros((len(unlabelled_y), class_num), dtype=np.float)
    # for i, label in enumerate(unlabelled_y) :
        # unlabelled_y_vec[i, unlabelled_y[i]] = 1.0

    test_labels_vec = np.zeros((len(test_labels), class_num), dtype=np.float)
    for i, label in enumerate(test_labels) :
        test_labels_vec[i, test_labels[i]] = 1.0

    labelled_x = torch.from_numpy(labelled_x).permute(0, 3, 1, 2)
    labelled_y_vec = torch.from_numpy(labelled_y_vec).argmax(dim=1)
    # unlabelled_x = torch.from_numpy(unlabelled_x).permute(0, 3, 1, 2)
    # unlabelled_y_vec = torch.from_numpy(unlabelled_y_vec).argmax(dim=1)
    test_data = torch.from_numpy(test_data).permute(0, 3, 1, 2)
    test_labels_vec = torch.from_numpy(test_labels_vec).argmax(dim=1)

    labelled_dataset = torch.utils.data.TensorDataset(labelled_x, labelled_y_vec)
    # unlabelled_dataset = torch.utils.data.TensorDataset(unlabelled_x, unlabelled_y_vec)
    test_dataset = torch.utils.data.TensorDataset(test_data, test_labels_vec)

    labelled = torch.utils.data.DataLoader(labelled_dataset, batch_size=batch_size,  num_workers=workers, pin_memory=True)
    # validation = None
    # unlabelled = torch.utils.data.DataLoader(unlabelled_dataset, batch_size=batch_size, num_workers=workers, pin_memory=True)
    test = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)

    print(labelled_y_vec.shape)
    # print(unlabelled_y_vec.shape)
    print(test_labels_vec.shape)

    loaders.append(labelled)
    # loaders.append(unlabelled)
    loaders.append(test)
    return loaders


def get_face_loaders_200(batch_size=100, workers=8, dataset='facescrub100', labels_per_class=128):
    class_num = 100
    n = class_num * labels_per_class 
    data_dir = 'facescrub100/'
    loaders = []
    from collections import defaultdict
    import pickle
    def load_face_files(filenames):
        if isinstance(filenames, str):
            filenames = [filenames]
        images = []
        labels = []
        for fn in filenames:
            f = os.path.join(data_dir, fn)
            dataset = np.load(f)
            print(dataset)
        
        trainx = dataset['trainx']
        trainy = dataset['trainy']
        testx = dataset['testx']
        testy = dataset['testy']

        return trainx, trainy, testx, testy

    train_data, train_labels, test_data, test_labels = load_face_files('facescrub100_64.npz')
    
    print(train_data.shape)
    criteria = n // class_num
    input_dict, labelled_x, labelled_y, unlabelled_x, unlabelled_y = defaultdict(int), list(), list(), list(), list()

    for image, label in zip(train_data, train_labels) :
        if input_dict[int(label)] != criteria :
            input_dict[int(label)] += 1
            labelled_x.append(image)
            labelled_y.append(label)

        unlabelled_x.append(image)
        unlabelled_y.append(label)


    labelled_x = np.asarray(labelled_x)
    labelled_y = np.asarray(labelled_y)
    unlabelled_x = np.asarray(unlabelled_x)
    unlabelled_y = np.asarray(unlabelled_y)
    
    indices = np.random.permutation(len(labelled_x))
    labelled_x = labelled_x[indices]
    labelled_y = labelled_y[indices]

    indices = np.random.permutation(len(unlabelled_x))
    unlabelled_x = unlabelled_x[indices]
    unlabelled_y = unlabelled_y[indices]

    labelled_y_vec = np.zeros((len(labelled_y), class_num), dtype=np.float)
    for i, label in enumerate(labelled_y) :
        labelled_y_vec[i, labelled_y[i]] = 1.0

    unlabelled_y_vec = np.zeros((len(unlabelled_y), class_num), dtype=np.float)
    for i, label in enumerate(unlabelled_y) :
        unlabelled_y_vec[i, unlabelled_y[i]] = 1.0

    test_labels_vec = np.zeros((len(test_labels), class_num), dtype=np.float)
    for i, label in enumerate(test_labels) :
        test_labels_vec[i, test_labels[i]] = 1.0

    labelled_x = torch.from_numpy(labelled_x).permute(0, 3, 1, 2)
    labelled_y_vec = torch.from_numpy(labelled_y_vec).argmax(dim=1)
    unlabelled_x = torch.from_numpy(unlabelled_x).permute(0, 3, 1, 2)
    unlabelled_y_vec = torch.from_numpy(unlabelled_y_vec).argmax(dim=1)
    test_data = torch.from_numpy(test_data).permute(0, 3, 1, 2)
    test_labels_vec = torch.from_numpy(test_labels_vec).argmax(dim=1)

    labelled_dataset = torch.utils.data.TensorDataset(labelled_x, labelled_y_vec)
    unlabelled_dataset = torch.utils.data.TensorDataset(unlabelled_x, unlabelled_y_vec)
    test_dataset = torch.utils.data.TensorDataset(test_data, test_labels_vec)

    labelled = torch.utils.data.DataLoader(labelled_dataset, batch_size=batch_size,  num_workers=workers, pin_memory=True)
    validation = None
    unlabelled = torch.utils.data.DataLoader(unlabelled_dataset, batch_size=batch_size, num_workers=workers, pin_memory=True)
    test = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)

    print(labelled_y_vec.shape)
    print(unlabelled_y_vec.shape)
    print(test_labels_vec.shape)

    loaders.append(labelled)
    loaders.append(unlabelled)
    loaders.append(test)
    return loaders


    # print(len(labelled_x))
    # print(len(labelled_y))



def get_face_loaders_semi(batch_size=100, workers=8, dataset='facescrub100', labels_per_class=128):
    class_num = 100
    n = class_num * labels_per_class 
    data_dir = 'facescrub100/'
    loaders = []
    from collections import defaultdict
    import pickle
    def load_face_files(filenames):
        if isinstance(filenames, str):
            filenames = [filenames]
        images = []
        labels = []
        for fn in filenames:
            f = os.path.join(data_dir, fn)
            dataset = np.load(f)
            print(dataset)
        
        trainx = dataset['trainx']
        trainy = dataset['trainy']
        testx = dataset['testx']
        testy = dataset['testy']

        return trainx, trainy, testx, testy

    train_data, train_labels, test_data, test_labels = load_face_files('facescrub100_64.npz')
    
    print(train_data.shape)
    criteria = n // class_num
    input_dict, labelled_x, labelled_y, unlabelled_x, unlabelled_y = defaultdict(int), list(), list(), list(), list()

    for image, label in zip(train_data, train_labels) :
        if input_dict[int(label)] != criteria :
            input_dict[int(label)] += 1
            labelled_x.append(image)
            labelled_y.append(label)

        unlabelled_x.append(image)
        unlabelled_y.append(label)


    labelled_x = np.asarray(labelled_x)
    labelled_y = np.asarray(labelled_y)
    unlabelled_x = np.asarray(unlabelled_x)
    unlabelled_y = np.asarray(unlabelled_y)
    
    indices = np.random.permutation(len(labelled_x))
    labelled_x = labelled_x[indices]
    labelled_y = labelled_y[indices]

    indices = np.random.permutation(len(unlabelled_x))
    unlabelled_x = unlabelled_x[indices]
    unlabelled_y = unlabelled_y[indices]

    labelled_y_vec = np.zeros((len(labelled_y), class_num), dtype=np.float)
    for i, label in enumerate(labelled_y) :
        labelled_y_vec[i, labelled_y[i]] = 1.0

    unlabelled_y_vec = np.zeros((len(unlabelled_y), class_num), dtype=np.float)
    for i, label in enumerate(unlabelled_y) :
        unlabelled_y_vec[i, unlabelled_y[i]] = 1.0

    test_labels_vec = np.zeros((len(test_labels), class_num), dtype=np.float)
    for i, label in enumerate(test_labels) :
        test_labels_vec[i, test_labels[i]] = 1.0

    labelled_x = torch.from_numpy(labelled_x).permute(0, 3, 1, 2)
    labelled_y_vec = torch.from_numpy(labelled_y_vec).argmax(dim=1)
    unlabelled_x = torch.from_numpy(unlabelled_x).permute(0, 3, 1, 2)
    unlabelled_y_vec = torch.from_numpy(unlabelled_y_vec).argmax(dim=1)
    test_data = torch.from_numpy(test_data).permute(0, 3, 1, 2)
    test_labels_vec = torch.from_numpy(test_labels_vec).argmax(dim=1)

    labelled_dataset = torch.utils.data.TensorDataset(labelled_x, labelled_y_vec)
    unlabelled_dataset = torch.utils.data.TensorDataset(unlabelled_x, unlabelled_y_vec)
    test_dataset = torch.utils.data.TensorDataset(test_data, test_labels_vec)

    labelled = torch.utils.data.DataLoader(labelled_dataset, batch_size=batch_size,  num_workers=workers, pin_memory=True)
    validation = None
    unlabelled = torch.utils.data.DataLoader(unlabelled_dataset, batch_size=batch_size, num_workers=workers, pin_memory=True)
    test = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)

    print(labelled_y_vec.shape)
    print(unlabelled_y_vec.shape)
    print(test_labels_vec.shape)

    loaders.append(labelled)
    loaders.append(unlabelled)
    loaders.append(test)
    return loaders

