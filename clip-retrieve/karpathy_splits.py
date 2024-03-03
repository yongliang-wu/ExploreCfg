import os

class COCOKSDataset():
    def __init__(self):
        self.train_data_dir = "/data/wyl/coco_data/train2014"
        self.val_data_dir = "/data/wyl/coco_data/val2014"

        self.train_name = []
        with open('../karpathy-splits/coco_train.txt', 'r', encoding='utf-8') as f:
            self.train_name = [i.strip("\n") for i in f.readlines()]
        with open('../karpathy-splits/coco_restval.txt', 'r', encoding='utf-8') as f:
            self.train_name.extend([i.strip("\n") for i in f.readlines()])      
        print(len(self.train_name))

        self.train_data = {}
        for tn in self.train_name:
            self.train_data[self.name2id(tn)] = os.path.join(self.train_data_dir, tn) if tn.find("train") != -1 else os.path.join(self.val_data_dir, tn)

        self.test_name = []
        with open('../karpathy-splits/coco_test.txt', 'r', encoding='utf-8') as f:
            self.test_name = [i.strip("\n") for i in f.readlines()]
        print(len(self.test_name))

        self.test_data = {}
        for tn in self.test_name:
            self.test_data[self.name2id(tn)] = os.path.join(self.val_data_dir, tn)

        self.val_name = []
        with open('../karpathy-splits/coco_val.txt', 'r', encoding='utf-8') as f:
            self.val_name = [i.strip("\n") for i in f.readlines()]
        print(len(self.val_name))

        self.val_data = {}
        for tn in self.val_name:
            self.val_data[self.name2id(tn)] = os.path.join(self.val_data_dir, tn)

    def name2id(self, name):
        return int(name.split("_")[2].split(".")[0])

    def get_train_data(self):
        return self.train_data

    def get_test_data(self):
        return self.test_data
    
    def get_val_data(self):
        return self.val_data