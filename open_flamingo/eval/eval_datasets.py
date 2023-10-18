import json
import os

from PIL import Image
from torch.utils.data import Dataset

class COCOTrainDataset(Dataset):
    def __init__(
        self,
        image_dir_path="coco_data/train2014",
        val_image_dir_path="coco_data/val2014",
        annotations_path="coco_data/annotations/captions_train2014.json",
        val_annotations_path="coco_data/annotations/captions_val2014.json",
        train_split_path = 'karpathy-splits/coco_train.txt',
        train_split_path_res = 'karpathy-splits/coco_restval.txt',
        WC_captions_path="MGC/wc_vis_80.json",
        WC_best_gt_path="MGCA/best_gt_WC(80).json",
    ):
        self.image_dir_path = image_dir_path
        self.val_image_dir_path = val_image_dir_path
        # use karpathy split train 113287
        self.train_names = []
        with open(train_split_path, 'r', encoding='utf-8') as f:
            self.train_names = [i.strip("\n") for i in f.readlines()]
        with open(train_split_path_res, 'r', encoding='utf-8') as f:
            self.train_names.extend([i.strip("\n") for i in f.readlines()])      
        print(len(self.train_names))

        self.imgs = {}
        for tn in self.train_names:
            image_id = int(tn.split("_")[2].split(".")[0])
            self.imgs[image_id] = {"image_id": image_id,
                                   "captions": [],
                                   "image": os.path.join(self.image_dir_path, tn) if tn.find("train") != -1 else os.path.join(val_image_dir_path, tn)}
            
        # add annotation
        self.annotations = json.load(open(annotations_path, "r"))["annotations"]
        self.annotations.extend(json.load(open(val_annotations_path, "r"))["annotations"])
        for ann in self.annotations:
            if self.imgs.get(ann["image_id"]):
                self.imgs[ann["image_id"]]["captions"].append(ann["caption"])

        # add caption generate from image caption model
        self.WC_captions = json.load(open(WC_captions_path, "r"))
        for wc in self.WC_captions:
            if self.imgs.get(wc['image_id']):
                self.imgs[wc['image_id']].update({"WC_captions": [wc["caption"]]})

        # add gt select form WC IP
        self.WC_best_gts = json.load(open(WC_best_gt_path, "r"))
        for wc_gt in self.WC_best_gts:
            if self.imgs.get(wc_gt['image_id']):
                self.imgs[wc_gt['image_id']].update({"WC_gt_idx": wc_gt["gt_idx"]})
            
        self.images = list(self.imgs.values())

    def __len__(self):
        return len(self.images)

    def id2item(self, idx):
        image = Image.open(self.imgs[idx]["image"])
        return {
            "image": image,
            "captions": self.imgs[idx]["captions"],
            "image_id": self.imgs[idx]["image_id"],
            "WC_captions": self.imgs[idx]["WC_captions"],
            "WC_gt_idx": self.imgs[idx]["WC_gt_idx"],
        }

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]["image"])
        return {
            "image": image,
            "captions": self.images[idx]["captions"],
            "image_id": self.images[idx]["image_id"],
            "WC_captions": self.images[idx]["WC_captions"],
            "WC_gt_idx": self.images[idx]["WC_gt_idx"],
        }
    
class COCOTestDataset(Dataset):
    def __init__(
        self,
        image_dir_path="coco_data/val2014",
        annotations_path="coco_data/annotations/captions_val2014.json",
        test_split_path = 'karpathy-splits/coco_test.txt',
        clip_ids_path = "train_set_clip.json",
    ):
        self.image_dir_path = image_dir_path
        self.annotations_path = annotations_path
        self.images = {}
        # use karpathy split 5000
        self.test_names = []
        with open(test_split_path, 'r', encoding='utf-8') as f:
            self.test_names = [i.strip("\n") for i in f.readlines()] 
        print(len(self.test_names))

        self.imgs = {}
        clip_json = json.load(open(clip_ids_path, "r"))
        for tn in self.test_names:
            image_id = int(tn.split("_")[2].split(".")[0])
            self.imgs[image_id] = {"image_id": image_id,
                                   "captions": [],
                                   "image": os.path.join(image_dir_path, tn),
                                    "clip_image_ids": clip_json[str(image_id)],
                                    }
    
        self.annotations = json.load(open(annotations_path, "r"))["annotations"]
        for ann in self.annotations:
            if self.imgs.get(ann["image_id"]):
                self.imgs[ann["image_id"]]["captions"].append(ann["caption"])
            
        self.images = list(self.imgs.values())

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]["image"])
        return {
            "image": image,
            "captions": self.images[idx]["captions"],
            "image_id": self.images[idx]["image_id"],
            "clip_image_ids": self.images[idx]["clip_image_ids"],
        }