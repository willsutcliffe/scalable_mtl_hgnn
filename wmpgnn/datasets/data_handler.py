from wmpgnn.datasets.graph_dataset import CustomDataset
from wmpgnn.datasets.hetero_graph_dataset import CustomHeteroDataset
from wmpgnn.datasets.neutrals_hetero_graph_dataset import CustomNeutralsHeteroDataset
from torch_geometric.loader import DataLoader
import glob
import re
import os

def natural_sort_key(s):
    """ Sort strings using a human-friendly key (e.g., input_2.npy before input_10.npy) """
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

class DataHandler:
    """ Class which loads data using an appropriate
        dataset class """

    def __init__(self, config, performance_mode=False):
        self.config_loader = config
        evt_max_train = self.config_loader.get("dataset.evt_max_train")
        evt_max_val = self.config_loader.get("dataset.evt_max_val")
        if evt_max_train is not None:
            print(f"Using {evt_max_train} events for training")
        if evt_max_val is not None:
            print(f"Using {evt_max_val} events for validation")
        self.batch_size =  self.config_loader.get("training.batch_size")
        data_path = self.config_loader.get("dataset.data_dir")
        polarity = self.config_loader.get("dataset.polarity")
        data_type = self.config_loader.get("dataset.data_type")
        if polarity == 'magall' :
            data_path_up = os.path.join(data_path, 'magup', data_type)
            data_path_down = os.path.join(data_path, 'magdown', data_type)

            # Training set: split across two folders
            files_input_tr_up_1 = sorted(glob.glob(f'{data_path_up}/training_dataset/input_*'), key=natural_sort_key)
            files_target_tr_up_1 = sorted(glob.glob(f'{data_path_up}/training_dataset/target_*'), key=natural_sort_key)
            files_input_tr_up_2 = sorted(glob.glob(f'{data_path_up}/training_dataset_2/input_*'), key=natural_sort_key)
            files_target_tr_up_2 = sorted(glob.glob(f'{data_path_up}/training_dataset_2/target_*'), key=natural_sort_key)
            size_tr_up =len(files_target_tr_up_1)+len(files_target_tr_up_2)

            files_input_tr_down_1 = sorted(glob.glob(f'{data_path_down}/training_dataset/input_*'), key=natural_sort_key)
            files_target_tr_down_1 = sorted(glob.glob(f'{data_path_down}/training_dataset/target_*'), key=natural_sort_key)
            files_input_tr_down_2 = sorted(glob.glob(f'{data_path_down}/training_dataset_2/input_*'), key=natural_sort_key)
            files_target_tr_down_2 = sorted(glob.glob(f'{data_path_down}/training_dataset_2/target_*'), key=natural_sort_key)
            size_tr_down =len(files_target_tr_down_1)+len(files_target_tr_down_2)
            
            # Combine the two lists and limit to evt_max_train
            files_input_tr = (files_input_tr_down_1 + files_input_tr_down_2+files_input_tr_up_1 + files_input_tr_up_2)[:evt_max_train]
            files_target_tr = (files_target_tr_down_1 + files_target_tr_down_2+files_target_tr_up_1 + files_target_tr_up_2)[:evt_max_train]
            
            files_input_up_vl = sorted(glob.glob(f'{data_path_up}/validation_dataset/input_*'), key=natural_sort_key)
            files_target_up_vl = sorted(glob.glob(f'{data_path_up}/validation_dataset/target_*'), key=natural_sort_key)
            size_vl_up =len(files_input_up_vl)
            files_input_down_vl = sorted(glob.glob(f'{data_path_down}/validation_dataset/input_*'), key=natural_sort_key)
            files_target_down_vl = sorted(glob.glob(f'{data_path_down}/validation_dataset/target_*'), key=natural_sort_key)
            size_vl_down =len(files_input_down_vl)

            files_input_vl = (files_input_up_vl + files_input_down_vl)[:evt_max_val]
            files_target_vl = (files_target_up_vl + files_target_down_vl)[:evt_max_val]

            files_input_tst = sorted(glob.glob(f'{data_path_up}/test_dataset/input_*'), key=natural_sort_key)
            files_target_tst = sorted(glob.glob(f'{data_path_up}/test_dataset/target_*'), key=natural_sort_key)

        elif (polarity=='magup' or polarity=='magdown' ):
            data_path = os.path.join(data_path, polarity, data_type)
            # Training set: split across two folders
            files_input_tr_1 = sorted(glob.glob(f'{data_path}/training_dataset/input_*'), key=natural_sort_key)
            files_target_tr_1 = sorted(glob.glob(f'{data_path}/training_dataset/target_*'), key=natural_sort_key)
            files_input_tr_2 = sorted(glob.glob(f'{data_path}/training_dataset_2/input_*'), key=natural_sort_key)
            files_target_tr_2 = sorted(glob.glob(f'{data_path}/training_dataset_2/target_*'), key=natural_sort_key)
            # Combine the two lists and limit to evt_max_train
            files_input_tr = (files_input_tr_1 + files_input_tr_2)[:evt_max_train]
            files_target_tr = (files_target_tr_1 + files_target_tr_2)[:evt_max_train]
            files_input_vl = sorted(glob.glob(f'{data_path}/validation_dataset/input_*'), key=natural_sort_key)[:evt_max_val]
            files_target_vl = sorted(glob.glob(f'{data_path}/validation_dataset/target_*'), key=natural_sort_key)[:evt_max_val]
            files_input_tst = sorted(glob.glob(f'{data_path}/test_dataset/input_*'), key=natural_sort_key)
            files_target_tst = sorted(glob.glob(f'{data_path}/test_dataset/target_*'), key=natural_sort_key)
        else :
            raise Exception(f"Unexpected magnet polarity {polarity}. Please use magdown, magup or magall.")

        
        if data_type == "homogeneous":
            LCA_classes = self.config_loader.get('model.LCA_classes')
            self.train_dataset = CustomDataset(files_input_tr, files_target_tr, performance_mode=performance_mode, n_classes=LCA_classes)
            self.val_dataset = CustomDataset(files_input_vl, files_target_vl, performance_mode=performance_mode, n_classes=LCA_classes)
            self.test_dataset = CustomDataset(files_input_tst, files_target_tst, performance_mode=performance_mode, n_classes=LCA_classes)
        elif data_type == "heterogeneous":
            LCA_classes = self.config_loader.get('model.LCA_classes')
            self.train_dataset = CustomHeteroDataset(files_input_tr, files_target_tr, performance_mode=performance_mode, n_classes=LCA_classes)
            self.val_dataset = CustomHeteroDataset(files_input_vl, files_target_vl, performance_mode=performance_mode, n_classes=LCA_classes)
            self.test_dataset = CustomHeteroDataset(files_input_tst, files_target_tst, performance_mode=performance_mode, n_classes=LCA_classes)
        elif data_type == "neutrals":
            neutrals_classes = self.config_loader.get('model.neutrals_classes') 
            print('creating custom datasets...')
            self.train_dataset = CustomNeutralsHeteroDataset(files_input_tr, files_target_tr, performance_mode=performance_mode, config_loader=self.config_loader, split="train", sizes=[size_tr_up, size_tr_down])
            self.val_dataset = CustomNeutralsHeteroDataset(files_input_vl, files_target_vl, performance_mode=performance_mode, config_loader=self.config_loader, split="val", sizes=[size_vl_up, size_vl_down])
            self.test_dataset = CustomNeutralsHeteroDataset(files_input_tst, files_target_tst, performance_mode=performance_mode, config_loader=self.config_loader, split="test")         
        else:
            LCA_classes = self.config_loader.get('model.LCA_classes')
            raise Exception(f"Unexpected data type {data_type}. Please use neutrals, homogeneous or heterogeneous.")

    def load_data(self):
        self.dataset_tr = self.train_dataset.get()
        self.dataset_vl = self.val_dataset.get()
        self.dataset_tst = self.test_dataset.get()

    def get_train_dataloader(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        return DataLoader(self.dataset_tr, batch_size=batch_size, drop_last=True)

    def get_val_dataloader(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        return DataLoader(self.dataset_vl, batch_size=batch_size, drop_last=True)

    def get_test_dataloader(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        return DataLoader(self.dataset_tst, batch_size=batch_size, drop_last=True)