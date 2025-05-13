from wmpgnn.datasets.graph_dataset import CustomDataset
from wmpgnn.datasets.hetero_graph_dataset import CustomHeteroDataset
from wmpgnn.datasets.neutrals_hetero_graph_dataset import CustomNeutraslHeteroDataset
from torch_geometric.loader import DataLoader
import glob



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
        data_path = self.config_loader.get("dataset.data_dir")
        data_type = self.config_loader.get("dataset.data_type")
        self.batch_size =  self.config_loader.get("training.batch_size")
        files_input_tr = glob.glob(f'{data_path}/training_dataset/input_*')[:evt_max_train]
        files_target_tr = glob.glob(f'{data_path}/training_dataset/target_*')[:evt_max_train]
        files_input_vl = glob.glob(f'{data_path}/validation_dataset/input_*')[:evt_max_val]
        files_target_vl = glob.glob(f'{data_path}/validation_dataset/target_*')[:evt_max_val]
        files_input_tst = glob.glob(f'{data_path}/test_dataset/input_*')
        files_target_tst = glob.glob(f'{data_path}/test_dataset/target_*')
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
            self.train_dataset = CustomNeutralsHeteroDataset(files_input_tr, files_target_tr, performance_mode=performance_mode, n_classes=neutrals_classes)
            self.val_dataset = CustomNeutralsHeteroDataset(files_input_vl, files_target_vl, performance_mode=performance_mode, n_classes=neutrals_classes)
            self.test_dataset = CustomNeutralsHeteroDataset(files_input_tst, files_target_tst, performance_mode=performance_mode, n_classes=neutrals_classes)
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