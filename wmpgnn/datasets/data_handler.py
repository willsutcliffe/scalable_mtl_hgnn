from wmpgnn.datasets.graph_dataset import CustomDataset
from wmpgnn.datasets.hetero_graph_dataset import CustomHeteroDataset
from torch_geometric.loader import DataLoader
import glob



class DataHandler:
    """ Class which loads data using an appropriate
        dataset class """

    def __init__(self, config):
        self.config_loader = config
        data_path = self.config_loader.get("dataset.data_dir")
        data_type = self.config_loader.get("dataset.data_type")
        self.batch_size =  self.config_loader.get("training.batch_size")
        files_input_tr = sorted(glob.glob(f'{data_path}/training_dataset/input_*'))
        files_target_tr = sorted(glob.glob(f'{data_path}/training_dataset/target_*'))
        files_input_vl = sorted(glob.glob(f'{data_path}/validation_dataset/input_*'))
        files_target_vl = sorted(glob.glob(f'{data_path}/validation_dataset/target_*'))

        if data_type == "homogeneous":
            self.train_dataset = CustomDataset(files_input_tr, files_target_tr)
            self.val_dataset = CustomDataset(files_input_vl, files_target_vl)
        elif data_type == "heterogeneous":
            self.train_dataset = CustomHeteroDataset(files_input_tr, files_target_tr)
            self.val_dataset = CustomHeteroDataset(files_input_vl, files_target_vl)
        else:
            raise Exception(f"Unexpected data type {data_type}. Please use homogeneous or heterogeneous.")

    def load_data(self):
        self.dataset_tr = self.train_dataset.get()
        self.dataset_vl = self.val_dataset.get()

    def get_train_dataloader(self):
        return DataLoader(self.dataset_tr, batch_size=self.batch_size, drop_last=True)

    def get_val_dataloader(self, batch_size=32):
        return DataLoader(self.dataset_vl, batch_size=self.batch_size, drop_last=True)