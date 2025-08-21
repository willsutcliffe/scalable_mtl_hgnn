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
    """
    DataHandler orchestrates loading and batching of graph datasets for training,
    validation, and testing. Supports both homogeneous and heterogeneous graph data.

    Attributes:
        config_loader (Config): Configuration object providing dataset paths and types.
        batch_size (int): Default batch size for data loaders.
        train_dataset, val_dataset, test_dataset:
            Instances of CustomDataset or CustomHeteroDataset or CustomNeutralsHeteroDataset representing
            training, validation, and test datasets respectively.
        dataset_tr, dataset_vl, dataset_tst:
            Loaded in-memory lists of Data or HeteroData objects for
              training, validation, and testing.
    """
         
    def __init__(self, config, performance_mode=False):
        """
        Initialize the DataHandler by creating dataset instances but not yet loading data.

        Args:
            config (Config): Configuration provider with keys:
                - "dataset.data_dir": Base directory containing dataset subfolders.
                - "dataset.data_type": "homogeneous" or "heterogeneous".
                - "training.batch_size": Default batch size.
            performance_mode (bool): If True, may enable accelerated data loading or
                reduced preprocessing in dataset implementations.

        Raises:
            Exception: If `dataset.data_type` is not "homogeneous" or "heterogeneous or neutrals".
        """
        self.config_loader = config
        self.batch_size = config.get("training.batch_size")

        evt_max_train = config.get("dataset.evt_max_train")
        evt_max_val = config.get("dataset.evt_max_val")
        if evt_max_train: print(f"Using {evt_max_train} events for training")
        if evt_max_val: print(f"Using {evt_max_val} events for validation")

        data_path = config.get("dataset.data_dir")
        polarity = config.get("dataset.polarity")
        data_type = config.get("dataset.data_type")
        load_graph = config.get("dataset.load_graph")
        
        ### DEBUG TODO !!!!
        data_subfolder = f"{data_type}_with_id"
        # data_subfolder = f"{data_type}"



        # Initialize input/target file lists
        files_input_tr, files_target_tr = [], []
        files_input_vl, files_target_vl = [], []
        files_input_tst, files_target_tst = [], []
        size_tr_up = size_tr_down = size_vl_up = size_vl_down = 0

        if polarity == 'magall':
            if not load_graph:
                # Construct paths for both polarities
                path_up = os.path.join(data_path, 'magup', data_subfolder)
                path_down = os.path.join(data_path, 'magdown', data_subfolder)

                # Training files (magup)
                input_up_1 = sorted(glob.glob(f'{path_up}/training_dataset/input_*'), key=natural_sort_key)
                target_up_1 = sorted(glob.glob(f'{path_up}/training_dataset/target_*'), key=natural_sort_key)
                input_up_2 = sorted(glob.glob(f'{path_up}/training_dataset_2/input_*'), key=natural_sort_key)
                target_up_2 = sorted(glob.glob(f'{path_up}/training_dataset_2/target_*'), key=natural_sort_key)
                size_tr_up = len(target_up_1) + len(target_up_2)

                # Training files (magdown)
                input_down_1 = sorted(glob.glob(f'{path_down}/training_dataset/input_*'), key=natural_sort_key)
                target_down_1 = sorted(glob.glob(f'{path_down}/training_dataset/target_*'), key=natural_sort_key)
                input_down_2 = sorted(glob.glob(f'{path_down}/training_dataset_2/input_*'), key=natural_sort_key)
                target_down_2 = sorted(glob.glob(f'{path_down}/training_dataset_2/target_*'), key=natural_sort_key)
                size_tr_down = len(target_down_1) + len(target_down_2)

                # Combine and truncate to evt_max
                files_input_tr = (input_down_1 + input_down_2 + input_up_1 + input_up_2)[:evt_max_train]
                files_target_tr = (target_down_1 + target_down_2 + target_up_1 + target_up_2)[:evt_max_train]

                # Validation files
                input_vl_up = sorted(glob.glob(f'{path_up}/validation_dataset/input_*'), key=natural_sort_key)
                target_vl_up = sorted(glob.glob(f'{path_up}/validation_dataset/target_*'), key=natural_sort_key)
                size_vl_up = len(input_vl_up)

                input_vl_down = sorted(glob.glob(f'{path_down}/validation_dataset/input_*'), key=natural_sort_key)
                target_vl_down = sorted(glob.glob(f'{path_down}/validation_dataset/target_*'), key=natural_sort_key)
                size_vl_down = len(input_vl_down)

                files_input_vl = (input_vl_up + input_vl_down)[:evt_max_val]
                files_target_vl = (target_vl_up + target_vl_down)[:evt_max_val]

                # Test files (only from magup)
                files_input_tst = sorted(glob.glob(f'{path_up}/test_dataset/input_*'), key=natural_sort_key)
                files_target_tst = sorted(glob.glob(f'{path_up}/test_dataset/target_*'), key=natural_sort_key)
            else:
                # Graphs are pre-generated and not loaded here
                size_tr_up, size_tr_down = 80000, 79500
                size_vl_up, size_vl_down = 10732, 10754

        elif polarity in ['magup', 'magdown']:
            if not load_graph:
                data_path = os.path.join(data_path, polarity, data_subfolder)
                input_tr_1 = sorted(glob.glob(f'{data_path}/training_dataset/input_*'), key=natural_sort_key)
                target_tr_1 = sorted(glob.glob(f'{data_path}/training_dataset/target_*'), key=natural_sort_key)
                input_tr_2 = sorted(glob.glob(f'{data_path}/training_dataset_2/input_*'), key=natural_sort_key)
                target_tr_2 = sorted(glob.glob(f'{data_path}/training_dataset_2/target_*'), key=natural_sort_key)
                files_input_tr = (input_tr_1 + input_tr_2)[:evt_max_train]
                files_target_tr = (target_tr_1 + target_tr_2)[:evt_max_train]

                files_input_vl = sorted(glob.glob(f'{data_path}/validation_dataset/input_*'), key=natural_sort_key)[:evt_max_val]
                files_target_vl = sorted(glob.glob(f'{data_path}/validation_dataset/target_*'), key=natural_sort_key)[:evt_max_val]
            else:
                # No loading from file list
                pass
        else:
            raise Exception(f"Unexpected magnet polarity {polarity}. Use magup, magdown or magall.")

        # Dataset instantiation
        if data_type == "homogeneous":
            n_classes = config.get('model.LCA_classes')
            self.train_dataset = CustomDataset(files_input_tr, files_target_tr, performance_mode, n_classes)
            self.val_dataset = CustomDataset(files_input_vl, files_target_vl, performance_mode, n_classes)
            self.test_dataset = CustomDataset(files_input_tst, files_target_tst, performance_mode, n_classes)

        elif data_type == "heterogeneous":
            n_classes = config.get('model.LCA_classes')
            self.train_dataset = CustomHeteroDataset(files_input_tr, files_target_tr, performance_mode, n_classes)
            self.val_dataset = CustomHeteroDataset(files_input_vl, files_target_vl, performance_mode, n_classes)
            self.test_dataset = CustomHeteroDataset(files_input_tst, files_target_tst, performance_mode, n_classes)

        elif data_type == "neutrals":
            print('Creating neutrals dataset...')
            self.train_dataset = CustomNeutralsHeteroDataset(files_input_tr, files_target_tr, performance_mode, config, "train", [size_tr_up, size_tr_down])
            self.val_dataset = CustomNeutralsHeteroDataset(files_input_vl, files_target_vl, performance_mode, config, "val", [size_vl_up, size_vl_down])
            # self.test_dataset = CustomNeutralsHeteroDataset(files_input_tst, files_target_tst, performance_mode, config, "test")
        else:
            raise Exception(f"Unexpected data type {data_type}. Use neutrals, homogeneous or heterogeneous.")

    def load_data(self):
        """
        Load the raw data from files into memory or internal representation.

        After calling this method, datasets are available via:
            - `self.dataset_tr`
            - `self.dataset_vl`
            - `self.dataset_tst`
        """        
        self.dataset_tr = self.train_dataset.get()
        self.dataset_vl = self.val_dataset.get()
        # self.dataset_tst = self.test_dataset.get()

    def get_train_dataloader(self, batch_size=None):
        """
        Create a DataLoader for the training dataset.

        Args:
            batch_size (int, optional): Number of samples per batch. If None,
                uses the default `self.batch_size`.

        Returns:
            DataLoader: Iterator over batches of training graphs.
        """
        if batch_size is None:
            batch_size = self.batch_size
        return DataLoader(self.dataset_tr, batch_size=batch_size, drop_last=True)

    def get_val_dataloader(self, batch_size=None):
        """
        Create a DataLoader for the validation dataset.

        Args:
            batch_size (int, optional): Number of samples per batch. If None,
                uses the default `self.batch_size`.

        Returns:
            DataLoader: Iterator over batches of validation graphs.
        """
        if batch_size is None:
            batch_size = self.batch_size
        return DataLoader(self.dataset_vl, batch_size=batch_size, drop_last=True)

    def get_test_dataloader(self, batch_size=None):
        """
        Create a DataLoader for the test dataset.

        Args:
            batch_size (int, optional): Number of samples per batch. If None,
                uses the default `self.batch_size`.

        Returns:
            DataLoader: Iterator over batches of test graphs.
        """
        if batch_size is None:
            batch_size = self.batch_size
        return DataLoader(self.dataset_tst, batch_size=batch_size, drop_last=True)
