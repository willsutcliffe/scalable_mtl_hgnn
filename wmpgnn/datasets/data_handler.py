from wmpgnn.datasets.graph_dataset import CustomDataset
from wmpgnn.datasets.hetero_graph_dataset import CustomHeteroDataset
from torch_geometric.loader import DataLoader
import glob



class DataHandler:
    """
    DataHandler orchestrates loading and batching of graph datasets for training,
    validation, and testing. Supports both homogeneous and heterogeneous graph data.

    Attributes:
        config_loader (Config): Configuration object providing dataset paths and types.
        batch_size (int): Default batch size for data loaders.
        train_dataset, val_dataset, test_dataset:
            Instances of CustomDataset or CustomHeteroDataset representing
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
            Exception: If `dataset.data_type` is not "homogeneous" or "heterogeneous".
        """

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
        LCA_classes = self.config_loader.get('model.LCA_classes')
        if data_type == "homogeneous":
            self.train_dataset = CustomDataset(files_input_tr, files_target_tr, performance_mode=performance_mode, n_classes=LCA_classes)
            self.val_dataset = CustomDataset(files_input_vl, files_target_vl, performance_mode=performance_mode, n_classes=LCA_classes)
            self.test_dataset = CustomDataset(files_input_tst, files_target_tst, performance_mode=performance_mode, n_classes=LCA_classes)
        elif data_type == "heterogeneous":
            self.train_dataset = CustomHeteroDataset(files_input_tr, files_target_tr, performance_mode=performance_mode, n_classes=LCA_classes)
            self.val_dataset = CustomHeteroDataset(files_input_vl, files_target_vl, performance_mode=performance_mode, n_classes=LCA_classes)
            self.test_dataset = CustomHeteroDataset(files_input_tst, files_target_tst, performance_mode=performance_mode, n_classes=LCA_classes)
        else:
            raise Exception(f"Unexpected data type {data_type}. Please use homogeneous or heterogeneous.")

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
        self.dataset_tst = self.test_dataset.get()

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