from wmpgnn.datasets.graph_dataset import CustomDataset
from wmpgnn.datasets.hetero_graph_dataset import CustomHeteroDataset
from torch_geometric.loader import DataLoader
import glob
import random


def random_shuffle(inputs, targets, seed):
    """
    Randomly shuffle lists of files preserving the correspondence
    
    Args:
        inputs (list): List of input file paths.
        targets (list): List of target file paths.
        seed (int): Random seed for reproducibility.
    """
    Nfiles = len(inputs)
    if Nfiles != len(targets):
        raise ValueError(f"Inputs and targets must have the same length, got {Nfiles} and {len(targets)}.")
    elif Nfiles != 0:
        files = list(zip(inputs, targets))
        random.shuffle(files)
        inputs, targets = zip(*files)
    return inputs, targets

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
                - "dataset.data_dir": Base directory containing dataset subfolders, or list of base direcories.
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
        evt_max_test = self.config_loader.get("dataset.evt_max_test")
        
        data_path = self.config_loader.get("dataset.data_dir")
        data_type = self.config_loader.get("dataset.data_type")
        # single or multiple data paths
        if isinstance(data_path, str):
            data_paths = [data_path]
        elif isinstance(data_path, list):
            data_paths = data_path
        else:
            raise Exception(f"Unexpected data path type {type(data_path)}. Please use str or list.")
        
        files_input_tr, files_target_tr = [], []
        files_input_vl, files_target_vl = [], []
        files_input_tst, files_target_tst = [], []
        # get all files in the data paths
        for data_path in data_paths:
            files_input_tr   += sorted(glob.glob(f'{data_path}/training_dataset/input_*'))
            files_target_tr  += sorted(glob.glob(f'{data_path}/training_dataset/target_*'))
            files_input_vl   += sorted(glob.glob(f'{data_path}/validation_dataset/input_*'))
            files_target_vl  += sorted(glob.glob(f'{data_path}/validation_dataset/target_*'))
            files_input_tst  += sorted(glob.glob(f'{data_path}/test_dataset/input_*'))
            files_target_tst += sorted(glob.glob(f'{data_path}/test_dataset/target_*'))
        
        # random shuffle with a fixed seed
        random_seed = self.config_loader.get("training.random_seed",default=None)
        if random_seed is not None:
            print(f"Shuffling files with random seed {random_seed}")
            files_input_tr, files_target_tr = random_shuffle(files_input_tr, files_target_tr, random_seed)
            files_input_vl, files_target_vl = random_shuffle(files_input_vl, files_target_vl, random_seed)
            files_input_tst, files_target_tst = random_shuffle(files_input_tst, files_target_tst, random_seed)
        
        #for i in range(5):
        #    print(f"Input {i}: {files_input_tr[i].split('/')[-1]}\t{files_target_tr[i].split('/')[-1]}")
        #for i in range(5):
        #    print(f"input {i}: {files_input_vl[i].split('/')[-1]}\t{files_target_vl[i].split('/')[-1]}")
        
        # reduce the number of events if needed
        if isinstance(evt_max_train, int) and evt_max_train >= 0:
            files_input_tr = files_input_tr[:evt_max_train]
            files_target_tr = files_target_tr[:evt_max_train]
        if isinstance(evt_max_val, int) and evt_max_val >= 0:
            files_input_vl = files_input_vl[:evt_max_val]
            files_target_vl = files_target_vl[:evt_max_val]
        if isinstance(evt_max_test, int) and evt_max_test >= 0:
            files_input_tst = files_input_tst[:evt_max_test]
            files_target_tst = files_target_tst[:evt_max_test]
        
        print(f"Using {len(files_input_tr)} events for training")
        print(f"Using {len(files_input_vl)} events for validation")
        print(f"Using {len(files_input_tst)} events for testing")
        
            
        self.batch_size =  self.config_loader.get("training.batch_size")
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