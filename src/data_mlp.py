import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.utils.data.dataloader import default_collate
from sklearn.model_selection import train_test_split
import os
import pandas as pd

def load_pracids(df):
    """
    Load train and validation practice IDs.

    Args:
        df (pd.DataFrame): DataFrame containing the practice IDs, with valid column

    Returns:
        tuple: Train and validation practice IDs (numpy arrays).
    """
    train_pracids = df[df['valid'] == 0]['pracid'].unique()
    valid_pracids = df[df['valid'] == 1]['pracid'].unique()

    return train_pracids, valid_pracids

def convert_datatypes(path, field_mapping, int_columns, float_columns):
    """
    Convert datatypes.

    Args:
        path (str): Path to the imputed data file.
        field_mapping (dict): Dictionary of field mapping
        int_columns (list): List of columns to be converted to integer type.
        float_columns (list): List of columns to be converted to float type.

    Returns:
        pd.DataFrame: Processed DataFrame.
    """
    df = pd.read_parquet(path)

    df = df.rename(columns=field_mapping)
    if 'nelal' in df.columns:
        df.drop(['nelal'], axis=1, inplace=True)
    
    for col in int_columns:
        df[col] = df[col].astype(int)
    
    for col in float_columns:
        df[col] = df[col].astype(float)
    
    return df


def extract_and_split(df, feature_columns, label_columns, train_pracids, valid_pracids, suffix, data_folder):
    """
    Extract features and labels, split the data, and save to files.

    Args:
        df (pd.DataFrame): Input DataFrame.
        feature_columns (list): List of feature column names.
        label_columns (list): List of label column names.
        train_pracids (np.array): Array of training practice IDs.
        valid_pracids (np.array): Array of validation practice IDs.
        suffix (str): Suffix for output file names (maggic or maggic_plus in this case).
        data_folder (str): Path to the data folder (to save the processed files).
    """
    label = df[label_columns]
    features = df[feature_columns]

    # def replace_ef(x):
    #     if x == 0:
    #         return np.random.uniform(0.2, 0.7)
    #     elif x == 1:
    #         return np.random.uniform(0.2, 0.4)
    #     else:
    #         return np.random.uniform(0.4, 0.7)
        
    # features['EF'] = features['EF'].apply(replace_ef)

    def replace_hftime(x):
        if x == 0:
            return 1
        elif x == 1:
            return 0
        
    # features['htime'] = features['hftime'].apply(replace_hftime)

    # from sklearn.preprocessing import MinMa
    # import pandas as pd
    
    # # Assuming you have a DataFrame 'df' with numerical columns to standardize
    # scaler = StandardScaler()
    # features = pd.DataFrame(scaler.fit_transform(features), columns=features.columns, index=features.index)
    
    features_train = torch.tensor(features[df['pracid'].isin(train_pracids)].to_numpy(), dtype=torch.float)
    labels_train = torch.tensor(label[df['pracid'].isin(train_pracids)].to_numpy(), dtype=torch.float)
    features_valid = torch.tensor(features[df['pracid'].isin(valid_pracids)].to_numpy(), dtype=torch.float)
    labels_valid = torch.tensor(label[df['pracid'].isin(valid_pracids)].to_numpy(), dtype=torch.float)
    
    features_train, features_test, labels_train, labels_test = train_test_split(
        features_train, labels_train, test_size=0.10, random_state=42
    )

    torch.save({'features': features_train, 'labels': labels_train}, os.path.join(data_folder, f'train_{suffix}.pt'))
    torch.save({'features': features_valid, 'labels': labels_valid}, os.path.join(data_folder, f'valid_{suffix}.pt'))
    torch.save({'features': features_test, 'labels': labels_test}, os.path.join(data_folder, f'test_{suffix}.pt'))


def process_imputed_data(config):
    """
    Process the imputed data using the provided configuration. Includes 
        - Converting datatypes
        - Splitting the data into train, valid, and test (depending on the prac ids)
        - Does for both maggic and maggic-plus predictor variables
        - Saves as numpy files

    Args:
        df_path (str): Imputed Dataframe path.
    """
    df = convert_datatypes(config['file_path'], config['field_mapping'], config['int_columns'], config['float_columns'])
    
    train_pracids, valid_pracids = load_pracids(df)
    
    extract_and_split(df, config['maggic_columns'], config['label_columns'], 
                      train_pracids, valid_pracids, 'maggic', config['data_folder'])
    
    extract_and_split(df, config['maggic_plus_columns'], config['label_columns'], 
                      train_pracids, valid_pracids, 'maggic_plus', config['data_folder'])


class DictDataset(Dataset):
    """
    A custom Dataset class that handles dictionary-based features and labels.
    """

    def __init__(self, features, labels):
        """
        Initialize the DictDataset.

        Args:
            features (dict): Dictionary of feature tensors.
            labels (torch.Tensor): Tensor of labels.
        """
        self.features = features
        self.labels = labels

    def __len__(self):
        """Return the number of samples in the dataset."""
        return self.labels.size(0)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: (sample_features, label)
        """
        sample_features = {key: self.features[key][idx] for key in self.features}
        return sample_features, self.labels[idx]


class OrderedBatchRandomSampler:
    """
    A custom sampler that yields randomly ordered batches of sorted indices.
    """

    def __init__(self, n, batch_size, seed=13):
        """
        Initialize the OrderedBatchRandomSampler.

        Args:
            n (int): Total number of samples.
            batch_size (int): Size of each batch.
            seed (int): Random seed for reproducibility.
        """
        self.n = n
        self.batch_size = batch_size
        self.seed = seed
        self.random_state = np.random.RandomState(seed)

    def __len__(self):
        """Return the number of batches."""
        return (self.n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        """
        Yield batches of sorted indices.

        Yields:
            list: Sorted batch indices.
        """
        batch = []
        for idx in self.random_state.permutation(self.n):
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield sorted(batch)
                batch = []
        if len(batch) > 0:
            yield sorted(batch)


def preprocess_data(x, y, std=0.0001):
    """
    Preprocess the input data.

    Args:
        x (np.ndarray): Input features.
        y (np.ndarray): Labels.
        std (float): To add for time adjustment (to avoid divide by 0 errors).

    Returns:
        tuple: Processed features, labels, and time.
    """
    label = y[:, 1]
    time2event = y[:, 0] + std

    # Sort
    idx = np.argsort(time2event)
    time2event = time2event[idx]
    labels = label[idx]
    x = x[idx]

    init_cond = np.zeros_like(time2event)

    return x, labels, time2event, init_cond


def get_dataloader(
    input_file, batch_size=2, random_state=np.random.RandomState(seed=0), std=0.0001
):
    """
    Create a DataLoader for the survival analysis task.

    Args:
        input_file (str): Path to the input file.
        batch_size (int): Size of each batch.
        random_state (np.random.RandomState): Random state for reproducibility.
        std (float): To add for time adjustment (to avoid divide by 0 errors).

    Returns:
        tuple: (DataLoader, feature_size)
    """
    dt = torch.load(input_file)
    x, y = dt["features"], dt["labels"]

    x, labels, time2event, init_cond = preprocess_data(x.numpy(), y.numpy(), std)

    N = len(time2event)
    feature_size = x.shape[1]

    labels = torch.tensor(labels, dtype=torch.float)

    features = {
        "t": torch.tensor(time2event, dtype=torch.float),
        "init_cond": torch.tensor(init_cond, dtype=torch.float),
        "features": torch.tensor(x, dtype=torch.float),
        "index": torch.arange(N, dtype=torch.long),
    }

    dataset = DictDataset(features, labels)
    sampler = OrderedBatchRandomSampler(N, batch_size)

    dataloader = DataLoader(dataset, batch_sampler=sampler, collate_fn=default_collate)

    return dataloader, feature_size
