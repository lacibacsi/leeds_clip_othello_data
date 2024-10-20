
import clip.utils
from clip.meter import AvgMeter
from data_handling import common
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm



class ToTensor:
    # Convert pd series to Tensors
    def __call__(self, sample):

        stringboard = sample['board'].to_list()[0]
        board_array = common.convert_board_array(stringboard)
        board = torch.FloatTensor(board_array)

        move = sample['move'].to_list()[0]
        move = common.convert_move_to_tensor(move)

        return board, move


class ClipDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.__df__ = df
        self.length = len(df)
        self.transform = ToTensor()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        row = self.__df__.iloc[[idx]]

        board, move = self.transform(row)
        item = {}
        item['board'] = board
        item['move'] = move

        return item

class ClipTrainer:
    '''
    Trains the clip model based on the input variables, parameters and config
    '''

    def __init__(self, source_path: str):
        self.source_path = source_path
        self.source = None

    def __read_source__(self):
        '''
        reads the source data from file into a dataframe
        :return: None
        '''
        self.source = common.read_dataframe(self.source_path)

    def make_train_validation_sets(self, split=0.2) -> (pd.DataFrame, pd.DataFrame):
        '''
        creates train and validation dataframes from the source data
        :param split: ration of validation set
        :return: dataframes for train and validation respectively
        '''
        if self.source is None:
            self.__read_source__()

        # there is no 'id' in the dataset, so train / validation split happens on a shuffled dataset
        # for large datasets this can take very long, TODO: optimize if need to rerun often
        df = self.source.sample(frac=1, random_state=42)

        split_index = int((1-split)*len(df))

        train, val = df[:split_index], df[split_index:]
        return train, val


    def getLoader(self, dataframe: pd.DataFrame, mode: str) -> DataLoader:
        '''
        creates a data loader for the frame for training
        :param dataframe:
        :param mode: 'Train' or 'Val'
        :return: torch dataloader
        '''

        ds = ClipDataset(dataframe)
        loader = DataLoader(ds,
                            batch_size=5,
                            num_workers=0,
                            shuffle=True if mode == 'Train' else False,
        )

        return loader


# other static methods for training
def train_epoch(device, model, train_loader, optimizer, lr_scheduler, step):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    for batch in tqdm_object:

        batch = {k: v.to(device) for k, v in batch.items()}
        loss = model(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == "batch":
            lr_scheduler.step()

        count = batch["board"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=clip.utils.get_lr(optimizer))
    return loss_meter

def valid_epoch(device, model, valid_loader):
    loss_meter = AvgMeter()

    tqdm_object = tqdm(valid_loader, total=len(valid_loader))
    for batch in tqdm_object:
        batch = {k: v.to(device) for k, v in batch.items()}
        loss = model(batch)

        count = batch["board"].size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(valid_loss=loss_meter.avg)
    return loss_meter