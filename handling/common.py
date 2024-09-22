
import hashlib
from os import listdir
from os.path import isfile, join
import os
import pandas as pd
import pyarrow as pa
import pyarrow.orc as orc

# common constants and static methods to be used across multiple data handling tasks

DEFAULT_INPUT_PATH = 'data/'
DEFAULT_BOOK_PATH = 'data/books/'
DEFAULT_OUTPUT_PATH = 'dataframe/'
DEFAULT_OUTPUT_FILE = 'othello_games.data'  #dataframe save
DEFAULT_BOOK_FILE = 'books.txt'
DF_COLUMNS = ['ID', 'Black', 'White', 'Result', 'Date', 'Source', 'Moves', 'Hash']


def get_files_from_directory(path: str) -> []:
    '''
    Returns the list of files in a given directory.
    :param path: input path
    :return: list of files in the directory, empty list if none
    '''
    files = [f for f in listdir(path) if isfile(join(path, f))]
    return files


def read_dataframe(filename: str) -> pd.DataFrame:
    '''
    Reads the daaframe from the input parameter. creates it if not found
    Using ORC to keep size and write speed manageable
    :param filename: path + filename of the file
    :return: read or created dataframe
    '''

    if os.path.isfile(filename):
        #df = pd.read_csv(filename, header=0, compression='gzip')
        df = pd.read_orc(filename, columns=DF_COLUMNS)
        #df = pd.read_orc(filename)

    else:
        df = pd.DataFrame(columns=DF_COLUMNS)

    return df

def save_dataframe(file_name: str, df: pd.DataFrame):
    # for now, csv
    # TODO move to ORC
    #df.to_csv(file_name, mode='a', index=False, compression='gzip')
    df.reset_index().to_orc(file_name)


def hash_moves(moves: str) -> str:
    '''
    Simple MD5 hash for moves duplicate checking
    :param moves: all the moves in one string
    :return: encoded value
    '''
    m = hashlib.md5()
    m.update(moves.encode('UTF-8'))
    return m.hexdigest()


def convert_to_notation(cell: int) -> str:
    '''
    the synthetic game uses a weird notation, 0 being the top left corner of the board, while 63 being the bottom right
    this is converted to a1 - h8 notation
    :param cell: input location from 0 to 63
    :return: board notation from a1 to h8
    '''

    columns = 'ABCDEFGH'
    column = columns[ cell % 8]
    row = (cell // 8) + 1
    return column + str(row)