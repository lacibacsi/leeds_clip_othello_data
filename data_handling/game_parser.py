
from data_handling import common
import os
from os.path import isfile, join
import pandas as pd
from data_handling.pypaya_pgn_parser.pgn_parser import PGNParser
import pyarrow as pa
import pyarrow.orc as orc
from io import StringIO
import uuid


class GameParser():
    '''
        Simple class to parse othello PGNs§
        cannot handle wthor files
        saves / appends games in the output file    
    '''

    def __init__(self, input_path : str = None, output_path: str = None, output_file_name: str = None ):
        self.input_path = common.DEFAULT_INPUT_PATH if input_path is None else input_path
        path = common.DEFAULT_OUTPUT_PATH if output_path is None else output_path
        file = common.DEFAULT_OUTPUT_FILE if output_file_name is None else output_file_name
        self.output_file = os.path.join(path, file)

    def __read_files(self, file_name=None, relativ_dir=None):
        '''
        Reads the source files from the source directory.
        :param file_name: name of the single file to be reads
        :param relativ_dir: relative path to base / default input dir. Set to read full directory
        :return: file path and list of file names
        '''
        read_dir = file_name is None
        if read_dir and relativ_dir is None:
            ValueError("pass either a file name or a relative directory to read files from")

        file_path = os.path.join(self.input_path, relativ_dir) if relativ_dir is not None else self.input_path
        files = []

        if read_dir:
            files = common.get_files_from_directory( file_path )
            print(f'read {len(files)} files')
        else:
            files = file_name

        return file_path, files

    def transpose_coords(self, moves):
        files = '12345678'
        ranks = 'ABCDEFGH'

        ind_moves = [moves[i:i + 2] for i in range(0, len(moves), 2)]
        transposed = ''

        for i, v in enumerate(ind_moves):
            r_index = ranks.index(v[0])
            f_index = files.index(v[1])
            x = str(ranks[f_index]) + str(files[r_index])
            transposed += x
        return transposed

    def add_pgn_to_df(self, df: pd.DataFrame, game_info: [], game_moves: [], source: str, check_duplicate=False, transpose_coords=False):
        '''
        Adds a record of a game to the parsed_data. Converts to required format, calculates hash of the moves and check dupes if set
        :param df: parsed_data to which the record needs to be added
        :param game_info: game header -> fixed position for important headers, i.e. Result, white, etc.
        :param game_moves: list of moves
        :param: source of the game, i.e. human or synthetic
        :param check_duplicate: set to True if only save the game if no other games with the same moves are present. Only game moves are checked
        :return: updated parsed_data
        '''

        id =  str(uuid.uuid4())
        game_date = game_info[2]
        white = game_info[4]
        black = game_info[5]
        result = game_info[6]
        moves = str(game_moves).replace(' ','')

        # major hack -> the source pgn files from wthor use transposed coords
        # massively slow algo
        if transpose_coords:
            moves = self.transpose_coords(moves)

        row = {
            'ID' : id,
            'Black': black,
            'White': white,
            'Result': result,
            'Date': game_date,
            'Source': source,
            'Moves': moves,
            'Hash': common.hash_moves(moves)
        }

        # using concat for now, change to using loc for speed purposes if
        df = pd.concat([df, pd.DataFrame.from_records([row])])

        return df

    def read_pgn_files(self, save_output=False, file_name=None, relativ_dir=None, transpose_coords = True):
        '''
        Read file or files from the input directory and saves / appends if required
        use this method to cons
        :param save_output: if set, it saves / appends the files.
        :param file_name: if set, only this file is added if left empty, the whole directory is processed
        :param relativ_dir: directory to read files from. Relative to the default input path. if using the default, pass ''
        :param transpose_coords: some sources use a faulty board representation, i.e. ranks are denoted by character and not files
        :return: int: number of games parsed
        '''

        df = common.read_dataframe(self.output_file)
        game_counter = 0

        file_path, files = self.__read_files(file_name=file_name, relativ_dir=relativ_dir)

        for file in files:
            with open(join(file_path, file)) as f:
                source = f.read()

            # example from https://github.com/PypayaTech/pypaya-pgn-parser
            pgn_stringio = StringIO(source)
            parser = PGNParser()

            pgn_games = 0

            while True:
                parsed = parser.parse(pgn_stringio)
                if parsed is None:
                    break
                else:
                    (game_info, game_moves) = parsed
                    game_counter += 1
                    pgn_games += 1

                    if save_output:
                        df = self.add_pgn_to_df(df, game_info, game_moves, 'human', transpose_coords)

            print(f'{pgn_games} games parsed in file')
            #print(f'total {game_counter} games parsed')

        if save_output:
            common.save_dataframe(self.output_file, df)

        return game_counter

    def read_csv_files(self, save_output=False, file_name=None, relativ_dir=None):
        '''
        Reads source othello games from csv files... The mapping is hard coded
        :param save_output: if set, it saves / appends the files.
        :param file_name: if set, only this file is added if left empty, the whole directory is processed
        :param relativ_dir: directory to read files from. Relative to the default input path. if using the default, pass ''
        :return: int: number of games parsed
        '''

        df = common.read_dataframe(self.output_file)
        game_counter = 0

        file_path, files = self.__read_files(file_name=file_name, relativ_dir=relativ_dir)

        for file in files:
            df_csv = pd.read_csv(join(file_path, file))

            if save_output:
                # this is hardcoded for the liveothello format
                # TODO change to generic
                df_csv.drop(['eOthello_game_id'], axis=1, inplace=True)
                df_csv['ID'] = df_csv.apply(lambda row: str(uuid.uuid4()), axis=1)
                df_csv['Black'] = 'Unknown'
                df_csv['White'] = 'Unknown'
                df_csv['Date'] = 'Unknown'
                df_csv['Source'] = 'human'
                df_csv = df_csv.rename(columns={'game_moves': 'Moves'})

                df_csv['Result'] = df_csv['winner'].apply(lambda x: '0-1' if x < 0 else ('1-0' if x > 0 else '0-0') )
                df_csv.drop(['winner'], axis=1, inplace=True)
                df_csv['Hash'] = df_csv.apply(lambda row: common.hash_moves(row['Moves']), axis=1)

                # concat and saving
                game_counter = len(df_csv.index)
                df = pd.concat([df, df_csv])
                common.save_dataframe(self.output_file, df)

        return game_counter

    def read_pickle_files(self, save_output=False, file_name=None, relativ_dir=None):
        df = common.read_dataframe(self.output_file)
        game_counter = 0

        file_path, files = self.__read_files(file_name=file_name, relativ_dir=relativ_dir)
        file_index = 0

        for file in files:
            games = pd.read_pickle(join(file_path, file))
            game_notations = []
            file_game_counter = 0
            file_index += 1

            # need to parse all games as the notation is index based and not conforming with standard notation
            for game in games:
                # game is a list of moves
                game_notations.append(''.join([ common.convert_to_notation(x) for x in game ]))

            # appending if needed
            if save_output:
                df_pck = pd.DataFrame()
                df_pck['Moves'] = game_notations
                df_pck['Black'] = 'Unknown'
                df_pck['White'] = 'Unknown'
                df_pck['Date'] = 'Unknown'
                df_pck['Source'] = 'synthetic'
                df_pck['Result'] = 'Unknown' #TODO
                df_pck['Hash'] = df_pck.apply(lambda row: common.hash_moves(row['Moves']), axis=1)
                df_pck["ID"] = str(uuid.uuid4())

                # concat and saving
                file_game_counter = len(df_pck.index)
                game_counter += file_game_counter
                df = pd.concat([df, df_pck])
                common.save_dataframe(self.output_file, df)
                print(f'added : {file_game_counter} games ({file_index} / {len(files)})')

        return game_counter

