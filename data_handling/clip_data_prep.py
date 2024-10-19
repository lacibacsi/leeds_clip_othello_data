import sys

import pandas as pd

from data_handling import common
from othello_game import othello


class ClipDataPreparator():
    '''
    Class to generate input data for CLIP model training
    It takes input game records and creates a set of image (vector) and text pairs (tuples)

    Init with a file name, outputs a saved file with list of tuples, where each tuple is a (vector, nextmove)
    For now all game data is in memory, for huge files need to implement chunked writes
    End of game is denoted with 'Z9' as both are illegal (arbitrary choice)
    '''

    def __init__(self, filename: str, output: str):
        '''
        Ctor for ClipDataPreparator
        :param filename: full path for the game file to parse
        :param output: full path for the game file to save
        '''

        # TODO: set a default
        self.output = output

        # just need the Moves column, saving memory for large files
        self._data = list(common.read_dataframe(filename)['Moves'])
        print(f'{len(self._data)} games loaded')

    def parse(self):
        '''
        Parses the loaded data file. For each position creates a board vector and next move tuple
        Prints updates throughout the process as it can take significant time for 100k + games
        :return: None, throws error if fails. In case of error nothing is saved
        '''

        no_of_games = len(self._data)
        game_counter = 0
        error_counter = 0
        success_counter = 0

        # list to store all tuples
        output_content = []

        for line in self._data:
            game_counter += 1

            # split moves
            moves = [line[i:i + 2] for i in range(0, len(line), 2)]

            # create a new othello engine for each game
            game = othello.OthelloGame(8, 8, 'B', 'W', 'M')

            for index,move in enumerate(moves):
                # get the vector before making the move
                output_content.append((game.current_board_as_vector(), move))
                row, col = common.get_move_coords(move)

                # if an illegal move is made, the game is skipped
                try:
                    game.move(row, col)
                except othello.InvalidMoveException:
                    print(f'error moving {move}, move # {index + 1} move out of {len(moves)}, found an illegal move, game counter: {game_counter}')
                    error_counter += 1
                    break

                if index == len(moves) - 1:
                    # add last move
                    output_content.append((game.current_board_as_vector(), common.END_OF_GAME))
                    success_counter += 1

            if game_counter % 50 == 0 or game_counter == no_of_games:
                print(f'parsed {game_counter} games out of {no_of_games}')

        print(f'parsing has finished, success: {success_counter}, error: {error_counter}')

        # saving the output
        df = pd.DataFrame(output_content, columns=['board','move'])
        common.save_dataframe(self.output,df)

        #print(f'number of vectors: {len(output_content)}, sample: {output_content[0]}')