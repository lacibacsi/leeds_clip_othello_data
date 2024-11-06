import numpy as np
import torch

from clip.model import CLIPModel
from clip.utils import get_device
from data_handling import common
from othello_game import othello
import torch.nn.functional as F


# code for getting the best text embedding (moves) on input image (board)

def get_text_tensors() -> []:
    '''
    Returns a list of tensors (64) with all possible moves to be used in inference, including end-of-game one
    :return: list of torch tensors
    '''
    tensor_list = []

    board_array = [0] * 64

    # adding empty / end of game
    tensor_list.append( torch.FloatTensor(board_array))

    for i in range(64):
        board_array[i] = 1
        tensor_list.append( torch.FloatTensor(board_array).to(get_device()))
        board_array[i] = 0  #restoring

    return tensor_list

class ClipInference():

    def __init__(self, model_path: str = None):
        self.text_embeddings = get_text_tensors()
        if model_path is not None:
            self.model = self.load_model(model_path)

    def load_model(self, souce_path: str) -> CLIPModel:
        '''
        loads and returns a saved model
        :param souce_path: path to the saved model
        :return: loaded and instantiated model on the available device
        '''

        model = CLIPModel().to(get_device())
        model.load_state_dict(torch.load(souce_path, map_location=get_device()))

        return model

    def __make_moves_in_game(self, move_list: str):
        '''
        Makes the moves in an othello game
        :param move_list:
        :return: game
        '''
        moves = [move_list[i:i + 2] for i in range(0, len(move_list), 2)]
        game = othello.OthelloGame(8, 8, 'B', 'W', 'M')

        for index, move in enumerate(moves):
            row, col = common.get_move_coords(move)
            # if an illegal move is made, the game is skipped
            try:
                game.move(row, col)
            except othello.InvalidMoveException:
                break

        return game

    def get_value_for_position(self, move_list: str) :
        '''
        Given an input list of moves in a game it returns 64 values of image-text distance for the resulting position
        :param: model: CLIP model to use
        :param move_list: all the moves until the position
        :return: best move / value pair for the end position in the game (move list)
        '''

        # the methods is rather simple and probably slow for huge number of eval positions
        # 1. get the position
        # 2. get all possible encoded moves through the text encoder
        # 3. generate all distances / clip results

        game = self.__make_moves_in_game(move_list)
        # the resulting position is to be fed into the image encoder
        board = game.current_board_as_vector()
        #print(f'board after the moves: {board}')
        # adding a dummy wrapper as there is no batch now
        board = board.reshape(1, 2, 8, 8 )

        boardTensor = torch.FloatTensor(board)
        #print(boardTensor)

        with torch.no_grad():

            image_features = self.model.image_encoder(boardTensor.to(get_device()))
            text_features = get_text_tensors()

            # dot product of the image fetures and all the text features (categories) will yield the model's results
            image_proj = self.model.image_projection(image_features)

            image_proj_norm = F.normalize(image_proj)

            max_value = -1000
            move_value = None

            result = {}

            for index, text_feature in enumerate(text_features):
                text_feature = text_feature.reshape(1,64)
                text_proj = self.model.text_projection(text_feature.to(get_device()))
                text_proj_norm = F.normalize(text_proj)

                dot_similarity = 100 * image_proj_norm @ text_proj_norm.T

                result[index] = dot_similarity.squeeze(0)

                #print(f'index: {index}, dot similarity value: {dot_similarity.item()}')
                if dot_similarity.item() > max_value:
                    max_value = dot_similarity.item()
                    move_value = index

        return move_value, max_value, result

    def run_eval_file(self, filepath: str)-> []:
        '''
        loads and runs inference on the input file, returns the result in a tabular format
        :param filepath:
        :return:
        '''

        df = common.read_dataframe(filepath, use_compression=False)
        df.reset_index()

        result = {}

        for index, row in df.iterrows():
            # structure: moves, expected move
            moves = row[0]
            expected = row[1]

            move, model_result, values = self.get_value_for_position(moves)
            result[index] = (moves, expected, common.convert_to_notation(move), model_result)

        return result

    def run_validity_check(self, eval_result):
        '''
        Takes a list of evaluation results (see above) and checks if the model result is a valid move or not
        Extends and input with the True/False value and returns it
        :param eval_result:
        :return:
        '''

        validity_result = {}

        for key, eval_position in eval_result.items():
            moves = eval_position[0]
            predicted_move =eval_position[2]

            # getting to the position
            game = self.__make_moves_in_game(moves)

            # making the move
            row, col = common.get_move_coords(predicted_move)
            valid = True
            try:
                game.move(row, col)
            except othello.InvalidMoveException:
                valid = False

            validity_result[key] = (eval_position[0], eval_position[1], eval_position[2], valid)

        return validity_result