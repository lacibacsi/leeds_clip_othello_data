import datetime
import os.path
import time

import clip
import data_handling.common
from clip.inference import ClipInference
from clip.model import CLIPModel
from data_handling.clip_data_prep import ClipDataPreparator
from data_handling.game_parser import GameParser
from data_handling.book_parser import BookParser
from data_handling.common import convert_to_notation
import torch
from othello_game import othello
import data_handling.common as cm
from clip.train import ClipTrainer
from clip import utils

def parse_games():
    parser = GameParser()
    counter = 0
    # parse all three sources
    #counter = parser.read_pgn_files(relativ_dir='pgn/', save_output=True)
    counter = parser.read_pgn_files(relativ_dir='pgn_test/', save_output=True)
    #counter += parser.read_csv_files(relativ_dir='csv/', save_output=True)
    #counter += parser.read_pickle_files(relativ_dir='othello_synthetic/', save_output=True)

    print(f'parsed {counter} games')

def parse_books():
    parser = BookParser(output_path='parsed_data/book/')
    files = parser.clean_files
    parser.generate_instructions(filenames=files)
    print('done')

def prepare_clip_training():
    prepper = ClipDataPreparator('parsed_data/othello_games.source_data', 'parsed_data/clip/clip_training_source_int')
    prepper.parse()

def play_game():
    # plays a sample game of othello based on the move string

    cols = "ABCDEFGH"

    #moves = "f5d6c4d3e6f4e3f6c5b4e7f3c6d7b5a5c3b3g5h5g4h4e2g6b6d8c7c8a4a6a7f1a3c2d2b2e1b7g3h3f2d1a1a2b1a8c1g1f7g8e8f8b8g7h8h7h6h2g2h1"
    moves = 'E3F2E2D2E1F6D3D6G4G3G5H3E6H4H6G6F7E7D7F1H5H7G1C8G2F3C6D8E8C5B5F8C7B4B6C3A3B8C2C4B3B2G7A4A1A5B7A2A6D1B1C1H2H8G8H1A8A7'
    moves = 'E6F4C3C4F3D3E3E2E1D2C5G3H3C2D1C1B1B3A3B4G6F2A4B5A5A6A7B6F1G4H4D6C6B2A1A2A8B7B8C8C7D7D8E8E7F8F7G8H8G7H7G5H6H5H2G2G1H1'
    game = othello.OthelloGame(8, 8, 'B', 'W', 'M')

    allmoves = [moves[i:i +2] for i in range(0, len(moves), 2)]
    for move in allmoves:
        # notation is column + row, the game takes them separately
        row = int(move[1])-1
        column = int(cols.index(str.upper(move[0])))
        game.move(row, column)

    print(f'game played, final board: {game.current_board}')

def sample_source_data():
    df = cm.read_dataframe('parsed_data/clip/clip_training_source')
    print(f'dataframe read: {df.shape}')
    df100k = df.sample(n=100000, random_state=42)
    df500k = df.sample(n=500000, random_state=42)
    df1M = df.sample(n=1000000, random_state=42)
    df3M = df.sample(n=3000000, random_state=42)

    print('dataframes sampled')

    cm.save_dataframe('clip_training_source_100k', df100k)
    cm.save_dataframe('clip_training_source_500k', df500k)
    cm.save_dataframe('clip_training_source_1M', df1M)
    cm.save_dataframe('clip_training_source_3M', df3M)


def train_clip_model(source_file: str):

    trainer = ClipTrainer(source_file)
    df_train, df_val = trainer.make_train_validation_sets()

    # creating data loader
    train_loader = trainer.getLoader(df_train, 'Train')
    val_loader = trainer.getLoader(df_val, 'Val')

    model = CLIPModel().to(utils.get_device())

    optimizer = torch.optim.AdamW(
        #model.parameters(), lr=1e-3, weight_decay=1e-3
        model.parameters(), lr=0.1, weight_decay=1e-3
    )
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=2, factor=0.5
    )
    step = "epoch"

    best_loss = float('inf')

    epochs_count = 10

    for epoch in range(epochs_count):
        print(f"Epoch: {epoch + 1}")
        model.train()
        train_loss = clip.train.train_epoch(utils.get_device(), model, train_loader, optimizer, lr_scheduler, step)
        model.eval()
        with torch.no_grad():
            valid_loss = clip.train.valid_epoch(utils.get_device(), model, val_loader)

        if valid_loss.avg < best_loss:
            best_loss = valid_loss.avg
            torch.save(model.state_dict(), os.path.join(data_handling.common.DEFAULT_MODEL_OUTPUT_PATH,"best.pt"))
            print("Saved Best Model!")

def run_inference(model_file: str, eval_set: str):

    inference = ClipInference(model_file)

    result = inference.run_eval_file(eval_set)
    print(result)
    for p in result:
        print(result[p][2])

    # test validity of moves
    validity_result = inference.run_validity_check(result)
    print(validity_result)
    for r in validity_result:
        print(f'{validity_result[r][2]},{validity_result[r][3]}')

    '''
    test_position = "F5F6E6"
    move, model_result, values = inference.get_value_for_position(test_position)
    move_notation = convert_to_notation(move)
    print(f'prediction: move: {move_notation}, move int {move}, result: {model_result}')
    print(f'all values: {values}')
    '''

if __name__ == '__main__':

    # prepare data
    #prepare_clip_training()
    #parse_games()

    # test if model setup works
    #clip.model.testClip()

    # train
    #train_clip_model('parsed_data/clip/clip_training_source')

    # inference
    #run_inference("clip/models/clip_trained_10k_10epoch_lr01.pt","source_data/eval_sets/eval_openings.csv")
    run_inference("clip/models/clip_trained_full_3epoch_lr01_l4.pt", "source_data/eval_sets/eval_endgame.csv")


