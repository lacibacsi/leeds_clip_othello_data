from handling.game_parser import GameParser
from handling.common import convert_to_notation

# Press ⌃R to execute it or replace it with your handling.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

if __name__ == '__main__':
    parser = GameParser()
    counter = 0
    #counter = parser.read_pgn_files(relativ_dir='pgn/', save_output=True)
    #counter += parser.read_csv_files(relativ_dir='csv/', save_output=True)
    counter += parser.read_pickle_files(relativ_dir='othello_synthetic/', save_output=True)

    print(f'parsed {counter} games')




