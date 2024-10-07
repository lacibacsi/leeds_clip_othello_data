import re
from enum import Enum
from typing import List, Tuple
from dataclasses import dataclass
from data_handling.pypaya_pgn_parser.movetext_tokenizer import MovetextTokenizer


class PlayerColor(Enum):
    # OTHELLO starts with Black
    BLACK = 1
    WHITE = 2


@dataclass
class ParseResult:
    moves: List[str]
    comments: List[Tuple[int, PlayerColor, str]]


class MovetextParser:
    # Othello denotes moves with capital file names, i.e. A, B, C ... H instead of a, b, c ... h - the rest is indifferent
    # as the is no capture, promotion, check, etc.
    PATTERNS = {
        "move_number": r'^\d+\.\.?\.?$',
        #"move": r'^([PNBRQK]?[a-h]?[1-8]?x?[a-h][1-8](=[NBRQ])?[+#]?|O-O(-O)?[+#]?)([!?]{1,2})?$',
        "move": r'^([PNBRQK]?[A-h]?[1-8]?x?[A-H][1-8](=[NBRQ])?[+#]?|O-O(-O)?[+#]?)([!?]{1,2})?$',
        "annotation": r'^([!?]{1,2}|\$\d+)$',
        "result": r'^(1-0|0-1|1/2-1/2|\*)$'
    }

    def __init__(self):
        self.tokenizer = MovetextTokenizer()
        self.regexes = {key: re.compile(pattern) for key, pattern in self.PATTERNS.items()}

    def parse(self, movetext: str) -> ParseResult:
        tokens = self.tokenizer.tokenize(movetext)
        moves = []
        comments = []
        current_move_number = 0
        #current_color = PlayerColor.WHITE
        #last_move_color = PlayerColor.BLACK
        current_color = PlayerColor.BLACK
        last_move_color = PlayerColor.WHITE
        current_comment = ""

        if not tokens:
            return ParseResult(moves, comments)

        if all(token.startswith('{') and token.endswith('}') for token in tokens):
            combined_comment = ' '.join(tokens)
            comments.append((0, PlayerColor.WHITE, combined_comment))
        else:
            for token in tokens:
                if self._is_move_number(token):
                    if current_comment:
                        comments.append((current_move_number, last_move_color, current_comment.strip()))
                        current_comment = ""
                    if "..." in token:
                        current_color = PlayerColor.BLACK
                    else:
                        current_move_number = int(token.rstrip('.'))
                        current_color = PlayerColor.WHITE
                elif self._is_move(token):
                    if current_comment:
                        comments.append((current_move_number, last_move_color, current_comment.strip()))
                        current_comment = ""
                    move, annotation = self._split_move_and_annotation(token)
                    moves.append(move)
                    if annotation:
                        comments.append((current_move_number, current_color, annotation))
                    last_move_color = current_color
                    current_color = PlayerColor.BLACK if current_color == PlayerColor.WHITE else PlayerColor.WHITE
                elif token.startswith('{') and token.endswith('}'):
                    current_comment += f" {token}"
                elif self.regexes["annotation"].match(token):
                    current_comment += f" {token}"
                elif self.regexes["result"].match(token):
                    current_comment += f" {token}"
                elif token.startswith('(') and token.endswith(')'):
                    current_comment += f" {token}"

            if current_comment:
                comments.append((current_move_number, last_move_color, current_comment.strip()))

        return ParseResult(moves, comments)

    def _is_move_number(self, token: str) -> bool:
        return bool(self.regexes["move_number"].match(token))

    def _is_move(self, token: str) -> bool:
        return bool(self.regexes["move"].match(token))

    def _split_move_and_annotation(self, token: str) -> Tuple[str, str]:
        match = self.regexes["move"].match(token)
        if match:
            move = match.group(1)
            annotation = match.group(4) or ""
            return move, annotation
        return token, ""
