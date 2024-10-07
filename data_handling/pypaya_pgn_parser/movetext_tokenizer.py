import re
from enum import Enum, auto
from typing import List, Union


class TokenType(Enum):
    MOVE_NUMBER = auto()
    MOVE = auto()
    COMMENT = auto()
    VARIATION = auto()
    RESULT = auto()
    ANNOTATION = auto()


class Token:
    def __init__(self, type: TokenType, value: str):
        self.type = type
        self.value = value

    def __repr__(self):
        return f"Token({self.type}, '{self.value}')"

    def __eq__(self, other):
        if isinstance(other, Token):
            return self.type == other.type and self.value == other.value
        return False


class State(Enum):
    NORMAL = auto()
    IN_COMMENT = auto()
    IN_VARIATION = auto()


class MovetextTokenizer:
    # same as for the parser, changing the regex to accommodate differet notation
    MOVE_NUMBER_PATTERN = re.compile(r'^\d+\.\.?\.?$')
    #MOVE_PATTERN = re.compile(r'^[KQRBNP]?[a-h]?[1-8]?x?[a-h][1-8](=[QRBN])?[+#]?$|^O-O(-O)?[+#]?$')
    MOVE_PATTERN = re.compile(r'^[KQRBNP]?[A-H]?[1-8]?x?[A-H][1-8](=[QRBN])?[+#]?$|^O-O(-O)?[+#]?$')
    RESULT_PATTERN = re.compile(r'^(1-0|0-1|1/2-1/2|\*)$')
    ANNOTATION_PATTERN = re.compile(r'^\$?\d+$|^[!?]+$')

    @classmethod
    def tokenize(cls, movetext: str) -> List[str]:
        tokens = cls._tokenize_internal(movetext)
        return [cls._token_to_string(token) for token in tokens]

    @classmethod
    def _tokenize_internal(cls, movetext: str) -> List[Union[str, Token]]:
        state = State.NORMAL
        tokens = []
        current_token = []
        variation_depth = 0

        for char in movetext:
            if state == State.NORMAL:
                if char == '{':
                    cls._add_token(current_token, tokens)
                    current_token = [char]
                    state = State.IN_COMMENT
                elif char == '(':
                    cls._add_token(current_token, tokens)
                    current_token = [char]
                    state = State.IN_VARIATION
                    variation_depth = 1
                elif char.isspace():
                    cls._add_token(current_token, tokens)
                    current_token = []
                else:
                    current_token.append(char)
            elif state == State.IN_COMMENT:
                current_token.append(char)
                if char == '}':
                    cls._add_token(current_token, tokens)
                    current_token = []
                    state = State.NORMAL
            elif state == State.IN_VARIATION:
                current_token.append(char)
                if char == '(':
                    variation_depth += 1
                elif char == ')':
                    variation_depth -= 1
                    if variation_depth == 0:
                        cls._add_token(current_token, tokens)
                        current_token = []
                        state = State.NORMAL

        cls._add_token(current_token, tokens)
        return tokens

    @classmethod
    def _add_token(cls, current_token: List[str], tokens: List[Union[str, Token]]) -> None:
        if not current_token:
            return

        value = ''.join(current_token)
        if value.startswith('{') and value.endswith('}'):
            tokens.append(Token(TokenType.COMMENT, value))
        elif value.startswith('(') and value.endswith(')'):
            tokens.append(Token(TokenType.VARIATION, value))
        elif cls.MOVE_NUMBER_PATTERN.match(value):
            tokens.append(Token(TokenType.MOVE_NUMBER, value))
        elif cls.MOVE_PATTERN.match(value):
            tokens.append(Token(TokenType.MOVE, value))
        elif cls.RESULT_PATTERN.match(value):
            tokens.append(Token(TokenType.RESULT, value))
        elif cls.ANNOTATION_PATTERN.match(value):
            tokens.append(Token(TokenType.ANNOTATION, value))
        else:
            tokens.append(value)  # Unrecognized token, keep as string

    @staticmethod
    def _token_to_string(token: Union[str, Token]) -> str:
        if isinstance(token, Token):
            return token.value
        return token
