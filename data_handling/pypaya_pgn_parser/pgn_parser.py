import io
from typing import Optional, Dict, List, Tuple
import re
from data_handling.pypaya_pgn_parser.movetext_parser import MovetextParser
from data_handling.pypaya_pgn_parser.headers import HEADERS, DEFAULT_VALUES


class PGNParser:
    HEADER_REGEX = re.compile(r'\[(\w+)\s+"(.*)"\]')

    # NOT True for Othello
    GAME_RESULT_REGEX = re.compile(r'\s(1-0|0-1|1/2-1/2|\*)\s*$')

    def __init__(self):
        self.movetext_parser = MovetextParser()

    def parse(self, stream: io.StringIO) -> Optional[Tuple[List[str], str]]:
        """Parse the PGN file."""
        try:
            return self._parse_custom(stream)
        except Exception as e:
            print(f"Error parsing PGN: {e}")
            return None

    def _parse_custom(self, stream: io.StringIO) -> Optional[Tuple[List[str], str]]:
        """Parse the PGN file using custom parsing logic."""
        start_pos = stream.tell()
        headers = self._parse_headers(stream)

        # If no headers were found, we might be at the end of the file
        if not headers:
            if stream.tell() == start_pos:
                return None  # We're at the end of the stream
            else:
                stream.seek(start_pos)  # Reset to start of current game

        movetext = self._extract_movetext(stream)

        # Even if movetext is empty, we should still return the headers
        game_info = self._create_game_info(headers)

        if movetext:
            parse_result = self.movetext_parser.parse(movetext)
            moves_str = " ".join(parse_result.moves)
            self._update_result(game_info, movetext)
        else:
            moves_str = ""

        return game_info, moves_str

    def _parse_headers(self, stream: io.StringIO) -> Dict[str, str]:
        """Parse the headers from the PGN file."""
        headers = {}
        for line in stream:
            line = line.strip()
            if not line.startswith('['):
                break
            match = self.HEADER_REGEX.match(line)
            if match:
                key, value = match.groups()
                headers[key] = value
        return headers

    def _extract_movetext(self, stream: io.StringIO) -> str:
        """Extract the movetext from the PGN file."""
        movetext = []
        for line in stream:
            line = line.strip()
            if line.startswith('['):
                stream.seek(stream.tell() - len(line) - 1)
                break
            movetext.append(line)
        return " ".join(movetext).strip()

    def _create_game_info(self, headers: Dict[str, str]) -> List[str]:
        """Create the game info list from parsed headers."""
        return [headers.get(header, DEFAULT_VALUES[i]) for i, header in enumerate(HEADERS[:-1])]

    def _update_result(self, game_info: List[str], movetext: str):
        """Update the game result in game_info if found in movetext."""
        match = self.GAME_RESULT_REGEX.search(movetext)
        if match:
            game_info[6] = match.group(1)
