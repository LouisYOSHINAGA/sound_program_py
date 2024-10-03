import numpy as np
from typing import Callable, Any


BITS_OF_BYTE: int = 8
N_NOTES: int = 128

CURRENT_TIME_IN_NOTEBUF: int = 0
VELOCITY_IN_NOTEBUF: int = 1
N_DATA_IN_NOTEBUF: int = 2  # current time, velocity

TRACK_IN_SCORE: int = 0
NOTEON_IN_SCORE: int = 1
CURRENT_TIME_IN_SCORE: int = 2
VELOCITY_IN_SCORE: int = 3
GATE_IN_SCORE: int =4


verbose: bool = False

tempo: int = 0
eotrack: int = 0
notebuf: np.ndarray = np.zeros((N_NOTES, N_DATA_IN_NOTEBUF), dtype=np.int16)
score: list[list[int]] = []  # list of [track, note no, current time, velocity, gate]


def vprint(msg: str) -> None:
    if verbose:
        print(msg)

def parse_value(midi: list[int], offset: int, n_byte: int) -> int:
    value: int = 0
    for i in range(n_byte):
        value += midi[offset+i] << (n_byte-1-i) * BITS_OF_BYTE
    return value

def parse_mthd(midi: list[int]) -> tuple[int, int, int]:
    BYTE_MTHD_ID: int = 4
    BYTE_MTHD_SIZE: int = 4
    BYTE_FORMAT_TYPE: int = 2
    BYTE_N_TRACKS: int = 2
    BYTE_DIVISION: int = 2

    offset: int = BYTE_MTHD_ID  # 'M', 'T', 'h', 'd'
    offset += BYTE_MTHD_SIZE  # 6

    format_type = parse_value(midi, offset, BYTE_FORMAT_TYPE)
    offset += BYTE_FORMAT_TYPE
    vprint(f"format: {format_type}")

    n_tracks: int = parse_value(midi, offset, BYTE_N_TRACKS)
    offset += BYTE_N_TRACKS
    vprint(f"number of tracks: {n_tracks}")

    division: int = parse_value(midi, offset, BYTE_DIVISION)
    offset += BYTE_DIVISION
    vprint(f"time division: {division}")

    return offset, n_tracks, division

def parse_mtrk_header(midi: list[int], offset: int) -> tuple[int, int]:
    BYTE_MTRK_ID: int = 4
    BYTE_MTRK_SIZE: int = 4

    offset += BYTE_MTRK_ID  # 'M', 'T', 'r', 'k'

    mtrk_size: int = parse_value(midi, offset, BYTE_MTRK_SIZE)
    offset += BYTE_MTRK_SIZE
    vprint(f"MTrk size: {mtrk_size}")

    return mtrk_size, offset

def parse_vardata(midi: list[int], offset: int) -> tuple[int, int]:
    """
        format: fxxx_xxxx (flag bit + 7bit data)
        if flag bit is 1 then continue else end
    """
    vardata: int = 0
    while True:
        fd: int = midi[offset]  # fxxx_xxxx
        offset += 1
        vardata = (vardata << 7) + (fd & 0x7F)
        if (fd >> 7) == 0:
            break
    return vardata, offset

def parse_message(midi: list[int], offset: int, current: int, track: int) -> tuple[int, int]:
    status: int = midi[offset]
    offset += 1

    stype: int = status >> 4
    # channel: int = status & 0x0F

    if status == 0xFF:
        offset = parse_meta_event(midi, offset, current)
    else:
        for value, parser in stype_parser_map.items():
            if stype == value:
                offset = parser(midi, offset, current, track)
    return offset

def parse_meta_event(midi: list[int], offset: int, current: int) -> int:
    global tempo, eotrack

    vprint(f"event type: meta event")

    # copyright notice: 0x_02_xx_y...y (2byte length, variable length data)
    if midi[offset] == 0x02:
        offset += 1
        n_byte, offset = parse_vardata(midi, offset)
        notice: str = parse_string(midi, offset, n_byte)
        offset += n_byte
        vprint(f"copyright notice: {notice}")

    # sequence  name / track name: 0x_02_xx_y...y (2byte length, variable length data)
    elif midi[offset] == 0x03:
        offset += 1
        n_byte, offset = parse_vardata(midi, offset)
        name: str = parse_string(midi, offset, n_byte)
        offset += n_byte
        vprint(f"name: {name}")

    # marker: 0x_02_xx_y...y (2byte length, variable length data)
    elif midi[offset] == 0x06:
        offset += 1
        n_byte, offset = parse_vardata(midi, offset)
        marker: str = parse_string(midi, offset, n_byte)
        offset += n_byte
        vprint(f"marker: {marker}")

    # port prefix: 0x_21_00_xx (1byte port prefix)
    elif midi[offset] == 0x21 and midi[offset+1] == 0x01:
        offset += 2
        vprint(f"port: {midi[offset]}")
        offset += 1

    # end of track: 0x_2F_00
    elif midi[offset] ==  0x2F and midi[offset+1] == 0x00:
        offset += 2
        vprint(f"end of track")
        if eotrack < current:
            eotrack = current

    # tempo: 0x_51_03_xx_xx_xx (3byte tempo value)
    elif midi[offset] == 0x51 and midi[offset+1] == 0x03:
        offset += 2
        tempo = parse_value(midi, offset, 3)
        offset += 3
        vprint(f"tempo: {tempo}")

    # beat
    elif midi[offset] == 0x58 and midi[offset+1] == 0x04:
        offset += 2
        vprint(f"beat: {midi[offset]}/{2**midi[offset+1]}")
        vprint(f"clocks per beat: {midi[offset+2]}")
        vprint(f"number of 32nd note in quarter note: {midi[offset+3]}")
        offset += 4

    # otherwise
    else:
        print(f"unsupported meta event: 0x{midi[offset]:02x}")
        offset += 1
        n_byte, offset = parse_vardata(midi, offset)
        offset += n_byte

    return offset

def parse_string(midi: list[int], offset, n_byte: int) -> str:
    string: str = ""
    for i in range(n_byte):
        string += chr(midi[offset+i])
    return string

def parse_note_off(midi: list[int], offset: int, current: int, track: int) -> int:
    global score

    noteno: int = midi[offset]
    velocity: int = midi[offset+1]  # note off velocity is not supported
    offset += 2

    vprint(f"event type: note off")
    vprint(f"note number: {noteno}")
    vprint(f"velocity: {velocity}")

    gate: int = current - notebuf[noteno, CURRENT_TIME_IN_NOTEBUF]
    score.append([track, noteno, *notebuf[noteno], gate])
    return offset

def parse_note_on(midi: list[int], offset: int, current: int, *disposed: tuple[Any]) -> int:
    global notebuf

    noteno: int = midi[offset]
    velocity: int = midi[offset+1]
    offset += 2

    vprint(f"event type: note on")
    vprint(f"note number: {noteno}")
    vprint(f"velocity: {velocity}")

    if velocity > 0:
        notebuf[noteno] = [current, velocity]
    else:
        parse_note_off(midi, offset-2, current)
    return offset

def parse_polyphonic_keypressure(midi: list[int], offset: int, *disposed: tuple[Any]) -> int:
    offset += 2
    return offset

def parse_control_change(midi: list[int], offset: int, *disposed: tuple[Any]) -> int:
    control_map: dict[int, str] = {
         1: "modulation (MSB)",
         7: "channel volume (MSB)",
        10: "pan (MSB)",
        11: "expression (MSB)",
    }

    ctrlno: int = midi[offset]
    value: int = midi[offset+1]
    offset += 2

    vprint(f"event type: control change")
    vprint(f"{control_map[ctrlno]} = {value}")
    return offset

def parse_program_change(midi: list[int], offset: int, *disposed: tuple[Any]) -> int:
    program_number: int = midi[offset]
    offset += 1

    vprint(f"even type: program change")
    vprint(f"program number: {program_number}")
    return offset

def parse_channel_pressure(midi: list[int], offset: int, *disposed: tuple[Any]) -> int:
    offset += 1
    return offset

def parse_pitch_bend(midi: list[int], offset: int, *disposed: tuple[Any]) -> int:
    lsb: int = midi[offset]
    msb: int = midi[offset+1]
    offset += 2

    vprint(f"event type: pitch bend")
    vprint(f"data = {(msb << 7 | lsb) - 8192}")
    return offset

def parse_system_exclusive(midi: list[int], offset: int, *disposed: tuple[Any]) -> int:
    _, offset = parse_vardata(midi, offset)
    vprint(f"event type: system exclusive\n")
    return offset

stype_parser_map: dict[int, Callable[[list[int], int], tuple[int, int]]] = {
    0x8: parse_note_off,
    0x9: parse_note_on,
    0xA: parse_polyphonic_keypressure,
    0xB: parse_control_change,
    0xC: parse_program_change,
    0xD: parse_channel_pressure,
    0xE: parse_pitch_bend,
    0xF: parse_system_exclusive,
}


def decode(file: str, is_verbose: bool =False) -> tuple[int, int, int, int, np.ndarray]:
    global verbose
    verbose = is_verbose

    with open(file, "rb") as f:
        midi: list[str] = f.read()

    vprint(f"================ header ================")
    offset, n_tracks, division = parse_mthd(midi)

    for track in range(n_tracks):
        vprint(f"\n================ track {track:02d} ================")
        mtrk_size, offset = parse_mtrk_header(midi, offset)
        eodata: int = mtrk_size + offset
        current: int = 0
        vprint(f"-----------------------")

        while offset < eodata:
            delta, offset = parse_vardata(midi, offset)
            current += delta
            vprint(f"delta time: {delta}")

            offset = parse_message(midi, offset, current, track)
            vprint(f"-----------------------")

    return division, tempo, n_tracks, eotrack, np.array(score, dtype=np.int16)