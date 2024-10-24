A4_FREQ: float = 440.0
A4_NOTE: int = 69

def calc_freq(note_no: int) -> float:
    return A4_FREQ * 2 ** ((note_no - A4_NOTE) / 12)