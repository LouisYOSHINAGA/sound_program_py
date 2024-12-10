import sys
sys.path.append("..")
from instruments.strings import acoustic_guitar
from instruments.utils import render_pad_save

render_pad_save(acoustic_guitar, save_title="p1106_acoustic_guitar.wav",
                note_no=60, velocity=100, gate=4.0, duration=5.0)