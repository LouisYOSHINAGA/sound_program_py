import sys
sys.path.append("..")
from instruments.strings import cello
from instruments.utils import render_pad_save

render_pad_save(cello, save_title="p1103_cello.wav",
                note_no=60, velocity=100, gate=1.0, duration=2.0)