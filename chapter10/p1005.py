import sys
sys.path.append("..")
from instruments.winds import bassoon
from instruments.utils import render_pad_save

render_pad_save(bassoon, save_title="p1005_bassoon.wav",
                note_no=34, velocity=100, gate=1.0, duration=2.0)