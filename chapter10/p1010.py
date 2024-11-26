import sys
sys.path.append("..")
from instruments.brasses import tuba
from instruments.utils import render_pad_save

render_pad_save(tuba, save_title="p1010_tuba.wav",
                note_no=36, velocity=100, gate=1.0, duration=2.0)