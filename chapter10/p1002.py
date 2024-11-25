import sys
sys.path.append("..")
from instruments.winds import piccolo
from instruments.utils import render_pad_save

render_pad_save(piccolo, save_title="p1002_piccolo.wav",
                note_no=74, velocity=100, gate=1.0, duration=2.0)