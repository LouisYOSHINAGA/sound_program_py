import sys
sys.path.append("..")
from instruments.percussions import glockenspiel
from instruments.utils import render_pad_save

render_pad_save(glockenspiel, save_title="p0901_glockenspiel.wav",
                note_no=72, velocity=100, gate=0.1, duration=5.0)