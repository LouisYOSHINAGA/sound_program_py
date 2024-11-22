import sys
sys.path.append("..")
from instruments.percussions import tubular_bells
from instruments.utils import render_pad_save

render_pad_save(tubular_bells, save_title="p0903_tubular_bells.wav",
                note_no=60, velocity=100, gate=0.1, duration=5.0)