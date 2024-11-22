import sys
sys.path.append("..")
from instruments.percussions import cymbal
from instruments.utils import render_pad_save

render_pad_save(cymbal, save_title="p0907_cymbal.wav",
                velocity=100, gate=0.1, duration=4.0)