import sys
sys.path.append("..")
from instruments.percussions import hihat
from instruments.utils import render_pad_save

render_pad_save(hihat, save_title="p0909_hihat.wav",
                velocity=100, gate=0.1, duration=1.1)