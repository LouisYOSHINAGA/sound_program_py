import sys
sys.path.append("..")
from instruments.percussions import kick
from instruments.utils import render_pad_save

render_pad_save(kick, save_title="p0910_kick.wav",
                velocity=100, gate=0.1, duration=1.3)