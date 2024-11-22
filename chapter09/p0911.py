import sys
sys.path.append("..")
from instruments.percussions import tom
from instruments.utils import render_pad_save

render_pad_save(tom, save_title="p0911_tom.wav",
                velocity=100, gate=0.1, duration=1.4)