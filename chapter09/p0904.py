import sys
sys.path.append("..")
from instruments.percussions import marimba
from instruments.utils import render_pad_save

render_pad_save(marimba, save_title="p0904_marimba.wav",
                note_no=48, velocity=100, gate=0.1, duration=1.8)