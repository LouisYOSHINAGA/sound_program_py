import sys
sys.path.append("..")
from instruments.percussions import timpani
from instruments.utils import render_pad_save

render_pad_save(timpani, save_title="p0906_timpani.wav",
                note_no=36, velocity=100, gate=0.1, duration=3.0)