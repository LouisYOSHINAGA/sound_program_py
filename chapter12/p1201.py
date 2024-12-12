import sys
sys.path.append("..")
from instruments.keys import pipe_organ
from instruments.utils import render_pad_save

render_pad_save(pipe_organ, save_title="p1201_pipe_organ.wav",
                note_no=36, velocity=100, gate=1.0, duration=3.0)