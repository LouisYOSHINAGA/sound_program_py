import sys
sys.path.append("..")
from instruments.percussions import triangle_in, triangle_out
from instruments.utils import render_pad_save

render_pad_save(triangle_in, save_title="p0902_triangle_in.wav",
                velocity=100, gate=0.1, duration=9.0)
render_pad_save(triangle_out, save_title="p0902_triangle_out.wav",
                velocity=100, gate=0.1, duration=9.0)