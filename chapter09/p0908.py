import sys
sys.path.append("..")
from instruments.percussions import tamtam
from instruments.utils import render_pad_save

render_pad_save(tamtam, save_title="p0908_tamtam.wav",
                velocity=100, gate=2.5, duration=9.0)