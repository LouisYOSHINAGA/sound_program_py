import sys
sys.path.append("..")
from instruments.percussions import xylophone 
from instruments.utils import render_pad_save

render_pad_save(xylophone, save_title="p0905_xylophone.wav",
                note_no=65, velocity=100, gate=0.1, duration=1.8)