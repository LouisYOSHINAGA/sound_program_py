import sys
sys.path.append("..")
from instruments.keys import electric_piano
from instruments.utils import render_pad_save

render_pad_save(electric_piano, save_title="p1205_electric_piano.wav",
                note_no=60, velocity=100, gate=4.0, duration=5.0)