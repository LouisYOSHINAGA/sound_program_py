import sys
sys.path.append("..")
from instruments.strings import slap_bass
from instruments.utils import render_pad_save

render_pad_save(slap_bass, save_title="p1109_slap_bass.wav",
                note_no=36, velocity=100, gate=4.0, duration=5.0)