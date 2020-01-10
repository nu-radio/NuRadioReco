import matplotlib.pyplot as plt

import NuRadioReco.modules.io.lofar.readLOFAR
from NuRadioReco.utilities import units

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('readARAexample')

# Initialize the reader with the file you want to read
LOFARreader = NuRadioReco.modules.io.lofar.readLOFAR.readLOFAR(['/Users/anelles/Experiments/LOFAR/A_NuRadioReco_Test/L78862_D20121205T051644.039Z_CS002_R000_tbb.h5'],DAL=1)

for iE, event in enumerate(LOFARreader.run()):
    print(event.get_id())


