import argparse

import NuRadioReco.modules.io.aera.adstReader
import NuRadioReco.modules.io.eventWriter
import NuRadioReco.modules.io.eventReader

from NuRadioReco.framework.parameters import showerParameters as shp

# Parse eventfile as argument
parser = argparse.ArgumentParser(description='ADST root')
parser.add_argument('inputfilename', type=str, nargs='*', help='path')

args = parser.parse_args()

# initialize modules
ADSTReader = NuRadioReco.modules.io.aera.adstReader.ADSTReader()
ADSTReader.begin(args.inputfilename)

# eventWriter = NuRadioReco.modules.io.eventWriter.eventWriter()
# output_filename = "Full_CoREAS_event.nur"
# eventWriter.begin(output_filename)

# for iE, event in enumerate(eventReader.run()):
for iE, event in enumerate(ADSTReader.run()):
    # print('Event: {} {}'.format(event.get_run_number(), event.get_id()))
    #
    # sim_shower = event.get_hybrid_information().get_hybrid_shower('SdRecShower')
    #
    # print('CR energy:', sim_shower.get_parameter(shp.energy))
    # # print('electromagnetic energy:', sim_shower.get_parameter(shp.electromagnetic_energy))

    pass
