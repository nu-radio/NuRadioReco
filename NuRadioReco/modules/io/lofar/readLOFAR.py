import sys
import numpy as np
import matplotlib.pyplot as plt

#LOFAR library
import NuRadioReco.modules.io.lofar.raw_tbb_IO as IO
import NuRadioReco.modules.io.lofar.metadata as md

import NuRadioReco.framework.event
import NuRadioReco.framework.station
import NuRadioReco.framework.channel

import logging
logger = logging.getLogger("readLOFAR")
logging.basicConfig()


def to_lofar_id(station_id):
    """
    Convert NuRadioReco event id (int) to LOFAR str event id

    Parameters:
    ----------
    station_id: int
        NuRadioReco ID
    """
    id = str(station_id)[-3:]
    if station_id > 3000:
        logger.error("Station ID {} too high, not implemented".format(station_id))
    elif station_id > 2000:
        LOFAR_id = 'RS'+id
    else:
        LOFAR_id = 'CS' + id
    return LOFAR_id


def from_lofar_id(LOFAR_id):
    """
    Convert LOFAR str event id to NuRadioReco event id

    Parameters:
    ----------
    station_id: str
        LOFAR id, in form of CS002 etc.

    """
    if 'RS' in LOFAR_id:
        station_id = 2000
        station_id += int(LOFAR_id[-3])
    elif "CS" in LOFAR_id:
        station_id = 1000
        station_id += int(LOFAR_id[-3:])
    else:
        logger.error("Station type {} not implemented".format(LOFAR_id))
    return station_id

def get_event_id_from_time(timestamp):
    """
    Get LOFAR standard event ID

    Parameters:
    ------------
    timestamp: int
        unix timestamp of event

    """
    timestamp_0 = 1262304000  # Unix timestamp on Jan 1, 2010 (date -u --date "Jan 1, 2010 00:00:00" +"%s")
    result = int(timestamp - timestamp_0)

    return result


class readLOFAR:

    """
    Reads LOFAR data
    """


    def __init__(self,filenames = [], DAL=1):
        """
        Init read LOFAR class

        -------------
        Parameters:
        filenames: list
            list of filenames belonging to one event
            will be sorted by station
        DAL: int
            1: data access library version of LOFAR
            currently implemented 1
            default: 1

        """
        self._DAL = DAL
        if DAL not in [1]:
            logger.error("Version of DAL {} not implemented".format(DAL))
            sys.exit()

        filenames_dict = {}
        for filename in filenames:
            if self._DAL == 1:
                file = IO.TBBData_Dal1(filename)
                LOFAR_id = (file.get_station_name())
                if LOFAR_id in filenames_dict.keys():
                    filenames_dict[LOFAR_id].append(filename)
                else:
                    filenames_dict[LOFAR_id] = [filename]


        self.filenames = filenames_dict


    def run(self):
        for station in self.filenames.keys():
            if self._DAL == 1:
                file_object = IO.MultiFile_Dal1(self.filenames[station])
            else:
                logger.error("Version of DAL {} not implemented".format(self._DAL))

            itrf_positions = file_object.get_ITRF_antenna_positions()
            positions = md.convertITRFToLocal(itrf_positions)

            antenna_set = file_object.get_antenna_set()
            if 'LBA' in antenna_set:
                antenna_model = 'LBA'
            elif 'HBA' in antenna_set:
                antenna_model = 'HBA'
            else:
                logger.warning("Antenna set {} not recognized.".format(antenna_set))

            evt_number = get_event_id_from_time(file_object.get_timestamp())

            # LOFAR does not operate with run numbers
            evt = NuRadioReco.framework.event.Event(1, evt_number)

            yield evt

    def end(self):
        pass



if __name__ == "__main__":

    f = readLOFAR(['/Users/anelles/Experiments/LOFAR/A_NuRadioReco_Test/L78862_D20121205T051644.039Z_CS002_R000_tbb.h5'],DAL=1)

    f.run()