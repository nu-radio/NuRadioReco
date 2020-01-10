First attempt in writing an LOFAR reader to look into using NuRadioReco for LOFAR

The files:
metadata.py, raw_tbb_IO.py, readLOFAR.py,  utilities.py
have been copied from Brian Hare
https://github.com/Bhare8972/LOFAR-LIM/blob/master/LIM_scripts/

His python package is very useful, but does not follow python convention
for modules, which it is copied rather than imported. This should be worked
on in the future.

The files also need LOFAR metadata to be downloaded:
https://github.com/Bhare8972/LOFAR-LIM/blob/master/LIM_scripts/data.tar.gz
and un-tarred in the folder.


