import os
import sys


class Quantities:

    def __init__(self):
        self.elements = {}

    def __repr__(self):
        ret = ''
        for k, v in self.__dict__.iteritems():
            if k == 'elements':
                continue
            ret += str(k)
            ret += ' = '
            ret += str(v)
            ret += '\n'
        return ret


if "AUGEROFFLINEROOT" in os.environ:
    pathPrefix = os.path.join(os.environ["AUGEROFFLINEROOT"], "include/adst/")
else:
    sys.exit("Environment variable AUGEROFFLINEROOT not set! Aborting ...")

# Objects which contain Enums
rdstQ = Quantities()
rdshQ = Quantities()
rdchQ = Quantities()

for ifile in (os.path.join(pathPrefix, 'StationRRecDataQuantities.h'),
              os.path.join(pathPrefix, 'ShowerRRecDataQuantities.h'),
              os.path.join(pathPrefix, 'ChannelRRecDataQuantities.h')):

    if not os.path.exists(ifile):
        raise IOError("File does not exist!\n" + ifile)

    if ifile.find('Station') > -1:
        elem = rdstQ
    elif ifile.find('Shower') > -1:
        elem = rdshQ
    elif ifile.find('Channel') > -1:
        elem = rdchQ
    else:
        raise IOError('failed to load,' + ifile)

    # parse through *Quantities.h
    for iline in open(ifile):
        iline = iline.strip()
        if iline.find('=') > -1 and iline[0] == 'e':
            iline = iline.split(',')[0]
            iline = iline.split('=')
            idx = iline[0]
            for i in iline[1].split(' '):
                if i != '':
                    val = i
                    break
            elem.__dict__[idx.strip()] = int(val.strip())
            elem.elements[idx.strip()] = int(val.strip())

# Example code
if __name__ == "__main__":

    import imp


    ce = imp.load_source('', './convertenums.py')
    rdstQ = ce.rdstQ  # fuer stations
    rdshQ = ce.rdshQ  # fue showers
    print(rdstQ.eGeomagneticEnergyFluence)
