from NuRadioReco.modules.io.snowshovel import readARIANNADataCalib as CreadARIANNAData

"""
Takes a calibrated snowShovel .root file and prints out the times and trigger type for each event.
"""

readARIANNAData = CreadARIANNAData.readARIANNAData()


def printHeaderDetailsPerEvent(path,file):
	rootFile = path + file

	#tfile = ROOT.TFile(rootFile)
	#RawTree = tfile.Get("RawTree")
	ConsecutiveEntryNum = 0
	n_events = readARIANNAData.begin([rootFile])
	print n_events
	event_count = 0
	for evt in readARIANNAData.run():
		for station_object in evt.get_stations():
			status = 'file: ' + rootFile + ' event: ' + str(event_count) + ' utc_time: ' + str(station_object.get_station_time()) + ' Thermal? ' + str(station_object.has_triggered())
			print status
			time = str(station_object.get_station_time())[11:]
			event_count += 1

def main():
	printHeaderDetailsPerEvent('/home/geoffrey/ARIANNA/Jan8Spice/','CalTree.RawTree.SnEvtsM0002F7F2E7B9r00212s00084.root')

if __name__== "__main__":
	main()
