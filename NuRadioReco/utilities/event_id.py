import six
import numpy as np
from NuRadioReco.utilities.metaclasses import Singleton

@six.add_metaclass(Singleton)
class eventIdProvider():
    """
    Class to keep track of which event IDs are already in use. Any process that
    introduces an event with a new ID should have calls to this class to tell it
    which new IDs have been added. If a new unique event ID is needed, it can be
    provided by this class
    """
    def __init__(self):
        self.__event_ids = []

    def add_event_id(self, run_number, event_id):
        """
        Add a run_number, event_id pair to the list to register that these IDs
        are already in use and cannot be used when creating a new event.

        Parameters
        ---------------
        run_number: int
            Run number of the new event
        event_id: int
            Event ID of the new event
        """
        if [run_number, event_id] not in self.__event_ids:
            self.__event_ids.append([run_number, event_id])

    def get_unique_event_id(self, run_number=None):
        """
        Returns a new, yet unused event ID.
        For simplicity, we take the largest event ID, increment by 1 and return
        the result

        Parameters
        ---------------
        run_number: int
        Only IDs with the same run number are considered.
        """
        if len(self.__event_ids) == 0:
            return 0
        event_ids = np.array(self.__event_ids)
        if run_number is None:
            return np.max(event_ids[:,1]) + 1
        else:
            masked_event_ids = event_ids[:,1][event_ids[:,0] == run_number]
            if len(masked_event_ids) == 0:
                return 0
            else:
                return np.max(masked_event_ids) + 1
