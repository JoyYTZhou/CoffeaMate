#!/usr/bin/env python
import dask, logging
import pandas as pd
import awkward as ak
from typing import Optional
from src.utils.coffeautil import weightedSelection, PackedSelection, sequentialSelection
from src.analysis.objutil import ObjectMasker, ObjectProcessor
from functools import lru_cache

class BaseEventSelections:
    """Base class for handling event selections in particle physics analysis.

    This class provides the fundamental structure for applying various selection criteria
    to particle physics event data. It handles trigger configurations, object selections,
    and name mapping between different data formats (AOD/NANOAOD).

    Attributes:
        _trigcfg (dict): Configuration for trigger selections
        _objselcfg (dict): Configuration for object selections
        _mapcfg (dict): Name mapping between different data formats
        _sequential (bool): Flag for sequential selection application
        objsel (Optional[weightedSelection]): Selection object for storing selection results
        objcollect (dict): Dictionary for collected objects
        cfno: Cutflow number object
        cfobj: Cutflow object
    """
    def __init__(self,
                 trigcfg: dict[str, bool],
                 objselcfg: dict[str, dict[str, float]],
                 mapcfg: dict[str, dict[str, str]],
                 is_mc: bool,
                 sequential: bool) -> None:
        """Initialize the event selection object.

        Args:
            trigcfg: Trigger configuration mapping trigger names to boolean flags
            objselcfg: Object selection configuration mapping AOD prefixes to selection criteria
            mapcfg: Mapping configuration for converting between AOD and NANOAOD names
            sequential: Whether selections should be applied sequentially
        """
        self._trigcfg = trigcfg
        self._objselcfg = objselcfg
        self._mapcfg = mapcfg
        self._sequential = sequential
        self.objsel: Optional[weightedSelection] = None
        self.objcollect: dict = {}
        self._with_wgt = is_mc
        self.cfno = None
        self.cfobj = None

    def __del__(self):
        """Cleanup method called when the instance is being destroyed."""
        print(f"Deleting instance of {self.__class__.__name__}")

    def __enter__(self):
        """Context manager entry point."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit point. Cleans up resources."""
        self.objsel = None
        self.objcollect.clear()
        self.cfno = None
        self.cfobj = None

    @property
    def trigcfg(self):
        """Returns the trigger configuration dictionary."""
        return self._trigcfg

    @property
    def objselcfg(self):
        """Returns the object selection configuration dictionary."""
        return self._objselcfg

    @property
    def mapcfg(self):
        """Returns the name mapping configuration dictionary."""
        return self._mapcfg

    def callevtsel(self, events, wgtname='Generator_weight') -> tuple[ak.Array, 'BaseEventSelections']:
        """Apply all selections to the events sequentially.

        Args:
            events: Input events array
            wgtname: Name of the weight column
            compute: Whether to compute results immediately

        Returns:
            tuple: (Passed events, self reference)
        """
        logging.debug(f"Performing selections on events with {len(events)} entries!")
        self._setobjsel(events)
        self._triggersel(events)
        self._setevtsel(events)
        self._getcutflow()
        return self._getpassed(events), self
    
    def _setobjsel(self, events):
        """Initialize selection object based on data type.

        Uses weightedSelection for MC and PackedSelection for real data.
        """
        if self._with_wgt:
            logging.debug("Using weighted selections")
            self.objsel = weightedSelection(events['Generator_weight'])
        else:
            if self._sequential:
                logging.debug("Using unweighted, sequential selections.")
                self.objsel = sequentialSelection()
            else:
                logging.debug("Using unweighted, PackedSelections")
                self.objsel = PackedSelection()

    def _getcutflow(self) -> pd.DataFrame:
        """Calculate and store cutflow information."""
        self.cfobj = self.objsel.cutflow(*self.objsel.names)
        self.cfno = self.cfobj.result()
    
    def _getpassed(self, events) -> ak.Array:
        raise NotImplementedError("Method must be implemented in derived class!")
    
    def _triggersel(self, events):
        raise NotImplementedError("Method must be implemented in derived class!")

    def _setevtsel(self, events):
        raise NotImplementedError("Method must be implemented in derived class!")
    
    def getObjMasker(self, events, name, **kwargs) -> ObjectMasker:
        return ObjectMasker(events, name, self.objselcfg[name], self.mapcfg[name], **kwargs)
    
    def getObjProc(self, name) -> ObjectProcessor:
        return ObjectProcessor(name, self.mapcfg[name])

    @lru_cache(maxsize=32)
    def cf_to_df(self) -> pd.DataFrame:
        """Convert cutflow information to a pandas DataFrame.

        Returns:
            DataFrame containing raw counts and (if applicable) weighted counts
        """
        row_names = self.cfno.labels
        dfdata = {}
        if self._with_wgt:
            wgt_number = dask.compute(self.cfno.wgtevcutflow)[0]
            dfdata['wgt'] = wgt_number
        number = dask.compute(self.cfno.nevcutflow)[0]
        dfdata['raw'] = number
        return pd.DataFrame(dfdata, index=row_names)

class SkimSelections(BaseEventSelections):
    """Class for handling both Monte Carlo and real data skim selections.

    This class combines functionality for both MC and real data skims,
    with weight handling determined by a flag at initialization.
    """
    def _triggersel(self, events):
        """Apply trigger selections based on configuration."""
        for trigname, value in self.trigcfg.items():
            if value:
                self.objsel.add(trigname, events[trigname])
            else:
                inverted = ~events[trigname]
                self.objsel.add(trigname, inverted)
    
    def _getpassed(self, events):
        return events[self.cfno.maskscutflow[-1]]
 
class PreselSelections(BaseEventSelections):
    """Class for handling selections that produce n-tuples."""
    def _getpassed(self, events):
        return self.objcollect_to_df()
    
    def _triggersel(self, events):
        pass

    def _saveAttributes(self, events, attribute_names) -> None:
        """Save attributes to the collected objects."""
        self.objcollect.update({attr: events[attr] for attr in attribute_names})

    def objcollect_to_df(self) -> pd.DataFrame:
        """Convert collected objects to a pandas DataFrame.

        Returns:
            DataFrame containing all collected object information
        """
        listofdf = [None] * len(self.objcollect)
        for i, (prefix, zipped) in enumerate(self.objcollect.items()):
            listofdf[i] = ObjectProcessor.object_to_df(zipped, f"{prefix}_")
        return pd.concat(listofdf, axis=1)
    
    def saveWeights(self, events: ak.Array, weights=['Generator_weight', 'LHEReweightingWeight']) -> None:
        """Save weights to the collected objects."""
        weights = ['Generator_weight', 'LHEReweightingWeight']
        self._saveAttributes(events, weights)
    
    def handle_selection_masks(self, name: str, mask: ak.Array) -> None:
        """Handle selection masks and update selection history.

        Args:
            name (str): Name of the selection being applied
            mask (ak.Array): Boolean mask array for selection

        This method:
        1. Applies selection and updates selection history
        2. Filters previously collected objects with the mask
        """
        # Apply selection and update selection history
        if self._sequential and self.objsel.names:
            previous_mask = self.objsel.any(self.objsel.names[-1])
            self.objsel.add_sequential(name, mask, previous_mask)
        else:
            self.objsel.add(name, mask)

        # Filter previously collected objects
        if self.objcollect:
            self.objcollect = {
                key: val[mask] for key, val in self.objcollect.items()
            }

    def selobjhelper(self, events: ak.Array, name: str, obj: ObjectMasker, mask: ak.Array) -> tuple[ObjectMasker, ak.Array]:
        """Apply selection mask to events and update object collections.

        This function handles both event-level and object-level selections by:
        1. Applying a selection mask to the events
        2. Updating the selection history in objsel
        3. Filtering previously collected objects with the same mask
        4. Updating the Object instance with filtered events

        Args:
            events (ak.Array): Input events to be filtered
            name (str): Name of the selection being applied
            obj (Object): Object instance to be updated
            mask (ak.Array): Boolean mask array with same length as events

        Returns:
            tuple[Object, ak.Array]: Updated (Object instance, filtered events)

        Examples:
            >>> events = ak.Array(...)
            >>> obj = Object(...)
            >>> mask = events.pt > 30
            >>> obj, filtered_events = selobjhelper(events, "pt_cut", obj, mask)

        Notes:
            - For sequential selections, the mask is combined with the previous selection
            - All previously collected objects in self.objcollect are filtered with the same mask
            - The input Object instance is updated with the filtered events
        """
        if isinstance(events, ak.Array):
            logging.debug(f"Applying selection mask to {len(events)} events!")
        
        self.handle_selection_masks(name, mask)
        
        # Filter events and update object
        filtered_events = events[mask]

        if isinstance(filtered_events, ak.Array):
            logging.debug(f"Selection {name} passed {len(filtered_events)} events!")

        obj.events = filtered_events
        return obj, filtered_events