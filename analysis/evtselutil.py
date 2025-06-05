#!/usr/bin/env python
import dask, logging
import pandas as pd
import awkward as ak
import operator as opr
from typing import Optional
from src.utils.coffeautil import weightedSelection, PackedSelection, sequentialSelection
from src.analysis.objutil import ObjectProcessor
from functools import lru_cache

class CommonObjSelMixin():
    def apply_dr_selections(self, events, objname, drcut, obj_mask, sortname='pt', selection_name='dR') -> tuple[ObjectProcessor, ak.Array, ak.Array, ak.Array, ak.Array]:
        """Apply delta R selections to objects in the event data.
        Returns:
            obj_proc: ObjectProcessor instance for the selected object
            events: Filtered events after applying delta R selections
            sortname: Name of the sorting column
            ld_obj: Leading object after applying delta R
            sd_obj: Subleading object after applying delta R"""
        obj_proc = self.getObjProc(events, objname, sortname=sortname)
        dR_mask_event = obj_proc.event_level_dr_mask(events, obj_mask, drcut)
        obj_proc, events = self.apply_selection_mask(events, selection_name, obj_proc, dR_mask_event)
        
        obj_mask = obj_mask[dR_mask_event]
        obj_proc = self.getObjProc(events, objname, sortname=sortname)
        ld_obj, sd_obj = obj_proc.apply_obj_level_dr(events, obj_mask, drcut)
        n_obj = ak.sum(obj_mask, axis=1)
        
        return obj_proc, events, ld_obj, sd_obj, n_obj
    
    def apply_objsel_trigger_match(self, events, objname, trigger_id, obj_base_cond, num_selname, dr_cut=0.1, match_selname='Trigger Match') -> tuple[ObjectProcessor, ak.Array, ak.Array]:
        """Apply object selection and trigger matching to events.
        Returns:
            obj_proc: ObjectProcessor instance for the selected object
            events: Filtered events after applying object selection and trigger matching
            obj_mask: Boolean mask representing selected objects"""
        obj_proc = self.getObjProc(events, objname, sortname=None)
        obj_mask = obj_proc.create_combined_mask(obj_base_cond)
        obj_nummask = obj_proc.numselmask(obj_mask, opr.ge)
        obj_proc, events = self.apply_selection_mask(events, num_selname, obj_proc, obj_nummask)

        obj_mask = (obj_proc.create_combined_mask(obj_base_cond) & obj_proc.match_trigger_object(trigger_id, dr_cut=dr_cut))
        tau_nummask = obj_proc.numselmask(obj_mask, opr.ge)
        obj_proc, events = self.apply_selection_mask(events, match_selname, obj_proc, tau_nummask)

        obj_mask = obj_mask[tau_nummask]

        return obj_proc, events, obj_mask
        
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
        logging.debug(f"Deleting instance of {self.__class__.__name__}")

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
    
    def getObjProc(self, events, name, **kwargs) -> ObjectProcessor:
        return ObjectProcessor(events, name, self.objselcfg[name], self.mapcfg[name], **kwargs)

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
 
class PreselSelections(CommonObjSelMixin, BaseEventSelections):
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

    def apply_selection_mask(self, events: ak.Array, name: str, obj: ObjectProcessor, mask: ak.Array) -> tuple[ObjectProcessor, ak.Array]:
        """Apply selection mask, update selection history, filter objcollect, and update object.

        Args:
            events (ak.Array): Input events to be filtered
            name (str): Name of the selection being applied
            obj (ObjectProc): Object instance to be updated
            mask (ak.Array): Boolean mask array with same length as events

        Returns:
            tuple[ObjectProcessor, ak.Array]: Updated (Object instance, filtered events)
        """
        # Update selection history and filter objcollect
        if self._sequential and self.objsel.names:
            previous_mask = self.objsel.any(self.objsel.names[-1])
            self.objsel.add_sequential(name, mask, previous_mask)
        else:
            self.objsel.add(name, mask)

        if self.objcollect:
            self.objcollect = {
                key: val[mask] for key, val in self.objcollect.items()
            }

        # Filter events and update object
        filtered_events = events[mask]
        obj.events = filtered_events
        if isinstance(filtered_events, ak.Array):
            logging.debug(f"Selection {name} passed {len(filtered_events)} events!")
        return obj, filtered_events
  