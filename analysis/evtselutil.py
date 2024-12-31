#!/usr/bin/env python
import dask
import pandas as pd
import awkward as ak

from src.utils.coffeautil import weightedSelection
from src.analysis.objutil import Object

class BaseEventSelections:
    """Base class for event selections.
    """
    def __init__(self, trigcfg, objselcfg, mapcfg, sequential=True) -> None:
        """Initialize the event selection object with the given selection configurations.
        
        Parameters
        - `trigcfg`: trigger configuration, {key=triggername, value=bool (whether to apply)}
        - `objselcfg`: object selection configuration, {key=AODPrefix, value={key=abbreviation, value=threshold}}. AODPrefix example: Electron, Muon, Jet
        - `mapcfg`: mapping configuration, {key=AODPrefix, value={key=abbreviation, value=nanoaodname}}
        - `sequential`: whether the selections will be applied sequentially"""
        self._trigcfg = trigcfg
        self._objselcfg = objselcfg
        self._mapcfg = mapcfg
        self._sequential = sequential
        self.objsel = None
        self.objcollect = {}
        self.cfno = None
        self.cfobj = None
    
    def __del__(self):
        print(f"Deleting instance of {self.__class__.__name__}")
    
    def __call__(self, events, wgtname='Generator_weight', **kwargs):
        """Apply all the selections in line on the events"""
        return self.callevtsel(events, wgtname=wgtname, **kwargs)

    @property
    def trigcfg(self):
        return self._trigcfg

    @property
    def objselcfg(self):
        return self._objselcfg
    
    @property
    def mapcfg(self):
        return self._mapcfg
    
    def triggersel(self, events):
        """Custom function to set the object selections on event levels based on config.
        Mask should be N*bool 1-D array.
        """
        pass

    def setevtsel(self, events):
        """Custom function to set the object selections based on config.
        Mask should be N*bool 1-D array.

        :param events: events loaded from a .root file
        """
        pass

    def callevtsel(self, events, wgtname, compute=False):
        """Apply all the selections in line on the events
        Parameters
        
        :return: passed events, vetoed events
        """
        self.objsel = weightedSelection(events[wgtname])
        self.triggersel(events)
        self.setevtsel(events)
        if self.objsel.names:
            self.cfobj = self.objsel.cutflow(*self.objsel.names)
            self.cfno = self.cfobj.result()
        else:
            raise NotImplementedError("Events selections not set, this is base selection!")
        if not self.objcollect:
            passed = events[self.cfno.maskscutflow[-1]]
            if compute: 
                vetoed = events[~(self.objsel.all())]
                result = (passed, vetoed)
            else:
                result = passed
            return result
        else:
            return self.objcollect_to_df() 

    def cf_to_df(self) -> pd.DataFrame:
        """Return a dataframe for a single EventSelections.cutflow object.
        DASK GETS COMPUTED!
        :return: cutflow df
        :rtype: pandas.DataFrame
        """
        row_names = self.cfno.labels
        dfdata = {}
        if self.cfno.wgtevcutflow is not None:
            wgt_number = dask.compute(self.cfno.wgtevcutflow)[0]
            dfdata['wgt'] = wgt_number
        number = dask.compute(self.cfno.nevcutflow)[0]
        dfdata['raw'] = number
        df_cf = pd.DataFrame(dfdata, index=row_names)
        return df_cf

    def objcollect_to_df(self) -> pd.DataFrame:
        """Return a dataframe for the collected objects."""
        listofdf = [Object.object_to_df(zipped, prefix+'_') for prefix, zipped in self.objcollect.items()]
        return pd.concat(listofdf, axis=1)
    
    def selobjhelper(self, events: ak.Array, name, obj: Object, mask: 'ak.Array') -> tuple[Object, ak.Array]:
        """Update event level and object level. Apply the reducing mask on the objects already booked as well as the events.
        
        - `mask`: event-shaped array."""
        print(f"Trying to add {name} mask!")
        if self._sequential and len(self.objsel.names) >= 1:
            lastmask = self.objsel.any(self.objsel.names[-1])
            self.objsel.add_sequential(name, mask, lastmask)
        else: self.objsel.add(name, mask)
        if self.objcollect:
            for key, val in self.objcollect.items():
                self.objcollect[key] = val[mask]
        events = events[mask]
        obj.events = events
        return obj, events
    
    def saveWeights(self, events: ak.Array, weights=['Generator_weight', 'LHEReweightingWeight']) -> None:
        """Save weights to the collected objects."""
        self.objcollect.update({weight: events[weight] for weight in weights})
    
    def getObj(self, name, events, **kwargs) -> Object:
        return Object(events, name, self.objselcfg[name], self.mapcfg[name], **kwargs)

class TriggerEventSelections(BaseEventSelections):
    """A class to skim the events based on the trigger selections."""
    def __init__(self, trigcfg, objselcfg, mapcfg, sequential=True):
        super().__init__(trigcfg, objselcfg, mapcfg, sequential)
    
    def triggersel(self, events):
        for trigname, value in self.trigcfg.items():
            if value:
                self.objsel.add(trigname, events[trigname])
            else:
                self.objsel.add(trigname, events[~trigname])