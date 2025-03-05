import weakref
import vector as vec
import pandas as pd
import operator as opr
import dask_awkward as dak
import awkward as ak
from typing import Optional, Union, List, Tuple, Callable

from src.utils.datautil import arr_handler

def _create_proxy_property(name):
    def getter(self):
        return getattr(self, f"_{name}")
    def setter(self, value):
        setattr(self, f"_{name}", weakref.proxy(value) if self.__weakref else value)
    return property(getter, setter)

MASK_CONFIGS = {
    'pt': {'func': None},
    'eta': {'func': abs},
    'dxy': {'func': abs},
    'dz': {'func': abs}
}

class Object:
    """Object class for handling object selections, meant as an observer of the events.

    Attributes
    - `name`: name of the object
    - `events`: a weak proxy of the events 
    - `selcfg`: selection configuration for the object, {key=abbreviation, value=threshold}
    - `mapcfg`: mapping configuration for the object
    - `fields`: list of fields in the object
    """
    events = _create_proxy_property('events')
    selcfg = _create_proxy_property('selcfg')

    def __init__(self, events, name, selcfg, mapcfg, weakrefEvt=True):
        """Construct an object from provided events with given selection configuration.
        
        Parameters
        - `name`: AOD prefix name of the object, e.g. Electron, Muon, Jet

        kwargs: 
        - `selcfg`: selection configuration for the object
        - `mapcfg`: mapping configuration for the object
        """
        self._name = name
        self.__weakref = weakrefEvt
        self._events = None
        self._selcfg = None
        self._mapcfg = mapcfg

        self._events = events
        self._selcfg = selcfg
        self.fields = list(self.mapcfg.keys())

    @property
    def name(self):
        return self._name
    
    @property
    def mapcfg(self):
        return self._mapcfg
        
    def custommask(self, propname: str, op: Callable, func: Optional[Callable]=None) -> ak.Array:
        """Create custom mask based on input.
        
        Parameters
        - `events`: events to apply the mask on
        - `propname`: name of the property mask is based on
        - `op`: operator to use for the mask
        - `func`: function to apply to the data. Defaults to None.

        Returns
        - `mask`: mask based on input
        """
        if self.selcfg.get(propname, None) is None:
            raise ValueError(f"threshold value {propname} is not given for object {self.name}")
        
        aodname = self.mapcfg.get(propname, None)
        if aodname is None:
            aodname = f'{self.name}_{propname}'
            if not aodname in self.events.fields:
                aodname = propname
                print(f"Consider adding the nanoaodname for {propname} in AOD namemap configuration file.")
                if not propname in self.events.fields:
                    raise ValueError(f"Nanoaodname is not given for {propname} of object {self.name}")

        selval = self.selcfg[propname]
        aodarr = self.events[aodname]
        if func is not None:
            return op(func(aodarr), selval)
        else:
            return op(aodarr, selval)
    
    def common_mask(self, prop_name, op):
        """Create object level mask with predefined configurations."""
        config = MASK_CONFIGS.get(prop_name)
        if not config:
            raise ValueError(f"Property {prop_name} is not a predefined mask configuration (e.g., pt, eta, dxy, etc.).")
        return self.custommask(prop_name, op, config['func'])
    
    def numselmask(self, mask, op):
        """Returns event-level boolean mask."""
        return Object.maskredmask(mask, op, self.selcfg.count)
    
    def vetomask(self, mask):
        """Returns the veto mask for events."""
        return Object.maskredmask(mask, opr.eq, 0)

    def evtosmask(self, selmask):
        """Create mask on events with OS objects.
        !!! Note that this mask is applied per event, not per object.
        1 for events with 2 OS objects that pass selmask"""
        aodname = self.mapcfg['charge']
        aodarr = self.events[aodname][selmask]
        sum_charge = abs(ak.sum(aodarr, axis=1))
        mask = (sum_charge < ak.num(aodarr, axis=1))
        return mask
    
    def trigObjmask(self, bit, trigger_obj, threshold=0.5, **kwargs):
        """Match Object to Trigger Object.
        
        Parameters
        - `bit`: trigger bit to be matched
        - `trigger_obj`: trigger object to be matched. Array of Objects.
        - `threshold`: threshold of dR for the matching
        """
        pass
    
    def dRwSelf(self, threshold, **kwargs):
        """Haphazard way to select pairs of objects"""
        object_lv = self.getfourvec(**kwargs)
        leading_lv = object_lv[:,0]
        subleading_lvs = object_lv[:,1:]
        dR_mask = Object.dRoverlap(leading_lv, subleading_lvs, threshold)
        return dR_mask
    
    def dRwOther(self, vec, threshold, **kwargs):
        object_lv = self.getfourvec(**kwargs)
        return Object.dRoverlap(vec, object_lv, threshold)

    def OSwSelf(self, **kwargs):
        zipped = self.getzipped(**kwargs)
        leading = zipped[:,0]
        subleading = zipped[:,1:]
        return leading['charge']*subleading['charge'] < 0
        
    def getfourvec(self, **kwargs) -> vec.Array:
        """Get four vector for the object from the currently observed events."""
        return Object.fourvector(self.events, self.name, **kwargs)
    
    def getzipped(self, mask=None, sort=True, sort_by='pt', **kwargs):
        """Get zipped object.
        
        Parameters
        - `mask`: mask must be same dimension as any object attributes."""
        zipped = Object.set_zipped(self.events, self.name, self.mapcfg)
        if mask is not None: zipped = zipped[mask]
        if sort: zipped = zipped[Object.sortmask(zipped[sort_by], **kwargs)]
        return zipped 
    
    def getldsd(self, **kwargs) -> tuple[ak.Array, ak.Array]:
        """Returns the zipped leading object and the rest of the objects (aka subleading candidates).
        All properties in obj setting included."""
        objs = self.getzipped(**kwargs) 
        return (objs[:,0], objs[:,1:])
    
    def getld(self, **kwargs) -> ak.Array:
        objs = self.getzipped(**kwargs) 
        return objs[:,0]
    
    def get_matched_jet(self):
        pass

    @staticmethod
    def sortmask(dfarr, **kwargs) -> ak.Array:
        """Wrapper around awkward argsort function.
        
        Parameters
        - `dfarr`: the data arr to be sorted

        kwargs: see ak.argsort
        """
        dfarr = arr_handler(dfarr)
        sortmask = ak.argsort(dfarr, 
                   axis=kwargs.get('axis', -1), 
                   ascending=kwargs.get('ascending', False),
                   highlevel=kwargs.get('highlevel', True)
                   )
        return sortmask
    
    @staticmethod
    def fourvector(events: 'ak.Array', objname: 'str'=None, mask=None, sort=True, sortname='pt', ascending=False, axis=-1) -> vec.Array:
        """Returns a fourvector from the events.
    
        Parameters
        - `events`: the events to extract the fourvector from. 
        - `objname`: the name of the field in the events that contains the fourvector information.
        - `sort`: whether to sort the fourvector
        - `sortname`: the name of the field to sort the fourvector by.
        - `ascending`: whether to sort the fourvector in ascending order.

        Return
        - a fourvector object.
        """
        to_be_zipped = Object.get_namemap(events, objname)
        object_ak = ak.zip(to_be_zipped) if mask is None else ak.zip(to_be_zipped)[mask] 
        if sort:
            object_ak = object_ak[ak.argsort(object_ak[sortname], ascending=ascending, axis=axis)]
        object_LV = vec.Array(object_ak)
        return object_LV

    @staticmethod
    def set_zipped(events, objname, namemap) -> ak.Array:
        """Given events, read only object-related observables and zip them into awkward/dask array.
        
        Parameters
        - `events`: events to extract the object from
        - `namemap`: mapping configuration for the object
        """
        dict_to_zip = Object.get_namemap(events, objname, namemap)
        zipped_object = dak.zip(dict_to_zip) if isinstance(events, dak.lib.core.Array) else ak.zip(dict_to_zip) 
        return zipped_object

    @staticmethod
    def object_to_df(zipped, prefix='') -> pd.DataFrame:
        """Take a zipped object, compute it if needed, turn it into a dataframe"""
        zipped = arr_handler(zipped, allow_delayed=False)
        objdf = ak.to_dataframe(zipped).add_prefix(prefix)
        return objdf

    @staticmethod
    def maskredmask(mask, op, count) -> ak.Array:
        """Reduces the mask to event level selections.
        Count is the number of objects per event that should return true in the mask. 
        
        Parameters
        - `mask`: the mask to be reduced
        - `op`: the operator to be used for the reduction
        - `count`: the count to be used for the reduction
        """
        return op(ak.sum(mask, axis=1), count)

    @staticmethod
    def dRoverlap(vec, veclist: 'vec.Array', threshold=0.4, op=opr.ge) -> ak.highlevel.Array:
        """Return deltaR mask. Default comparison threshold is 0.4. Default comparison is >=. 
        
        Parameters
        - `vec`: the vector to compare with
        - `veclist`: the list of vectors to compare against vec
        - `threshold`: the threshold for the comparison
        
        Return
        - a mask of the veclist that satisfies the comparison condition."""
        return op(abs(vec.deltaR(veclist)), threshold)

    @staticmethod
    def get_namemap(events: 'ak.Array', col_name: Optional[str], namemap: dict = {}) -> dict:
        """Get mapping between names and event arrays.
        
        Args:
            events: Event array
            col_name: Column prefix for vector components, e.g. 'Electron'
            namemap: Additional name mappings
            
        Returns:
            Dictionary mapping names to event arrays
        """
        VECTOR_COMPONENTS = ['pt', 'eta', 'phi', 'mass']
        
        result = {}
        for component in VECTOR_COMPONENTS:
            field = f"{col_name}_{component}" if col_name else component
            result[component] = events[field]
        
        result.update({
            name: events[aod_name] 
            for name, aod_name in namemap.items()
        })
        
        return result


# For NANOAOD V12
trigger_obj_map = {"id": "TrigObj_id", "eta": "TrigObj_eta", "phi": "TrigObj_phi", "pt": "TrigObj_pt",
                   "filterBits": "TrigObj_filterBits", "charge": "TrigObj_l1charge"}

class TriggerObject(Object):
    """Trigger Object class for handling trigger object selections, meant as an observer of the events.

    Attributes
    - `name`: name of the object
    - `events`: a weak proxy of the events 
    - `selcfg`: selection configuration for the object, {key=abbreviation, value=threshold}
    """

    def __init__(self, events, selcfg, weakrefEvt=True):
        """Construct an object from provided events with given selection configuration.
        
        Parameters
        - `name`: AOD prefix name of the object, e.g. Electron, Muon, Jet
        kwargs: 
        - `selcfg`: selection configuration for the object
        """
        self._name = "TrigObj"
        self.__weakref = weakrefEvt
        self._selcfg = selcfg
        self.events = events
        self._mapcfg = trigger_obj_map
        self.fields = list(self.mapcfg.keys())
        self._maskcollec = {}
    
    def custommask(self, propname, op, value, func=None):
        """Create custom mask based on input."""
        mask = super().custommask(propname, op, value, func)
        self._maskcollec.append(mask)
        
    def bitmask(self, bit):
        """Create mask based on trigger bit."""
        aodarr = self.events[self.mapcfg['filterBits']]
        return (aodarr & bit) > 0

    def passedTrigObj(self):
        mask = ak.all(ak.stack(self._maskcollec), axis=0)
        pass

    
    
    
    

