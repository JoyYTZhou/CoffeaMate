import weakref
import vector as vec
import pandas as pd
import operator as opr
import dask_awkward as dak
import awkward as ak

from src.utils.datautil import arr_handler

class Object:
    """Object class for handling object selections, meant as an observer of the events.

    Attributes
    - `name`: name of the object
    - `events`: a weak proxy of the events 
    - `selcfg`: selection configuration for the object, {key=abbreviation, value=threshold}
    - `mapcfg`: mapping configuration for the object
    - `fields`: list of fields in the object
    """

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
        self.events = events
        self._selcfg = selcfg
        self._mapcfg = mapcfg
        self.fields = list(self.mapcfg.keys())
    
    @property
    def events(self):
        return self._events

    @events.setter
    def events(self, value):
        self._events = weakref.proxy(value) if self.__weakref else value
    
    @property
    def selcfg(self):
        return self._selcfg
    @selcfg.setter
    def selcfg(self, value):
        self._selcfg = weakref.proxy(value)

    @property
    def name(self):
        return self._name

    @property
    def selcfg(self):
        return self._selcfg
    
    @property
    def mapcfg(self):
        return self._mapcfg
        
    def custommask(self, propname: 'str', op, func=None):
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
        if not f'{self.name}_{propname}' in self.events.fields:
            if self.mapcfg.get(propname, None) is None:
                raise ValueError(f"Nanoaodname is not given for {propname} of object {self.name}")
            else:
                aodname = self.mapcfg[propname]
        else:
            aodname = f'{self.name}_{propname}' 
        selval = self.selcfg[propname]
        aodarr = self.events[aodname]
        if func is not None:
            return op(func(aodarr), selval)
        else:
            return op(aodarr, selval)

    def ptmask(self, op):
        """Object Level mask for pt."""
        return self.custommask('pt', op)

    def absetamask(self, op):
        """Object Level mask for |eta|."""
        return self.custommask('eta', op, abs)

    def absdxymask(self, op):
        """Object Level mask for |dxy|."""
        return self.custommask('dxy', op, abs)

    def absdzmask(self, op):
        """Object Level mask for |dz|."""
        return self.custommask('dz', op, abs)
    
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
        """Given events, read only object-related observables and zip them into ak. 
        Then zip the dict into an object.
        
        Parameters
        - `events`: events to extract the object from
        - `namemap`: mapping configuration for the object
        """
        zipped_dict = Object.get_namemap(events, objname, namemap)
        zipped_object = dak.zip(zipped_dict) if isinstance(events, dak.lib.core.Array) else ak.zip(zipped_dict) 
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
    def get_namemap(events: 'ak.Array', col_name: 'str', namemap: 'dict' = {}):
        vec_type = ['pt', 'eta', 'phi', 'mass']
        to_be_zipped = {cop: events[col_name+"_"+cop] 
                        for cop in vec_type} if col_name is not None else {cop: events[cop] for cop in vec_type}
        if namemap: to_be_zipped.update({name: events[nanoaodname] 
                       for name, nanoaodname in namemap.items()})
        return to_be_zipped