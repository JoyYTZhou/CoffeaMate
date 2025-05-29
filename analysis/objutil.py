import weakref, logging
import vector as vec
import pandas as pd
import operator as opr
import dask_awkward as dak
import numpy as np
import awkward as ak

from src.utils.datautil import arr_handler

trigger_obj_namemap = {"id": "TrigObj_id", "filterBits": "TrigObj_filterBits", "charge": "TrigObj_l1charge"}

class ObjectSelMixin:
    @staticmethod
    def delta_r_match(A, B, dr_cut=0.1):
        """
        For each object in A (per event), check if it matches any object in B by ΔR < dr_cut.
        
        A, B: ak.Arrays of objects with .eta and .phi (same # of events)
        Returns: ak.Array[bool] of shape (events x #A)
        """
        # Cartesian pairs (axis=2 keeps event structure)
        pairs = ak.cartesian([A, B], axis=1)
        a, b = ak.unzip(pairs)

        # ΔR computation
        deta = a.eta - b.eta
        dphi = np.abs(a.phi - b.phi)
        dphi = ak.where(dphi > np.pi, 2 * np.pi - dphi, dphi)
        deltaR = np.sqrt(deta**2 + dphi**2)

        # For each object in A, check if it matches *any* B in same event
        matched = ak.any(deltaR < dr_cut, axis=-1)
        return matched

    @staticmethod
    def get_namemap(events: 'ak.Array', col_name: 'str', namemap: 'dict' = {}) -> dict:
        """Create a dictionary mapping field names to their corresponding arrays in events."""
        vec_type = ['pt', 'eta', 'phi', 'mass'] if not col_name == 'TrigObj' else ['pt', 'eta', 'phi']

        base = {c: events[f"{col_name}_{c}"] if col_name else events[c] for c in vec_type}
        if namemap:
            base.update({k: events[v] for k, v in namemap.items()})
        to_be_zipped = base

        return to_be_zipped
    
    @staticmethod
    def dRoverlap(vec: 'vec.Array', veclist: 'vec.Array', threshold=0.4, op=opr.ge) -> ak.highlevel.Array:
        """Return deltaR mask."""
        return op(abs(vec.deltaR(veclist)), threshold)
    
    def _get_prop_name(self, propname) -> str:
        """Maps internal property name to NanoAOD branch name."""
        aodname = self._mapcfg.get(propname, None)
        if aodname is None:
            aodname = f'{self._name}_{propname}'
            if not aodname in self.events.fields:
                aodname = propname
                if not propname in self.events.fields:
                    raise ValueError(f"Nanoaodname is not given for {propname} of object {self._name}. Have tried querying for both {self._name}_{propname} and {propname}.")
                else:
                    logging.info(f"Consider adding the nanoaodname for {propname} in AOD namemap configuration file.")
        return aodname
    
    @staticmethod
    def sortmask(dfarr, axis=-1, ascending=False) -> ak.Array:
        """Creates a sorting mask for an awkward array. wraps awkward's argsort function."""
        dfarr = arr_handler(dfarr)
        return ak.argsort(dfarr, axis=axis, ascending=ascending)

    @staticmethod
    def fourvector(events: 'ak.Array', objname: 'str'=None, mask=None, sort=True,
                   sortname='pt', ascending=False, axis=-1) -> tuple[vec.Array, ak.Array]:
        """Creates a four-vector representation from event data.

        This method constructs a four-vector (momentum vector with energy) from the event data
        using pt, eta, phi, and mass components. It can handle both flat events and pre-zipped
        object collections.

        Parameters
        ----------
        events : awkward.Array
            The input events containing the object properties (pt, eta, phi, mass).
        Can be either flat events or pre-zipped object collections.
        objname : str, optional
            The prefix name of the object in the events (e.g., 'Electron', 'Muon').
            If None, assumes the properties are directly accessible.
        mask : awkward.Array, optional
            Boolean mask to filter the objects. Must match the dimension of object attributes.
        sort : bool, default=True
            Whether to sort the resulting four-vectors.
        sortname : str, default='pt'
            The field name to sort by when sort=True.
        ascending : bool, default=False
            Sort in ascending order if True, descending if False.
        axis : int, default=-1
            Axis along which to perform the sorting.

        Returns
        -------
        vector.Array
            A four-vector representation of the objects, potentially masked and sorted.
        ak.Array
            The sorting mask if sort=True, otherwise None

        Examples
        --------
        >>> # For flat events structure
        >>> el_vecs = ObjectProcessor.fourvector(events, 'Electron')

        >>> # For pre-zipped object collections
        >>> el_vecs = ObjectProcessor.fourvector(events.electron, sort=False)
        """
        vec_components = ['pt', 'eta', 'phi', 'mass']

        # Handle pre-zipped object collections
        if objname is None and all(comp in events.fields for comp in vec_components):
            object_ak = ak.zip({comp: events[comp] for comp in vec_components})

        # Handle flat events structure
        else:
            to_be_zipped = ObjectProcessor.get_namemap(events, objname)
            object_ak = ak.zip(to_be_zipped)

        # Apply mask if provided
        if mask is not None:
            object_ak = object_ak[mask]

        # Sort if requested
        if sort:
            sort_mask = ak.argsort(object_ak[sortname], ascending=ascending, axis=axis)
            object_ak = object_ak[sort_mask]
            return vec.Array(object_ak), sort_mask
        else:
            return vec.Array(object_ak), None

    @staticmethod
    def set_zipped(events, objname, namemap) -> ak.Array:
        """Given events, read only object-related observables and zip them."""
        zipped_dict = ObjectSelMixin.get_namemap(events, objname, namemap)
        return dak.zip(zipped_dict) if isinstance(events, dak.lib.core.Array) else ak.zip(zipped_dict)
    
    def get_trig_zipped(self, events) -> ak.Array:
        """Get zipped trigger objects from events."""
        return self.set_zipped(events, 'TrigObj', trigger_obj_namemap)

class ObjectMasker(ObjectSelMixin):
    """Handles object selections and mask creation for physics event data.

    Creates and combines boolean masks for filtering physics objects based on
    configured selection criteria and properties.
    Attributes
        ----------
    name : str
        Name identifier for the physics object (e.g. 'Electron', 'Muon')
    events : awkward.Array
        Event data array (optionally stored as weak reference)
    selcfg : dict
        Selection thresholds {property_name: threshold_value}
    mapcfg : dict
        Maps internal names to NanoAOD branch names

    Examples
    --------
    >>> masker = ObjectMasker(events, "Electron",
    ...                      selcfg={"pt": 20, "eta": 2.4},
    ...                      mapcfg={"pt": "Electron_pt"})
    >>> pt_mask = masker.custommask("pt", operator.ge)
    >>> eta_mask = masker.custommask("eta", operator.le, abs)
    >>> combined = masker.create_combined_mask({"pt": (operator.ge,),
    ...                                        "eta": (operator.le, abs)})
        """

    def __init__(self, events, name, selcfg, mapcfg, weakrefEvt=True, sortname=None):
        self._name = name
        self.__weakref = weakrefEvt
        self.events = events  # This will use the property setter
        self._selcfg = selcfg
        self._mapcfg = mapcfg
        self._sortname = sortname

    @property
    def events(self):
        return self._events

    @events.setter
    def events(self, value):
        self._events = weakref.proxy(value) if self.__weakref else value

    def custommask(self, propname: str, op, func=None) -> ak.Array:
        """Creates a boolean mask based on property value comparison.

        Parameters
        ----------
        propname : str
            Property to apply selection on
        op : callable
            Comparison operator function
        func : callable, optional
            Function to apply to property values before comparison
        """
        aodname = self._get_prop_name(propname)
        if self._selcfg.get(propname, None) is None:
            raise ValueError(f"threshold value {propname} is not given for object {self._name}")
        selval = self._selcfg[propname]
        aodarr = self.events[aodname]
        if func is not None:
            return op(func(aodarr), selval)
        else:
            return op(aodarr, selval)

    def create_combined_mask(self, conditions):
        """Combines multiple selection conditions into a single mask.

        Parameters
        ----------
        conditions : dict
            {field: (operator,)} or {field: (operator, function)} selection criteria
            where operator is a comparison operator and function is optional
        Returns
        -------
        awkward.Array
            Combined boolean mask for obj-level selection
        """
        if not conditions:
            return None

        # Get first field and create initial mask
        field, op_func = next(iter(conditions.items()))
        # Handle tuple of (operator,) or (operator, function)
        op = op_func[0]
        func = op_func[1] if len(op_func) > 1 else None
        combined_mask = self.custommask(field, op, func)

        # Combine remaining conditions
        for field, op_func in list(conditions.items())[1:]:
            # Handle tuple of (operator,) or (operator, function)
            op = op_func[0]
            func = op_func[1] if len(op_func) > 1 else None
            mask = self.custommask(field, op, func)
            combined_mask = combined_mask & mask  # Creates new array instead of modifying in place
        
        if self._sortname is not None:
            sort_mask = self.getsortmask(self.events)
            combined_mask = combined_mask[sort_mask]
            
        return combined_mask

    def numselmask(self, mask, op):
        """Creates event-level mask based on number of selected objects."""
        return ObjectMasker.maskredmask(mask, op, self._selcfg.count)

    def vetomask(self, mask):
        """Creates mask for events with no selected objects."""
        return ObjectMasker.maskredmask(mask, opr.eq, 0)

    def evtosmask(self, selmask):
        """Creates mask for events with opposite-sign object pairs."""
        aodname = self._mapcfg['charge']
        aodarr = self.events[aodname][selmask]
        sum_charge = abs(ak.sum(aodarr, axis=1))
        return (sum_charge < ak.num(aodarr, axis=1))
    
    def getsortmask(self, events) -> ak.Array:
        """Get sorting mask for the events based on the configured sortname."""
        if self._sortname is None:
            return None
        sortname = self._get_prop_name(self._sortname)
        return self.sortmask(events[sortname], axis=-1, ascending=False)
    
    def getzipped(self, events, mask=None, **kwargs) -> ak.Array:
        """Get zipped object with optional masking and sorting if sortname property is set. 
        Mask is applied before sorting.

        Parameters:
        - `mask`: mask must be same dimension as any object attributes
        - `**kwargs`: additional arguments passed to sortmask

        Returns:
        - Zipped awkward array of object attributes
        """
        zipped = self.set_zipped(events, self._name, self._mapcfg)
        if mask is not None:
            zipped = zipped[mask]
        if self._sortname is not None:
            zipped = zipped[self.sortmask(zipped[self._sortname], **kwargs)]
        return zipped

    def match_trigger(self, lepton, trigger_id, dr_cut=0.1) -> ak.Array:
        """
        Check if the lepton match at least one trigger objects by deltaR and trigger ID.
        
        Parameters:
        - lepton: vec.Array of reco objects (e.g., electrons, muons, taus, jets, etc.)
        - trig_objs: ak.Array of trigger objects (NanoAOD 'TrigObj')
        - trigger_id: int, e.g., 13 for muons, 11 for electrons, 15 = Tau
        - dr_cut: float, deltaR threshold for matching
        
        Returns:
        - ak.Array of booleans (mask)
        """
        trig_objs = self.set_zipped(self.events, 'TrigObj', trigger_obj_namemap)

        # Filter trigger objects by ID
        trig_sel = trig_objs[abs(trig_objs['id']) == trigger_id]
        deta = trig_sel.eta - lepton.eta
        dphi = np.abs(trig_sel.phi - lepton.phi)
        dphi = np.where(dphi > np.pi, 2*np.pi - dphi, dphi)
        deltaR = np.sqrt(deta**2 + dphi**2)
        
        return ak.any(deltaR < dr_cut, axis=-1)

    def match_trigger_object(self, trigger_id, dr_cut=0.1) -> ak.Array:
        """
        Match each lepton (per event) to trigger objects by ΔR and trigger ID.

        Parameters:
        - trigger_id: int, e.g., 13 for muons, 11 for electrons, 15 for taus
        - dr_cut: float, deltaR threshold for matching

        Returns:
        - ak.Array of booleans, same shape as leptons (True if matched)
        """
        trig_objs = self.set_zipped(self.events, 'TrigObj', trigger_obj_namemap)
        leptons = self.getzipped(self.events, mask=None)

        trig_sel = trig_objs[abs(trig_objs['id']) == trigger_id]
        
        return self.delta_r_match(leptons, trig_sel, dr_cut)

    @staticmethod
    def maskredmask(mask, op, count) -> ak.Array:
        """Reduces per-object mask to event-level selections."""
        return op(ak.sum(mask, axis=1), count)

class ObjectProcessor(ObjectMasker):
    """Processes physics objects in events, applying selections and delta R requirements."""
    def event_level_dr_mask(self, events, objmask, dr_threshold) -> tuple[ak.Array, ak.Array]:
        """Compute delta R (ΔR) mask for event-level selections (dim = (#events))."""
        dr_mask, _ = self.dRwSelf(events, dr_threshold, objmask)
        dr_mask_events = self.maskredmask(dr_mask, opr.ge, 1)
        
        return dr_mask_events
    
    def apply_obj_level_dr(self, events, objmask, dr_threshold):
        dr_mask, _ = self.dRwSelf(events, dr_threshold, objmask)
        zipped = self.getzipped(events, mask=objmask)

        leading = zipped[:,0]
        subleading = zipped[:,1:][dr_mask][:,0]

        return leading, subleading

    def get_dr_selection_results(self, events, objmask, dr_threshold=0.5, **kwargs):
        """Apply delta R (ΔR) separation requirements and return comprehensive results.
            
        Parameters
        ----------
        events : awkward.Array
            Input events containing the physics objects
        objmask : awkward.Array
            Boolean mask for initial object selection (e.g. pT, eta cuts)
        dr_threshold : float, default=0.5
            Minimum required ΔR separation between objects
        **kwargs : dict
        Additional keyword arguments for fourvector creation and sorting

        Returns
        -------
        tuple
            - event_mask : awkward.Array
                Boolean mask indicating events that pass the selection
            - filtered_events : awkward.Array
                Events filtered by the selection mask
            - leading_objects : awkward.Array
                Leading object in each event
            - subleading_objects : awkward.Array
                Subleading object in each event
        Examples
        --------
        >>> results = processor.get_dr_selection_results(
        ...     events, pt_mask, dr_threshold=0.4
        ... )
        >>> passing_events = results['filtered_events']
        >>> event_mask = results['event_mask']
        >>> leading_obj = results['leading_objects']
        """
        # Get deltaR mask between leading and subleading objects
        dr_mask, sort_mask = self.dRwSelf(events, dr_threshold, objmask)

        dr_mask_events = self.maskredmask(dr_mask, opr.ge, 1)
        events = events[dr_mask_events]
        objmask = objmask[sort_mask][dr_mask_events]

        # Get sorted objects passing initial selection
        zipped = self.getzipped(events, mask=objmask)

        new_dr = dr_mask[dr_mask_events]

        # Separate leading object (always kept)
        leading = zipped[:,0]

        # Initialize empty subleading array
        subleading = zipped[:,1:][new_dr][:,0]

        return dr_mask_events, events, leading, subleading

    def dRwSelf(self, events, threshold, mask, **kwargs) -> ak.Array:
        """Calculate delta R between the leading object and subleading objects in a collection.

        Parameters
        ----------
        events : awkward.Array
            The input events containing the object properties.
        threshold : float
            The delta R threshold value to compare against.
        mask : awkward.Array
            Boolean mask to filter the objects.
        **kwargs : dict
            Additional keyword arguments passed to fourvector and sorting:
            - ascending : bool, default=False
                Sort direction (False=descending, True=ascending)

        Returns
        -------
        awkward.Array
            Boolean mask indicating pairs of objects that pass the delta R threshold.
        awkward.Array
            Sorting mask if sort=True, otherwise None

        Examples
        --------
        >>> # Get dR mask for muons sorted by pt in descending order
        >>> dR_mask = obj.dRwSelf(events, 0.4, pt_mask, sort=True, sortname='pt')

        >>> # Get dR mask without sorting
        >>> dR_mask = obj.dRwSelf(events, 0.4, pt_mask, sort=False)
        """
        sort = False if self._sortname is None else True
        obj_lvs, sort_mask = self.fourvector(events, self._name, mask, sort=sort, sortname=self._sortname)
        ld_lv, sd_lvs = obj_lvs[:, 0], obj_lvs[:, 1:]
        dR_mask = self.dRoverlap(ld_lv, sd_lvs, threshold)
        return dR_mask, sort_mask

    def dRwOther(self, events, vec, threshold):
        sort = False if self._sortname is None else True 
        object_lv, sort_mask = self.fourvector(events, self._name, sort=sort, sortname=self._sortname)
        return self.dRoverlap(vec, object_lv, threshold), sort_mask
    
    @staticmethod
    def object_to_df(zipped, prefix='') -> pd.DataFrame:
        """Take a zipped object, compute it if needed, turn it into a dataframe"""
        zipped = arr_handler(zipped, allow_delayed=False)
        return ak.to_dataframe(zipped).add_prefix(prefix)







# class TriggerObject(Object):
#     """Trigger Object class for handling trigger object selections, meant as an observer of the events.

#     Attributes
#     - `name`: name of the object
#     - `events`: a weak proxy of the events 
#     - `selcfg`: selection configuration for the object, {key=abbreviation, value=threshold}
#     """

#     def __init__(self, events, selcfg, weakrefEvt=True):
#         """Construct an object from provided events with given selection configuration.
        
#         Parameters
#         - `name`: AOD prefix name of the object, e.g. Electron, Muon, Jet
#         kwargs: 
#         - `selcfg`: selection configuration for the object
#         """
#         self._name = "TrigObj"
#         self.__weakref = weakrefEvt
#         self._selcfg = selcfg
#         self.events = events
#         self._mapcfg = trigger_obj_map
#         self.fields = list(self._mapcfg.keys())
#         self._maskcollec = {}
    
#     def custommask(self, propname, op, value, func=None):
#         """Create custom mask based on input."""
#         mask = super().custommask(propname, op, value, func)
#         self._maskcollec.append(mask)
        
#     def bitmask(self, bit):
#         """Create mask based on trigger bit."""
#         aodarr = self.events[self._mapcfg['filterBits']]
#         return (aodarr & bit) > 0

#     def passedTrigObj(self):
#         mask = ak.all(ak.stack(self._maskcollec), axis=0)
#         pass

    
    
    
    

