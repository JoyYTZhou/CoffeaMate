import weakref, logging
import vector as vec
import pandas as pd
import operator as opr
import dask_awkward as dak
import awkward as ak

from src.utils.datautil import arr_handler

class ObjectMasker:
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

    def __init__(self, events, name, selcfg, mapcfg, weakrefEvt=True):
        self._name = name
        self.__weakref = weakrefEvt
        self.events = events  # This will use the property setter
        self._selcfg = selcfg
        self._mapcfg = mapcfg

    @property
    def events(self):
        return self._events

    @events.setter
    def events(self, value):
        self._events = weakref.proxy(value) if self.__weakref else value

    def _get_prop_name(self, propname) -> str:
        """Maps internal property name to NanoAOD branch name."""
        aodname = self._mapcfg.get(propname, None)
        if aodname is None:
            aodname = f'{self._name}_{propname}'
            if not aodname in self.events.fields:
                aodname = propname
                if not propname in self.events.fields:
                    raise ValueError(f"Nanoaodname is not given for {propname} of object {self._name}")
                else:
                    logging.info(f"Consider adding the nanoaodname for {propname} in AOD namemap configuration file.")
        return aodname

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
            Combined boolean mask where True indicates passing all conditions
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

    @staticmethod
    def maskredmask(mask, op, count) -> ak.Array:
        """Reduces per-object mask to event-level selections."""
        return op(ak.sum(mask, axis=1), count)

class ObjectProcessor:
    """Static class for processing object data and creating zipped collections."""
    def __init__(self, name, mapcfg):
        self._name = name
        self._mapcfg = mapcfg
        
    @staticmethod
    def sortmask(dfarr, **kwargs) -> ak.Array:
        """Creates a sorting mask for an awkward array.

        This method creates an index array that would sort the input array according to specified parameters.
        It's a wrapper around awkward's argsort function with preset default values.

        Parameters:
        ----------
        dfarr : awkward.Array
            The array to be sorted. Will be computed first if it's a delayed array.
        **kwargs : dict
            Keyword arguments to customize sorting behavior:
            - axis : int, default=-1
                Axis along which to sort. The default -1 means sort along the last axis.
            - ascending : bool, default=False
                Sort in ascending order if True, descending if False.
            - highlevel : bool, default=True
                Return a high-level awkward Array if True, a low-level layout if False.

        Returns:
        -------
        awkward.Array
            An array of indices that would sort the input array according to the specified parameters.

        Examples:
        --------
        >>> arr = ak.Array([[1, 5, 2], [3, 1, 4]])
        >>> mask = ObjectProcessor.sortmask(arr)
        >>> arr[mask]  # Will return sorted array [[5, 2, 1], [4, 3, 1]]
        """
        dfarr = arr_handler(dfarr)
        return ak.argsort(dfarr,
                   axis=kwargs.get('axis', -1),
                   ascending=kwargs.get('ascending', False),
                   highlevel=kwargs.get('highlevel', True))

    @staticmethod
    def fourvector(events: 'ak.Array', objname: 'str'=None, mask=None, sort=True,
                   sortname='pt', ascending=False, axis=-1) -> vec.Array:
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
            object_ak = object_ak[ak.argsort(object_ak[sortname], ascending=ascending, axis=axis)]

        return vec.Array(object_ak)

    @staticmethod
    def set_zipped(events, objname, namemap) -> ak.Array:
        """Given events, read only object-related observables and zip them."""
        zipped_dict = ObjectProcessor.get_namemap(events, objname, namemap)
        return dak.zip(zipped_dict) if isinstance(events, dak.lib.core.Array) else ak.zip(zipped_dict)

    def getzipped(self, events, mask=None, sort=True, sort_by='pt', **kwargs) -> ak.Array:
        """Get zipped object with optional masking and sorting.

        Parameters:
        - `mask`: mask must be same dimension as any object attributes
        - `sort`: whether to sort the zipped object (default: True)
        - `sort_by`: field to sort by (default: 'pt')
        - `**kwargs`: additional arguments passed to sortmask

        Returns:
        - Zipped awkward array of object attributes
        """
        zipped = ObjectProcessor.set_zipped(events, self._name, self._mapcfg)
        if mask is not None:
            zipped = zipped[mask]
        if sort:
            zipped = zipped[ObjectProcessor.sortmask(zipped[sort_by], **kwargs)]
        return zipped

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
        dr_mask = self.dRwSelf(events, dr_threshold, objmask, **kwargs)

        dr_mask_events = ObjectMasker.maskredmask(dr_mask, opr.ge, 1)
        events = events[dr_mask_events]

        # Get sorted objects passing initial selection
        zipped = self.getzipped(events, mask=objmask)

        new_dr = dr_mask[dr_mask_events]

        # Separate leading object (always kept)
        leading = zipped[:,0]

        # Initialize empty subleading array
        subleading = zipped[:,1:][new_dr]

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
            - sort : bool, default=True
                Whether to sort the objects by pt
            - sortname : str, default='pt'
                Field name to sort by
            - ascending : bool, default=False
                Sort direction (False=descending, True=ascending)
            - axis : int, default=-1
                Axis along which to sort

        Returns
        -------
        awkward.Array
            Boolean mask indicating pairs of objects that pass the delta R threshold.

        Examples
        --------
        >>> # Get dR mask for muons sorted by pt in descending order
        >>> dR_mask = obj.dRwSelf(events, 0.4, pt_mask, sort=True, sortname='pt')

        >>> # Get dR mask without sorting
        >>> dR_mask = obj.dRwSelf(events, 0.4, pt_mask, sort=False)
        """
        obj_lvs = self.fourvector(events, self._name, mask, **kwargs)
        ld_lv, sd_lvs = obj_lvs[:, 0], obj_lvs[:, 1:]
        dR_mask = ObjectProcessor.dRoverlap(ld_lv, sd_lvs, threshold)
        return dR_mask

    def dRwOther(self, events, vec, threshold, **kwargs):
        object_lv = self.fourvector(events, self._name, **kwargs)
        return self.dRoverlap(vec, object_lv, threshold)
    
    @staticmethod
    def object_to_df(zipped, prefix='') -> pd.DataFrame:
        """Take a zipped object, compute it if needed, turn it into a dataframe"""
        zipped = arr_handler(zipped, allow_delayed=False)
        return ak.to_dataframe(zipped).add_prefix(prefix)

    @staticmethod
    def dRoverlap(vec, veclist: 'vec.Array', threshold=0.4, op=opr.ge) -> ak.highlevel.Array:
        """Return deltaR mask."""
        return op(abs(vec.deltaR(veclist)), threshold)

    @staticmethod
    def get_namemap(events: 'ak.Array', col_name: 'str', namemap: 'dict' = {}) -> dict:
        """Create a dictionary mapping field names to their corresponding arrays in events."""
        vec_type = ['pt', 'eta', 'phi', 'mass']

        if col_name is not None:
            to_be_zipped = {
                component: events[f"{col_name}_{component}"]
                for component in vec_type
            }
        else:
            to_be_zipped = {
                component: events[component]
                for component in vec_type
            }

        if namemap:
            to_be_zipped.update({
                name: events[nanoaodname]
                for name, nanoaodname in namemap.items()
            })

        return to_be_zipped

trigger_obj_map = {"id": "TrigObj_id", "eta": "TrigObj_eta", "phi": "TrigObj_phi", "pt": "TrigObj_pt",
                   "filterBits": "TrigObj_filterBits", "charge": "TrigObj_l1charge"}

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

    
    
    
    

