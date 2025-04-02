from coffea.analysis_tools import PackedSelection, Cutflow
import coffea.util
import dask_awkward, dask
from collections import namedtuple
import awkward as ak
import logging
import time
import numpy as np
from functools import wraps

class weightedCutflow(Cutflow):
    """An inherited class that represents a set of selections on a set of events with weights"""
    def __init__(
        self, names, nevonecut, nevcutflow, wgtevcutflow, masksonecut, maskscutflow, delayed_mode
    ):
        self._names = names
        self._nevonecut = nevonecut
        self._nevcutflow = nevcutflow
        self._wgtevcutflow = wgtevcutflow
        self._masksonecut = masksonecut
        self._maskscutflow = maskscutflow
        self._delayed_mode = delayed_mode
    
    __init__.__doc__ = Cutflow.__init__.__doc__
    
    def __add__(self, cutflow2):
        if self._delayed_mode != cutflow2._delayed_mode:
            raise TypeError("Concatenation of delayed and computed cutflows are not supported now!")
        names = self._names + cutflow2._names
        nevonecut = self._nevonecut + cutflow2._nevonecut
        nevcutflow = self._nevcutflow + cutflow2._nevcutflow
        wgtevcutflow = self._wgtevcutflow + cutflow2._wgtevcutflow
        masksonecut = self._masksonecut + cutflow2._masksonecut
        maskscutflow = self._maskscutflow + cutflow2._maskscutflow

        return weightedCutflow(names, nevonecut, nevcutflow, wgtevcutflow, masksonecut, maskscutflow, self._delayed_mode)

    def result(self):
        CutflowResult = namedtuple(
            "CutflowResult",
            ["labels", "nevonecut", "nevcutflow", "wgtevcutflow", "masksonecut", "maskscutflow"],
        )
        labels = ["initial"] + list(self._names)
        return CutflowResult(
            labels,
            self._nevonecut,
            self._nevcutflow,
            self._wgtevcutflow,
            self._masksonecut,
            self._maskscutflow,
        )

    result.__doc__ = (Cutflow.result.__doc__ or "") + """
    Additional Returns for weightedCutflow:
        wgtevcutflow: list of integers or dask_awesome.lib.core.Scalar objects
            The number of events that survive the weighted cutflow as a list of integers or delayed integers
    """

class sequentialSelection(PackedSelection):
    def add_sequential(self, name: str, thissel, lastsel, fill_value: bool = False) -> None:
        """Add a sequential selection to the existing selection set.

        This method applies a new selection ('thissel') only to events that passed
        the previous selection ('lastsel'). The result is stored under 'name'.

        Parameters
        ----------
        name : str
            The name to identify this selection in the selection set
        thissel : array-like
            The current selection mask to be applied
        lastsel : array-like
            The previous selection mask that defines which events to consider
        fill_value : bool, optional
            The value to use for events that don't pass the previous selection,
            defaults to False

        Raises
        ------
        ValueError
            If either thissel or lastsel are dask arrays instead of dask_awkward arrays

        Notes
        -----
        - Both selection masks must be either numpy arrays or dask_awkward arrays
        - The method flattens both selection masks before processing
        - The result maintains the shape of lastsel, filling non-selected events with fill_value
        """
        # Validate input arrays aren't dask arrays
        if isinstance(thissel, dask.array.Array) or isinstance(lastsel, dask.array.Array):
            raise ValueError(
            "Dask arrays are not supported, please convert them to dask_awkward.Array "
            "by using dask_awkward.from_dask_array()"
            )

        start_time = time.time()
        logging.debug(f"Starting sequential selection computation for {name}")
        # Compute the selections if they are delayed
        if isinstance(thissel, dask_awkward.Array):
            thissel = thissel.compute()
        if isinstance(lastsel, dask_awkward.Array):
            lastsel = lastsel.compute()
        end_time = time.time()
        logging.debug(f"Sequential computation took {end_time - start_time} seconds")

        # Ensure inputs are flat arrays
        thissel_flat = coffea.util._ensure_flat(thissel, allow_missing=True)
        lastsel_flat = coffea.util._ensure_flat(lastsel, allow_missing=True)
        logging.debug(f"The shape of thissel_flat is {thissel_flat.shape}")
        logging.debug(f"The shape of lastsel_flat is {lastsel_flat.shape}")

        result = np.full_like(lastsel_flat, fill_value, dtype=bool)
        true_indices = np.where(lastsel_flat)[0]
        result[true_indices] = thissel_flat
        self._PackedSelection__add_eager(name, result, fill_value)
  
class weightedSelection(sequentialSelection):
    def __init__(self, perevtwgt, dtype="uint32"):
        """An inherited class that represents a set of selections on a set of events with weights

        Parameters
        - ``perevtwgt`` : dask.array.Array that represents the weights of the events
        """
        super().__init__(dtype)
        self._perevtwgt = perevtwgt
     
    def cutflow(self, *names) -> weightedCutflow:
        for cut in names:
            if not isinstance(cut, str) or cut not in self._names:
                raise ValueError(
                    "All arguments must be strings that refer to the names of existing selections"
                )
        masksonecut, maskscutflow, maskwgtcutflow = [], [], []
        for i, cut in enumerate(names):
            mask1 = self.any(cut)
            mask2 = self.all(*(names[: i + 1]))
            maskwgt = self._perevtwgt[mask2]

            masksonecut.append(mask1)
            maskscutflow.append(mask2)
            maskwgtcutflow.append(maskwgt)

        if not self.delayed_mode:
            logging.debug(f"Using eager mode for cutflow computation")
            nevonecut = [len(self._data)]
            nevcutflow = [len(self._data)]
            nevonecut.extend(np.sum(masksonecut, axis=1, initial=0))
            nevcutflow.extend(np.sum(maskscutflow, axis=1, initial=0))
            if self._perevtwgt is not None:
                initial_weight = np.sum(self._perevtwgt)
                wgtevcutflow = [initial_weight]
                logging.debug(f"Initial weight: {initial_weight}")
                logging.debug(f"Initial weight in list: {wgtevcutflow[0]}")
                logging.debug(f"Type of initial weight: {type(initial_weight)}")
                logging.debug(f"Type of weight in list: {type(wgtevcutflow[0])}")
                wgtevcutflow.extend([np.sum(ak.to_numpy(maskwgt), initial=0) for maskwgt in maskwgtcutflow])
                logging.debug("Weight cutflow: %s", wgtevcutflow)
            else:
                wgtevcutflow = None

        else:
            nevonecut = [dask_awkward.count(self._data, axis=0)]
            nevcutflow = [dask_awkward.count(self._data, axis=0)]
            
            def catchZeroArr(mask):
                try:
                    return dask_awkward.sum(mask)
                except:
                    return np.sum(mask.compute(), initial=0)

            nevonecut.extend([catchZeroArr(mask1) for mask1 in masksonecut])
            nevcutflow.extend([catchZeroArr(mask2) for mask2 in maskscutflow])

            if self._perevtwgt is not None:
                wgtevcutflow = [catchZeroArr(self._perevtwgt)] 
                wgtevcutflow.extend([catchZeroArr(self._perevtwgt[mask2]) for mask2 in maskscutflow])
            else:
                wgtevcutflow = None

        return weightedCutflow(
            names, nevonecut, nevcutflow, wgtevcutflow, masksonecut, maskscutflow, self.delayed_mode
        )
