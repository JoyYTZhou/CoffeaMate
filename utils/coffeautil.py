from coffea.analysis_tools import PackedSelection, Cutflow
import coffea.util
import dask_awkward, dask
from collections import namedtuple
import awkward as ak
import numpy as np

class weightedCutflow(Cutflow):
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
        """Returns the results of the cutflow as a namedtuple

        Returns
        -------
            result : CutflowResult
                A namedtuple with the following attributes:

                nevonecut : list of integers or dask_awkward.lib.core.Scalar objects
                    The number of events that survive each cut alone as a list of integers or delayed integers
                nevcutflow : list of integers or dask_awkward.lib.core.Scalar objects
                    The number of events that survive the cumulative cutflow as a list of integers or delayed integers
                wgtevcutflow: list of integers or dask_awesome.lib.core.Scalar objects
                    The number of events that survive the weighted cutflow as a list of integers or delayed integers
                masksonecut : list of boolean numpy.ndarray or dask_awkward.lib.core.Array objects
                    The boolean mask vectors of which events pass each cut alone as a list of materialized or delayed boolean arrays
                maskscutflow : list of boolean numpy.ndarray or dask_awkward.lib.core.Array objects
                    The boolean mask vectors of which events pass the cumulative cutflow a list of materialized or delayed boolean arrays
        """
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

class weightedSelection(PackedSelection):
    def __init__(self, perevtwgt, dtype="uint32"):
        """An inherited class that represents a set of selections on a set of events with weights
        Parameters
        - ``perevtwgt`` : dask.array.Array that represents the weights of the events
        """
        super().__init__(dtype)
        self._perevtwgt = perevtwgt

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

        # Ensure inputs are flat arrays
        thissel_flat = coffea.util._ensure_flat(thissel, allow_missing=True)
        lastsel_flat = coffea.util._ensure_flat(lastsel, allow_missing=True)

        # Apply sequential selection
        selected_events = lastsel_flat == True
        passed_previous = thissel_flat[selected_events]
        combined_result = passed_previous & thissel_flat

        # Create final mask with same shape as input
        final_mask = np.full(lastsel_flat.shape, fill_value)
        final_mask[selected_events] = combined_result

        # Add to selection set based on array type
        if isinstance(final_mask, np.ndarray):
            self._PackedSelection__add_eager(name, final_mask, fill_value)
        elif isinstance(final_mask, dask_awkward.Array):
            self._PackedSelection__add_delayed(name, final_mask, fill_value)
        
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
            nevonecut = [len(self._data)]
            nevcutflow = [len(self._data)]
            nevonecut.extend(np.sum(masksonecut, axis=1, initial=0))
            nevcutflow.extend(np.sum(maskscutflow, axis=1, initial=0))
            if self._perevtwgt is not None:
                wgtevcutflow = [np.sum(self._perevtwgt)]
                wgtevcutflow.extend([np.sum(ak.to_numpy(maskwgt), initial=0) for maskwgt in maskwgtcutflow])
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
