# adapted from https://github.com/bu-cms/projectcoffea/blob/master/projectcoffea/helpers/helpers.py
import numpy as np
import os
import pandas as pd
from scipy.stats import pearsonr
from src.analysis.objutil import ObjectProcessor

pjoin = os.path.join

class MathUtil:
    @staticmethod
    def clopper_pearson_error(passed, total, level=0.6827):
        """
        matching TEfficiency::ClopperPearson(),
        >>> ROOT.TEfficiency.ClopperPearson(total, passed, level, is_upper)
        """
        import scipy.stats

        alpha = 0.5 * (1.0 - level)
        low = scipy.stats.beta.ppf(alpha, passed, total - passed + 1)
        high = scipy.stats.beta.ppf(1 - alpha, passed + 1, total - passed)
        return low, high
    
    @staticmethod
    def add_f_momentum(df, obj_name:'str'):
        """Convert the four-momentum of the object to the dataframe.
        
        Parameters
        - df: DataFrame
        - obj_name: Object name
        """
        fv, _ = ObjectProcessor.fourvector(df, obj_name, sort=False)
        df[f'{obj_name}_px'] = fv.px
        df[f'{obj_name}_py'] = fv.py
        df[f'{obj_name}_pz'] = fv.pz
    
    @staticmethod
    def add_inv_M(df, objx:'str', objy:'str', inv_m_name:'str') -> None:
        """Add invariant mass column to the dataframe.
        
        Parameters
        - df: DataFrame
        - objx: Object 1 name
        - objy: Object 2 name
        - inv_m_name: Name of the invariant mass column
        """
        fvx, _ = ObjectProcessor.fourvector(df, objx, sort=False)
        fvy, _ = ObjectProcessor.fourvector(df, objy, sort=False)
        df[inv_m_name] = (fvx+fvy).mass
    
    @staticmethod
    def add_dR(df, objx:'str', objy:'str', dR_name:'str') -> None:
        """Add delta R column to the dataframe.
        
        Parameters
        - df: DataFrame
        - objx: Object 1 name
        - objy: Object 2 name
        - dR_name: Name of the delta R column
        """
        fvx, _ = ObjectProcessor.fourvector(df, objx, sort=False)
        fvy, _ = ObjectProcessor.fourvector(df, objy, sort=False)
        df[dR_name] = fvx.deltaR(fvy)
    
    @staticmethod
    def add_system_4vec(df, objx:'str', objy:'str', sys_name:'str') -> None:
        """Add system fourvector column to the dataframe.
        """
        fvx, _ = ObjectProcessor.fourvector(df, objx, sort=False)
        fvy, _ = ObjectProcessor.fourvector(df, objy, sort=False)
        inv_sys = fvx + fvy
        df[f'{sys_name}_pt'] = inv_sys.pt
        df[f'{sys_name}_eta'] = inv_sys.eta
        df[f'{sys_name}_phi'] = inv_sys.phi
        df[f'{sys_name}_mass'] = inv_sys.mass
    
    @staticmethod
    def add_HT(df, jets:'list', HT_name:'str') -> None:
        """Add HT column to the dataframe.
        
        Parameters
        - df: DataFrame
        - jets: List of jet names
        - HT_name: Name of the HT column
        """
        jet_pt_names = [f'{jet}_pt' for jet in jets]
        df[HT_name] = df[jet_pt_names].sum(axis=1)

class ABCDUtil:
    def __init__(self, dfA, dfB, dfC, dfD, weight=None):
        self.dfA = dfA
        self.dfB = dfB
        self.dfC = dfC
        self.dfD = dfD
        self.weight = weight
    
    def get_all_stats(self):
        """Get all stats of the ABCD method."""
        print(self.ABCD_table(self.dfA, self.dfB, self.dfC, self.dfD, weight=self.weight))
        print("====================================")
        print("The closure stats for Events: ")
        Raw_stats = self.ABCD_closure(len(self.dfA), len(self.dfB), len(self.dfC), len(self.dfD))
        Weighted_stats = self.ABCD_closure(self.dfA['weight'].sum(), self.dfB['weight'].sum(), self.dfC['weight'].sum(), self.dfD['weight'].sum())
        total_stats = pd.concat([Raw_stats, Weighted_stats], axis=1)
        total_stats.columns = ['Raw', 'Weighted']
        print(total_stats)
    
    def shape_reweight(self):
        """Derive Signal Region Prediction by average of two estimates
        A = C * B / D
        A' = B * C / D
        A_prediction = (A + A') / 2"""
        rwgt_C = self.dfC.copy()
        rwgt_C['weight'] *= self.dfB['weight'].sum() / self.dfD['weight'].sum() / 2
        rwgt_B = self.dfB.copy()
        rwgt_B['weight'] *= self.dfC['weight'].sum() / self.dfD['weight'].sum() / 2
        avg_A = pd.concat([rwgt_C, rwgt_B], axis=0)
        return avg_A
    
    @staticmethod
    def split_dataframe(df_ori, condition_func):
        """
        Generic function to split a dataframe based on a condition
        
        Args:
            df_ori: Original dataframe
            condition_func: Function that takes df_ori and returns a boolean mask
            
        Returns:
            tuple: (negative_condition_df, positive_condition_df)
        """
        df = df_ori.copy()
        condition = condition_func(df)
        negative = df[~condition]
        positive = df[condition]
        return negative, positive
        
    @staticmethod
    def ABCD_closure(N_A, N_B, N_C, N_D) -> pd.DataFrame:
        """Calculate the closure of the ABCD method."""
        N_A_Pred = int(N_B * N_C / N_D)
        sigma = N_A_Pred * np.sqrt(1/N_B + 1/N_C + 1/N_D) 
        deviation = (N_A - N_A_Pred) / sigma
        difference = abs((N_A - N_A_Pred)/N_A) * 100
        pull = (N_A - N_A_Pred) / np.sqrt(N_A + N_A_Pred)
        ratio = N_A_Pred / N_A
        results = {
            'Metric': [
                'Predicted events in A',
                'Prediction/Actual ratio',
                'Pull significance',
                'Deviation',
                'Percentage difference'
            ],
            'Value': [
                f'{N_A_Pred} ± {sigma:.2f}',
                f'{ratio:.3f}',
                f'{pull:.3f}',
                f'{deviation:.3f}',
                f'{difference:.1f}%'
            ]
        }
        return pd.DataFrame(results).set_index('Metric')
    
    @staticmethod
    def ABCD_plot(dfA, dfB, dfC, dfD):
        """Plot the ABCD method."""
        pass

    @staticmethod
    def ABCD_table(dfA, dfB, dfC, dfD, weight=None) -> pd.DataFrame:
        """Print the ABCD method report."""
        count_A = len(dfA)
        count_B = len(dfB)
        count_C = len(dfC)
        count_D = len(dfD)

        if weight is None:
            table = {
                'Region': ['Region A', 'Region B', 'Region C', 'Region D'],
                'Events': [count_A, count_B, count_C, count_D]
            }
        else:
            weighted_events = [
            int(dfA[weight].sum()),
            int(dfB[weight].sum()),
            int(dfC[weight].sum()),
            int(dfD[weight].sum())
            ]
            uncertainties = [
                np.sqrt((dfA[weight] ** 2).sum()),
                np.sqrt((dfB[weight] ** 2).sum()),
                np.sqrt((dfC[weight] ** 2).sum()),
                np.sqrt((dfD[weight] ** 2).sum())
            ]
            table = {
                'Region': ['Region A', 'Region B', 'Region C', 'Region D'],
                'Events (Raw)': [count_A, count_B, count_C, count_D],
                'Events (Weighted)': weighted_events,
                'Stat. Uncertainty': uncertainties
            }
        return pd.DataFrame(table).set_index('Region')
    
    @staticmethod
    def pearson_corr(df, X, Y):
        df['X_num'] = df[X].astype(int)
        corr_coeff, pval = pearsonr(df['X_num'], df[Y])
        print(f'Pearson Correlation Coefficient: {corr_coeff}')
        print(f'P-value: {pval}')

def poisson_errors(obs, alpha=1 - 0.6827) -> tuple[np.ndarray, np.ndarray]:
    """
    Taken from https://github.com/aminnj/yahist/blob/master/yahist/utils.py 
    Return poisson low and high values for a series of data observations
    """
    from scipy.stats import gamma

    lows = np.nan_to_num(gamma.ppf(alpha / 2, np.array(obs)))
    highs = np.nan_to_num(gamma.ppf(1.0 - alpha / 2, np.array(obs) + 1))
    return lows, highs
    
def simplifyError(passed,total,level=0.6827):
    low,high=clopper_pearson_error(passed, total, level)
    err=high-passed
    return err

def dphi(phi1, phi2):
    """Calculates delta phi between objects"""
    x = np.abs(phi1 - phi2)
    sign = x<=np.pi
    dphi = sign* x + ~sign * (2*np.pi - x)
    return dphi

def min_dphi_jet_met(jets, met_phi, njet=4, ptmin=30, etamax=2.4):
    """Calculate minimal delta phi between jets and met

    :param jets: Jet candidates to use, must be sorted by pT
    :type jets: JaggedCandidateArray
    :param met_phi: MET phi values, one per event
    :type met_phi: array
    :param njet: Number of leading jets to consider, defaults to 4
    :type njet: int, optional
    """

    assert(met_phi.shape!=())
    

    jets=jets[(jets.pt>ptmin)&(jets.abseta < etamax)]
    jets = jets[:,:njet]

    return dphi(jets.phi, met_phi).min()

def mt(pt1, phi1, pt2, phi2):
    """Calculates MT of two objects"""
    return np.sqrt(2 * pt1 * pt2 * (1-np.cos(phi1-phi2)))

def pt_phi_to_px_py(pt, phi):
    """Convert pt and phi to px and py."""
    x = pt * np.cos(phi)
    y = pt * np.sin(phi)

    return x, y

def recoil(met_pt, met_phi, eles, mus, photons):
    """Calculates hadronic recoil by removing leptons from MET

    :param met_pt: MET pt values
    :type met_pt: array
    :param met_phi: MET phi values
    :type met_phi: array
    :param eles: Electron candidates
    :type eles: JaggedCandidateArray
    :param mus: Muon candidates
    :type mus: JaggedCandidateArray
    :return: Pt and phi of the recoil
    :rtype: tuple of arrays (pt, phi)
    """
    met_x, met_y = pt_phi_to_px_py(met_pt, met_phi)
    ele_x, ele_y = pt_phi_to_px_py(eles.pt, eles.phi)
    gam_x, gam_y = pt_phi_to_px_py(photons.pt, photons.phi)
    mu_x, mu_y = pt_phi_to_px_py(mus.pt, mus.phi)

    recoil_x = met_x + ele_x.sum() + mu_x.sum() + gam_x.sum()
    recoil_y = met_y + ele_y.sum() + mu_y.sum() + gam_y.sum()

    recoil_pt = np.hypot(recoil_x, recoil_y)
    recoil_phi = np.arctan2(recoil_y, recoil_x)
    return recoil_pt, recoil_phi


def weight_shape(values, weight):
    """Broadcasts weight array to right shape for given values"""
    return (~np.isnan(values) * weight).flatten()

def object_overlap(toclean, cleanagainst, dr=0.4):
    """Generate a mask to use for overlap removal

    Parameters
    - `toclean`: Candidates that should be cleaned (lower priority candidats)
    - `cleanagainst`: Candidate that should be cleaned against (higher priority)
    - `dr`: Delta R parameter, defaults to 0.4

    Return
    Mask to select non-overlapping candidates in the collection to be cleaned
    """
    delta_r = toclean.deltaR(cleanagainst)
    return delta_r.min() > dr

def sigmoid(x,a,b,c,d):
    """
    Sigmoid function for trigger turn-on fits.

    f(x) = c + (d-c) / (1 + np.exp(-a * (x-b)))
    """
    return c + (d-c) / (1 + np.exp(-a * (x-b)))

def sigmoid3(x,a,b,c):
    '''
    Sigmoid function with three parameters.
    '''
    return c / (1 + np.exp(-a * (x-b)))

def exponential(x, a, b, c):
    """Exponential function for scale factor fits."""
    return a * np.exp(-b * x) + c


def candidates_in_hem(candidates):
    """Returns a mask telling you which candidates are in the HEM region"""
    return (-3.0 < candidates.eta) & (candidates.eta < -1.3) & (-1.8 < candidates.phi) & (candidates.phi < -0.6)

def electrons_in_hem(electrons):
    """Returns a mask telling you which electrons (different region def compared to jets) are in the HEM region"""
    return (-3.0 < electrons.eta) & (electrons.eta < -1.39) & (-1.6 < electrons.phi) & (electrons.phi < -0.9)

def calculate_vecB(ak4, met_pt, met_phi):
    '''Calculate vecB (balance) quantity, based on jets and MET.'''
    mht_p4 = ak4[ak4.pt>30].p4.sum()
    mht_x = - mht_p4.pt * np.cos(mht_p4.phi) 
    mht_y = - mht_p4.pt * np.sin(mht_p4.phi) 

    met_x = met_pt * np.cos(met_phi)
    met_y = met_pt * np.sin(met_phi)

    vec_b = np.hypot(met_x-mht_x, met_y-mht_y) / np.hypot(met_x+mht_x, met_y+mht_y) 

    return vec_b

def calculate_vecDPhi(ak4, met_pt, met_phi, tk_met_phi):
    '''Calculate vecDPhi quantitity.'''
    vec_b = calculate_vecB(ak4, met_pt, met_phi)
    dphitkpf = dphi(met_phi, tk_met_phi)
    vec_dphi = np.hypot(3.33 * vec_b, dphitkpf)

    return vec_dphi