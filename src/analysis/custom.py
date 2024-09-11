# This file contains custom event selection classes for the src.analysis.
# The classes are inherited from the BaseEventSelections class
# TECHNICALLY THIS SHOULD BE THE ONLY FILE THAT NEEDS TO BE MODIFIED FOR CUSTOM EVENT SELECTIONS
from .evtselutil import BaseEventSelections
from .objutil import Object
import operator as opr
import awkward as ak

def switch_selections(sel_name):
    selections = {
        'vetoskim': skimEvtSel,
        'prelim_onelooseb': ControlEvtSel,
        'prelim_twolooseb': SignalEvtSel,
        'prelim_total': PrelimEvtSel
    }
    return selections.get(sel_name, BaseEventSelections)

class skimEvtSel(BaseEventSelections):
    """Reduce event sizes"""
    def triggersel(self, events):
        for trigname, value in self.trigcfg.items():
            if value:
                self.objsel.add(trigname, events[trigname])
            else:
                self.objsel.add(trigname, events[~trigname])

    def setevtsel(self, events) -> None:
        muon = Object(events, "Muon")
        electron = Object(events, "Electron")
        e_mask = (electron.ptmask(opr.ge) & \
                electron.absdxymask(opr.le) & \
                electron.absetamask(opr.le) & \
                electron.absdzmask(opr.le) & \
                electron.custommask('mvaisoid', opr.eq)
                )
        elec_nummask = electron.vetomask(e_mask)

        m_mask = (muon.ptmask(opr.ge) & \
                muon.absdxymask(opr.le) & \
                muon.absetamask(opr.le) & \
                muon.absdzmask(opr.le) & \
                muon.custommask('looseid', opr.eq) & \
                muon.custommask('isoid', opr.ge))
        muon_nummask = muon.vetomask(m_mask)

        self.objsel.add_multiple({"Electron Veto": elec_nummask,
                                "Muon Veto": muon_nummask})
        electron = None
        muon = None

class twoTauEvtSel(BaseEventSelections):
    def seltwotaus(self, events) -> ak.Array:
        tau = Object(events, "Tau", weakrefEvt=True)
        def tauobjmask(tau: 'Object'):
            tau_mask = (tau.ptmask(opr.ge) & \
                        tau.absetamask(opr.le) & \
                        tau.absdzmask(opr.lt) & \
                        tau.custommask('idvsjet', opr.ge) & \
                        tau.custommask('idvsmu', opr.ge) & \
                        tau.custommask('idvse', opr.ge))
            return tau_mask

        tau_nummask = tau.numselmask(tauobjmask(tau), opr.ge)

        tau, events = self.selobjhelper(events, '>= 2 Medium hadronic Taus', tau, tau_nummask)
        leading_tau, sd_cand = tau.getldsd(mask=tauobjmask(tau))
        self.objcollect['LDTau'] = leading_tau

        dR_mask = tau.dRwSelf(threshold=0.5, mask=tauobjmask(tau))
        sd_cand = sd_cand[dR_mask]

        tau_dRmask = Object.maskredmask(dR_mask, opr.ge, 1)
        tau, events = self.selobjhelper(events,'Tau dR >= 0.5', tau, tau_dRmask)

        sd_cand = sd_cand[tau_dRmask][:,0]
        self.objcollect['SDTau'] = sd_cand

        return events

class ControlEvtSel(twoTauEvtSel):
    def setevtsel(self, events) -> None:
        events = self.seltwotaus(events)
    
        jet = Object(events, 'Jet')
        def jobjmask(jet: 'Object'):
            j_mask = (jet.ptmask(opr.ge) & jet.absetamask(opr.le))
            tau_ldvec = Object.fourvector(self.objcollect['LDTau'], sort=False)
            tau_sdvec = Object.fourvector(self.objcollect['SDTau'], sort=False)
            jetdR_mask = jet.dRwOther(tau_ldvec, 0.5) & jet.dRwOther(tau_sdvec, 0.5)
            return j_mask & jetdR_mask
        
        jet_nummask = jet.numselmask(jobjmask(jet), opr.ge)
        jet, events = self.selobjhelper(events, '>=2 ak4 jets', jet, jet_nummask)

        jet_nummask = jet.maskredmask((jobjmask(jet) & jet.custommask('btag', opr.ge)), opr.eq, count=1)
        jet, events = self.selobjhelper(events, '==1 Loose B-tagged', jet, jet_nummask)
        
        jet_mask = (jobjmask(jet) & jet.custommask('btag', opr.ge))
        ld_j = jet.getld(mask=jet_mask)
        sd_j = jet.getld(mask=(~jet_mask) & jobjmask(jet), sort_by='btag')
        self.objcollect['LDBjetBYtag'] = ld_j
        self.objcollect['SDBjetBYtag'] = sd_j

        sd_j = jet.getld(mask=(~jet_mask) & jobjmask(jet), sort_by='pt')
        self.objcollect['SDBjetBYpt'] = sd_j
        self.saveWeights(events)

class SignalEvtSel(twoTauEvtSel):
    def setevtsel(self, events) -> None:
        events = self.seltwotaus(events)
        
        jet = Object(events, 'Jet')
        
        def jobjmask(jet: 'Object'):
            j_mask = (jet.ptmask(opr.ge) & jet.absetamask(opr.le))
            tau_ldvec = Object.fourvector(self.objcollect['LDTau'], sort=False)
            tau_sdvec = Object.fourvector(self.objcollect['SDTau'], sort=False)
            jetdR_mask = jet.dRwOther(tau_ldvec, 0.5) & jet.dRwOther(tau_sdvec, 0.5)
            return j_mask & jetdR_mask
        
        jet_nummask = jet.numselmask(jobjmask(jet), opr.ge)
        jet, events = self.selobjhelper(events, '>=2 ak4 jets', jet, jet_nummask)

        jet_nummask = jet.numselmask((jobjmask(jet) & jet.custommask('btag', opr.ge)), opr.eq)
        jet, events = self.selobjhelper(events, '>=2 Loose B-tagged', jet, jet_nummask)
        
        jet_mask = (jobjmask(jet) & jet.custommask('btag', opr.ge))
        ld_j, sd_j = jet.getldsd(sort_by='btag', mask=jet_mask)
        self.objcollect['LDBjetBYtag'] = ld_j
        self.objcollect['SDBjetBYtag'] = sd_j[:,0]

        ld_j, sd_j = jet.getldsd(sort_by='pt', mask=jet_mask)
        self.objcollect['LDBjetBYpt'] = ld_j
        self.objcollect['SDBjetBYpt'] = sd_j[:,0]

        self.saveWeights(events)

class PrelimEvtSel(twoTauEvtSel):
    def setevtsel(self, events):
        events = self.seltwotaus(events)

        jet = Object(events, 'Jet')
        
        def jobjmask(jet: 'Object'):
            j_mask = (jet.ptmask(opr.ge) & jet.absetamask(opr.le))
            tau_ldvec = Object.fourvector(self.objcollect['LDTau'], sort=False)
            tau_sdvec = Object.fourvector(self.objcollect['SDTau'], sort=False)
            jetdR_mask = jet.dRwOther(tau_ldvec, 0.5) & jet.dRwOther(tau_sdvec, 0.5)
            return j_mask & jetdR_mask
        
        jet_nummask = jet.numselmask(jobjmask(jet), opr.ge)
        jet, events = self.selobjhelper(events, '>=2 ak4 jets', jet, jet_nummask)

        jet_nummask = jet.maskredmask((jobjmask(jet) & jet.custommask('btag', opr.ge)), opr.ge, count=1)
        jet, events = self.selobjhelper(events, '>=1 Loose B-tagged', jet, jet_nummask)
        
        jet_mask = (jobjmask(jet) & jet.custommask('btag', opr.ge))
        ld_j = jet.getld(sort_by='btag', mask=jet_mask)
        self.objcollect['LDBjetBYtag'] = ld_j

        self.saveWeights(events)