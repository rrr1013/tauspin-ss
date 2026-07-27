#ifndef TAUSPINNTUPLE_TAUSPINNTUPLEALG_H
#define TAUSPINNTUPLE_TAUSPINNTUPLEALG_H

#include <SelectionHelpers/SysReadSelectionHandle.h>
#include <SystematicsHandles/SysListHandle.h>
#include <SystematicsHandles/SysReadHandle.h>
#include <xAODTau/TauJetContainer.h>
#include <AnaAlgorithm/AnaAlgorithm.h>
#include <string>

// Algorithmのclassを定義
class TauSpinNtupleAlg : public EL::AnaAlgorithm{
public:
    // コンストラクタ
    TauSpinNtupleAlg(
        const std::string& name,
        ISvcLocator* pSvcLocator
    );

    StatusCode initialize() override;

    StatusCode execute() override;

    StatusCode finalize() override;

private:
    // jobで処理するsystematicsたち
    CP::SysListHandle m_systematicsList {this};

    CP::SysReadHandle<xAOD::TauJetContainer> m_taus {
        this,
        "taus",
        "AnaTauJets_%SYS%",
        "Calibrated tau container"
    };

    CP::SysReadSelectionHandle m_tauSelection {
        this,
        "tauSelection",
        "selected_tau_loose,as_bits",
        "GNTau Loose and kinematic selection"
    };
    
};

#endif