#include "TauSpinNtuple/TauSpinNtupleAlg.h"
#include <xAODTau/TauJetContainer.h>

// コンストラクタ
TauSpinNtupleAlg::TauSpinNtupleAlg(
    const std::string& name,
    ISvcLocator* pSvcLocator
)
    : EL::AnaAlgorithm(name, pSvcLocator)
{
}

StatusCode TauSpinNtupleAlg::initialize(){

    return StatusCode::SUCCESS;
}

StatusCode TauSpinNtupleAlg::execute(){
    const xAOD::TauJetContainer* taus = nullptr;
    ANA_CHECK(evtStore()->retrieve(taus, "TauJets"));
    ANA_MSG_INFO("Number of tau candidates: " << taus->size());
    
    return StatusCode::SUCCESS;
}

StatusCode TauSpinNtupleAlg::finalize(){
    
    return StatusCode::SUCCESS;
}