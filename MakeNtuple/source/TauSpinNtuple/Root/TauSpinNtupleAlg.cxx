#include "TauSpinNtuple/TauSpinNtupleAlg.h"

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
    
    return StatusCode::SUCCESS;
}

StatusCode TauSpinNtupleAlg::finalize(){
    
    return StatusCode::SUCCESS;
}