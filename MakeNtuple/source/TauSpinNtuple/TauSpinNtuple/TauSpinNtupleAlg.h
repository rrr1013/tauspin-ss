#ifndef TAUSPINNTUPLE_TAUSPINNTUPLEALG_H
#define TAUSPINNTUPLE_TAUSPINNTUPLEALG_H

#include <AnaAlgorithm/AnaAlgorithm.h>
#include <string>

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
    
};

#endif