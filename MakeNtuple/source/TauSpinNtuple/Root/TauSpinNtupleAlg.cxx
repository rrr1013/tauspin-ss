#include "TauSpinNtuple/TauSpinNtupleAlg.h"
#include <xAODTau/TauJetContainer.h>
#include <vector>

// コンストラクタ
TauSpinNtupleAlg::TauSpinNtupleAlg(
    const std::string& name,
    ISvcLocator* pSvcLocator
)
    : EL::AnaAlgorithm(name, pSvcLocator)
{
}

// initialize
StatusCode TauSpinNtupleAlg::initialize(){

    ANA_CHECK(m_taus.initialize(m_systematicsList));
    ANA_CHECK(
        m_tauSelection.initialize(
            m_systematicsList,
            m_taus
        )
    );
    ANA_CHECK(m_systematicsList.initialize());

    return StatusCode::SUCCESS;
}

// execute
StatusCode TauSpinNtupleAlg::execute(){

    for (const auto& sys : m_systematicsList.systematicsVector()){

        // CP Algorithm後のcontainer取得
        const xAOD::TauJetContainer* taus = nullptr;
        ANA_CHECK(m_taus.retrieve(taus,sys));

        // selection通ったtauを保存
        std::vector<const xAOD::TauJet*> passingTaus;
        passingTaus.reserve(taus->size());

        for (const xAOD::TauJet* tau : *taus){
            if (m_tauSelection.getBool(*tau, sys)){
                passingTaus.push_back(tau);
            }
        }

        ANA_MSG_INFO(
            "Raw taus: " << taus->size()
            << ", selected taus: " << passingTaus.size()
        );

        // tauがちょうど2個のやつだけ残す
        if (passingTaus.size() != 2) {
            continue;
        }

        const xAOD::TauJet* tau0 = passingTaus.at(0);
        const xAOD::TauJet* tau1 = passingTaus.at(1);

        // 異符号だけ残す
        if (tau0->charge() * tau1->charge() >= 0.0) {
            continue;
        }

        // マイナス→プラスの順
        const xAOD::TauJet* tauMinus = nullptr;
        const xAOD::TauJet* tauPlus = nullptr;
        if (tau0->charge() < 0.0) {
            tauMinus = tau0;
            tauPlus = tau1;
        } else {
            tauMinus = tau1;
            tauPlus = tau0;
        }

        ANA_MSG_INFO(
            "Selected OS tau pair:"
            << " tau- pT = " << tauMinus->pt() / 1000.0 << " GeV,"
            << " tau+ pT = " << tauPlus->pt() / 1000.0 << " GeV"
        );


    }


    // const xAOD::TauJetContainer* taus = nullptr;
    // ANA_CHECK(evtStore()->retrieve(taus, "TauJets"));
    // ANA_MSG_INFO("Number of tau candidates: " << taus->size());
    
    return StatusCode::SUCCESS;
}

// finalize
StatusCode TauSpinNtupleAlg::finalize(){
    
    return StatusCode::SUCCESS;
}