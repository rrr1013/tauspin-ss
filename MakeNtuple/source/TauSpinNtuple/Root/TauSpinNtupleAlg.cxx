#include "TauSpinNtuple/TauSpinNtupleAlg.h"

#include <AthContainers/AuxElement.h>
#include <TauAnalysisTools/HelperFunctions.h>
#include <TList.h>
#include <TNamed.h>
#include <TTree.h>
#include <xAODEventInfo/EventInfo.h>
#include <xAODMissingET/MissingET.h>
#include <xAODPFlow/PFO.h>
#include <xAODPFlow/PFODefs.h>
#include <xAODTau/TauDefs.h>
#include <xAODTau/TauTrack.h>
#include <xAODTracking/TrackParticle.h>
#include <xAODTracking/TrackingPrimitives.h>
#include <xAODTracking/VertexContainer.h>
#include <xAODTruth/TruthParticleContainer.h>

#include <cmath>
#include <cstdint>
#include <unordered_map>
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
    ANA_CHECK(m_tauSelection.initialize(m_systematicsList, m_taus));
    ANA_CHECK(m_met.initialize(m_systematicsList));
    ANA_CHECK(m_systematicsList.initialize());

    // TTree作る
    ANA_CHECK(book(TTree("tauspin", "Tau spin ntuple")));
    TTree* outputTree = tree("tauspin");

    // event
    outputTree->Branch("eventNumber", &m_eventNumber);
    outputTree->Branch("runNumber", &m_runNumber);
    outputTree->Branch("mcChannelNumber", &m_mcChannelNumber);
    outputTree->Branch("averageInteractionsPerCrossing", &m_averageInteractionsPerCrossing);
    outputTree->Branch("actualInteractionsPerCrossing", &m_actualInteractionsPerCrossing);
    outputTree->Branch("met_et", &m_metEt);
    outputTree->Branch("met_phi", &m_metPhi);
    outputTree->Branch("met_sumet", &m_metSumet);

    // tau
    outputTree->Branch("tau_pt", m_tauPt, "tau_pt[2]/F");
    outputTree->Branch("tau_eta", m_tauEta, "tau_eta[2]/F");
    outputTree->Branch("tau_phi", m_tauPhi, "tau_phi[2]/F");
    outputTree->Branch("tau_charge", m_tauCharge, "tau_charge[2]/F");
    outputTree->Branch("tau_m", m_tauM, "tau_m[2]/F");
    outputTree->Branch("tau_nTracks", m_tauNTracks, "tau_nTracks[2]/I");
    outputTree->Branch("tau_nChargedTracks", m_tauNChargedTracks, "tau_nChargedTracks[2]/I");
    outputTree->Branch("tau_nIsolatedTracks", m_tauNIsolatedTracks, "tau_nIsolatedTracks[2]/I");
    outputTree->Branch("tau_nAllTracks", m_tauNAllTracks, "tau_nAllTracks[2]/I");
    outputTree->Branch("tau_decayMode", m_tauDecayMode, "tau_decayMode[2]/I");
    outputTree->Branch("tau_panTauDecayMode", m_tauPanTauDecayMode, "tau_panTauDecayMode[2]/I");
    outputTree->Branch("tau_nnDecayMode", m_tauNNDecayMode, "tau_nnDecayMode[2]/I");
    outputTree->Branch("tau_rnnJetScore", m_tauRNNJetScore, "tau_rnnJetScore[2]/F");
    outputTree->Branch("tau_rnnJetScoreSigTrans", m_tauRNNJetScoreSigTrans, "tau_rnnJetScoreSigTrans[2]/F");
    outputTree->Branch("tau_gntauScore_v0", m_tauGNTauScoreV0, "tau_gntauScore_v0[2]/F");
    outputTree->Branch("tau_gntauScoreSigTrans_v0", m_tauGNTauScoreSigTransV0, "tau_gntauScoreSigTrans_v0[2]/F");
    outputTree->Branch("tau_gntauVL_v0", m_tauGNTauVLV0, "tau_gntauVL_v0[2]/I");
    outputTree->Branch("tau_gntauL_v0", m_tauGNTauLV0, "tau_gntauL_v0[2]/I");
    outputTree->Branch("tau_gntauM_v0", m_tauGNTauMV0, "tau_gntauM_v0[2]/I");
    outputTree->Branch("tau_gntauT_v0", m_tauGNTauTV0, "tau_gntauT_v0[2]/I");

    // tau track
    outputTree->Branch("track_tauIndex", &m_trackTauIndex);
    outputTree->Branch("track_indexInTau", &m_trackIndexInTau);
    outputTree->Branch("track_pt", &m_trackPt);
    outputTree->Branch("track_eta", &m_trackEta);
    outputTree->Branch("track_phi", &m_trackPhi);
    outputTree->Branch("track_dEta", &m_trackDEta);
    outputTree->Branch("track_dPhi", &m_trackDPhi);
    outputTree->Branch("track_ptFraction", &m_trackPtFraction);
    outputTree->Branch("track_d0", &m_trackD0);
    outputTree->Branch("track_z0", &m_trackZ0);
    outputTree->Branch("track_z0SinTheta", &m_trackZ0SinTheta);
    outputTree->Branch("track_theta", &m_trackTheta);
    outputTree->Branch("track_qOverP", &m_trackQOverP);
    outputTree->Branch("track_charge", &m_trackCharge);
    outputTree->Branch("track_isCore", &m_trackIsCore);
    outputTree->Branch("track_isIsolation", &m_trackIsIsolation);
    outputTree->Branch("track_isConversion", &m_trackIsConversion);
    outputTree->Branch("track_isFake", &m_trackIsFake);
    outputTree->Branch("track_passTrkSelector", &m_trackPassTrkSelector);
    outputTree->Branch("track_numberOfPixelHits", &m_trackNumberOfPixelHits);
    outputTree->Branch("track_numberOfSCTHits", &m_trackNumberOfSCTHits);
    outputTree->Branch("track_numberOfTRTHits", &m_trackNumberOfTRTHits);

    // neutral/pi0 PFO
    outputTree->Branch("pfo_tauIndex", &m_pfoTauIndex);
    outputTree->Branch("pfo_indexInTau", &m_pfoIndexInTau);
    outputTree->Branch("pfo_pt", &m_pfoPt);
    outputTree->Branch("pfo_eta", &m_pfoEta);
    outputTree->Branch("pfo_phi", &m_pfoPhi);
    outputTree->Branch("pfo_e", &m_pfoE);
    outputTree->Branch("pfo_dEta", &m_pfoDEta);
    outputTree->Branch("pfo_dPhi", &m_pfoDPhi);
    outputTree->Branch("pfo_ptFraction", &m_pfoPtFraction);
    outputTree->Branch("pfo_charge", &m_pfoCharge);
    outputTree->Branch("pfo_isPi0", &m_pfoIsPi0);
    outputTree->Branch("pfo_bdtPi0Score", &m_pfoBDTPi0Score);
    outputTree->Branch("pfo_nPi0Proto", &m_pfoNPi0Proto);

    // vertex
    outputTree->Branch("primaryVertex_x", &m_primaryVertexX);
    outputTree->Branch("primaryVertex_y", &m_primaryVertexY);
    outputTree->Branch("primaryVertex_z", &m_primaryVertexZ);
    outputTree->Branch("primaryVertex_nTracks", &m_primaryVertexNTracks);
    outputTree->Branch("nPrimaryVertices", &m_nPrimaryVertices);
    outputTree->Branch("tau_vertexIndex", m_tauVertexIndex, "tau_vertexIndex[2]/I");
    outputTree->Branch("tau_vertexDeltaZ", m_tauVertexDeltaZ, "tau_vertexDeltaZ[2]/F");

    // truth診断
    outputTree->Branch("truth_hasHiggs", &m_truthHasHiggs);
    outputTree->Branch("truth_hasZ", &m_truthHasZ);
    outputTree->Branch("truth_boson_pdgId", &m_truthBosonPdgId);
    outputTree->Branch("truth_boson_pt", &m_truthBosonPt);
    outputTree->Branch("truth_boson_eta", &m_truthBosonEta);
    outputTree->Branch("truth_boson_phi", &m_truthBosonPhi);
    outputTree->Branch("truth_boson_m", &m_truthBosonM);
    outputTree->Branch("tau_truthMatched", m_tauTruthMatched, "tau_truthMatched[2]/I");
    outputTree->Branch("tau_truthPdgId", m_tauTruthPdgId, "tau_truthPdgId[2]/I");
    outputTree->Branch("tau_truthPt", m_tauTruthPt, "tau_truthPt[2]/F");
    outputTree->Branch("tau_truthEta", m_tauTruthEta, "tau_truthEta[2]/F");
    outputTree->Branch("tau_truthPhi", m_tauTruthPhi, "tau_truthPhi[2]/F");
    outputTree->Branch("tau_truthDecayMode", m_tauTruthDecayMode, "tau_truthDecayMode[2]/I");
    outputTree->Branch("tau_truthIsHadronic", m_tauTruthIsHadronic, "tau_truthIsHadronic[2]/I");

    // ntuple定義
    outputTree->GetUserInfo()->Add(new TNamed("tau_id_algorithm", "JETIDRNNLOOSE"));
    outputTree->GetUserInfo()->Add(new TNamed("use_gntau_selection", "false"));
    outputTree->GetUserInfo()->Add(new TNamed("gntau_decoration_version", "v0"));
    outputTree->GetUserInfo()->Add(new TNamed("input_sim_flavour", "ATLFAST3_QS"));
    outputTree->GetUserInfo()->Add(new TNamed("input_mc_campaign", "mc20e"));
    outputTree->GetUserInfo()->Add(new TNamed("input_ami_tag", "r14861"));
    outputTree->GetUserInfo()->Add(new TNamed("tau_selection_config", "tau_selection_rnn_loose_nocrack.conf"));
    outputTree->GetUserInfo()->Add(new TNamed("met_definition", "CP AnaMET_NOSYS Final AntiKt4EMPFlow"));
    outputTree->GetUserInfo()->Add(new TNamed("transformer_baseline_inputs", "tau_summary,tracks,pfos,final_met"));

    return StatusCode::SUCCESS;
}

// execute
StatusCode TauSpinNtupleAlg::execute(){

    const xAOD::EventInfo* eventInfo = nullptr;
    ANA_CHECK(evtStore()->retrieve(eventInfo, "EventInfo"));

    for (const auto& sys : m_systematicsList.systematicsVector()){

        // CP Algorithm後のcontainer取得
        const xAOD::TauJetContainer* taus = nullptr;
        const xAOD::MissingETContainer* met = nullptr;
        ANA_CHECK(m_taus.retrieve(taus, sys));
        ANA_CHECK(m_met.retrieve(met, sys));

        // selection通ったtauを保存
        std::vector<const xAOD::TauJet*> passingTaus;
        passingTaus.reserve(taus->size());

        for (const xAOD::TauJet* tau : *taus){
            if (!m_tauSelection.getBool(*tau, sys)){
                continue;
            }

            // chargeとprong条件を明示的に再確認
            if (std::abs(tau->charge()) != 1.0F){
                continue;
            }
            if (tau->nTracks() != 1 && tau->nTracks() != 3){
                continue;
            }
            passingTaus.push_back(tau);
        }

        ANA_MSG_INFO(
            "Raw taus: " << taus->size()
            << ", selected taus: " << passingTaus.size()
        );

        // tauがちょうど2個のやつだけ残す
        if (passingTaus.size() != 2){
            continue;
        }

        const xAOD::TauJet* tau0 = passingTaus.at(0);
        const xAOD::TauJet* tau1 = passingTaus.at(1);

        // 異符号だけ残す
        if (tau0->charge() * tau1->charge() >= 0.0){
            continue;
        }

        // マイナス→プラスの順
        const xAOD::TauJet* tauMinus = tau0->charge() < 0.0 ? tau0 : tau1;
        const xAOD::TauJet* tauPlus = tau0->charge() < 0.0 ? tau1 : tau0;

        resetBranches();
        m_eventNumber = eventInfo->eventNumber();
        m_runNumber = eventInfo->runNumber();
        m_mcChannelNumber = eventInfo->mcChannelNumber();
        m_averageInteractionsPerCrossing = eventInfo->averageInteractionsPerCrossing();
        m_actualInteractionsPerCrossing = eventInfo->actualInteractionsPerCrossing();

        // Final MET
        const xAOD::MissingET* finalMET = (*met)["Final"];
        if (!finalMET){
            ANA_MSG_ERROR("AnaMET does not contain the Final term");
            return StatusCode::FAILURE;
        }
        m_metEt = finalMET->met() / 1000.0F;
        m_metPhi = finalMET->phi();
        m_metSumet = finalMET->sumet() / 1000.0F;

        fillTau(tauMinus, 0);
        fillTau(tauPlus, 1);
        fillTracks(tauMinus, 0);
        fillTracks(tauPlus, 1);
        fillPFOs(tauMinus, 0);
        fillPFOs(tauPlus, 1);

        // primary vertex
        const xAOD::Vertex* primaryVertex = nullptr;
        if (evtStore()->contains<xAOD::VertexContainer>("PrimaryVertices")){
            const xAOD::VertexContainer* vertices = nullptr;
            ANA_CHECK(evtStore()->retrieve(vertices, "PrimaryVertices"));
            for (const xAOD::Vertex* vertex : *vertices){
                if (vertex->vertexType() == xAOD::VxType::PriVtx){
                    ++m_nPrimaryVertices;
                    if (!primaryVertex){
                        primaryVertex = vertex;
                    }
                }
            }
        }

        if (primaryVertex){
            m_primaryVertexX = primaryVertex->x();
            m_primaryVertexY = primaryVertex->y();
            m_primaryVertexZ = primaryVertex->z();
            m_primaryVertexNTracks = static_cast<int>(primaryVertex->nTrackParticles());
        }

        const xAOD::TauJet* selectedTaus[2] = {tauMinus, tauPlus};
        for (std::size_t i = 0; i < 2; ++i){
            const xAOD::Vertex* tauVertex = selectedTaus[i]->vertex();
            if (tauVertex){
                m_tauVertexIndex[i] = static_cast<int>(tauVertex->index());
                if (primaryVertex){
                    m_tauVertexDeltaZ[i] = tauVertex->z() - primaryVertex->z();
                }
            }
        }

        // H/Z boson truth
        if (eventInfo->eventType(xAOD::EventInfo::IS_SIMULATION)
            && evtStore()->contains<xAOD::TruthParticleContainer>("TruthBoson")){
            const xAOD::TruthParticleContainer* truthBosons = nullptr;
            ANA_CHECK(evtStore()->retrieve(truthBosons, "TruthBoson"));

            const xAOD::TruthParticle* selectedBoson = nullptr;
            int bestScore = -1;
            for (const xAOD::TruthParticle* boson : *truthBosons){
                const int absPdgId = std::abs(boson->pdgId());
                if (absPdgId != 23 && absPdgId != 25){
                    continue;
                }
                if (!hasTauDescendant(boson)){
                    continue;
                }

                if (absPdgId == 23){
                    m_truthHasZ = 1;
                }
                if (absPdgId == 25){
                    m_truthHasHiggs = 1;
                }

                int directTauChildren = 0;
                bool hasSameBosonChild = false;
                for (std::size_t i = 0; i < boson->nChildren(); ++i){
                    const xAOD::TruthParticle* child = boson->child(i);
                    if (!child){
                        continue;
                    }
                    if (std::abs(child->pdgId()) == 15){
                        ++directTauChildren;
                    }
                    if (child->pdgId() == boson->pdgId()){
                        hasSameBosonChild = true;
                    }
                }

                const int score = 10 * directTauChildren
                    + static_cast<int>(!hasSameBosonChild);
                if (score > bestScore){
                    selectedBoson = boson;
                    bestScore = score;
                }
            }

            if (selectedBoson){
                m_truthBosonPdgId = selectedBoson->pdgId();
                m_truthBosonPt = selectedBoson->pt() / 1000.0F;
                m_truthBosonEta = selectedBoson->eta();
                m_truthBosonPhi = selectedBoson->phi();
                m_truthBosonM = selectedBoson->m() / 1000.0F;
            }
        }

        tree("tauspin")->Fill();
    }

    return StatusCode::SUCCESS;
}

// finalize
StatusCode TauSpinNtupleAlg::finalize(){
    return StatusCode::SUCCESS;
}

void TauSpinNtupleAlg::resetBranches(){

    m_averageInteractionsPerCrossing = -999.0F;
    m_actualInteractionsPerCrossing = -999.0F;
    m_metEt = -999.0F;
    m_metPhi = -999.0F;
    m_metSumet = -999.0F;

    for (std::size_t i = 0; i < 2; ++i){
        m_tauPt[i] = 0.0F;
        m_tauEta[i] = 0.0F;
        m_tauPhi[i] = 0.0F;
        m_tauCharge[i] = 0.0F;
        m_tauM[i] = -999.0F;
        m_tauNTracks[i] = -1;
        m_tauNChargedTracks[i] = -1;
        m_tauNIsolatedTracks[i] = -1;
        m_tauNAllTracks[i] = -1;
        m_tauDecayMode[i] = -1;
        m_tauPanTauDecayMode[i] = -1;
        m_tauNNDecayMode[i] = -1;
        m_tauRNNJetScore[i] = -999.0F;
        m_tauRNNJetScoreSigTrans[i] = -999.0F;
        m_tauGNTauScoreV0[i] = -999.0F;
        m_tauGNTauScoreSigTransV0[i] = -999.0F;
        m_tauGNTauVLV0[i] = -1;
        m_tauGNTauLV0[i] = -1;
        m_tauGNTauMV0[i] = -1;
        m_tauGNTauTV0[i] = -1;
        m_tauVertexIndex[i] = -1;
        m_tauVertexDeltaZ[i] = -999.0F;
        m_tauTruthMatched[i] = -1;
        m_tauTruthPdgId[i] = 0;
        m_tauTruthPt[i] = -999.0F;
        m_tauTruthEta[i] = -999.0F;
        m_tauTruthPhi[i] = -999.0F;
        m_tauTruthDecayMode[i] = -1;
        m_tauTruthIsHadronic[i] = -1;
    }

    m_trackTauIndex.clear();
    m_trackIndexInTau.clear();
    m_trackPt.clear();
    m_trackEta.clear();
    m_trackPhi.clear();
    m_trackDEta.clear();
    m_trackDPhi.clear();
    m_trackPtFraction.clear();
    m_trackD0.clear();
    m_trackZ0.clear();
    m_trackZ0SinTheta.clear();
    m_trackTheta.clear();
    m_trackQOverP.clear();
    m_trackCharge.clear();
    m_trackIsCore.clear();
    m_trackIsIsolation.clear();
    m_trackIsConversion.clear();
    m_trackIsFake.clear();
    m_trackPassTrkSelector.clear();
    m_trackNumberOfPixelHits.clear();
    m_trackNumberOfSCTHits.clear();
    m_trackNumberOfTRTHits.clear();

    m_pfoTauIndex.clear();
    m_pfoIndexInTau.clear();
    m_pfoPt.clear();
    m_pfoEta.clear();
    m_pfoPhi.clear();
    m_pfoE.clear();
    m_pfoDEta.clear();
    m_pfoDPhi.clear();
    m_pfoPtFraction.clear();
    m_pfoCharge.clear();
    m_pfoIsPi0.clear();
    m_pfoBDTPi0Score.clear();
    m_pfoNPi0Proto.clear();

    m_primaryVertexX = -999.0F;
    m_primaryVertexY = -999.0F;
    m_primaryVertexZ = -999.0F;
    m_primaryVertexNTracks = -1;
    m_nPrimaryVertices = 0;

    m_truthHasHiggs = 0;
    m_truthHasZ = 0;
    m_truthBosonPdgId = 0;
    m_truthBosonPt = -999.0F;
    m_truthBosonEta = -999.0F;
    m_truthBosonPhi = -999.0F;
    m_truthBosonM = -999.0F;
}

void TauSpinNtupleAlg::fillTau(const xAOD::TauJet* tau, std::size_t tauIndex){

    static const SG::AuxElement::ConstAccessor<int> nChargedTracks("nChargedTracks");
    static const SG::AuxElement::ConstAccessor<int> nIsolatedTracks("nIsolatedTracks");
    static const SG::AuxElement::ConstAccessor<int> nAllTracks("nAllTracks");
    static const SG::AuxElement::ConstAccessor<int> panTauDecayMode("PanTau_DecayMode");
    static const SG::AuxElement::ConstAccessor<int> nnDecayMode("NNDecayMode");
    static const SG::AuxElement::ConstAccessor<float> rnnJetScore("RNNJetScore");
    static const SG::AuxElement::ConstAccessor<float> rnnJetScoreSigTrans("RNNJetScoreSigTrans");
    static const SG::AuxElement::ConstAccessor<float> gnTauScore("GNTauScore");
    static const SG::AuxElement::ConstAccessor<float> gnTauScoreSigTrans("GNTauScoreSigTrans_v0");
    static const SG::AuxElement::ConstAccessor<char> gnTauVL("GNTauVL_v0");
    static const SG::AuxElement::ConstAccessor<char> gnTauL("GNTauL_v0");
    static const SG::AuxElement::ConstAccessor<char> gnTauM("GNTauM_v0");
    static const SG::AuxElement::ConstAccessor<char> gnTauT("GNTauT_v0");
    static const SG::AuxElement::ConstAccessor<char> truthIsHadronic("IsHadronicTau");
    static const SG::AuxElement::ConstAccessor<ElementLink<xAOD::TruthParticleContainer>>
        truthParticleLink("truthParticleLink");

    m_tauPt[tauIndex] = tau->pt() / 1000.0F;
    m_tauEta[tauIndex] = tau->eta();
    m_tauPhi[tauIndex] = tau->phi();
    m_tauCharge[tauIndex] = tau->charge();
    m_tauM[tauIndex] = tau->m() / 1000.0F;
    m_tauNTracks[tauIndex] = static_cast<int>(tau->nTracks());

    if (nChargedTracks.isAvailable(*tau)){
        m_tauNChargedTracks[tauIndex] = nChargedTracks(*tau);
    }
    if (nIsolatedTracks.isAvailable(*tau)){
        m_tauNIsolatedTracks[tauIndex] = nIsolatedTracks(*tau);
    }
    if (nAllTracks.isAvailable(*tau)){
        m_tauNAllTracks[tauIndex] = nAllTracks(*tau);
    }
    if (panTauDecayMode.isAvailable(*tau)){
        m_tauPanTauDecayMode[tauIndex] = panTauDecayMode(*tau);
    }
    if (nnDecayMode.isAvailable(*tau)){
        m_tauNNDecayMode[tauIndex] = nnDecayMode(*tau);
        m_tauDecayMode[tauIndex] = nnDecayMode(*tau);
    } else {
        m_tauDecayMode[tauIndex] = m_tauPanTauDecayMode[tauIndex];
    }
    if (rnnJetScore.isAvailable(*tau)){
        m_tauRNNJetScore[tauIndex] = rnnJetScore(*tau);
    }
    if (rnnJetScoreSigTrans.isAvailable(*tau)){
        m_tauRNNJetScoreSigTrans[tauIndex] = rnnJetScoreSigTrans(*tau);
    }
    if (gnTauScore.isAvailable(*tau)){
        const float score = gnTauScore(*tau);
        if (std::isfinite(score)){
            m_tauGNTauScoreV0[tauIndex] = score;
        }
    }
    if (gnTauScoreSigTrans.isAvailable(*tau)){
        m_tauGNTauScoreSigTransV0[tauIndex] = gnTauScoreSigTrans(*tau);
    }
    if (gnTauVL.isAvailable(*tau)){
        m_tauGNTauVLV0[tauIndex] = static_cast<int>(gnTauVL(*tau));
    }
    if (gnTauL.isAvailable(*tau)){
        m_tauGNTauLV0[tauIndex] = static_cast<int>(gnTauL(*tau));
    }
    if (gnTauM.isAvailable(*tau)){
        m_tauGNTauMV0[tauIndex] = static_cast<int>(gnTauM(*tau));
    }
    if (gnTauT.isAvailable(*tau)){
        m_tauGNTauTV0[tauIndex] = static_cast<int>(gnTauT(*tau));
    }

    // truth linkが保存されている場合だけmatchingを判定
    if (!truthParticleLink.isAvailable(*tau)){
        return;
    }

    m_tauTruthMatched[tauIndex] = 0;
    const auto& link = truthParticleLink(*tau);
    const xAOD::TruthParticle* truthTau = link.isValid() ? *link : nullptr;
    if (truthTau){
        m_tauTruthPdgId[tauIndex] = truthTau->pdgId();
    }
    if (truthTau && std::abs(truthTau->pdgId()) == 15){
        m_tauTruthMatched[tauIndex] = 1;
        m_tauTruthPt[tauIndex] = truthTau->pt() / 1000.0F;
        m_tauTruthEta[tauIndex] = truthTau->eta();
        m_tauTruthPhi[tauIndex] = truthTau->phi();
        m_tauTruthDecayMode[tauIndex] = static_cast<int>(TauAnalysisTools::getTruthDecayMode(*truthTau));
        if (truthIsHadronic.isAvailable(*truthTau)){
            m_tauTruthIsHadronic[tauIndex] = static_cast<int>(truthIsHadronic(*truthTau));
        }
    }
}

void TauSpinNtupleAlg::fillTracks(const xAOD::TauJet* tau, std::size_t tauIndex){

    int trackIndex = 0;
    for (const auto& link : tau->allTauTrackLinks()){
        if (!link.isValid()){
            if (!m_warnedInvalidTrackLink){
                ANA_MSG_WARNING("Invalid tau track links are skipped");
                m_warnedInvalidTrackLink = true;
            }
            ++trackIndex;
            continue;
        }

        const xAOD::TauTrack* tauTrack = *link;
        const xAOD::TrackParticle* track = tauTrack ? tauTrack->track() : nullptr;
        if (!track){
            ++trackIndex;
            continue;
        }

        std::uint8_t pixelHits = 0;
        std::uint8_t sctHits = 0;
        std::uint8_t trtHits = 0;
        const int nPixelHits = track->summaryValue(pixelHits, xAOD::numberOfPixelHits)
            ? static_cast<int>(pixelHits) : -1;
        const int nSCTHits = track->summaryValue(sctHits, xAOD::numberOfSCTHits)
            ? static_cast<int>(sctHits) : -1;
        const int nTRTHits = track->summaryValue(trtHits, xAOD::numberOfTRTHits)
            ? static_cast<int>(trtHits) : -1;

        m_trackTauIndex.push_back(static_cast<int>(tauIndex));
        m_trackIndexInTau.push_back(trackIndex);
        m_trackPt.push_back(track->pt() / 1000.0F);
        m_trackEta.push_back(track->eta());
        m_trackPhi.push_back(track->phi());
        m_trackDEta.push_back(track->eta() - tau->eta());
        m_trackDPhi.push_back(deltaPhi(track->phi(), tau->phi()));
        m_trackPtFraction.push_back(tau->pt() != 0.0 ? track->pt() / tau->pt() : -999.0F);
        m_trackD0.push_back(track->d0());
        m_trackZ0.push_back(track->z0());
        m_trackZ0SinTheta.push_back(track->z0() * std::sin(track->theta()));
        m_trackTheta.push_back(track->theta());
        m_trackQOverP.push_back(track->qOverP());
        m_trackCharge.push_back(track->charge());
        m_trackIsCore.push_back(static_cast<int>(tauTrack->flag(xAOD::TauJetParameters::coreTrack)));
        m_trackIsIsolation.push_back(static_cast<int>(tauTrack->flag(xAOD::TauJetParameters::classifiedIsolation)));
        m_trackIsConversion.push_back(static_cast<int>(tauTrack->flag(xAOD::TauJetParameters::classifiedConversion)));
        m_trackIsFake.push_back(static_cast<int>(tauTrack->flag(xAOD::TauJetParameters::classifiedFake)));
        m_trackPassTrkSelector.push_back(static_cast<int>(tauTrack->flag(xAOD::TauJetParameters::passTrkSelector)));
        m_trackNumberOfPixelHits.push_back(nPixelHits);
        m_trackNumberOfSCTHits.push_back(nSCTHits);
        m_trackNumberOfTRTHits.push_back(nTRTHits);
        ++trackIndex;
    }
}

void TauSpinNtupleAlg::fillPFOs(const xAOD::TauJet* tau, std::size_t tauIndex){

    static const SG::AuxElement::ConstAccessor<float> pfoCharge("charge");
    static const SG::AuxElement::ConstAccessor<float> bdtPi0Score("bdtPi0Score");

    std::unordered_map<const xAOD::PFO*, std::size_t> pfoIndices;
    int pfoIndex = 0;

    auto addPFO = [&](const xAOD::PFO* pfo, bool isPi0){
        const auto existing = pfoIndices.find(pfo);
        if (existing != pfoIndices.end()){
            if (isPi0){
                m_pfoIsPi0[existing->second] = 1;
            }
            return;
        }

        const std::size_t outputIndex = m_pfoIsPi0.size();
        pfoIndices[pfo] = outputIndex;

        int nPi0Proto = -1;
        pfo->attribute<int>(xAOD::PFODetails::nPi0Proto, nPi0Proto);

        m_pfoTauIndex.push_back(static_cast<int>(tauIndex));
        m_pfoIndexInTau.push_back(pfoIndex);
        m_pfoPt.push_back(pfo->pt() / 1000.0F);
        m_pfoEta.push_back(pfo->eta());
        m_pfoPhi.push_back(pfo->phi());
        m_pfoE.push_back(pfo->e() / 1000.0F);
        m_pfoDEta.push_back(pfo->eta() - tau->eta());
        m_pfoDPhi.push_back(deltaPhi(pfo->phi(), tau->phi()));
        m_pfoPtFraction.push_back(tau->pt() != 0.0 ? pfo->pt() / tau->pt() : -999.0F);
        m_pfoCharge.push_back(pfoCharge.isAvailable(*pfo) ? pfoCharge(*pfo) : -999.0F);
        m_pfoIsPi0.push_back(static_cast<int>(isPi0));
        m_pfoBDTPi0Score.push_back(bdtPi0Score.isAvailable(*pfo) ? bdtPi0Score(*pfo) : -999.0F);
        m_pfoNPi0Proto.push_back(nPi0Proto);
        ++pfoIndex;
    };

    for (const auto& link : tau->neutralPFOLinks()){
        if (!link.isValid()){
            if (!m_warnedInvalidPFOLink){
                ANA_MSG_WARNING("Invalid tau PFO links are skipped");
                m_warnedInvalidPFOLink = true;
            }
            continue;
        }
        addPFO(*link, false);
    }

    for (const auto& link : tau->pi0PFOLinks()){
        if (!link.isValid()){
            if (!m_warnedInvalidPFOLink){
                ANA_MSG_WARNING("Invalid tau PFO links are skipped");
                m_warnedInvalidPFOLink = true;
            }
            continue;
        }
        addPFO(*link, true);
    }
}

float TauSpinNtupleAlg::deltaPhi(float phi1, float phi2){
    const float difference = phi1 - phi2;
    return std::atan2(std::sin(difference), std::cos(difference));
}

bool TauSpinNtupleAlg::hasTauDescendant(const xAOD::TruthParticle* particle, int depth){
    if (!particle || depth > 20){
        return false;
    }
    for (std::size_t i = 0; i < particle->nChildren(); ++i){
        const xAOD::TruthParticle* child = particle->child(i);
        if (!child){
            continue;
        }
        if (std::abs(child->pdgId()) == 15 || hasTauDescendant(child, depth + 1)){
            return true;
        }
    }
    return false;
}
