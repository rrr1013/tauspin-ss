#ifndef TAUSPINNTUPLE_TAUSPINNTUPLEALG_H
#define TAUSPINNTUPLE_TAUSPINNTUPLEALG_H

#include <SelectionHelpers/SysReadSelectionHandle.h>
#include <SystematicsHandles/SysListHandle.h>
#include <SystematicsHandles/SysReadHandle.h>
#include <xAODMissingET/MissingETContainer.h>
#include <xAODTau/TauJetContainer.h>
#include <xAODTruth/TruthParticle.h>
#include <AnaAlgorithm/AnaAlgorithm.h>
#include <cstdint>
#include <string>
#include <vector>

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
        "RNN Loose and kinematic selection"
    };

    CP::SysReadHandle<xAOD::MissingETContainer> m_met {
        this,
        "met",
        "AnaMET_%SYS%",
        "Reconstructed MET container"
    };

    // イベント情報
    std::uint64_t m_eventNumber = 0;
    std::uint32_t m_runNumber = 0;
    std::uint32_t m_mcChannelNumber = 0;
    float m_averageInteractionsPerCrossing = -999.0F;
    float m_actualInteractionsPerCrossing = -999.0F;
    float m_metEt = -999.0F;
    float m_metPhi = -999.0F;
    float m_metSumet = -999.0F;

    // tau[0] = tau-
    // tau[1] = tau+
    float m_tauPt[2] = {0.0F, 0.0F};
    float m_tauEta[2] = {0.0F, 0.0F};
    float m_tauPhi[2] = {0.0F, 0.0F};
    float m_tauCharge[2] = {0.0F, 0.0F};
    float m_tauM[2] = {-999.0F, -999.0F};
    int m_tauNTracks[2] = {-1, -1};
    int m_tauNChargedTracks[2] = {-1, -1};
    int m_tauNIsolatedTracks[2] = {-1, -1};
    int m_tauNAllTracks[2] = {-1, -1};
    int m_tauDecayMode[2] = {-1, -1};
    int m_tauPanTauDecayMode[2] = {-1, -1};
    int m_tauNNDecayMode[2] = {-1, -1};
    float m_tauRNNJetScore[2] = {-999.0F, -999.0F};
    float m_tauRNNJetScoreSigTrans[2] = {-999.0F, -999.0F};
    float m_tauGNTauScoreV0[2] = {-999.0F, -999.0F};
    float m_tauGNTauScoreSigTransV0[2] = {-999.0F, -999.0F};
    int m_tauGNTauVLV0[2] = {-1, -1};
    int m_tauGNTauLV0[2] = {-1, -1};
    int m_tauGNTauMV0[2] = {-1, -1};
    int m_tauGNTauTV0[2] = {-1, -1};

    // tau track
    std::vector<int> m_trackTauIndex;
    std::vector<int> m_trackIndexInTau;
    std::vector<float> m_trackPt;
    std::vector<float> m_trackEta;
    std::vector<float> m_trackPhi;
    std::vector<float> m_trackDEta;
    std::vector<float> m_trackDPhi;
    std::vector<float> m_trackPtFraction;
    std::vector<float> m_trackD0;
    std::vector<float> m_trackZ0;
    std::vector<float> m_trackZ0SinTheta;
    std::vector<float> m_trackTheta;
    std::vector<float> m_trackQOverP;
    std::vector<float> m_trackCharge;
    std::vector<int> m_trackIsCore;
    std::vector<int> m_trackIsIsolation;
    std::vector<int> m_trackIsConversion;
    std::vector<int> m_trackIsFake;
    std::vector<int> m_trackPassTrkSelector;
    std::vector<int> m_trackNumberOfPixelHits;
    std::vector<int> m_trackNumberOfSCTHits;
    std::vector<int> m_trackNumberOfTRTHits;

    // neutral/pi0 PFO
    std::vector<int> m_pfoTauIndex;
    std::vector<int> m_pfoIndexInTau;
    std::vector<float> m_pfoPt;
    std::vector<float> m_pfoEta;
    std::vector<float> m_pfoPhi;
    std::vector<float> m_pfoE;
    std::vector<float> m_pfoDEta;
    std::vector<float> m_pfoDPhi;
    std::vector<float> m_pfoPtFraction;
    std::vector<float> m_pfoCharge;
    std::vector<int> m_pfoIsPi0;
    std::vector<float> m_pfoBDTPi0Score;
    std::vector<int> m_pfoNPi0Proto;

    // vertex
    float m_primaryVertexX = -999.0F;
    float m_primaryVertexY = -999.0F;
    float m_primaryVertexZ = -999.0F;
    int m_primaryVertexNTracks = -1;
    int m_nPrimaryVertices = 0;
    int m_tauVertexIndex[2] = {-1, -1};
    float m_tauVertexDeltaZ[2] = {-999.0F, -999.0F};

    // truth診断
    int m_truthHasHiggs = 0;
    int m_truthHasZ = 0;
    int m_truthBosonPdgId = 0;
    float m_truthBosonPt = -999.0F;
    float m_truthBosonEta = -999.0F;
    float m_truthBosonPhi = -999.0F;
    float m_truthBosonM = -999.0F;
    int m_tauTruthMatched[2] = {-1, -1};
    int m_tauTruthPdgId[2] = {0, 0};
    float m_tauTruthPt[2] = {-999.0F, -999.0F};
    float m_tauTruthEta[2] = {-999.0F, -999.0F};
    float m_tauTruthPhi[2] = {-999.0F, -999.0F};
    int m_tauTruthDecayMode[2] = {-1, -1};
    int m_tauTruthIsHadronic[2] = {-1, -1};

    bool m_warnedInvalidTrackLink = false;
    bool m_warnedInvalidPFOLink = false;

    void resetBranches();
    void fillTau(const xAOD::TauJet* tau, std::size_t tauIndex);
    void fillTracks(const xAOD::TauJet* tau, std::size_t tauIndex);
    void fillPFOs(const xAOD::TauJet* tau, std::size_t tauIndex);
    static float deltaPhi(float phi1, float phi2);
    static bool hasTauDescendant(const xAOD::TruthParticle* particle, int depth = 0);
};

#endif
