import argparse
import sys
import ROOT

# 全体実行
def main():
    parser = argparse.ArgumentParser(description = "Create ntuple from DAOD_PHYS.")

    # DAOD_PHYS file
    parser.add_argument("--input", required=True)
    args = parser.parse_args()

    ROOT.xAOD.Init().ignore()
    sample_handler = ROOT.SH.SampleHandler()
    sample_handler.setMetaString("nc_tree", "CollectionTree")
    sample = ROOT.SH.SampleLocal("input")
    sample.add(args.input)
    sample_handler.add(sample)
    sample_handler.printContent()

    # job
    from pathlib import Path
    from AnaAlgorithm.AlgSequence import AlgSequence
    from AnalysisAlgorithmsConfig.ConfigAccumulator import DataType
    from AnalysisAlgorithmsConfig.ConfigText import makeSequence as makeTextSequence
    from AnaAlgorithm.DualUseConfig import createAlgorithm

    job = ROOT.EL.Job()
    job.sampleHandler(sample_handler) #sample登録
    job.options().setDouble(ROOT.EL.Job.optMaxEvents,5) #先頭 5 event
    job.options().setString(ROOT.EL.Job.optSubmitDirMode, "unique-link")

    config_path = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "config.yaml"
    )

    # YAMLからCP Algorithmを作るsequence
    alg_seq = AlgSequence("AnalysisSequence")

    makeTextSequence(
        configPath=str(config_path),
        dataType=DataType.FastSim,
        algSeq=alg_seq,
        geometry="RUN3",
        noSystematics=True,
    )

    alg_seq.addSelfToJob(job)

    # 自作Algorithmを登録
    algorithm = createAlgorithm("TauSpinNtupleAlg", "TauSpinAlg")
    job.algsAdd(algorithm)
    driver = ROOT.EL.DirectDriver()
    driver.submit(job, "submitDir")

    return 0

if __name__ == "__main__":
    sys.exit(main())