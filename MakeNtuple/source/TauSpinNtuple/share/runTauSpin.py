import argparse
import sys
from pathlib import Path
import ROOT

# 全体実行
def main():
    parser = argparse.ArgumentParser(description = "Create ntuple from DAOD_PHYS.")

    # DAOD_PHYS file
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input")
    input_group.add_argument("--input-list")
    parser.add_argument("--max-events", type=int, default=5)
    parser.add_argument("--submit-dir", default="submitDir")
    args = parser.parse_args()

    # 入力ファイル一覧
    if args.input:
        input_files = [args.input]
    else:
        input_list_path = Path(args.input_list)
        if not input_list_path.is_file():
            raise FileNotFoundError(f"Input list was not found: {input_list_path}")

        input_files = []
        for line in input_list_path.read_text().splitlines():
            file_path = line.strip()
            if not file_path or file_path.startswith("#"):
                continue
            if not Path(file_path).is_file():
                print(f"Skip unavailable input: {file_path}")
                continue
            input_files.append(file_path)

    if not input_files:
        raise RuntimeError("No valid input files were found")

    print(f"Input files: {len(input_files)}")

    ROOT.xAOD.Init().ignore()
    sample_handler = ROOT.SH.SampleHandler()
    sample_handler.setMetaString("nc_tree", "CollectionTree")
    sample = ROOT.SH.SampleLocal("input")
    for input_file in input_files:
        sample.add(input_file)
    sample_handler.add(sample)
    sample_handler.printContent()

    # job
    from AnaAlgorithm.AlgSequence import AlgSequence
    from AnalysisAlgorithmsConfig.ConfigAccumulator import DataType
    from AnalysisAlgorithmsConfig.ConfigText import makeSequence as makeTextSequence
    from AnaAlgorithm.DualUseConfig import createAlgorithm
    from AthenaConfiguration.AllConfigFlags import initConfigFlags

    job = ROOT.EL.Job()
    job.sampleHandler(sample_handler) #sample登録

    job.outputAdd(ROOT.EL.OutputStream("ANALYSIS")) # 自作TTreeの書き出し先

    job.options().setDouble(ROOT.EL.Job.optMaxEvents, args.max_events)
    job.options().setString(ROOT.EL.Job.optSubmitDirMode, "unique-link")

    data_path = Path(__file__).resolve().parents[1] / "data"
    config_path = data_path / "config.yaml"
    tau_selection_path = data_path / "tau_selection_rnn_loose_nocrack.conf"

    # 入力metadataからcampaignとPRW設定を決める
    config_flags = initConfigFlags()
    config_flags.Input.Files = input_files
    config_flags.lock()

    # YAMLからCP Algorithmを作るsequence
    alg_seq = AlgSequence("AnalysisSequence")

    makeTextSequence(
        configPath=str(config_path),
        dataType=DataType.FastSim,
        algSeq=alg_seq,
        geometry="RUN2",
        autoconfigFromFlags=config_flags,
        noSystematics=True,
    )

    # 標準RNN Looseからeta crack vetoだけ外す
    found_tau_selection = False
    for alg in alg_seq:
        if alg.getName() == "TauSelectionAlg_loose":
            alg.selectionTool.ConfigPath = str(tau_selection_path)
            found_tau_selection = True
            break
    if not found_tau_selection:
        raise RuntimeError("TauSelectionAlg_loose was not found")

    alg_seq.addSelfToJob(job)

    # 自作Algorithmを登録
    algorithm = createAlgorithm("TauSpinNtupleAlg", "TauSpinAlg")
    job.algsAdd(algorithm)
    driver = ROOT.EL.DirectDriver()
    driver.submit(job, args.submit_dir)

    return 0

if __name__ == "__main__":
    sys.exit(main())