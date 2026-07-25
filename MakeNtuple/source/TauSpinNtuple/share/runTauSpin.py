import argparse
import sys
import ROOT

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
    job = ROOT.EL.Job()
    job.sampleHandler(sample_handler)
    job.options().setDouble(ROOT.EL.Job.optMaxEvents,5)
    job.options().setString(ROOT.EL.Job.optSubmitDirMode, "unique-link")

    from AnaAlgorithm.DualUseConfig import createAlgorithm

    algorithm = createAlgorithm("TauSpinNtupleAlg", "TauSpinAlg")

    job.algsAdd(algorithm)
    driver = ROOT.EL.DirectDriver()
    driver.submit(job, "submitDir")

    print(f"Input file:{args.input}")

    return 0

if __name__ == "__main__":
    sys.exit(main())