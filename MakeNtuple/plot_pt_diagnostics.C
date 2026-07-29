#include <TChain.h>
#include <TCanvas.h>
#include <TLegend.h>
#include <TH1D.h>
#include <TH2D.h>
#include <TROOT.h>
#include <TStyle.h>
#include <TSystem.h>
#include <TTreeReader.h>
#include <TTreeReaderArray.h>
#include <TTreeReaderValue.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct SampleData {
  std::vector<double> truthPt;
  std::vector<double> tauMinusPt;
  std::vector<double> tauPlusPt;
  std::vector<double> leadingTauPt;
  std::vector<double> subleadingTauPt;
  std::vector<double> visibleDitauPt;
  std::vector<double> leadingVsTruthX;
  std::vector<double> leadingVsTruthY;
  std::vector<double> subleadingVsTruthX;
  std::vector<double> subleadingVsTruthY;
  long long pdgMatches = 0;
  long long pdgMismatches = 0;
  std::vector<double> masses;
};

std::string join(const std::string &dir, const std::string &name) {
  return dir + "/" + name;
}

double median(std::vector<double> values) {
  if (values.empty()) return std::numeric_limits<double>::quiet_NaN();
  std::sort(values.begin(), values.end());
  const size_t n = values.size();
  if (n % 2) return values[n / 2];
  return 0.5 * (values[n / 2 - 1] + values[n / 2]);
}

double mean(const std::vector<double> &values) {
  if (values.empty()) return std::numeric_limits<double>::quiet_NaN();
  double sum = 0.0;
  for (double x : values) sum += x;
  return sum / static_cast<double>(values.size());
}

double finiteMin(const std::vector<double> &values) {
  if (values.empty()) return std::numeric_limits<double>::quiet_NaN();
  return *std::min_element(values.begin(), values.end());
}

double finiteMax(const std::vector<double> &values) {
  if (values.empty()) return std::numeric_limits<double>::quiet_NaN();
  return *std::max_element(values.begin(), values.end());
}

SampleData readSample(const std::string &dir, int expectedPdg) {
  SampleData result;
  TChain chain("tauspin");
  chain.Add((dir + "/*.root").c_str());

  TTreeReader reader(&chain);
  TTreeReaderValue<int> pdg(reader, "truth_boson_pdgId");
  TTreeReaderValue<float> truthPt(reader, "truth_boson_pt");
  TTreeReaderValue<float> truthMass(reader, "truth_boson_m");
  TTreeReaderArray<float> tauPt(reader, "tau_pt");
  TTreeReaderArray<float> tauEta(reader, "tau_eta");
  TTreeReaderArray<float> tauPhi(reader, "tau_phi");
  TTreeReaderArray<float> tauMass(reader, "tau_m");

  while (reader.Next()) {
    if (!std::isfinite(*truthPt) || !std::isfinite(*truthMass) || tauPt.GetSize() < 2 ||
        tauEta.GetSize() < 2 || tauPhi.GetSize() < 2 || tauMass.GetSize() < 2) {
      continue;
    }
    result.truthPt.push_back(*truthPt);
    result.masses.push_back(*truthMass);
    if (*pdg == expectedPdg) ++result.pdgMatches;
    else ++result.pdgMismatches;

    const double pt0 = tauPt[0];
    const double pt1 = tauPt[1];
    result.tauMinusPt.push_back(pt0);
    result.tauPlusPt.push_back(pt1);
    const double lead = std::max(pt0, pt1);
    const double sub = std::min(pt0, pt1);
    result.leadingTauPt.push_back(lead);
    result.subleadingTauPt.push_back(sub);
    result.leadingVsTruthX.push_back(*truthPt);
    result.leadingVsTruthY.push_back(lead);
    result.subleadingVsTruthX.push_back(*truthPt);
    result.subleadingVsTruthY.push_back(sub);

    const double px = pt0 * std::cos(tauPhi[0]) + pt1 * std::cos(tauPhi[1]);
    const double py = pt0 * std::sin(tauPhi[0]) + pt1 * std::sin(tauPhi[1]);
    result.visibleDitauPt.push_back(std::sqrt(px * px + py * py));
  }
  return result;
}

void style(TH1 *h, int color, const char *xTitle, const char *yTitle = "Events") {
  h->SetLineColor(color);
  h->SetLineWidth(2);
  h->SetStats(false);
  h->GetXaxis()->SetTitle(xTitle);
  h->GetYaxis()->SetTitle(yTitle);
}

void drawOverlay(TH1 *h, TH1 *z, const std::string &path, const char *xTitle,
                 bool normalize, bool logy, double xmin = -1, double xmax = -1) {
  auto hc = static_cast<TH1 *>(h->Clone("h_draw"));
  auto zc = static_cast<TH1 *>(z->Clone("z_draw"));
  if (normalize) {
    if (hc->Integral() > 0) hc->Scale(1.0 / hc->Integral());
    if (zc->Integral() > 0) zc->Scale(1.0 / zc->Integral());
  }
  style(hc, kRed + 1, xTitle, normalize ? "Normalized entries" : "Events");
  style(zc, kBlue + 1, xTitle, normalize ? "Normalized entries" : "Events");
  if (xmin >= 0 && xmax > xmin) {
    hc->GetXaxis()->SetRangeUser(xmin, xmax);
    zc->GetXaxis()->SetRangeUser(xmin, xmax);
  }
  const double ymax = std::max(hc->GetMaximum(), zc->GetMaximum());
  hc->SetMaximum(logy ? std::max(1.0, ymax * 30.0) : ymax * 1.2);
  TCanvas canvas("canvas", "canvas", 1000, 700);
  canvas.SetLogy(logy);
  hc->Draw("hist");
  zc->Draw("hist same");
  TLegend legend(0.72, 0.78, 0.9, 0.9);
  legend.AddEntry(hc, "H", "l");
  legend.AddEntry(zc, "Z", "l");
  legend.Draw();
  canvas.SaveAs(path.c_str());
  delete hc;
  delete zc;
}

void drawRatio(TH1D *h, TH1D *z, const std::string &path, const char *xTitle) {
  auto hn = static_cast<TH1D *>(h->Clone("h_norm_ratio"));
  auto zn = static_cast<TH1D *>(z->Clone("z_norm_ratio"));
  if (hn->Integral() > 0) hn->Scale(1.0 / hn->Integral());
  if (zn->Integral() > 0) zn->Scale(1.0 / zn->Integral());
  hn->Divide(zn);
  hn->SetStats(false);
  hn->SetLineColor(kBlack);
  hn->SetMarkerStyle(20);
  hn->GetXaxis()->SetTitle(xTitle);
  hn->GetYaxis()->SetTitle("Normalized H / Z");
  hn->GetYaxis()->SetRangeUser(0.0, 2.5);
  TCanvas canvas("ratio", "ratio", 1000, 700);
  hn->Draw("hist p");
  canvas.SaveAs(path.c_str());
  delete hn;
  delete zn;
}

void draw2D(const std::vector<double> &x, const std::vector<double> &y,
            const std::string &path, const char *yTitle, const char *title) {
  TH2D hist("h2", title, 50, 0.0, 1000.0, 50, 0.0, 500.0);
  for (size_t i = 0; i < x.size(); ++i) hist.Fill(x[i], y[i]);
  hist.SetStats(false);
  hist.GetXaxis()->SetTitle("truth boson p_{T} [GeV]");
  hist.GetYaxis()->SetTitle(yTitle);
  TCanvas canvas("h2canvas", "h2canvas", 1000, 800);
  canvas.SetRightMargin(0.14);
  hist.Draw("colz");
  canvas.SaveAs(path.c_str());
}

void writeJson(const SampleData &h, const SampleData &z, const std::string &path) {
  std::ofstream out(path);
  out << std::setprecision(10);
  out << "{\n";
  auto emit = [&](const char *name, const SampleData &d) {
    out << "  \"" << name << "\": {\n";
    out << "    \"entries\": " << d.truthPt.size() << ",\n";
    out << "    \"pdg_matches\": " << d.pdgMatches << ",\n";
    out << "    \"pdg_mismatches\": " << d.pdgMismatches << ",\n";
    out << "    \"truth_boson_pt_GeV\": {\"min\": " << finiteMin(d.truthPt)
        << ", \"median\": " << median(d.truthPt) << ", \"mean\": " << mean(d.truthPt)
        << ", \"max\": " << finiteMax(d.truthPt) << "},\n";
    out << "    \"truth_boson_mass_GeV\": {\"mean\": " << mean(d.masses)
        << ", \"min\": " << finiteMin(d.masses) << ", \"max\": " << finiteMax(d.masses) << "}\n";
    out << "  }";
  };
  emit("H", h);
  out << ",\n";
  emit("Z", z);
  out << "\n}\n";
}

}  // namespace

void plot_pt_diagnostics(const char *hDir, const char *zDir, const char *outDir) {
  gROOT->SetBatch(true);
  gStyle->SetOptStat(0);
  gSystem->mkdir(outDir, true);

  const SampleData h = readSample(hDir, 25);
  const SampleData z = readSample(zDir, 23);
  writeJson(h, z, join(outDir, "pt_summary.json"));

  TH1D hTruth("h_truth", "", 50, 0.0, 1000.0);
  TH1D zTruth("z_truth", "", 50, 0.0, 1000.0);
  TH1D hTauMinus("h_tau_minus", "", 50, 0.0, 500.0);
  TH1D zTauMinus("z_tau_minus", "", 50, 0.0, 500.0);
  TH1D hTauPlus("h_tau_plus", "", 50, 0.0, 500.0);
  TH1D zTauPlus("z_tau_plus", "", 50, 0.0, 500.0);
  TH1D hTauLead("h_tau_lead", "", 50, 0.0, 500.0);
  TH1D zTauLead("z_tau_lead", "", 50, 0.0, 500.0);
  TH1D hTauSub("h_tau_sub", "", 50, 0.0, 500.0);
  TH1D zTauSub("z_tau_sub", "", 50, 0.0, 500.0);
  TH1D hDitau("h_ditau", "", 50, 0.0, 1000.0);
  TH1D zDitau("z_ditau", "", 50, 0.0, 1000.0);
  for (double v : h.truthPt) hTruth.Fill(v);
  for (double v : z.truthPt) zTruth.Fill(v);
  for (double v : h.tauMinusPt) hTauMinus.Fill(v);
  for (double v : z.tauMinusPt) zTauMinus.Fill(v);
  for (double v : h.tauPlusPt) hTauPlus.Fill(v);
  for (double v : z.tauPlusPt) zTauPlus.Fill(v);
  for (double v : h.leadingTauPt) hTauLead.Fill(v);
  for (double v : z.leadingTauPt) zTauLead.Fill(v);
  for (double v : h.subleadingTauPt) hTauSub.Fill(v);
  for (double v : z.subleadingTauPt) zTauSub.Fill(v);
  for (double v : h.visibleDitauPt) hDitau.Fill(v);
  for (double v : z.visibleDitauPt) zDitau.Fill(v);

  drawOverlay(&hTruth, &zTruth, join(outDir, "truth_boson_pt_raw.png"), "truth boson p_{T} [GeV]", false, false);
  drawOverlay(&hTruth, &zTruth, join(outDir, "truth_boson_pt_normalized.png"), "truth boson p_{T} [GeV]", true, false);
  drawRatio(&hTruth, &zTruth, join(outDir, "truth_boson_pt_ratio.png"), "truth boson p_{T} [GeV]");
  drawOverlay(&hTruth, &zTruth, join(outDir, "truth_boson_pt_logy.png"), "truth boson p_{T} [GeV]", false, true);
  drawOverlay(&hTruth, &zTruth, join(outDir, "truth_boson_pt_zoom.png"), "truth boson p_{T} [GeV]", true, false, 150.0, 500.0);
  drawOverlay(&hTauMinus, &zTauMinus, join(outDir, "tau_minus_pt_normalized.png"), "#tau^{-} p_{T} [GeV]", true, false);
  drawOverlay(&hTauPlus, &zTauPlus, join(outDir, "tau_plus_pt_normalized.png"), "#tau^{+} p_{T} [GeV]", true, false);
  drawOverlay(&hTauLead, &zTauLead, join(outDir, "tau_leading_pt_normalized.png"), "leading tau p_{T} [GeV]", true, false);
  drawOverlay(&hTauSub, &zTauSub, join(outDir, "tau_subleading_pt_normalized.png"), "subleading tau p_{T} [GeV]", true, false);
  drawOverlay(&hDitau, &zDitau, join(outDir, "visible_ditau_pt_normalized.png"), "visible di-tau p_{T} [GeV]", true, false);
  draw2D(h.leadingVsTruthX, h.leadingVsTruthY, join(outDir, "H_truth_boson_pt_vs_leading_tau_pt.png"), "leading tau p_{T} [GeV]", "H");
  draw2D(z.leadingVsTruthX, z.leadingVsTruthY, join(outDir, "Z_truth_boson_pt_vs_leading_tau_pt.png"), "leading tau p_{T} [GeV]", "Z");
  draw2D(h.subleadingVsTruthX, h.subleadingVsTruthY, join(outDir, "H_truth_boson_pt_vs_subleading_tau_pt.png"), "subleading tau p_{T} [GeV]", "H");
  draw2D(z.subleadingVsTruthX, z.subleadingVsTruthY, join(outDir, "Z_truth_boson_pt_vs_subleading_tau_pt.png"), "subleading tau p_{T} [GeV]", "Z");
}
