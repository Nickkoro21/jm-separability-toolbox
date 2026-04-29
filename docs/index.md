---
layout: default
---

A browser-based, sensor-agnostic toolkit for quantifying class separability in multispectral remote-sensing data. Built as a complementary deliverable to an MSc thesis at the University of the Aegean.

## Quick links

- 🚀 **[Try the live app](https://huggingface.co/spaces/NickKoro21/jm-separability-toolbox)** — interactive Gradio app on Hugging Face Spaces
- 📂 **[Source code on GitHub](https://github.com/Nickkoro21/jm-separability-toolbox)** — full implementation, README, license
- 📖 **[Full documentation](https://github.com/Nickkoro21/jm-separability-toolbox#readme)** — theory, input format, validation rules, six-step workflow

## What the tool does

Spectral Separability Explorer accepts any CSV with per-sample band values and a class label, and produces:

- Per-class spectral signatures
- Boxplots and violin plots per band (units-aware: reflectance, height in metres, temperature in °C)
- Jeffries–Matusita distance matrices with discrete 4-bucket colour coding (Poor / Moderate / Good / Excellent)
- Comparative analysis across band subsets (e.g. RGB vs 5 MS vs 7D = RGB + RedEdge + NIR + nDSM + Thermal)
- Ranked separability table for pair-by-pair drill-down
- ZIP export bundle with results CSV plus an HTML interpretation guide

Eight built-in sensor presets cover the most common multispectral and satellite platforms: MicaSense Altum-PT, RedEdge-MX, RedEdge-MX Dual, DJI Phantom 4 Multispectral, DJI Mavic 3 Multispectral, Parrot Sequoia, Sentinel-2 MSI, Landsat 8/9 OLI+TIRS.

For full theory, input format, validation rules, and the six-step workflow, see the [GitHub README](https://github.com/Nickkoro21/jm-separability-toolbox#readme).

## Citation

If you use this tool in academic work:

```bibtex
@software{koroniadis2026spectral,
  author    = {Koroniadis, Nikolaos},
  title     = {{Spectral Separability Explorer:
                Sensor-agnostic Jeffries--Matusita analysis for multispectral data}},
  year      = {2026},
  publisher = {Hugging Face},
  url       = {https://huggingface.co/spaces/NickKoro21/jm-separability-toolbox},
  note      = {MSc thesis deliverable, University of the Aegean}
}
```

## Author

**Nikolaos Koroniadis** — MSc Geography and Applied Geoinformatics, [University of the Aegean](https://geography.aegean.gr/geoinformatics/), [RSGIS Lab](https://rsgis.aegean.gr/).

Thesis Supervisor: Dr. Christos Vasilakos.

Released under the [MIT License](https://github.com/Nickkoro21/jm-separability-toolbox/blob/main/LICENSE).