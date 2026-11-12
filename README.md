# PDF Astrological & Visual Chart Extractor

This project scans PDF documents (such as astrological Kundli reports, birth charts, divisional charts, and solar return charts) and automatically detects and extracts **only authentic charts** in high resolution, filtering out advertisements, data tables, headers, and decorative elements.

## Features

- **Accurate Chart Detection**: Uses connected-component vector geometry (and OpenCV raster diamond cross-detection fallback) to specifically identify North Indian Kundli chart structures (outer borders, inner diamond, and 12-house diagonal divisions).
- **Zero Noise / No False Positives**: Automatically eliminates tables (which only contain orthogonal horizontal/vertical grids) and advertisements (raster banner ads and cover graphics).
- **Automatic Title Association**: Extracts and associates the exact chart name from the document (such as `Varshphal Chart (Solar Return Chart)`, `Lagna Chart`, `Navamasa Chart`, `Karakamsa Chart`, `Swamsa Chart`, `Shodashvarga Charts`, etc.).
- **Chart Content Analysis**: Extracts the planets (e.g., Sun, Moon, Mars, Mercury, Jupiter, Venus, Saturn, Rahu, Ketu, etc.) and zodiac house signs (1–12) located inside each chart.
- **Structured Markdown Output**: Generates a clean `output/charts.md` with a clickable Table of Contents, chart titles, metadata, planetary placements, and embedded high-resolution PNGs.
- **Detailed JSON Report**: Outputs `output/detection_report.json` with coordinate bounding boxes, reasons, and metadata for every extracted chart.

## Outputs Generated

- `output/charts/*.png` — High-resolution cropped images of each chart with descriptive filenames (e.g. `page_016_chart_0004_varshphal_chart_solar_return_chart.png`).
- `output/charts.md` & `output/extracted_images.md` — Complete Markdown document containing all extracted charts, titles, and details.
- `output/detection_report.json` — Machine-readable extraction metadata report.

## Setup (Windows PowerShell / CMD)

From the project root:

```bash
# 1. Activate existing virtual environment
.venv\Scripts\activate

# 2. Install dependencies (if not already installed)
pip install -r requirements.txt
```

## Running the Extractor

Extract all charts from `input/sample.pdf`:

```bash
python scripts/extract_charts.py
```

Or specify custom paths and parameters:

```bash
python scripts/extract_charts.py "input/sample.pdf" --output output --dpi 300
```

To extract only a specific chart (e.g. Varshphal / Solar Return):

```bash
python scripts/extract_charts.py input/sample.pdf --filter "Varshphal"
```
