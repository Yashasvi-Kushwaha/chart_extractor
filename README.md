# PDF Astrological & Visual Chart Extractor

A Python-based PDF extraction tool designed to detect and extract astrological charts, particularly Vedic and North Indian Kundli charts, from PDF documents.

The program analyzes PDF vector drawings to identify chart structures, extracts chart regions as high-resolution PNG images, identifies chart titles, and maps readable planetary and sign information to approximate house positions.

It generates Markdown, HTML, JSON, and ZIP outputs for each processed PDF.

## Features

- **Vector-based chart detection:** Identifies chart structures by analyzing diagonal lines and connected drawing components in PDF pages.
- **Raster-based fallback detection:** Uses OpenCV edge detection and Hough Line Transform to identify chart-like structures in scanned or image-based PDFs.
- **Automatic chart title detection:** Attempts to identify chart headings and associated section names.
- **House and planetary mapping:** Extracts readable planetary labels, sign numbers, and other numerical values using PDF text coordinates.
- **High-resolution image extraction:** Exports detected chart regions as PNG images, with a default resolution of 300 DPI.
- **Multiple report formats:** Generates standard Markdown, standalone Markdown with embedded Base64 images, standalone HTML, and JSON metadata.
- **Separate outputs for different PDFs:** Creates an independent output directory for each input PDF to prevent report and image files from being overwritten.

## Technologies Used

| Technology | Purpose                                                                    |
| ---------- | -------------------------------------------------------------------------- |
| Python     | Main programming language                                                  |
| PyMuPDF    | PDF reading, text extraction, vector drawing analysis, and image rendering |
| OpenCV     | Raster image processing and line detection                                 |
| NumPy      | Image array processing                                                     |
| HTML       | Standalone visual reports                                                  |
| Markdown   | Structured chart reports                                                   |
| JSON       | Extraction metadata and detection reports                                  |

## Project Structure

```text
PDF_Extraction_Project/
│
├── input/
│   └── sample.pdf
│
├── scripts/
│   └── script.py
│
├── output/
│   └── sample/
│       ├── charts/
│       │   ├── page_001_chart_0001_*.png
│       │   └── page_002_chart_0002_*.png
│       │
│       ├── chartssample.md
│       ├── extracted_imagessample.md
│       ├── chartssample_standalone.md
│       ├── chartssample.html
│       └── detection_report.json
│
├── extracted_charts/
│   └── sample.zip
│
├── README.md
└── requirements.txt
```

*The example filenames above assume the input PDF is named `sample.pdf`. Actual chart image filenames depend on the detected page, chart number, and chart title.*

## Installation

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd PDF_Extraction_Project
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

If a `requirements.txt` file is not available, install the dependencies directly:

```bash
pip install pymupdf opencv-python numpy
```

## Usage

### 1. Add your PDF

Place the PDF document inside the `input/` directory.

Example:

```text
input/
└── sample.pdf
```

### 2. Run the extractor

From the project root:

```bash
python scripts/script.py input/sample.pdf
```

If no PDF path is provided, the program uses its default input path:

```bash
python scripts/script.py
```

Default input:

```text
input/sample.pdf
```

### 3. Optional arguments

**Specify a custom output directory:**

```bash
python scripts/script.py input/sample.pdf --output results
```

**Change the image export resolution:**

```bash
python scripts/script.py input/sample.pdf --dpi 400
```

**Filter charts by title keyword:**

```bash
python scripts/script.py input/sample.pdf --filter Varshphal
```

### Command-line arguments

| Argument   | Description                         | Default            |
| ---------- | ----------------------------------- | ------------------ |
| `pdf`      | Input PDF file path                 | `input/sample.pdf` |
| `--output` | Base output directory               | `output`           |
| `--dpi`    | PNG export resolution               | `300`              |
| `--filter` | Optional chart title keyword filter | None               |

## Output Files

For an input PDF named `sample.pdf`, the program creates the following outputs.

| Output                        | Description                                                       |
| ----------------------------- | ----------------------------------------------------------------- |
| `charts/`                     | Directory containing extracted chart PNG images                   |
| `chartssample.md`             | Main Markdown report containing chart images and placement tables |
| `extracted_imagessample.md`   | Backward-compatible copy of the main Markdown report              |
| `chartssample_standalone.md`  | Self-contained Markdown with embedded Base64 images               |
| `chartssample.html`           | Standalone HTML report with embedded chart images                 |
| `detection_report.json`       | Structured chart detection metadata                               |
| `extracted_charts/sample.zip` | ZIP archive containing the PDF's output directory                 |

## Detection Workflow

The extraction process follows these stages:

1. **PDF loading:** Opens the input PDF and examines its pages.
2. **Detection method selection:** Checks whether the document contains vector drawings.
3. **Chart detection:** Uses vector drawing analysis when vector drawings are present; otherwise, uses raster-based detection.
4. **Geometric filtering:** Evaluates chart dimensions, diagonal lines, aspect ratios, and bounding regions.
5. **Title identification:** Searches for nearby text that may identify the chart and its section.
6. **House mapping:** Uses PDF text coordinates and predefined North Indian chart geometry to associate readable values with house regions.
7. **Image extraction:** Renders detected chart regions as high-resolution PNG images.
8. **Report generation:** Creates Markdown, HTML, JSON, and ZIP outputs.

## Important Notes and Limitations

- The detector is primarily designed for North Indian-style Kundli charts and similar geometric chart layouts.
- Detection performance depends on the PDF's drawing structure, image quality, and chart layout.
- Scanned PDFs may require raster-based detection.
- Planetary and sign recognition depends on text being readable and available in the PDF's text layer.
- House placement is based on geometric position mapping and should be verified against the original chart.
- The program does not guarantee perfect chart detection or interpretation across arbitrary PDF documents.
- When a document contains vector drawings, the current implementation selects vector detection for the document rather than independently selecting a method for every page.

## Future Improvements

- Improve detection across different Kundli chart layouts.
- Improve chart-title and section-heading association.
- Improve detection of charts embedded in scanned PDFs.
- Add more robust text recognition for image-only documents.
- Improve chart boundary detection and reduce false positives.
- Add automated validation against manually labelled chart datasets.

##
