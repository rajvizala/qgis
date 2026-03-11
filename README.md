# 🛰️ Automated Geospatial Chipping Pipeline for Building Detection

<p align="center">
  <img src="https://img.shields.io/badge/QGIS-3.x-green?style=for-the-badge&logo=qgis&logoColor=white" alt="QGIS"/>
  <img src="https://img.shields.io/badge/PyQGIS-Python%203-blue?style=for-the-badge&logo=python&logoColor=white" alt="PyQGIS"/>
  <img src="https://img.shields.io/badge/Computer%20Vision-Mask%20R--CNN%20%7C%20YOLO-orange?style=for-the-badge&logo=opencv&logoColor=white" alt="Computer Vision"/>
  <img src="https://img.shields.io/badge/GeoAI-Satellite%20Imagery-purple?style=for-the-badge&logo=satellite&logoColor=white" alt="GeoAI"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" alt="License"/>
</p>

---

## 📖 Overview

This repository provides a **fully automated, end-to-end geospatial chipping pipeline** for generating labeled training datasets from satellite imagery — purpose-built for training deep learning object detection and instance segmentation models such as **Mask R-CNN**, **YOLOv8**, and **Detectron2**.

### What This Pipeline Does

The pipeline executes three core operations in sequence:

```
OSM Building Footprints  →  Satellite Basemap Overlay  →  256×256 PNG Chips
        (Vector)                   (Raster / XYZ Tiles)          (Image + Mask Pairs)
```

1. **Scrape OSM Building Footprints** using [QuickOSM](https://plugins.qgis.org/plugins/QuickOSM/) inside QGIS. The plugin queries the [Overpass API](https://overpass-api.de/) to pull structured `building=*` polygon features for any geographic bounding box on Earth — no manual digitizing required.

2. **Overlay on Satellite Basemaps** by loading a high-resolution XYZ tile service (e.g., Google Satellite, ESRI World Imagery, or Bing Maps) as a raster backdrop within QGIS. The building polygons are rendered on top, aligning vector labels spatially to the pixel grid of the imagery.

3. **Slice into 256×256 Training Chips** using the included PyQGIS script. The script tiles the full AOI (Area of Interest) extent into a regular grid of non-overlapping (or configurable-overlap) chips, exports each chip as:
   - A **raw satellite image** PNG (the model input `X`)
   - A **binary or multi-class raster mask** PNG (the ground-truth label `y`)

   These image-mask pairs are then ready for ingestion into any PyTorch / TensorFlow data loader.

### Why This Approach?

| Advantage | Detail |
|---|---|
| **Zero-cost labels** | OSM has >500M mapped buildings globally. No manual annotation needed. |
| **Any geography** | Works for any city or region with OSM coverage. |
| **Reproducible** | All parameters are version-controlled in `config/dataset_params.yaml`. |
| **Framework-agnostic** | Outputs standard PNG + mask pairs compatible with COCO, Pascal VOC, or custom loaders. |
| **Scalable** | Batch process multiple AOIs by looping over bounding boxes. |

---

## 🗂️ Repository Structure

```
geospatial-chipping-pipeline/
│
├── 📄 README.md                    ← You are here
├── 📄 .gitignore                   ← Excludes QGIS files, shapefiles, large imagery
│
├── 📁 config/
│   └── dataset_params.yaml         ← All tunable parameters for the pipeline
│
├── 📁 scripts/
│   └── generate_tiles.py           ← Core PyQGIS chipping script
│
├── 📁 data/                        ← (git-ignored) Local data lives here
│   ├── raw/
│   │   ├── osm_buildings.geojson   ← Fetched OSM building footprints
│   │   └── basemap_cache/          ← XYZ tile cache from QGIS
│   ├── chips/
│   │   ├── images/                 ← Output image chips (256×256 PNG)
│   │   └── masks/                  ← Output mask chips (256×256 PNG)
│   └── annotations/
│       └── coco_annotations.json   ← (Optional) COCO-format label export
│
├── 📁 notebooks/                   ← (Optional) EDA and visualization
│   └── explore_chips.ipynb
│
├── 📁 qgis_project/                ← (git-ignored) .qgz project files
│   └── building_detection.qgz
│
└── 📁 docs/
    ├── pipeline_diagram.png        ← Architecture visual
    └── sample_chips.png            ← Example image/mask pairs
```

---

## 🏗️ Pipeline Architecture

The pipeline is composed of three distinct stages. Each stage can be run independently, making it easy to resume from any step.

### Stage 1 — Vector Fetching (QuickOSM)

**Goal:** Obtain building polygon geometries for your target AOI.

**Method:** QuickOSM sends a structured [Overpass QL](https://wiki.openstreetmap.org/wiki/Overpass_API/Overpass_QL) query to the OSM API:

```
[out:json][timeout:60];
(
  way["building"]({{bbox}});
  relation["building"]["type"="multipolygon"]({{bbox}});
);
out body;
>;
out skel qt;
```

**Outputs:**
- A temporary or saved QGIS vector layer (`.geojson` or `.shp`) of building polygons
- Attributes include: `osm_id`, `building` type, `name`, `addr:*` tags (where available)

**Tips:**
- For large cities, split your AOI into multiple bounding boxes to avoid Overpass timeout limits.
- Use `building=yes` to filter only confirmed buildings and exclude `building=construction`.

---

### Stage 2 — Rasterization & Basemap Setup

**Goal:** Load a pixel-accurate satellite basemap aligned to the same CRS as your OSM vectors.

**Method:**
1. In QGIS, add an XYZ tile layer via `Layer → Add Layer → Add XYZ Layer`:
   ```
   # Google Satellite
   https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}

   # ESRI World Imagery
   https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}
   ```
2. Set the **project CRS** to `EPSG:3857` (Web Mercator) to match XYZ tile coordinates. For metric accuracy, reproject to a local UTM zone.
3. Use `Raster → Miscellaneous → Merge` or GDAL's `gdal_translate` to export the visible basemap canvas to a GeoTIFF at your target resolution (e.g., 0.5 m/pixel at zoom level 18).
4. Run `Raster → Conversion → Rasterize (Vector to Raster)` on the OSM building layer with:
   - **Burn value:** `1` (building class)
   - **Background value:** `0` (background class)
   - **Resolution:** Identical to the basemap GeoTIFF

---

### Stage 3 — XYZ Tile Generation & Chip Export

**Goal:** Slice the full AOI raster + mask into a regular grid of 256×256 chips.

**Method:** The `scripts/generate_tiles.py` PyQGIS script:

1. Reads `config/dataset_params.yaml` for all parameters
2. Computes a regular grid of tile extents over the AOI bounding box
3. For each grid cell:
   - Clips the satellite GeoTIFF to the tile extent → exports `images/chip_RRRRR_CCCCC.png`
   - Clips the binary mask GeoTIFF to the tile extent → exports `masks/chip_RRRRR_CCCCC.png`
   - Optionally skips chips where `building_pixel_coverage < min_positive_ratio` to avoid all-background chips
4. Logs chip statistics (total chips, positive chip ratio, class pixel distribution)

**Chip Naming Convention:**
```
chip_{row:05d}_{col:05d}.png
   │              │
   │              └── Column index (0-padded to 5 digits)
   └── Row index (0-padded to 5 digits)
```

---

## ⚡ Quickstart

### Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| QGIS Desktop | ≥ 3.28 | With PyQGIS / OSGeo4W shell |
| Python | ≥ 3.9 | Bundled with QGIS |
| QuickOSM Plugin | Latest | Install via QGIS Plugin Manager |
| PyYAML | Any | `pip install pyyaml` in OSGeo4W |
| Pillow (PIL) | ≥ 9.0 | `pip install Pillow` in OSGeo4W |

### Step-by-Step

#### 1. Clone the Repository
```bash
git clone https://github.com/your-username/geospatial-chipping-pipeline.git
cd geospatial-chipping-pipeline
```

#### 2. Fetch OSM Buildings in QGIS

Open QGIS, navigate to `Vector → QuickOSM → QuickOSM`, and run:
- **Key:** `building`
- **Value:** *(leave blank for all buildings)*
- **Extent:** Draw your AOI bounding box or enter coordinates manually

Save the resulting layer to `data/raw/osm_buildings.geojson`.

#### 3. Export the Basemap as GeoTIFF

In QGIS, load your XYZ tile layer and go to `Project → Import/Export → Export Map to Image`. Save to `data/raw/basemap.tif`. For programmatic export, use:

```bash
# Example: Export via GDAL (run in OSGeo4W Shell)
gdal_translate -of GTiff -projwin <xmin> <ymax> <xmax> <ymin> \
  "WMTS:https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}" \
  data/raw/basemap.tif
```

#### 4. Configure the Dataset Parameters
```bash
# Edit config/dataset_params.yaml to match your AOI and requirements
nano config/dataset_params.yaml
```

Key parameters to set:
```yaml
aoi:
  xmin: -0.1276   # Your AOI bounding box
  ymin: 51.5074
  xmax: -0.1176
  ymax: 51.5174
```

#### 5. Run the Chipping Script

**Option A — From the QGIS Python Console:**
```python
# In QGIS → Plugins → Python Console
exec(open('/path/to/scripts/generate_tiles.py').read())
```

**Option B — From the OSGeo4W Shell (headless):**
```bash
cd C:\OSGeo4W
o4w_env.bat
python-qgis.bat /path/to/geospatial-chipping-pipeline/scripts/generate_tiles.py
```

**Option C — As a QGIS Processing Script:**

Copy `generate_tiles.py` into QGIS's Processing Toolbox scripts folder and run it via the GUI with form-based parameter input.

#### 6. Verify Outputs

```bash
data/chips/
├── images/
│   ├── chip_00000_00000.png   # 256×256 RGB satellite chip
│   ├── chip_00000_00001.png
│   └── ...
└── masks/
    ├── chip_00000_00000.png   # 256×256 binary mask (0=bg, 255=building)
    ├── chip_00000_00001.png
    └── ...
```

#### 7. Load Into Your ML Training Pipeline

```python
# Example: PyTorch Dataset class
from torch.utils.data import Dataset
from PIL import Image
import os

class BuildingChipDataset(Dataset):
    def __init__(self, chips_dir, transform=None):
        self.image_dir = os.path.join(chips_dir, "images")
        self.mask_dir  = os.path.join(chips_dir, "masks")
        self.filenames = sorted(os.listdir(self.image_dir))
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname  = self.filenames[idx]
        image  = Image.open(os.path.join(self.image_dir, fname)).convert("RGB")
        mask   = Image.open(os.path.join(self.mask_dir,  fname)).convert("L")
        if self.transform:
            image, mask = self.transform(image, mask)
        return image, mask
```

---

## 📊 Expected Dataset Statistics

For a typical urban AOI (e.g., 5 km²):

| Metric | Typical Value |
|---|---|
| Total chips generated | 3,000 – 15,000 |
| Positive chips (≥1 building) | 40% – 75% |
| Chips discarded (background-only) | 25% – 60% |
| Avg. buildings per positive chip | 3 – 12 |
| Pixel-level class balance | ~15–35% building, ~65–85% background |

---

## 🧠 Model Compatibility

Chips generated by this pipeline are compatible with:

| Model | Framework | Format Needed |
|---|---|---|
| **YOLOv8** | Ultralytics | YOLO `.txt` annotation files |
| **Mask R-CNN** | Detectron2 / MMDetection | COCO JSON |
| **U-Net / SegFormer** | PyTorch / HuggingFace | Image + mask PNG pairs ✅ (direct) |
| **SAM (Segment Anything)** | Meta AI | Image PNG ✅ (direct) |

---

## 📄 License

This project is licensed under the **MIT License**. OSM data is licensed under the [Open Database License (ODbL)](https://opendatacommons.org/licenses/odbl/).

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you'd like to change. Please ensure your contributions include docstrings and update the config schema if new parameters are added.


```
