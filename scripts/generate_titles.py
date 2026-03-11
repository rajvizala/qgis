"""
generate_tiles.py
=================
PyQGIS script for generating synthetic training chips (image + mask pairs)
from a satellite raster basemap and a vector layer of OSM building polygons.

Author  : Senior ML Engineer — GeoAI Division
Purpose : Automated geospatial data preparation for deep learning pipelines
          targeting building detection models (Mask R-CNN, YOLOv8, U-Net, etc.)

Usage (QGIS Python Console):
    exec(open('/path/to/scripts/generate_tiles.py').read())

Usage (OSGeo4W Shell / headless):
    python-qgis.bat /path/to/scripts/generate_tiles.py

Dependencies:
    - QGIS >= 3.28 (PyQGIS API)
    - PyYAML  (`pip install pyyaml`  in OSGeo4W shell)
    - Pillow  (`pip install Pillow`  in OSGeo4W shell)

Pipeline Overview:
    1. Load config/dataset_params.yaml for all tunable parameters.
    2. Open the satellite GeoTIFF and the building mask GeoTIFF.
    3. Compute a regular chip grid over the AOI bounding box.
    4. For each grid cell (row, col):
           a. Clip the satellite GeoTIFF  → images/chip_RRRRR_CCCCC.png
           b. Clip the binary mask GeoTIFF → masks/chip_RRRRR_CCCCC.png
           c. Optionally skip all-background chips below a coverage threshold.
    5. Log final statistics for dataset QA.
"""

# =============================================================================
# SECTION 1 — IMPORTS
# Standard library, QGIS API (qgis.core), and lightweight helpers.
# NOTE: All QGIS imports are available when this script runs inside QGIS or
#       via python-qgis.bat. They are NOT available in a plain Python env.
# =============================================================================

import os
import sys
import math
import logging
import pathlib
from datetime import datetime

# --- PyYAML: for reading our version-controlled config file ---
try:
    import yaml
except ImportError:
    raise ImportError(
        "PyYAML is required. Install it with: pip install pyyaml\n"
        "In OSGeo4W shell: pip install pyyaml"
    )

# --- Pillow: for PNG chip validation and optional augmentation previews ---
try:
    from PIL import Image
    import numpy as np
except ImportError:
    raise ImportError(
        "Pillow and NumPy are required. Install with: pip install Pillow numpy"
    )

# --- QGIS Core API ---
# These are injected into the Python environment by QGIS itself.
from qgis.core import (
    QgsApplication,          # QGIS application singleton (headless init)
    QgsProject,              # Access the current QGIS project
    QgsRasterLayer,          # Raster layer (GeoTIFF, XYZ tiles, etc.)
    QgsVectorLayer,          # Vector layer (GeoJSON, Shapefile, etc.)
    QgsCoordinateReferenceSystem,  # CRS definitions (EPSG codes)
    QgsCoordinateTransform,  # On-the-fly CRS reprojection
    QgsRectangle,            # Axis-aligned bounding box
    QgsPointXY,              # 2D spatial point
    QgsMapSettings,          # Map rendering configuration
    QgsMapRendererParallelJob,  # Parallel raster rendering job
    QgsLayoutExporter,       # Layout export utilities
    QgsRasterBlock,          # Raw raster pixel block access
    QgsProcessingFeedback,   # Progress and log feedback for Processing
    QgsMessageLog,           # QGIS application log
)
from qgis import processing  # QGIS Processing Framework (gdal:cliprasterbyextent, etc.)

# =============================================================================
# SECTION 2 — LOGGING SETUP
# We use Python's standard logging module so output is captured both in the
# QGIS Python console and in any log file redirect.
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger("generate_tiles")


# =============================================================================
# SECTION 3 — CONFIGURATION LOADER
# Reads dataset_params.yaml so that ALL tunable knobs are version-controlled
# and NOT hard-coded in this script. This is critical ML best practice:
# every experiment should be fully reproducible from its config snapshot.
# =============================================================================

def load_config(config_path: str) -> dict:
    """
    Load and validate the dataset configuration YAML file.

    Parameters
    ----------
    config_path : str
        Absolute or relative path to dataset_params.yaml.

    Returns
    -------
    dict
        Parsed configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist at the given path.
    KeyError
        If a required top-level key is missing from the YAML.
    """
    config_path = pathlib.Path(config_path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # --- Validate required top-level keys ---
    required_keys = ["tile", "dataset", "paths", "aoi", "rasterization", "classes"]
    for key in required_keys:
        if key not in cfg:
            raise KeyError(
                f"Missing required config key: '{key}'. "
                f"Check config/dataset_params.yaml."
            )

    log.info(f"Config loaded from: {config_path}")
    return cfg


# =============================================================================
# SECTION 4 — RASTER UTILITY FUNCTIONS
# Low-level helpers for interacting with GeoTIFF rasters via QGIS and GDAL.
# =============================================================================

def open_raster_layer(path: str, layer_name: str) -> QgsRasterLayer:
    """
    Open a GeoTIFF as a QGIS raster layer and validate it loaded correctly.

    In ML data prep context, a corrupt or misaligned raster will silently
    produce blank chips — catching this early saves hours of debugging.

    Parameters
    ----------
    path : str
        Path to the GeoTIFF file.
    layer_name : str
        Display name for the layer in QGIS.

    Returns
    -------
    QgsRasterLayer
        Validated, opened raster layer.
    """
    layer = QgsRasterLayer(path, layer_name)
    if not layer.isValid():
        raise ValueError(
            f"Failed to open raster layer '{layer_name}' at path: {path}\n"
            "Verify the file exists, is a valid GeoTIFF, and has a defined CRS."
        )
    log.info(
        f"Raster layer '{layer_name}' opened | "
        f"CRS: {layer.crs().authid()} | "
        f"Extent: {layer.extent()} | "
        f"Size: {layer.width()}×{layer.height()} px"
    )
    return layer


def open_vector_layer(path: str, layer_name: str) -> QgsVectorLayer:
    """
    Open a vector file (GeoJSON / Shapefile) as a QGIS vector layer.

    Building polygon quality is critical for mask accuracy.  This function
    logs the feature count so you immediately know if the OSM fetch was empty.

    Parameters
    ----------
    path : str
        Path to the GeoJSON or Shapefile.
    layer_name : str
        Display name in QGIS.

    Returns
    -------
    QgsVectorLayer
        Validated, opened vector layer.
    """
    layer = QgsVectorLayer(path, layer_name, "ogr")
    if not layer.isValid():
        raise ValueError(
            f"Failed to open vector layer '{layer_name}' at path: {path}"
        )
    log.info(
        f"Vector layer '{layer_name}' opened | "
        f"CRS: {layer.crs().authid()} | "
        f"Features: {layer.featureCount()}"
    )
    return layer


def reproject_layer_if_needed(layer, target_crs_str: str):
    """
    Reproject a vector layer to the target CRS using QGIS Processing if
    the source CRS does not already match.

    Why this matters for ML chips:
        The satellite basemap (usually EPSG:3857) and OSM buildings
        (exported in EPSG:4326) must share the same CRS before we can
        compute pixel-accurate chip extents. Mismatched CRS = misaligned
        image/mask pairs = corrupted training labels.

    Parameters
    ----------
    layer : QgsVectorLayer
        Input vector layer.
    target_crs_str : str
        Target CRS as EPSG string, e.g. "EPSG:3857".

    Returns
    -------
    QgsVectorLayer
        Reprojected vector layer (may be same object if CRS already matched).
    """
    target_crs = QgsCoordinateReferenceSystem(target_crs_str)

    if layer.crs() == target_crs:
        log.info(f"Layer '{layer.name()}' already in {target_crs_str}. No reprojection needed.")
        return layer

    log.info(f"Reprojecting '{layer.name()}' from {layer.crs().authid()} → {target_crs_str}...")
    result = processing.run(
        "native:reprojectlayer",
        {
            "INPUT": layer,
            "TARGET_CRS": target_crs,
            "OUTPUT": "memory:",  # Keep in memory; write to disk only if chip count is large
        },
        feedback=QgsProcessingFeedback()
    )
    reprojected = result["OUTPUT"]
    log.info(f"Reprojection complete. Features retained: {reprojected.featureCount()}")
    return reprojected


def rasterize_building_layer(
    vector_layer,
    reference_raster_path: str,
    output_mask_path: str,
    burn_value: int = 1,
) -> str:
    """
    Burn building polygon footprints onto a raster grid to create a binary
    segmentation mask GeoTIFF aligned pixel-for-pixel with the satellite image.

    ML Context:
        The output mask is the ground-truth `y` label for each chip.
        Pixel value == 1  →  building (positive class)
        Pixel value == 0  →  background (negative class)

        The mask MUST be spatially identical (same extent, resolution, and CRS)
        as the image raster. We achieve this by using the satellite GeoTIFF as
        the spatial reference for the rasterization operation.

    Parameters
    ----------
    vector_layer : QgsVectorLayer
        Building polygon layer to rasterize.
    reference_raster_path : str
        Path to the satellite GeoTIFF. Used to define output grid parameters.
    output_mask_path : str
        Where to save the output binary mask GeoTIFF.
    burn_value : int
        Pixel value to burn for building pixels. Default 1.

    Returns
    -------
    str
        Path to the generated mask GeoTIFF.
    """
    log.info("Rasterizing building footprints → binary mask GeoTIFF...")

    # Open reference raster to extract grid parameters
    ref_raster = open_raster_layer(reference_raster_path, "reference")
    extent = ref_raster.extent()
    x_res = ref_raster.rasterUnitsPerPixelX()
    y_res = ref_raster.rasterUnitsPerPixelY()

    # GDAL rasterize via QGIS Processing
    # This is more robust than calling gdal directly: it handles CRS
    # reprojection, polygon filling, and edge cases automatically.
    result = processing.run(
        "gdal:rasterize",
        {
            "INPUT": vector_layer,
            "FIELD": "",                  # No attribute field; use burn value
            "BURN": burn_value,           # Value for building pixels
            "USE_Z": False,
            "UNITS": 1,                   # 1 = georeferenced units (meters or degrees)
            "WIDTH": x_res,
            "HEIGHT": y_res,
            "EXTENT": f"{extent.xMinimum()},{extent.xMaximum()},"
                       f"{extent.yMinimum()},{extent.yMaximum()}"
                       f" [{ref_raster.crs().authid()}]",
            "NODATA": 0,
            "OPTIONS": "COMPRESS=DEFLATE",
            "DATA_TYPE": 0,               # Byte (uint8) — sufficient for binary mask
            "INIT": None,
            "INVERT": False,
            "EXTRA": "",
            "OUTPUT": output_mask_path,
        },
        feedback=QgsProcessingFeedback()
    )

    mask_path = result["OUTPUT"]
    log.info(f"Binary mask GeoTIFF saved to: {mask_path}")
    return mask_path


# =============================================================================
# SECTION 5 — CHIP GRID COMPUTATION
# Computes the regular grid of tile extents that tiles the AOI.
# This is the spatial arithmetic core of the chipping pipeline.
# =============================================================================

def compute_chip_grid(
    aoi_extent: QgsRectangle,
    tile_size_px: int,
    pixel_size_m: float,
    overlap: float = 0.0,
) -> list:
    """
    Compute a list of (row, col, QgsRectangle) tuples covering the AOI in a
    regular grid of equal-size chips.

    ML Context:
        - `tile_size_px` defines the spatial context each model inference sees.
          256×256 is standard for Mask R-CNN and most YOLO variants.
        - `overlap` introduces redundancy between adjacent chips, which is
          important during inference (sliding-window detection) to avoid
          missing buildings that straddle chip boundaries.
        - During TRAINING, overlap = 0.0 is typically preferred to avoid
          duplicate building annotations in the training set.
        - During INFERENCE, overlap = 0.2–0.5 with NMS post-processing
          is recommended.

    Parameters
    ----------
    aoi_extent : QgsRectangle
        Full extent of the Area of Interest in the raster's CRS.
    tile_size_px : int
        Chip size in pixels (e.g., 256).
    pixel_size_m : float
        Ground sampling distance in map units per pixel (e.g., 0.5 for 50 cm/px).
    overlap : float
        Fractional overlap between adjacent chips. 0.0 = no overlap, 0.2 = 20%.
        Must be in range [0.0, 0.9].

    Returns
    -------
    list of (row: int, col: int, extent: QgsRectangle)
        Ordered list of chip grid cells.
    """
    if not (0.0 <= overlap < 1.0):
        raise ValueError(f"Overlap must be in [0.0, 1.0). Got: {overlap}")

    # Chip footprint in map units (e.g., meters in EPSG:3857)
    chip_size_m = tile_size_px * pixel_size_m
    step_size_m = chip_size_m * (1.0 - overlap)  # Step < chip size when overlap > 0

    aoi_width  = aoi_extent.xMaximum() - aoi_extent.xMinimum()
    aoi_height = aoi_extent.yMaximum() - aoi_extent.yMinimum()

    # Number of rows/cols needed to cover the full AOI (ceiling division ensures
    # full coverage even if the AOI isn't evenly divisible by the step size)
    n_cols = math.ceil(aoi_width  / step_size_m)
    n_rows = math.ceil(aoi_height / step_size_m)

    log.info(
        f"Chip grid: {n_rows} rows × {n_cols} cols = {n_rows * n_cols} total chips | "
        f"Chip size: {chip_size_m:.1f} m ({tile_size_px} px) | "
        f"Step size: {step_size_m:.1f} m | "
        f"Overlap: {overlap * 100:.0f}%"
    )

    chips = []
    for row in range(n_rows):
        for col in range(n_cols):
            x_min = aoi_extent.xMinimum() + col * step_size_m
            y_max = aoi_extent.yMaximum() - row * step_size_m
            x_max = x_min + chip_size_m
            y_min = y_max - chip_size_m

            # Clamp chip extents to AOI boundary to avoid requesting pixels
            # outside the raster extent (which would produce black borders)
            x_min = max(x_min, aoi_extent.xMinimum())
            y_min = max(y_min, aoi_extent.yMinimum())
            x_max = min(x_max, aoi_extent.xMaximum())
            y_max = min(y_max, aoi_extent.yMaximum())

            chips.append((row, col, QgsRectangle(x_min, y_min, x_max, y_max)))

    return chips


# =============================================================================
# SECTION 6 — CHIP EXPORT FUNCTIONS
# Extract and save individual image/mask chip pairs using GDAL clip operations.
# =============================================================================

def clip_raster_to_chip(
    raster_path: str,
    chip_extent: QgsRectangle,
    output_path: str,
    crs_str: str,
    target_size_px: int,
) -> bool:
    """
    Clip a raster GeoTIFF to the given chip extent and save as a PNG.

    Uses `gdal:cliprasterbyextent` which is robust, handles CRS, and preserves
    NoData values. The output is then resized to exactly `target_size_px` ×
    `target_size_px` using Pillow to guarantee uniform tensor dimensions
    for batched deep learning training.

    Parameters
    ----------
    raster_path : str
        Source GeoTIFF (satellite image or binary mask).
    chip_extent : QgsRectangle
        Spatial extent of the chip in the raster's CRS.
    output_path : str
        Where to save the PNG chip.
    crs_str : str
        CRS of the raster as EPSG string.
    target_size_px : int
        Output PNG dimensions (both width and height).

    Returns
    -------
    bool
        True if clip succeeded and file was written, False otherwise.
    """
    extent_str = (
        f"{chip_extent.xMinimum()},{chip_extent.xMaximum()},"
        f"{chip_extent.yMinimum()},{chip_extent.yMaximum()}"
        f" [{crs_str}]"
    )

    # --- Step 1: Clip to extent → temporary GeoTIFF ---
    tmp_path = output_path.replace(".png", "_tmp.tif")
    try:
        result = processing.run(
            "gdal:cliprasterbyextent",
            {
                "INPUT": raster_path,
                "PROJWIN": extent_str,
                "NODATA": None,
                "OPTIONS": "",
                "DATA_TYPE": 0,      # 0 = Use input layer data type
                "EXTRA": "",
                "OUTPUT": tmp_path,
            },
            feedback=QgsProcessingFeedback()
        )
    except Exception as e:
        log.warning(f"Clip failed for extent {extent_str}: {e}")
        return False

    # --- Step 2: Open the clipped GeoTIFF with Pillow and resize to target ---
    # This guarantees all chips are exactly target_size_px × target_size_px,
    # which is required for batched tensor operations in PyTorch/TensorFlow.
    try:
        with Image.open(tmp_path) as img:
            # LANCZOS for image chips (high-quality downsampling)
            # NEAREST for mask chips (preserves hard class boundaries — critical!)
            resample_method = (
                Image.NEAREST if "mask" in output_path else Image.LANCZOS
            )
            img_resized = img.resize(
                (target_size_px, target_size_px),
                resample=resample_method
            )
            img_resized.save(output_path, format="PNG", optimize=True)

        # Clean up temp GeoTIFF
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

        return True

    except Exception as e:
        log.warning(f"Failed to save chip PNG at {output_path}: {e}")
        return False


def compute_building_coverage(mask_path: str) -> float:
    """
    Compute the fraction of pixels in a mask chip that belong to the building class.

    ML Context:
        This is used to filter out 'empty' chips — tiles that contain no
        building pixels at all. Training on excessive background-only chips
        causes class imbalance, which degrades precision on the positive class.

        Typical threshold: discard chips where coverage < 0.005 (0.5%)
        This value is controlled by `dataset.min_positive_ratio` in the config.

    Parameters
    ----------
    mask_path : str
        Path to the binary mask PNG (0 = background, 255 = building).

    Returns
    -------
    float
        Fraction of pixels in [0.0, 1.0] that are positive (building) class.
    """
    try:
        with Image.open(mask_path) as mask_img:
            mask_arr = np.array(mask_img)
        # Handle both binary (0/1) and uint8 (0/255) masks
        positive_pixels = np.count_nonzero(mask_arr)
        total_pixels = mask_arr.size
        return positive_pixels / total_pixels if total_pixels > 0 else 0.0
    except Exception:
        return 0.0


# =============================================================================
# SECTION 7 — MAIN PIPELINE ORCHESTRATOR
# Ties all components together and handles progress reporting + statistics.
# =============================================================================

def run_chipping_pipeline(config_path: str) -> dict:
    """
    Main entry point for the geospatial chipping pipeline.

    Executes the full sequence:
        1. Load config
        2. Open raster and vector layers
        3. Reproject vector to raster CRS
        4. Rasterize buildings → binary mask GeoTIFF
        5. Compute chip grid
        6. Export image + mask chip pairs
        7. Return dataset statistics

    Parameters
    ----------
    config_path : str
        Path to dataset_params.yaml.

    Returns
    -------
    dict
        Statistics about the generated dataset (chip counts, coverage, etc.)
    """
    # ------------------------------------------------------------------
    # 7.1 — Load Configuration
    # ------------------------------------------------------------------
    cfg = load_config(config_path)

    tile_size   = cfg["tile"]["size"]
    overlap     = cfg["tile"]["overlap"]
    pixel_size  = cfg["tile"]["pixel_size_m"]
    min_ratio   = cfg["dataset"]["min_positive_ratio"]
    crs_str     = cfg["dataset"]["crs"]

    image_path  = cfg["paths"]["satellite_geotiff"]
    vector_path = cfg["paths"]["osm_buildings"]
    output_dir  = pathlib.Path(cfg["paths"]["output_dir"])

    # Create output directories for images and masks
    img_out_dir  = output_dir / "images"
    mask_out_dir = output_dir / "masks"
    img_out_dir.mkdir(parents=True, exist_ok=True)
    mask_out_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("  Geospatial Chipping Pipeline — Starting")
    log.info(f"  Run timestamp : {datetime.now().isoformat()}")
    log.info(f"  Tile size     : {tile_size}×{tile_size} px")
    log.info(f"  Overlap       : {overlap * 100:.0f}%")
    log.info(f"  Pixel size    : {pixel_size} m/px")
    log.info(f"  CRS           : {crs_str}")
    log.info(f"  Output dir    : {output_dir}")
    log.info("=" * 60)

    # ------------------------------------------------------------------
    # 7.2 — Open Layers
    # ------------------------------------------------------------------
    satellite_layer = open_raster_layer(image_path, "satellite_basemap")
    building_layer  = open_vector_layer(vector_path, "osm_buildings")

    # ------------------------------------------------------------------
    # 7.3 — Reproject Building Layer to Match Raster CRS
    # ------------------------------------------------------------------
    building_layer = reproject_layer_if_needed(building_layer, crs_str)

    # ------------------------------------------------------------------
    # 7.4 — Rasterize Buildings to Binary Mask GeoTIFF
    # ------------------------------------------------------------------
    mask_geotiff_path = str(output_dir / "building_mask_full.tif")
    rasterize_building_layer(
        vector_layer=building_layer,
        reference_raster_path=image_path,
        output_mask_path=mask_geotiff_path,
        burn_value=1,
    )

    # ------------------------------------------------------------------
    # 7.5 — Compute Chip Grid
    # ------------------------------------------------------------------
    # Use the AOI extent from config (not the full raster extent) so you
    # can focus chipping on a specific sub-region without re-exporting
    # the satellite GeoTIFF.
    aoi = cfg["aoi"]
    if all(aoi.get(k) is not None for k in ["xmin", "ymin", "xmax", "ymax"]):
        aoi_extent = QgsRectangle(
            aoi["xmin"], aoi["ymin"], aoi["xmax"], aoi["ymax"]
        )
        log.info(f"Using AOI extent from config: {aoi_extent}")
    else:
        # Fall back to full raster extent if AOI not specified
        aoi_extent = satellite_layer.extent()
        log.info(f"No AOI in config. Using full raster extent: {aoi_extent}")

    chips = compute_chip_grid(
        aoi_extent=aoi_extent,
        tile_size_px=tile_size,
        pixel_size_m=pixel_size,
        overlap=overlap,
    )

    # ------------------------------------------------------------------
    # 7.6 — Export Chip Pairs
    # ------------------------------------------------------------------
    stats = {
        "total_chips_computed"  : len(chips),
        "chips_exported"        : 0,
        "chips_skipped_empty"   : 0,
        "chips_failed"          : 0,
        "building_coverage_sum" : 0.0,
    }

    log.info(f"Starting chip export for {len(chips)} grid cells...")

    for i, (row, col, extent) in enumerate(chips):

        chip_name = f"chip_{row:05d}_{col:05d}.png"
        img_chip_path  = str(img_out_dir  / chip_name)
        mask_chip_path = str(mask_out_dir / chip_name)

        # --- Export satellite image chip ---
        img_ok = clip_raster_to_chip(
            raster_path=image_path,
            chip_extent=extent,
            output_path=img_chip_path,
            crs_str=crs_str,
            target_size_px=tile_size,
        )

        if not img_ok:
            stats["chips_failed"] += 1
            log.debug(f"  [{i+1}/{len(chips)}] FAILED  → {chip_name}")
            continue

        # --- Export binary mask chip ---
        mask_ok = clip_raster_to_chip(
            raster_path=mask_geotiff_path,
            chip_extent=extent,
            output_path=mask_chip_path,
            crs_str=crs_str,
            target_size_px=tile_size,
        )

        if not mask_ok:
            stats["chips_failed"] += 1
            continue

        # --- Filter empty chips based on building pixel coverage ---
        # This is a key data quality step for ML:
        #   Too many empty chips → model learns to predict "nothing"
        #   Too few  empty chips → model never learns the background
        # Recommended: discard chips with < 0.5% building pixels,
        # then optionally sub-sample remaining background chips to
        # achieve your target positive:negative ratio (e.g., 1:3).
        coverage = compute_building_coverage(mask_chip_path)

        if coverage < min_ratio:
            # Remove both chips if we're discarding this sample
            os.remove(img_chip_path)
            os.remove(mask_chip_path)
            stats["chips_skipped_empty"] += 1
            log.debug(
                f"  [{i+1}/{len(chips)}] SKIPPED (coverage={coverage:.4f}) → {chip_name}"
            )
            continue

        # --- Chip accepted ---
        stats["chips_exported"] += 1
        stats["building_coverage_sum"] += coverage

        # Progress log every 100 chips
        if (i + 1) % 100 == 0 or (i + 1) == len(chips):
            pct = (i + 1) / len(chips) * 100
            log.info(
                f"  Progress: {i+1}/{len(chips)} ({pct:.1f}%) | "
                f"Exported: {stats['chips_exported']} | "
                f"Skipped: {stats['chips_skipped_empty']} | "
                f"Failed: {stats['chips_failed']}"
            )

    # ------------------------------------------------------------------
    # 7.7 — Final Statistics
    # ------------------------------------------------------------------
    n_exported = stats["chips_exported"]
    avg_coverage = (
        stats["building_coverage_sum"] / n_exported
        if n_exported > 0 else 0.0
    )

    log.info("")
    log.info("=" * 60)
    log.info("  PIPELINE COMPLETE — Dataset Statistics")
    log.info("=" * 60)
    log.info(f"  Total grid cells computed : {stats['total_chips_computed']}")
    log.info(f"  Chips exported            : {n_exported}")
    log.info(f"  Chips skipped (empty)     : {stats['chips_skipped_empty']}")
    log.info(f"  Chips failed (I/O error)  : {stats['chips_failed']}")
    log.info(f"  Avg. building coverage    : {avg_coverage * 100:.2f}%")
    log.info(f"  Output directory          : {output_dir}")
    log.info("=" * 60)

    stats["avg_building_coverage"] = avg_coverage
    return stats


# =============================================================================
# SECTION 8 — HEADLESS QGIS INIT (for running outside QGIS Desktop)
# When running via `python-qgis.bat`, we must initialize a minimal QGIS
# application before calling any qgis.core functions.
# =============================================================================

def init_qgis_headless():
    """
    Initialize a minimal QGIS application for headless (non-GUI) use.

    This is ONLY needed when running the script from the command line
    (OSGeo4W Shell, CI/CD pipeline, Docker container, etc.).
    When running inside QGIS Desktop (Python Console or Script Editor),
    the application is already initialized — do NOT call this function.

    Returns
    -------
    QgsApplication
        The initialized application instance. Keep a reference to prevent
        garbage collection (which would crash the QGIS C++ backend).
    """
    # Supply_prefix_path = path to QGIS installation directory
    # This is typically handled automatically by python-qgis.bat on Windows
    # or by sourcing /usr/share/qgis/python on Linux.
    app = QgsApplication([], False)  # False = headless, no GUI
    app.initQgis()
    log.info(f"QGIS initialized (headless) | Version: {QgsApplication.QGIS_VERSION}")
    return app


# =============================================================================
# SECTION 9 — ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    """
    Script entry point — detects headless vs. in-QGIS context automatically.

    Headless (command line):
        python generate_tiles.py [config_path]

    In QGIS Python Console:
        exec(open('generate_tiles.py').read())
        # Config path defaults to ../config/dataset_params.yaml
    """
    import sys

    # Determine if we are running inside QGIS Desktop already
    # (QgsApplication.instance() returns a valid object inside QGIS)
    running_in_qgis = QgsApplication.instance() is not None

    qgs_app = None
    if not running_in_qgis:
        qgs_app = init_qgis_headless()

    # --- Resolve config path from CLI arg or default ---
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    else:
        # Default: config lives one level up from scripts/
        script_dir  = pathlib.Path(__file__).parent if "__file__" in dir() else pathlib.Path(".")
        config_file = str(script_dir.parent / "config" / "dataset_params.yaml")

    log.info(f"Using config: {config_file}")

    try:
        stats = run_chipping_pipeline(config_path=config_file)
        exit_code = 0
    except Exception as e:
        log.error(f"Pipeline failed with error: {e}", exc_info=True)
        exit_code = 1
    finally:
        # Always clean up the QGIS application to avoid C++ memory leaks
        if qgs_app is not None:
            qgs_app.exitQgis()

    sys.exit(exit_code)
