# IFCB datasets

## What is IFCB data?

The **Imaging FlowCytobot (IFCB)** is an instrument that continuously samples seawater and takes images of individual particles as they flow past a camera. Each detected particle is saved as a small image called a **Region of Interest (ROI)**. An ROI usually represents a single phytoplankton cell, zooplankton, detritus particle, or other object in the water.

For each sampling period (called a **bin**), IFCB records thousands of ROIs along with metadata describing when, where, and how much water was analyzed. Automated classifiers are often applied to the ROI images, producing a table of **class scores** that estimate how likely each ROI belongs to different species or functional groups.

## How do we compute abundance (objects per mL)?

Abundance is calculated by combining **object counts** with the **volume of water analyzed**:

1. **Count objects**  
   Each row in a `*_class_scores.csv` file corresponds to **one ROI (one detected object)**.  
   To estimate species-level counts, each ROI is assigned to the class with the highest classification score (“winner”), sometimes requiring the score to exceed a confidence threshold.

2. **Get analyzed volume**  
   For each bin, IFCB metadata includes volume analyzed (`ml_analyzed` in our dataset), the total volume of seawater (in milliliters) that passed the detector during that sample.

3. **Compute abundance**  

   $$
   \text{objects per mL} = \frac{\text{number of ROIs (or ROIs of a given species)}}{\text{ml\_analyzed}}
   $$

This converts image-based counts into a physically meaningful concentration. Abundance per mL.

## FCB files per bin

For a given bin `DYYYYMMDDTHHMMSS_IFCBXXX`, we have:

* *_class_scores.csv
    - One row per ROI (detected object)
    - Columns = classifier scores (probabilities)

* *_features.csv
    - One row per ROI (object)
    - Contains morphological / size features, e.g.: area, equivalent spherical diameter (ESD), major/minor axis, perimeter, biovolume

This notebook is just using the class_scores.csv file.

## How to get started

Start with one of these notebooks:
- `notebooks/ifcb.ipynb`

## Provenance and metadata

Data prep
- `notebooks/Get_WHOI_IFCB.ipynb`

Formal metadata for each file is provided via STAC:
- See `collection.json`
- See individual item JSONs in `items/`
