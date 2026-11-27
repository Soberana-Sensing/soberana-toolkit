# Configuration and processing flow

These diagrams describe how a consuming project uses the reusable toolkit package. Config,
AOI, and product paths are relative to the consuming project's root; they are not bundled in
this repository.

```mermaid
flowchart TD
    subgraph Config["Consuming project configuration"]
        CubeConfig["config/datasets/s2_l2a_monthly.yaml<br/>source, dates, AOI path, processing, indices, output bands"]
        GridConfig["config/grids/utm28_30m.yaml<br/>CRS, resolution, alignment"]
        AOI["data/aoi/diattang.geojson<br/>area of interest"]
    end

    subgraph Processing["Toolkit package"]
        Script["soberana_toolkit.openeo.s2_l2a_processing"]
        GridModule["soberana_toolkit.grid<br/>load_grid()"]
    end

    subgraph ProjectOutputs["Consuming project outputs"]
        Graph["data/derived/s2_l2a_monthly/process_graph.json"]
        OpenEO["openEO backend"]
        Split["Split downloaded NetCDF<br/>into one file per selected band"]
        Outputs["data/derived/s2_l2a_monthly/<br/>NDVI.nc, SI1.nc, ..."]
        Metadata["data/derived/s2_l2a_monthly/run_metadata.json"]
    end

    subgraph Verification["Toolkit verification"]
        Verify["soberana_toolkit.verify"]
        Findings["CRS, pixel size, axes,<br/>and time-axis findings"]
    end

    CubeConfig --> Script
    AOI --> Script
    GridConfig --> GridModule
    GridModule --> Script
    Script --> Graph
    Script -->|"--run"| OpenEO
    OpenEO --> Split
    Split --> Outputs
    Script --> Metadata
    CubeConfig --> Verify
    GridModule --> Verify
    Outputs --> Verify
    Verify --> Findings
```

The dataset config and grid stay in the consuming project and are inputs to both the processing
module and verifier. The dataset config also tells the verifier where to find products. With
NetCDF output, the module downloads the combined openEO result temporarily and writes one `.nc`
file for each selected output band. `process_graph.json` is written when the graph is built; the
NetCDF products require `--run`.

## Generated openEO process graph

This is a readable summary of the actual 112-node graph generated in the consuming project's
`data/derived/s2_l2a_monthly/process_graph.json`. Repeated per-band arithmetic is grouped by
output; the openEO node IDs are retained so the chart can be traced back to that JSON.

```mermaid
flowchart TD
    Load["loadcollection1<br/>load_collection: SENTINEL2_L2A<br/>2016-01-01 to 2026-01-01 (end exclusive)<br/>AOI + cloud cover <= 15%"]
    Load --> Spectral["filterbands1<br/>select B02, B03, B04, B05,<br/>B08, B8A, B11, B12"]
    Load --> SCLMaskInput["filterbands2<br/>select SCL"]
    Spectral --> Scale["apply1<br/>scale reflectance by 0.0001"]
    SCLMaskInput --> Dilate["toscldilationmask1<br/>dilated SCL cloud/shadow mask"]
    Scale --> Mask["mask1<br/>apply SCL mask"]
    Dilate --> Mask
    Mask --> Reproject["resamplespatial1<br/>EPSG:32628, 30 m,<br/>bilinear, upper-left"]

    Reproject --> Formula["Index formula branches<br/>array_element / arithmetic / reduce_dimension nodes<br/>CRSI and normalized-ratio branches also mask zero denominators"]
    Formula --> IndexBands["add_dimension1-15<br/>SI1, SI2, SI3, CRSI, BSI, NDVI, SAVI,<br/>MSAVI2, WDVI, NDWI, MNDWI, NDMI,<br/>NDRE, SRCI, NDCI"]
    IndexBands --> IndexMonths["aggregate_temporal_period1-15<br/>monthly median"]

    Load --> SCLClasses["filterbands3<br/>SCL class indicators: water, vegetation, soil"]
    SCLClasses --> ClassMasks["apply4-6 -> mask12-14<br/>mask class indicators with dilated SCL mask"]
    Dilate --> ClassMasks
    ClassMasks --> ClassGrid["resamplespatial2-4<br/>30 m, nearest"]
    ClassGrid --> ClassNames["renamelabels1-3<br/>scl_water, scl_vegetation, scl_soil"]
    ClassNames --> ClassMonths["aggregate_temporal_period16-18<br/>monthly mean (class fractions)"]

    Reproject --> Valid["reducedimension26<br/>select first band for counting"]
    Valid --> ValidMonths["aggregatetemporalperiod19<br/>monthly count of valid observations"]
    ValidMonths --> ValidBand["adddimension16<br/>n_valid"]

    IndexMonths --> Merge["mergecubes1-18<br/>combine all index, SCL-class,<br/>and n_valid bands"]
    ClassMonths --> Merge
    ValidBand --> Merge
    Merge --> Save["saveresult1<br/>save_result: NetCDF"]
    Save --> Download["Script downloads the combined NetCDF"]
    Download --> Split["Local post-processing in<br/>s2_l2a_processing.py"]
    Split --> Files["One .nc per selected band<br/>e.g. NDVI.nc, scl_water.nc, n_valid.nc"]
```

The final three boxes are script-side handling after the openEO graph finishes; the splitter
is not an openEO process node. The chart groups the graph's many arithmetic nodes rather than
drawing each individual addition, multiplication, and division.

## `run_metadata.json` provenance

The metadata file is a summary of the inputs and settings used for the graph, not another
processing graph. This chart reflects the S2 example's metadata at
`data/derived/s2_l2a_monthly/run_metadata.json`:

```mermaid
flowchart LR
    Config["config/datasets/s2_l2a_monthly.yaml<br/>dataset name, source, dates, AOI,<br/>temporal settings, requested bands"]
    Grid["config/grids/utm28_30m.yaml<br/>grid name, CRS, resolution, policy"]
    Hash["SHA-256 of grid definition"]
    Meta["run_metadata.json"]
    Config --> Meta
    Grid --> Meta
    Grid --> Hash --> Meta
    Meta --> Identity["SENTINEL2_L2A<br/>Copernicus Data Space via openEO"]
    Meta --> Period["2016-01-01 to 2025-12-31<br/>monthly median"]
    Meta --> Spatial["EPSG:32628<br/>30 m, native_or_grid"]
    Meta --> Selection["requested bands + n_valid<br/>NetCDF"]
    Meta --> Provenance["AOI path + config path<br/>generation timestamp"]
```

The example metadata snapshot has no `job_id` field.
