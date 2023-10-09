## Deep learning balloon anomaly detection
Suite of deep learning solutions for balloon pressure data denoising and anomaly detection. This is work in progress by Quentin Brissaud at NORSAR.

## Data availability
- Balloon data from the 2019 Ridgecrest campaign can be found here: https://doi.org/10.6084/m9.figshare.14374067
- Balloon data from the the 15 January 2022 Hunga Eruption, Tonga can be found here: https://data.ipsl.fr/catalog/strateole2/eng/catalog.search#/search?from=1%26to%3D30
- Balloon data from Large Earthquakes can be found here: https://doi.org/10.5281/zenodo.6344454
- Balloon data from the MINI-BOOSTER campaign can be found here: https://snd.gu.se/en/catalogue/dataset/2021-257-1
- Balloon data from a Buried Chemical Explosion can be found here: https://ds.iris.edu/mda/21-021/

## Dataset construction and synthetics
- .h5 datasets can be built using the notebook "generate_h5_files.ipynb"
- Extraction of Rayleigh waveforms to build synthetics can be done using "get_rayleigh_waves.ipynb" 

## Models
- Deep learning anomaly detection: "anomaly_detection.ipynb"

## Extra
- The Fastmap algorithm for clustering was tested in routine "fastmap_test.ipynb"