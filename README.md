# osmviz
Visualizing OSM data in 3D, using vulkano.
Use `just prep` to initialize everything once you have the data sources.

### Data sources
#### Elevation data
[EuroDEM dataset](https://www.mapsforeurope.org/datasets/euro-dem)
Extract tif file, supply path as argument to `just prep`.

#### OSM map data
[Geofabrik](https://download.geofabrik.de/europe/germany.html)
Provide .osm.pbf as second argument to `just prep`.

#### Color data
<a xmlns:dct="http://purl.org/dc/terms/" href="https://s2maps.eu" property="dct:title">Sentinel-2 cloudless - https://s2maps.eu</a> by <a xmlns:cc="http://creativecommons.org/ns#" href="https://eox.at" property="cc:attributionName" rel="cc:attributionURL">EOX IT Services GmbH</a> (Contains modified Copernicus Sentinel data 2016 &amp; 2017) released under <a rel="license" href="https://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>.

Public API, no setup necessary.
