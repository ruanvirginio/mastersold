# pip install geopandas contextily shapely matplotlib pyproj

import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
from shapely.geometry import Point

substations = [
    {"name": "T1", "lat": -7.118452779363018,  "lon": -34.8671852904291},
    {"name": "T2", "lat": -7.167148023736754,  "lon": -34.87723241712038},
    {"name": "T3", "lat": -7.1153782009446545, "lon": -34.83401142939276},
]

gdf = gpd.GeoDataFrame(
    substations,
    geometry=[Point(p["lon"], p["lat"]) for p in substations],
    crs="EPSG:4326"  # lat/lon
).to_crs(epsg=3857)  # Web Mercator


minx, miny, maxx, maxy = gdf.total_bounds
pad = 1500 

dx = (maxx - minx) + 2 * pad
dy = (maxy - miny) + 2 * pad
side = max(dx, dy)

cx = (minx + maxx) / 2
cy = (miny + maxy) / 2

half = side / 2
sq_minx, sq_maxx = cx - half, cx + half
sq_miny, sq_maxy = cy - half, cy + half

fig, ax = plt.subplots(figsize=(6, 6))

ax.set_xlim(sq_minx, sq_maxx)
ax.set_ylim(sq_miny, sq_maxy)

ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)

gdf.plot(ax=ax, markersize=120, color="red", edgecolor="white", linewidth=1.5, zorder=3)

for x, y, name in zip(gdf.geometry.x, gdf.geometry.y, gdf["name"]):
    ax.text(x, y, f" {name}", fontsize=10, fontweight="bold",
            va="center", ha="left", color="black", zorder=4)

ax.set_axis_off()
plt.tight_layout()

plt.savefig("substations_joao_pessoa.svg", format="svg", dpi=300, bbox_inches="tight")

print("Salvo: substations_joao_pessoa.svg")
