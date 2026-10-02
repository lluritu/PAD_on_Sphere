# ------------------------------
# Import the needed libraries
# ------------------------------

# Import the PAD Python library
# The PAD python library file (PY_PAD_library.py) as well as the PAD C++ shared library file (PAD_Cxx_shared_library.so)
# need to be in the same folder as the script using them
from PAD_on_sphere import *

# Import netcdf library
import netCDF4
from netCDF4 import Dataset

# ------------------------------------------------------------------------------------------------------------------------
# Read sample netcdf fields - both fields are defined in a regular 0.25 deg lat-lon grid (i.e. same grid)
# ------------------------------------------------------------------------------------------------------------------------

nc_file_id = Dataset("sample_data/example_field_A.nc", "r")
lon1 = nc_file_id.variables["lon"][:].data
lat1 = nc_file_id.variables["lat"][:].data
field1 = nc_file_id.variables["precipitation"][:].data
nc_file_id.close()

nc_file_id = Dataset("sample_data/example_field_B.nc", "r")
field2 = nc_file_id.variables["precipitation"][:].data
nc_file_id.close()

# ------------------------------------------------------------------------------------------------------------------------
# Convert fields to one-dimensional arrays of lats, lons, and precipitation amounts for all grid points
# ------------------------------------------------------------------------------------------------------------------------

values1 = np.reshape(field1, (-1))
values2 = np.reshape(field2, (-1))
lons = np.tile(lon1, field1.shape[0])
lats = np.repeat(lat1, field1.shape[1])

# ------------------------------------------------------------------------------------------------------------------------
# Convert precipitation height to precipitation volume in m^3 (assuming the precipitation height values are in mm)
# ------------------------------------------------------------------------------------------------------------------------

dlat = 0.25  # resoltion of lat lon grid of the input fields
grid_point_area_size = (
    np.deg2rad(dlat)
    * Earth_radius
    * np.deg2rad(dlat)
    * Earth_radius
    * np.cos(np.deg2rad(lats))
)
values_prec_volume1 = values1 / 1000.0 * grid_point_area_size
values_prec_volume2 = values2 / 1000.0 * grid_point_area_size

# ------------------------------------------------------------------------------------------------------------------------
# Calculate PAD results using a 3000 km great-circle cutoff (specified in metres)
# ------------------------------------------------------------------------------------------------------------------------

# if the fields are on different grids, supply also lat2 and lon2 arrays, and set same_grid=False.
[PAD_attributions, non_attributed_values1, non_attributed_values2] = (
    calculate_attributions_from_numpy(
        values_prec_volume1,
        values_prec_volume2,
        lats,
        lons,
        same_grid=True,
        distance_cutoff=3000 * 1000,
    )
)

# The PAD_attributions array contains the data on all attributions.
# The dimension of the array is (number_of_all_attributions,4) with
# -  PAD_attributions(:,0) being the attributed distances
# -  PAD_attributions(:,1) being the attributed amounts
# -  PAD_attributions(:,2) being the indexes of the associated grid point in the first field
# -  PAD_attributions(:,3) being the indexes of the associated grid point in the second field

# The non_attributed_values1 and non_attributed_values2 arrays contain the data about precipitation that was not attributed.
# Both arrays are one-dimensional with the dimension equal to the values_prec_volume1 and values_prec_volume2 arrays, respectively

# ------------------------------------------------------------------------------------------------------------------------
# Calculate the PAD distance value - the result is in meters
# ------------------------------------------------------------------------------------------------------------------------

PAD_distance_value = calculate_PAD_on_sphere_from_attributions(PAD_attributions)
print("PAD distance value: " + str(np.round(PAD_distance_value / 1000, 1)) + " km")

# ------------------------------------------------------------------------------------------------------------------------
# Calculate the portion of unattributed precipitation
# ------------------------------------------------------------------------------------------------------------------------

print(
    "Portion of unattributed precipitation in field 1: "
    + str(
        np.round(100 * np.sum(non_attributed_values1) / np.sum(values_prec_volume1), 2)
    )
    + " %"
)
print(
    "Portion of unattributed precipitation in field 2: "
    + str(
        np.round(100 * np.sum(non_attributed_values2) / np.sum(values_prec_volume2), 2)
    )
    + " %"
)

# ------------------------------------------------------------------------------------------------------------------------
# Draw the attribution PDF
# ------------------------------------------------------------------------------------------------------------------------

# Import matplotlib library
import matplotlib
import matplotlib.pyplot as plt

# ---  PDF
hist = np.histogram(
    PAD_attributions[:, 0],
    bins=100,
    range=(0, np.max(PAD_attributions[:, 0])),
    density=True,
    weights=PAD_attributions[:, 1],
)
PAD_PDF = np.asarray([hist[1][:-1], hist[0][:]]).transpose(1, 0)
fig, ax = plt.subplots()
# Convert density per metre to density per kilometre to match the x-axis.
plt.fill_between(PAD_PDF[:, 0] / 1000, PAD_PDF[:, 1] * 1000, 0, linestyle="-")
plt.axvline(
    PAD_distance_value / 1000,
    color="navy",
    label="PAD_distance = " + str(int(np.round(PAD_distance_value / 1000, 0))) + " km",
    linestyle="--",
)
plt.ylim(bottom=0, top=1.2 * np.max(PAD_PDF[10:, 1]) * 1000)
plt.xlim(left=0)
plt.xlabel("PAD attribution distance [km]")
plt.ylabel("Probability density [km⁻¹]")
leg = plt.legend(loc="upper right")
plt.show()
plt.close()

# ------------------------------------------------------------------------------------------------------------------------
# Visualize PAD attributions in an Orthographic projection
# ------------------------------------------------------------------------------------------------------------------------

# Import additional libraries needed for visualization
import cartopy
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# number of attribution lines shown in the figure
number_of_shown_attributions = 2000
# cumulative distribution
cumulative = np.cumsum(PAD_attributions[:, 1] / np.sum(PAD_attributions[:, 1]))
# randomly select attributions - the probability of selection is affected by the attribution value
rand = np.random.rand(number_of_shown_attributions)
ind = np.searchsorted(cumulative, rand)
PAD_attributions_selected = PAD_attributions[ind]

projX = cartopy.crs.Orthographic(central_longitude=180, central_latitude=0, globe=None)
projX._threshold = projX._threshold / 50.0  # set finer threshold

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1, projection=projX)
ax.set_global()

plt.title("Visualization of PAD attributions")

fontsize = 10

cmap_b = matplotlib.colors.LinearSegmentedColormap.from_list(
    "rb_cmap", ["white", (0.7, 0.7, 1.0), "blue"], 512
)
cmap_r = matplotlib.colors.LinearSegmentedColormap.from_list(
    "rb_cmap", ["white", (1.0, 0.7, 0.7), "red"], 512
)

levels = [0.1, 1, 10, 100]
norm_b = matplotlib.colors.BoundaryNorm(
    boundaries=levels, ncolors=cmap_b.N, extend="both"
)
norm_r = matplotlib.colors.BoundaryNorm(
    boundaries=levels, ncolors=cmap_r.N, extend="both"
)

# conver to rgb values using colormaps
f1x = cmap_b(norm_b(field1))
f2x = cmap_r(norm_r(field2))

# get the combined colors using the multiply effect
fx = f1x * f2x

img_extent = (0, 360, -90, 90)
img = plt.imshow(
    fx,
    transform=cartopy.crs.PlateCarree(),
    interpolation="nearest",
    origin="lower",
    extent=img_extent,
)

axins_b = inset_axes(
    ax,
    width="30%",
    height="3%",
    bbox_to_anchor=(0.0, -0.08, 1, 1),
    loc="lower left",
    bbox_transform=ax.transAxes,
)
cb_b = matplotlib.colorbar.ColorbarBase(
    axins_b,
    orientation="horizontal",
    cmap=cmap_b,
    norm=norm_b,
    extend="both",
    label="This is a label",
    ticks=levels,
)
cb_b.set_label("Precipitation (mm/6h)", fontsize=fontsize, labelpad=3)
cb_b.ax.tick_params(labelsize=fontsize, length=2)

axins_r = inset_axes(
    ax,
    width="30%",
    height="3%",
    bbox_to_anchor=(0.0, -0.08, 1, 1),
    loc="lower right",
    bbox_transform=ax.transAxes,
)
cb_r = matplotlib.colorbar.ColorbarBase(
    axins_r,
    orientation="horizontal",
    cmap=cmap_r,
    norm=norm_r,
    extend="both",
    label="This is a label",
    ticks=levels,
)
cb_r.set_label("precipitation (mm/6h)", fontsize=fontsize, labelpad=3)
cb_r.ax.tick_params(labelsize=fontsize, length=2)

lons_of_attributions = np.asarray(
    [
        lons[PAD_attributions_selected[:, 2].astype(int)],
        lons[PAD_attributions_selected[:, 3].astype(int)],
    ]
)
lats_of_attributions = np.asarray(
    [
        lats[PAD_attributions_selected[:, 2].astype(int)],
        lats[PAD_attributions_selected[:, 3].astype(int)],
    ]
)
ax.plot(
    lons_of_attributions,
    lats_of_attributions,
    "-ok",
    transform=cartopy.crs.Geodetic(),
    alpha=0.3,
    linewidth=0.75,
    markersize=2,
    mfc="black",
    mec="black",
)

ax.coastlines(resolution="110m", color="grey", linestyle="-", alpha=1)
# ax.gridlines(draw_labels=False, alpha=0.5)

plt.show()
plt.close()

# ------------------------------------------------------------------------------------------------------------------------
# Visualize non-attributed precipitation in an Orthographic projection
# ------------------------------------------------------------------------------------------------------------------------

# Divide the non_attributed_values with area size and multiply by 1000 to convert from precipitation volume to precipitation height in mm
non_attributed_values_height1 = non_attributed_values1 / grid_point_area_size * 1000
non_attributed_values_height2 = non_attributed_values2 / grid_point_area_size * 1000

fn1 = np.reshape(non_attributed_values_height1, field1.shape)
fn2 = np.reshape(non_attributed_values_height2, field2.shape)

# treba na novo definitat projekcijo in ._threshold parameter, da lepo gladko narise povezovalne crte za attribucijo (drugace so prevec odsekoma linearne)
projX = cartopy.crs.Orthographic(central_longitude=0, central_latitude=0, globe=None)
projX._threshold = projX._threshold / 50.0  # set finer threshold

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1, projection=projX)
ax.set_global()

plt.title("Visualization of non-attributed values")

fontsize = 10

cmap_b = matplotlib.colors.LinearSegmentedColormap.from_list(
    "rb_cmap", ["white", (0.7, 0.7, 1.0), "blue"], 512
)
cmap_r = matplotlib.colors.LinearSegmentedColormap.from_list(
    "rb_cmap", ["white", (1.0, 0.7, 0.7), "red"], 512
)

levels = [0.1, 1, 10, 100]
norm_b = matplotlib.colors.BoundaryNorm(
    boundaries=levels, ncolors=cmap_b.N, extend="both"
)
norm_r = matplotlib.colors.BoundaryNorm(
    boundaries=levels, ncolors=cmap_r.N, extend="both"
)

# conver to rgb values using colormaps
f1x = cmap_b(norm_b(fn1))
f2x = cmap_r(norm_r(fn2))

# get the combined colors using the multiply effect
fx = f1x * f2x

img_extent = (0, 360, -90, 90)
img = plt.imshow(
    fx,
    transform=cartopy.crs.PlateCarree(),
    interpolation="nearest",
    origin="lower",
    extent=img_extent,
)

axins_b = inset_axes(
    ax,
    width="30%",
    height="3%",
    bbox_to_anchor=(0.0, -0.08, 1, 1),
    loc="lower left",
    bbox_transform=ax.transAxes,
)
cb_b = matplotlib.colorbar.ColorbarBase(
    axins_b,
    orientation="horizontal",
    cmap=cmap_b,
    norm=norm_b,
    extend="both",
    label="This is a label",
    ticks=levels,
)
cb_b.set_label("Precipitation (mm/6h)", fontsize=fontsize, labelpad=3)
cb_b.ax.tick_params(labelsize=fontsize, length=2)

axins_r = inset_axes(
    ax,
    width="30%",
    height="3%",
    bbox_to_anchor=(0.0, -0.08, 1, 1),
    loc="lower right",
    bbox_transform=ax.transAxes,
)
cb_r = matplotlib.colorbar.ColorbarBase(
    axins_r,
    orientation="horizontal",
    cmap=cmap_r,
    norm=norm_r,
    extend="both",
    label="This is a label",
    ticks=levels,
)
cb_r.set_label("precipitation (mm/6h)", fontsize=fontsize, labelpad=3)
cb_r.ax.tick_params(labelsize=fontsize, length=2)

ax.coastlines(resolution="110m", color="grey", linestyle="-", alpha=1)
# ax.gridlines(draw_labels=False, alpha=0.5)

plt.show()
plt.close()
