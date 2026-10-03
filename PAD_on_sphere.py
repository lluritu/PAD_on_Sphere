"""ctypes interface for PAD on a sphere (requires wrapper ABI version 3).

Cutoffs and returned distances are great-circle
distances. Invalid inputs raise Python exceptions. Rebuild the shared library
after updating the C++ wrapper.
"""
import ctypes as ct
import operator
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from PAD_postprocess import aggregate_transportplan_at_gridpoints, get_latlon_df

Earth_radius = 6371.0 * 1000.0
libc = ct.CDLL(str(Path(__file__).resolve().parent / "PAD_on_sphere_Cxx_shared_library.so"))
try:
    libc.PAD_wrapper_abi_version.argtypes = []
    libc.PAD_wrapper_abi_version.restype = ct.c_int
except AttributeError as exc:
    raise ImportError("Rebuild PAD_on_sphere_Cxx_shared_library.so: wrapper ABI 3 required.") from exc
if libc.PAD_wrapper_abi_version() != 3:
    raise ImportError("Incompatible PAD shared library: wrapper ABI 3 required.")

ND_POINTER_1D = np.ctypeslib.ndpointer(
    dtype=np.float64, ndim=1, flags=("C_CONTIGUOUS", "ALIGNED")
)
libc.free_mem_double_array.argtypes = [ct.POINTER(ct.c_double)]
libc.free_mem_double_array.restype = None
libc.PAD_last_error.argtypes = []
libc.PAD_last_error.restype = ct.c_char_p
libc.calculate_PAD_results_assume_same_grid_ctypes.argtypes = [
    ND_POINTER_1D, ND_POINTER_1D, ND_POINTER_1D, ND_POINTER_1D,
    ct.c_size_t, ct.POINTER(ct.c_size_t), ct.c_double, ct.c_int64,
]
libc.calculate_PAD_results_assume_same_grid_ctypes.restype = ct.POINTER(ct.c_double)
libc.calculate_PAD_results_assume_different_grid_ctypes.argtypes = [
    ND_POINTER_1D, ND_POINTER_1D, ND_POINTER_1D, ct.c_size_t,
    ND_POINTER_1D, ND_POINTER_1D, ND_POINTER_1D, ct.c_size_t,
    ct.POINTER(ct.c_size_t), ct.c_double, ct.c_int64,
]
libc.calculate_PAD_results_assume_different_grid_ctypes.restype = ct.POINTER(ct.c_double)


def check_input_array(f, name):
    """Validate a nonempty, finite, one-dimensional real NumPy array.

    Returns True on success; raises TypeError or ValueError on invalid input.
    Masked arrays are deliberately rejected.
    """
    if not isinstance(f, np.ndarray) or isinstance(f, np.ma.MaskedArray):
        raise TypeError(f"{name} must be an unmasked NumPy array.")
    if f.ndim != 1 or f.size == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional array.")
    if not np.issubdtype(f.dtype, np.number) or np.iscomplexobj(f):
        raise TypeError(f"{name} must contain real numeric values.")
    if not np.all(np.isfinite(f)):
        raise ValueError(f"{name} must contain only finite values.")
    return True


def _array(f, name):
    check_input_array(f, name)
    converted = np.require(f, dtype=np.float64, requirements=["C", "A"])
    if not np.all(np.isfinite(converted)):
        raise ValueError(f"{name} must be representable as finite float64 values.")
    return converted


def _seed(random_seed):
    if random_seed is None:
        return -1
    if isinstance(random_seed, (bool, np.bool_)):
        raise TypeError("random_seed must be an integer, not a boolean.")
    seed = operator.index(random_seed)
    if seed < -1 or seed > 0xFFFFFFFF:
        raise ValueError("random_seed must be None, -1, or an unsigned 32-bit integer.")
    return seed


def _cutoff(value):
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise ValueError("Great-circle distance cutoff must be finite and nonnegative.")
    return value


def calculate_attributions_from_numpy(
    values1, values2, lat1, lon1, lat2=None, lon2=None,
    same_grid=True, distance_cutoff=100 * 1000 * 1000, random_seed=None,
):
    """Calculate PAD attributions from one-dimensional arrays.

    Latitude/longitude are degrees. Amounts must be finite and nonnegative,
    with at least one positive amount per field. They are attributed as supplied:
    pass volumes for volume-weighted PAD; no area conversion occurs here.
    distance_cutoff is a great-circle distance in metres (zero is allowed).
    random_seed=None or -1 chooses and prints a random seed; an explicit uint32
    seed gives reproducibility with identical inputs and code/library versions.

    Returns [attributions, remaining1, remaining2]. Attribution shape is (N, 4),
    including (0, 4) when no matches satisfy the cutoff. Columns are great-circle
    distance in metres, amount, original index1, original index2. Indices in this
    homogeneous float64 array are exactly representable for the supported sizes.
    same_grid=True means identical coordinates in identical order.
    """
    if not isinstance(same_grid, (bool, np.bool_)):
        raise TypeError("same_grid must be a boolean.")
    seed = _seed(random_seed)
    cutoff = _cutoff(distance_cutoff)
    if same_grid:
        if lat2 is not None or lon2 is not None:
            raise ValueError("Do not supply lat2/lon2 when same_grid=True.")
    elif lat2 is None or lon2 is None:
        raise ValueError("lat2 and lon2 are required when same_grid=False.")

    lat1, lon1 = _array(lat1, "lat1"), _array(lon1, "lon1")
    values1, values2 = _array(values1, "values1"), _array(values2, "values2")
    if lat1.shape != lon1.shape or lat1.shape != values1.shape:
        raise ValueError("lat1, lon1 and values1 must have identical shapes.")
    if same_grid:
        if values1.shape != values2.shape:
            raise ValueError("Same-grid fields must have identical shapes.")
        lat2, lon2 = lat1, lon1
    else:
        lat2, lon2 = _array(lat2, "lat2"), _array(lon2, "lon2")
        if lat2.shape != lon2.shape or lat2.shape != values2.shape:
            raise ValueError("lat2, lon2 and values2 must have identical shapes.")
    for lat in (lat1, lat2):
        if np.any(np.abs(lat) > 90):
            raise ValueError("Latitudes must be between -90 and 90 degrees.")
    for values in (values1, values2):
        if np.any(values < 0) or not np.any(values > 0):
            raise ValueError("Each field must be nonnegative with at least one positive amount.")
        if values.size > np.iinfo(np.int32).max:
            raise ValueError("Too many grid points for the C++ tree.")

    count = ct.c_size_t()
    if same_grid:
        result = libc.calculate_PAD_results_assume_same_grid_ctypes(
            lat1, lon1, values1, values2, values1.size,
            ct.byref(count), cutoff, seed,
        )
    else:
        result = libc.calculate_PAD_results_assume_different_grid_ctypes(
            lat1, lon1, values1, values1.size,
            lat2, lon2, values2, values2.size, ct.byref(count), cutoff, seed,
        )
    if not result:
        message = libc.PAD_last_error()
        raise RuntimeError(message.decode("utf-8", errors="replace") if message else "PAD calculation failed.")
    try:
        n = count.value
        if n > values1.size + values2.size:
            raise RuntimeError("Invalid attribution count returned by PAD.")
        total = 4 * n + values1.size + values2.size
        # One owned copy, without creating millions of Python float objects.
        packed = np.ctypeslib.as_array(result, shape=(total,)).copy()
        attributions = packed[:4 * n].reshape(n, 4)
        remaining1 = packed[4 * n:4 * n + values1.size]
        remaining2 = packed[4 * n + values1.size:]
        return [attributions, remaining1, remaining2]
    finally:
        libc.free_mem_double_array(result)


def calculate_PAD_on_sphere_from_attributions(PAD_attributions):
    """Return the amount-weighted mean distance; reject undefined/invalid input."""
    rows = np.asarray(PAD_attributions, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] < 2 or rows.shape[0] == 0:
        raise ValueError("PAD requires a nonempty attribution array with at least two columns.")
    distances, weights = rows[:, 0], rows[:, 1]
    if (not np.all(np.isfinite(distances)) or not np.all(np.isfinite(weights))
            or np.any(distances < 0) or np.any(weights < 0) or not np.any(weights > 0)):
        raise ValueError("PAD requires finite nonnegative distances and positive total weight.")
    # Scale before products/sums to avoid overflow for large finite amounts.
    weights = weights / weights.max()
    scale = distances.max()
    if scale == 0:
        return 0.0
    fraction = np.sum((distances / scale) * weights) / np.sum(weights)
    return float(min(fraction, 1.0) * scale)


def _dataarray(array, name, coordinates=False):
    if not isinstance(array, xr.DataArray) or array.dims != ("gridpoint",):
        raise ValueError(f"{name} must be an xarray DataArray with only the gridpoint dimension.")
    if coordinates:
        for coordinate in ("lat", "lon"):
            if coordinate not in array.coords or array[coordinate].dims != ("gridpoint",):
                raise ValueError(f"{name} needs one-dimensional lat/lon coordinates.")
    return array


def calculate_attributions_from_xarrays(
    fcst, obs, area, area2=None, same_grid=True, cutoff=3000,
    gridded_output=True, random_seed=None,
):
    """Calculate PAD for precipitation (mm) and cell areas (km^2).

    cutoff is great-circle distance in km.
    random_seed follows calculate_attributions_from_numpy.
    distance_m is stored as an integer (truncated to whole metres) to save output
    storage; 1 m resolution is enough. Index columns are integers.
    Returns, depending on the options:
    - same_grid=True, gridded_output=True: (transport_dataframe, gridded_ds), with
      volume transported, distance transported and residual error (mm) at each grid point.
    - same_grid=True, gridded_output=False: (transport_dataframe, residual_dataframe)
      with the non-zero residual errors in mm.
    - same_grid=False: (transport_dataframe, residual_fcst_ds, residual_obs_ds) with
      the non-attributed precipitation in mm on each grid.
    For the gridded outputs, positive distances represent a water export at origin
    (fcst > obs) and negative distances a water import at destination (fcst < obs).
    Similarly, positive residual errors represent overforecasting (fcst > obs) and vice versa.
    """
    if not isinstance(same_grid, (bool, np.bool_)):
        raise TypeError("same_grid must be a boolean.")
    _dataarray(fcst, "fcst", True)
    _dataarray(obs, "obs", True)
    _dataarray(area, "area")
    if same_grid:
        if area2 is not None:
            raise ValueError("Do not supply area2 when same_grid=True.")
        area2 = area
    _dataarray(area2, "area2")

    # Exact joins reject different/reordered indexes instead of silently dropping
    # points during xarray arithmetic. Positional sizes must also match.
    fcst, area = xr.align(fcst, area, join="exact", copy=False)
    obs, area2 = xr.align(obs, area2, join="exact", copy=False)
    if fcst.sizes != area.sizes or obs.sizes != area2.sizes:
        raise ValueError("Each area array must match its field.")
    if same_grid:
        fcst, obs = xr.align(fcst, obs, join="exact", copy=False)
        if fcst.sizes != obs.sizes:
            raise ValueError("Same-grid fields must have identical lengths.")
        for coordinate in ("lat", "lon"):
            if not np.array_equal(fcst[coordinate].values, obs[coordinate].values):
                raise ValueError("Same-grid fields must have identical lat/lon coordinates in the same order.")

    amounts = []
    for field, cell_area, name in ((fcst, area, "fcst"), (obs, area2, "obs")):
        precipitation = _array(field.values, name)
        areas = _array(cell_area.values, name + " area")
        if np.any(precipitation < 0) or np.any(areas <= 0):
            raise ValueError("Precipitation must be nonnegative and cell areas strictly positive.")
        with np.errstate(over="raise", invalid="raise"):
            amounts.append(precipitation * areas * 1000.0)

    distance_cutoff = _cutoff(_cutoff(cutoff) * 1000.0)
    attributions, remaining1, remaining2 = calculate_attributions_from_numpy(
        amounts[0], amounts[1], fcst.lat.values, fcst.lon.values,
        lat2=None if same_grid else obs.lat.values,
        lon2=None if same_grid else obs.lon.values,
        same_grid=same_grid, distance_cutoff=distance_cutoff, random_seed=random_seed,
    )
    transport = pd.DataFrame(
        attributions, columns=["distance_m", "volume_m3", "gridpoint_fcst", "gridpoint_obs"]
    )
    # Store distance and gridpoint columns as integers. For distance_m this truncates
    # to whole metres, which saves output storage; 1 m resolution is enough here.
    for column in ("distance_m", "gridpoint_fcst", "gridpoint_obs"):
        transport[column] = transport[column].astype(np.int64)

    def residual_ds(remaining, field, cell_area):
        # Convert volume back to height: mm = m^3 / (km^2 * 1e6) * 1e3
        coords = {"lat": field.lat.astype(np.float64), "lon": field.lon.astype(np.float64)}
        if "gridpoint" in field.coords:
            coords["gridpoint"] = field.gridpoint
        return xr.Dataset(
            {"error": ("gridpoint", remaining / cell_area.values / 1000.0)},
            coords=coords,
        )

    if not same_grid:
        return (
            transport,
            residual_ds(remaining1, fcst, area),
            residual_ds(remaining2, obs, area2),
        )

    residual = residual_ds(remaining1 - remaining2, fcst, area)
    if gridded_output:
        # Transport-plan gridpoints are positional indices, so combine positionally.
        transport_ds = (
            aggregate_transportplan_at_gridpoints(transport, get_latlon_df(residual))
            .to_xarray()
            .set_coords(("lat", "lon"))
        )
        gridded_ds = xr.merge((
            transport_ds.drop_vars("gridpoint"),
            residual.drop_vars("gridpoint", errors="ignore"),
        ))
        return transport, gridded_ds

    residual = residual.to_dataframe()[["lat", "lon", "error"]]
    return transport, residual.loc[residual["error"] != 0]
