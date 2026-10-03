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

from PAD_postprocess import (
    aggregate_transportplan_at_gridpoints,
    aggregate_transportplan_at_gridpoints_unequal_grids,
    get_latlon_df,
)

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
    """Check that an input numpy array has the right dimension and contents.

    Parameters
    ----------
    f : numpy.ndarray
        Array to check.
    name : str
        Name of the parameter, used in error messages.

    Returns
    -------
    bool
        True if all checks pass.

    Raises
    ------
    TypeError
        If ``f`` is not a numpy array, is a masked array, or does not contain
        real numeric values.
    ValueError
        If ``f`` is empty, not one-dimensional, or contains non-finite values.
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
    """Compute precipitation attributions (i.e. the optimal transport plan) with the PAD-on-sphere method from numpy arrays.

    Parameters
    ----------
    values1, values2 : numpy.ndarray
        One-dimensional amounts of field1 and field2. They must be finite and
        nonnegative, with at least one positive value per field. Values are
        attributed as supplied, with no area weighting: pass volumes (e.g.
        precipitation in mm times grid-cell area) for a volume-weighted PAD.
    lat1, lon1 : numpy.ndarray
        One-dimensional latitudes and longitudes of field1, in degrees.
    lat2, lon2 : numpy.ndarray, optional
        Latitudes and longitudes of field2, in degrees. Required if
        ``same_grid=False`` and not allowed if ``same_grid=True``.
    same_grid : bool, default True
        Whether both fields are on the same grid, i.e. identical coordinates
        in identical order.
    distance_cutoff : float, default 1e8
        Great-circle cutoff distance in metres. The default (100,000 km)
        effectively means no cutoff. Zero is allowed; negative values are not.
    random_seed : int, optional
        Seed for the random choices made during attribution. ``None`` or -1
        chooses a random seed and prints it. An unsigned 32-bit integer gives
        reproducible results for identical inputs and code/library versions.

    Returns
    -------
    list of numpy.ndarray
        ``[attributions, remaining1, remaining2]``:

        - ``attributions``: float64 array of shape (N, 4), with columns
          great-circle distance (m), attributed amount, index in field1 and
          index in field2. Shape is (0, 4) if no attributions satisfy the cutoff.
        - ``remaining1``, ``remaining2``: non-attributed amounts of field1 and
          field2, with the same shapes as ``values1`` and ``values2``.

    Raises
    ------
    TypeError
        If inputs are not unmasked real numpy arrays, or ``same_grid`` or
        ``random_seed`` have the wrong type.
    ValueError
        If arrays are empty, not one-dimensional, non-finite or of mismatched
        shapes; if amounts are negative or all zero; if latitudes are outside
        [-90, 90]; or if ``lat2``/``lon2`` do not match ``same_grid``.
    RuntimeError
        If the C++ library reports an error.

    References
    ----------
    Skok, G. & Lledó, L. (2025) Spatial verification of global precipitation
    forecasts. Quarterly Journal of the Royal Meteorological Society.
    https://doi.org/10.1002/qj.5006
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
    """Compute the PAD value as the volume-weighted mean of the attribution distances.

    Parameters
    ----------
    PAD_attributions : array_like
        Two-dimensional array with distances in the first column and attributed
        amounts (weights) in the second, e.g. the ``attributions`` returned by
        `calculate_attributions_from_numpy`. Extra columns are ignored.

    Returns
    -------
    float
        Volume-weighted mean distance, in the units of the first column
        (metres for the arrays returned by this package).

    Raises
    ------
    ValueError
        If the array is empty or has fewer than two columns, contains
        non-finite or negative values, or all weights are zero.
    """
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
    """Compute precipitation attributions (i.e. the optimal transport plan) with the PAD-on-sphere method from xarray DataArrays.

    Parameters
    ----------
    fcst, obs : xarray.DataArray
        Total precipitation in mm, finite and nonnegative, with at least one
        positive value each. Must have a single ``gridpoint`` dimension and
        one-dimensional ``lat`` and ``lon`` coordinates in degrees.
    area : xarray.DataArray
        Grid-cell area of ``fcst`` in km^2, nonnegative, aligned with
        ``fcst``. Tiny negative areas, within 1e-6 of the largest area (e.g.
        float32 rounding of cos(lat) at the poles), are treated as zero.
        Zero-area cells hold no volume and get a residual error of 0 mm.
    area2 : xarray.DataArray, optional
        Grid-cell area of ``obs`` in km^2, with the same rules as ``area``.
        Required if ``same_grid=False`` and not allowed if ``same_grid=True``.
    same_grid : bool, default True
        Whether ``fcst`` and ``obs`` are on the same grid, i.e. identical
        lat/lon in identical order.
    cutoff : float, default 3000
        Great-circle cutoff distance in km.
    gridded_output : bool, default True
        If True, include the per-gridpoint volume and distance of the
        attributions in the returned datasets. If False, return only the
        residual errors, as a pandas DataFrame if ``same_grid=True``.
    random_seed : int, optional
        Seed for the random choices made during attribution. ``None`` or -1
        chooses a random seed and prints it. An unsigned 32-bit integer gives
        reproducible results for identical inputs and code/library versions.

    Returns
    -------
    transport : pandas.DataFrame
        All attributions, with columns ``distance_m`` (int64, great-circle
        distance in m), ``volume_m3`` (float64), ``gridpoint_fcst`` and
        ``gridpoint_obs`` (int64, positional indices 0..n-1 into ``fcst``
        and ``obs``).
    gridded_ds : xarray.Dataset
        Only if ``same_grid=True`` and ``gridded_output=True``. Variables
        ``volume`` (m^3 transported), ``distance`` (volume-weighted mean
        distance in m) and ``error`` (residual error in mm) at each grid point.
    residual_df : pandas.DataFrame
        Only if ``same_grid=True`` and ``gridded_output=False``. Columns
        ``lat``, ``lon`` and ``error`` (mm) for grid points with non-zero
        residual error, indexed by ``gridpoint``.
    gridded_fcst_ds, gridded_obs_ds : xarray.Dataset
        Only if ``same_grid=False`` and ``gridded_output=True``. Per-gridpoint
        summary on each grid: ``volume`` (m^3) and ``distance`` (m) of the water
        exported from each fcst grid point (positive) and imported into each obs
        grid point (negative), and ``error`` with the non-attributed
        precipitation (mm).
    residual_fcst_ds, residual_obs_ds : xarray.Dataset
        Only if ``same_grid=False`` and ``gridded_output=False``. Variable
        ``error`` with the non-attributed precipitation (mm) of ``fcst`` and
        ``obs`` on their own grids.

    Raises
    ------
    TypeError
        If ``same_grid`` or ``random_seed`` have the wrong type, or the data
        are not real numeric values.
    ValueError
        If inputs are not DataArrays with the expected dimension and
        coordinates, are not aligned with their areas, contain negative or
        non-finite values or non-positive areas, if a field is all zero, or if
        ``area2`` does not match ``same_grid``.
    RuntimeError
        If the C++ library reports an error.

    Notes
    -----
    In the gridded output, positive distances represent water exported from a
    grid point (fcst > obs) and negative distances water imported into it
    (fcst < obs). Positive residual errors represent overforecasting
    (fcst > obs, "false alarm") and negative ones underforecasting
    (fcst < obs, "miss").

    ``distance_m`` is truncated to whole metres and stored as an integer to
    save output storage; 1 m resolution is enough for attribution distances.

    References
    ----------
    Skok, G. & Lledó, L. (2025) Spatial verification of global precipitation
    forecasts. Quarterly Journal of the Royal Meteorological Society.
    https://doi.org/10.1002/qj.5006
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

    # Compute water volume (in m^3) from tp (or height in mm) and grid-cell area (in km^2)
    # vol_in_m3 = tp_in_mm / 1000 * area_in_km2 * 1000 * 1000
    amounts, cell_areas = [], []
    for field, cell_area, name in ((fcst, area, "fcst"), (obs, area2, "obs")):
        precipitation = _array(field.values, name)
        areas = _array(cell_area.values, name + " area")
        # Treat areas within rounding error of zero as zero, e.g. cos(90 deg) computed
        # from float32 latitudes is about -4e-8 instead of 0.
        areas = np.where(np.abs(areas) <= 1e-6 * np.abs(areas).max(), 0.0, areas)
        if np.any(precipitation < 0) or np.any(areas < 0):
            raise ValueError("Precipitation and cell areas must be nonnegative.")
        with np.errstate(over="raise", invalid="raise"):
            amounts.append(precipitation * areas * 1000.0)
        cell_areas.append(areas)

    # Convert cutoff from km to m
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
        # Convert residual error back from volume (in m^3) to height (in mm)
        # error_mm = error_m3 / (area_km2 * 1000 * 1000) * 1000
        # Zero-area cells hold no volume, so their error is 0 mm.
        error = np.divide(
            remaining, cell_area, out=np.zeros_like(remaining), where=cell_area > 0
        ) / 1000.0
        coords = {"lat": field.lat.astype(np.float64), "lon": field.lon.astype(np.float64)}
        if "gridpoint" in field.coords:
            coords["gridpoint"] = field.gridpoint
        return xr.Dataset(
            {"error": ("gridpoint", error)},
            coords=coords,
        )

    if not same_grid:
        residual_fcst = residual_ds(remaining1, fcst, cell_areas[0])
        residual_obs = residual_ds(remaining2, obs, cell_areas[1])
        if not gridded_output:
            return transport, residual_fcst, residual_obs
        # Transport-plan gridpoints are positional indices, so combine positionally,
        # keeping the residual's gridpoint coordinate as before.
        transport_fcst, transport_obs = aggregate_transportplan_at_gridpoints_unequal_grids(
            transport, get_latlon_df(residual_fcst), get_latlon_df(residual_obs)
        )
        gridded_fcst, gridded_obs = (
            xr.merge((transport_ds.drop_vars("gridpoint"), residual), compat="no_conflicts")
            for transport_ds, residual in (
                (transport_fcst, residual_fcst), (transport_obs, residual_obs)
            )
        )
        return transport, gridded_fcst, gridded_obs

    residual = residual_ds(remaining1 - remaining2, fcst, cell_areas[0])
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
        ), compat="no_conflicts")
        return transport, gridded_ds

    residual = residual.to_dataframe()[["lat", "lon", "error"]]
    return transport, residual.loc[residual["error"] != 0]
