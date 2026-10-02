import numpy as np
import xarray as xr
import pandas as pd
import os
from ctypes import *
from PAD_postprocess import aggregate_transportplan_at_gridpoints, get_latlon_df

# -----------------------------------------------------------------------------------------------------
# Definitions for the C++ library calls
# -----------------------------------------------------------------------------------------------------

# search for the PAD C++ shared library file (PAD_on_sphere_Cxx_shared_library.so) in the same folder
libc = CDLL(
    os.path.abspath(os.path.expanduser(os.path.dirname(__file__)))
    + os.path.sep
    + "PAD_on_sphere_Cxx_shared_library.so"
)

# define data types of library functions
ND_POINTER_1D = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C")
libc.free_mem_double_array.argtypes = [POINTER(c_double)]
libc.free_mem_double_array.restype = None
libc.calculate_PAD_results_assume_same_grid_ctypes.argtypes = [
    ND_POINTER_1D,
    ND_POINTER_1D,
    ND_POINTER_1D,
    ND_POINTER_1D,
    c_size_t,
    POINTER(c_size_t),
    c_double,
]
libc.calculate_PAD_results_assume_same_grid_ctypes.restype = POINTER(c_double)
libc.calculate_PAD_results_assume_different_grid_ctypes.argtypes = [
    ND_POINTER_1D,
    ND_POINTER_1D,
    ND_POINTER_1D,
    c_size_t,
    ND_POINTER_1D,
    ND_POINTER_1D,
    ND_POINTER_1D,
    c_size_t,
    POINTER(c_size_t),
    c_double,
]
libc.calculate_PAD_results_assume_different_grid_ctypes.restype = POINTER(c_double)

# -----------------------------------------------------------------------------------------------------
# Numpy wrapper functions
# -----------------------------------------------------------------------------------------------------

# Constant used for the Numpy examples
Earth_radius = 6371.0 * 1000.0


def calculate_attributions_from_numpy(
    values1,
    values2,
    lat1,
    lon1,
    lat2=None,
    lon2=None,
    same_grid=True,
    distance_cutoff=100 * 1000 * 1000,
):
    """Compute Precipitation Attributions (i.e. the Optimal Transport Plan) with the PAD-on-sphere method (Skok and Lledó 2025) from numpy arrays.

    :param ndarray values1: field1 total precipitation in mm.
    :param ndarray values2: field2 total precipitation in mm.
    :param ndarray lat1: latitudes of field1.
    :param ndarray lon1: longitudes of field1.
    :param ndarray lat2: latitudes of field2.
    :param ndarray lon2: longitudes of field2.
    :param bool same_grid: whether field1 and field2 are in the same grid.
    :param int distance_cutoff: cutoff distance in m.

    :return: a numpy array with attributed precipitation and two arrays with unattributed precipitation in each field.

    """

    if same_grid:
        if (lat2 is not None) or (lon2 is not None):
            print(
                'ERROR: lat2, lon2 were provided but same_grid=True !. Returning "None" as result!'
            )
            return None
    else:
        if (lat2 is None) or (lon2 is None):
            print(
                'ERROR: lat2, lon2 arrays not provided but same_grid=False !. Returning "None" as result!'
            )
            return None

    # check input arrays
    if check_input_array(lat1, "lat1") == False:
        return None
    if check_input_array(lon1, "lon1") == False:
        return None
    if check_input_array(values1, "values1") == False:
        return None
    if check_input_array(values2, "values2") == False:
        return None
    if not same_grid:
        if check_input_array(lat2, "lat2") == False:
            return None
        if check_input_array(lon2, "lon2") == False:
            return None

    # check dimensions of fields
    if lat1.shape != lon1.shape or lat1.shape != values1.shape:
        print(
            'ERROR: the lat1, lon1, values1 arrays do not have the same shape !. Returning "None" as result!'
        )
        return None
    if same_grid:
        if lat1.shape != values2.shape:
            print(
                'ERROR: the lat1, lon1, values2 arrays do not have the same shape !. Returning "None" as result!'
            )
            return None
    else:
        if lat2.shape != lon2.shape or lat2.shape != values2.shape:
            print(
                'ERROR: the lat2, lon2 and values2 arrays do not have the same shape !. Returning "None" as result!'
            )
            return None

    # detect negative values
    if values1[values1 < 0].size > 0 or values2[values2 < 0].size > 0:
        print(
            'ERROR: the values1 or values2 array contains some negative values, which is not allowed . Returning "None" as result!'
        )
        return None

    # check if all the values are zero
    if np.sum(values1) == 0 or np.sum(values2) == 0:
        print(
            'ERROR: the values1 or values2 array contains only zeroes, which is not allowed . Returning "None" as result!'
        )
        return None

    # cast distance to float64 and ensure it is positive
    distance_cutoff = np.float64(distance_cutoff)
    if distance_cutoff <= 0:
        distance_cutoff = 1e99

    # if needed make a copy and cast to float64 - just in case the input fields are integers
    if lat1.dtype != np.float64:
        lat1 = lat1.astype(np.float64, copy=True)
    if lon1.dtype != np.float64:
        lon1 = lon1.astype(np.float64, copy=True)
    if values1.dtype != np.float64:
        values1 = values1.astype(np.float64, copy=True)
    if values2.dtype != np.float64:
        values2 = values2.astype(np.float64, copy=True)
    if not same_grid:
        if lat2.dtype != np.float64:
            lat2 = lat2.astype(np.float64, copy=True)
        if lon2.dtype != np.float64:
            lon2 = lon2.astype(np.float64, copy=True)

    # convert to C_CONTIGUOUS arrays if needed (these are required by C++)
    if lat1.flags["C_CONTIGUOUS"] != True:
        lat1 = np.ascontiguousarray(lat1)
    if lon1.flags["C_CONTIGUOUS"] != True:
        lon1 = np.ascontiguousarray(lon1)
    if values1.flags["C_CONTIGUOUS"] != True:
        values1 = np.ascontiguousarray(values1)
    if values2.flags["C_CONTIGUOUS"] != True:
        values2 = np.ascontiguousarray(values2)
    if not same_grid:
        if lat2.flags["C_CONTIGUOUS"] != True:
            lat2 = np.ascontiguousarray(lat2)
        if lon2.flags["C_CONTIGUOUS"] != True:
            lon2 = np.ascontiguousarray(lon2)

    c_number_of_attributions = c_size_t()

    ngridpoints1 = lat1.shape[0]
    if same_grid:
        ngridpoints2 = ngridpoints1
        results = libc.calculate_PAD_results_assume_same_grid_ctypes(
            lat1,
            lon1,
            values1,
            values2,
            ngridpoints1,
            byref(c_number_of_attributions),
            distance_cutoff,
        )
    else:
        ngridpoints2 = lat2.shape[0]
        results = libc.calculate_PAD_results_assume_different_grid_ctypes(
            lat1,
            lon1,
            values1,
            ngridpoints1,
            lat2,
            lon2,
            values2,
            ngridpoints2,
            byref(c_number_of_attributions),
            distance_cutoff,
        )

    number_of_attributions = c_number_of_attributions.value

    # Deserialize
    attributions = np.asarray(results[0 : number_of_attributions * 4]).reshape(
        number_of_attributions, -1
    )
    non_attributed_values1 = np.asarray(
        results[
            (number_of_attributions * 4) : (number_of_attributions * 4 + ngridpoints1)
        ]
    )
    non_attributed_values2 = np.asarray(
        results[
            (number_of_attributions * 4 + ngridpoints1) : (
                number_of_attributions * 4 + ngridpoints1 + ngridpoints2
            )
        ]
    )

    libc.free_mem_double_array(results)

    return [attributions, non_attributed_values1, non_attributed_values2]


def calculate_PAD_on_sphere_from_attributions(PAD_attributions):
    """Compute the volume-weighted mean of the attribution distances.

    :param ndarray PAD_atributions: an array containing distances in the first column and volume in the second column.

    :return: the average PAD value of a list of attributions.

    """
    return np.sum(PAD_attributions[:, 0] * PAD_attributions[:, 1]) / np.sum(
        PAD_attributions[:, 1]
    )


def check_input_array(f, name):
    """Check the input numpy arrays for the right dimension and contents.

    :param ndarray f: a numpy array.
    :param str name: the name of the parameter.

    :return: True if all tests successful, otherwise False.

    """

    # test if fields are numpy arrays
    if type(f) is not np.ndarray:
        print(
            "ERROR: the "
            + name
            + ' input array is not a numpy array of type numpy.ndarray, which is not permitted! Perhaps it is a masked arrays, which is also not permitted. Returning "None" as result!'
        )
        return False

    # check dimensions of fields
    if f.ndim != 1:
        print(
            "ERROR: the "
            + name
            + ' input array is not one-dimensional, which is not permitted!. Returning "None" as result!'
        )
        return False

    # check if the array has some elements
    if f.size == 0:
        print(
            "ERROR: the "
            + name
            + ' array does not contain any elements. Returning "None" as result!'
        )
        return False

    # detect non-numeric values
    result = np.where(np.isfinite(f) == False)
    if len(result[0]) > 0:
        print(
            "ERROR: the "
            + name
            + ' arrays contains some non-numeric values, which is not permitted!. Returning "None" as result!'
        )
        return False

    # detect masked array
    if isinstance(f, np.ma.MaskedArray):
        print(
            "ERROR: the "
            + name
            + ' array is a masked array which is not permitted. Returning "None" as result!'
        )
        return False

    return True


# -----------------------------------------------------------------------------------------------------
# Xarray wrapper functions
# -----------------------------------------------------------------------------------------------------


def calculate_attributions_from_xarrays(
    fcst, obs, area, area2=None, same_grid=True, cutoff=3000, gridded_output=True
):
    """Compute Precipitation Attributions (i.e. the Optimal Transport Plan) with the PAD-on-sphere method (Skok and Lledó 2025) from xarray datasets.

    :param xarray fcst: should contain tp in mm.
    :param xarray obs: should contain tp in mm.
    :param xarray area: fcst grid cell area in km^2.
    :param xarray area2: obs grid cell area in km^2, if same_grid is False.
    :param int cutoff: cutoff distance in km.
    :param bool gridded_output: if True the output is recasted from pandas dataframes to a gridded xarray dataset. Only possible if same_grid=True.

    :return: a list with a pandas dataframe containing all attributions, and additionally, depending on the options: if same_grid=True and gridded_output=True an xarray dataset containing volume transported, distance transported, and residual error in mm at each grid point. if same_grid=True and gridded_output=False, a pandas dataframe with non-attributed precipitation. If same_grid=False, two xarray datasets with non-attributed precipitation from each field. For the gridded outputs, positive distances represent a water export at origin (fcst > obs) and negative distances represent a water import at destination (fcst < obs). Similarly, positive residual errors represent overforecasting (fcst>obs) and viceversa.

    """
    if same_grid:
        area2 = area

    # Check input data
    if not isinstance(fcst, xr.DataArray):
        print("fcst should be an xarray DataArray")
        return None
    if not isinstance(obs, xr.DataArray):
        print("obs should be an xarray DataArray")
        return None
    if not isinstance(area, xr.DataArray):
        print("area should be an xarray DataArray")
        return None
    if not isinstance(area2, xr.DataArray):
        print("area2  should be an xarray DataArray")
        return None

    if not fcst.dims == ("gridpoint",):
        print("fcst should have (only) a gridpoint dimension")
        return None
    if not obs.dims == ("gridpoint",):
        print("obs should have (only) a gridpoint dimension")
        return None
    if not area.dims == ("gridpoint",):
        print("area should have (only) a gridpoint dimension")
        return None
    if not area2.dims == ("gridpoint",):
        print("area2 should have (only) a gridpoint dimension")
        return None

    if fcst.sizes != area.sizes:
        print("The area and fcst DataArrays are not aligned")
        return None
    if obs.sizes != area2.sizes:
        print("The area2 and obs DataArrays are not aligned")
        return None

    # Compute water volume (in m^3) from tp (or height in mm) and grid-cell area (in km^2)
    # vol_in_m3 = fcst_in_mm / 1000 * area_in_km2 * 1000 * 1000
    fcst = fcst * area * 1000
    obs = obs * area2 * 1000

    # Convert cutoff from km to m
    # cast distance to float64 and ensure it is positive
    cutoff = np.float64(cutoff)
    cutoff *= 1000
    if cutoff <= 0:
        print("ERROR: negative cutoff not allowed")
        return None

    # Check for negative values
    if (fcst < 0).any() or (obs < 0).any():
        print("ERROR: negative values not allowed")
        return None
    # Check for nan, inf and a positive sum.
    total_fcst = fcst.sum(skipna=False)
    total_obs = obs.sum(skipna=False)
    if total_fcst <= 0 or total_obs <= 0:
        print("ERROR: all-zero fields not allowed")
        return None
    if total_fcst.isnull() or total_obs.isnull():
        print("ERROR: NaN values not allowed")
        return None
    if total_fcst == np.inf or total_obs == np.inf:
        print("ERROR: infinite values not allowed")
        return None

    # Cast the data to C_contiguous float64 type
    fcst = fcst.astype("float64", order="C")
    fcst["lat"] = fcst.lat.astype("float64", order="C")
    fcst["lon"] = fcst.lon.astype("float64", order="C")
    obs = obs.astype("float64", order="C")
    obs["lat"] = obs.lat.astype("float64", order="C")
    obs["lon"] = obs.lon.astype("float64", order="C")

    c_number_of_attributions = c_size_t()

    ngridpoints1 = fcst.sizes["gridpoint"]
    ngridpoints2 = obs.sizes["gridpoint"]

    if same_grid:
        results = libc.calculate_PAD_results_assume_same_grid_ctypes(
            fcst.lat.values,
            fcst.lon.values,
            fcst.values,
            obs.values,
            ngridpoints1,
            byref(c_number_of_attributions),
            cutoff,
        )
    else:
        results = libc.calculate_PAD_results_assume_different_grid_ctypes(
            fcst.lat.values,
            fcst.lon.values,
            fcst.values,
            ngridpoints1,
            obs.lat.values,
            obs.lon.values,
            obs.values,
            ngridpoints2,
            byref(c_number_of_attributions),
            cutoff,
        )

    number_of_attributions = c_number_of_attributions.value

    # Deserialize
    attributions = np.asarray(results[0 : number_of_attributions * 4]).reshape(
        number_of_attributions, -1
    )
    non_attributed_values1 = np.asarray(
        results[
            (number_of_attributions * 4) : (number_of_attributions * 4 + ngridpoints1)
        ]
    )
    non_attributed_values2 = np.asarray(
        results[
            (number_of_attributions * 4 + ngridpoints1) : (
                number_of_attributions * 4 + ngridpoints1 + ngridpoints2
            )
        ]
    )

    libc.free_mem_double_array(results)

    # Create a dataframe with the transport plan
    transportplan_df = pd.DataFrame(
        attributions,
        columns=["distance_m", "volume_m3", "gridpoint_fcst", "gridpoint_obs"],
    )
    # Round distance and gridpoint columns to integer
    transportplan_df[["distance_m", "gridpoint_fcst", "gridpoint_obs"]] = (
        transportplan_df[["distance_m", "gridpoint_fcst", "gridpoint_obs"]].astype(int)
    )

    if same_grid:
        # Compute residual error as fcst_nonattributed - obs_nonattributed
        residual_error_ds = xr.Dataset(
            data_vars=dict(
                error=(
                    ["gridpoint"],
                    (non_attributed_values1 - non_attributed_values2),
                ),
            ),
            coords={
                "gridpoint": fcst.gridpoint,
                "lat": fcst.lat,
                "lon": fcst.lon,
            },
        )

        # Convert residual error back from volume (in m^3) to height (in mm)
        # error_mm = error_m3 / (area_km2 * 1000 * 1000) * 1000
        residual_error_ds /= area * 1000

        if gridded_output:
            # Convert the attributions list to a gridded xarray
            latlon_df = get_latlon_df(obs)
            transportplan_ds = (
                aggregate_transportplan_at_gridpoints(transportplan_df, latlon_df)
                .to_xarray()
                .set_coords(("lat", "lon"))
            )
            gridded_ds = xr.merge((transportplan_ds, residual_error_ds)).drop_vars(
                "gridpoint"
            )
            return (transportplan_df, gridded_ds)

        else:
            # Conversion of residual error back to pandas df
            residual_df = residual_error_ds.to_dataframe()
            residual_df = residual_df[residual_df.error != 0]
            return (transportplan_df, residual_df)

    else:
        # Create two residual error datasets, one in each grid
        residual_error_fcst = xr.Dataset(
            data_vars=dict(error=(["gridpoint"], non_attributed_values1)),
            coords={
                "gridpoint": fcst.gridpoint,
                "lat": fcst.lat,
                "lon": fcst.lon,
            },
        )

        residual_error_obs = xr.Dataset(
            data_vars=dict(error=(["gridpoint"], non_attributed_values2)),
            coords={
                "gridpoint": obs.gridpoint,
                "lat": obs.lat,
                "lon": obs.lon,
            },
        )

        # Convert residual error back from volume (in m^3) to height (in mm)
        # error_mm = error_m3 / (area_km2 * 1000 * 1000) * 1000
        residual_error_fcst /= area * 1000
        residual_error_obs /= area2 * 1000

        return (transportplan_df, residual_error_fcst, residual_error_obs)
