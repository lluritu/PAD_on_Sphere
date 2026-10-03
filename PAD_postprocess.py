import pandas as pd
import numpy as np
import xarray as xr


def weighted_average(df, data_col, weight_col, by_col):
    """Compute weighted averages per group.

    Much faster than groupby/apply alternatives. Rows with a missing value in
    ``data_col`` are left out of both the numerator and the total weight. The
    input dataframe is not modified.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe containing the columns below.
    data_col : str
        Column to be averaged.
    weight_col : str
        Column to be used as weight.
    by_col : str
        Column or index level name to group by.

    Returns
    -------
    pandas.DataFrame
        Indexed by the groups of ``by_col``, with columns ``volume`` (sum of
        weights) and ``distance`` (weighted average of ``data_col``).
    """
    # Preserve grouping by either a column or an index level without modifying
    # the caller's data, even if it already contains our temporary column names.
    working = df[[by_col]].copy() if by_col in df.columns else pd.DataFrame(index=df.index)
    working["_data_times_weight"] = df[data_col] * df[weight_col]
    working["_weight_where_notnull"] = df[weight_col] * pd.notnull(df[data_col])
    g = working.groupby(by_col)
    total = g["_weight_where_notnull"].sum()
    result = g["_data_times_weight"].sum() / total

    return pd.concat([total.rename("volume"), result.rename("distance")], axis=1)


def aggregate_transportplan_at_gridpoints(transportplan_df, latlon_df):
    """Aggregate the transport plan into a single attribution value at each grid point.

    The transport plan can have several displacements starting or ending at the
    same grid point (Kantorovich relaxation). This computes the total volume
    displaced and the volume-weighted mean displacement at each grid point,
    which is useful for plots and grid-point statistics. Only meaningful when
    both fields are on the same grid.

    Parameters
    ----------
    transportplan_df : pandas.DataFrame
        Transport plan with columns ``distance_m``, ``volume_m3``,
        ``gridpoint_fcst`` and ``gridpoint_obs``, as returned by
        `PAD_on_sphere.calculate_attributions_from_xarrays`.
    latlon_df : pandas.DataFrame
        Coordinates of every grid point, indexed by ``gridpoint``, as returned
        by `get_latlon_df`.

    Returns
    -------
    pandas.DataFrame
        Indexed by ``gridpoint`` with one row per grid point in ``latlon_df``,
        and columns ``volume`` (m^3), ``distance`` (m), ``lat`` and ``lon``.
        Grid points without attributions have NaN volume and distance.

    Notes
    -----
    A grid point either exports or imports water, never both. Positive
    distances represent water exported from the grid point (fcst > obs), and
    negative distances water imported into it (fcst < obs).
    """
    # Aggregate all displacements that started (fcst) or ended (obs) in each grid point
    dist_df_at_fcst = weighted_average(
        transportplan_df, "distance_m", "volume_m3", "gridpoint_fcst"
    )
    dist_df_at_obs = weighted_average(
        transportplan_df, "distance_m", "volume_m3", "gridpoint_obs"
    )

    # The two df above can overlap, but only when one of the two fields has zero displacement.
    # A grid point either gives or receives water volume, but not both.
    # We can combine both into a single xarray dataset:
    # * positive distances represent a water source, i.e. water being exported from origin grid point (fcst > obs).
    # * negative distances represent a water sink, i.e. water being imported at destination grid point (fcst < obs).
    dist_df_at_obs.loc[:, "distance"] = dist_df_at_obs.loc[:, "distance"] * -1
    dist_df = pd.concat([dist_df_at_fcst, dist_df_at_obs]).rename_axis("gridpoint")

    # We need to aggregated again due to the overlap
    dist_df = weighted_average(dist_df, "distance", "volume", "gridpoint")

    # add lat lon coordinates to df
    dist_df = pd.merge(
        dist_df, latlon_df, how="right", left_on="gridpoint", right_on="gridpoint"
    )
    return dist_df


def aggregate_transportplan_at_gridpoints_unequal_grids(transportplan_df, latlon_fcst_df, latlon_obs_df):
    """Aggregate the transport plan into a single attribution value at each grid point, for unequal grids.

    Counterpart of `aggregate_transportplan_at_gridpoints` for fields on
    different grids. Exports are aggregated on the fcst grid and imports on the
    obs grid, giving one dataset per grid.

    Parameters
    ----------
    transportplan_df : pandas.DataFrame
        Transport plan with columns ``distance_m``, ``volume_m3``,
        ``gridpoint_fcst`` and ``gridpoint_obs``, as returned by
        `PAD_on_sphere.calculate_attributions_from_xarrays`.
    latlon_fcst_df, latlon_obs_df : pandas.DataFrame
        Coordinates of every fcst and obs grid point, indexed by
        ``gridpoint``, as returned by `get_latlon_df`.

    Returns
    -------
    fcst_ds, obs_ds : xarray.Dataset
        One dataset per grid, along ``gridpoint`` with ``lat``/``lon``
        coordinates, and variables ``volume`` (m^3) and ``distance`` (m).
        Grid points without attributions have NaN volume and distance.

    Notes
    -----
    Signs follow `aggregate_transportplan_at_gridpoints`: distances are
    positive on the fcst grid (water exported, fcst > obs) and negative on
    the obs grid (water imported, fcst < obs).
    """
    datasets = []
    for column, latlon_df, sign in (
        ("gridpoint_fcst", latlon_fcst_df, 1),
        ("gridpoint_obs", latlon_obs_df, -1),
    ):
        dist_df = weighted_average(transportplan_df, "distance_m", "volume_m3", column)
        dist_df["distance"] = dist_df["distance"] * sign
        dist_df = pd.merge(
            dist_df, latlon_df, how="right", left_on=column, right_on="gridpoint"
        )
        datasets.append(
            dist_df.rename_axis("gridpoint").to_xarray().set_coords(("lat", "lon"))
        )
    return tuple(datasets)


def postprocess_residue_df(residue_df, latlon_df):
    """Expand a residual-error dataframe to all grid points.

    Parameters
    ----------
    residue_df : pandas.DataFrame
        Non-attributed precipitation with an ``error`` column, indexed by
        ``gridpoint``, e.g. as returned by
        `PAD_on_sphere.calculate_attributions_from_xarrays` with
        ``gridded_output=False``.
    latlon_df : pandas.DataFrame
        Coordinates of every grid point, indexed by ``gridpoint``, as returned
        by `get_latlon_df`.

    Returns
    -------
    pandas.DataFrame
        Indexed by ``gridpoint`` with columns ``error``, ``lat`` and ``lon``
        for every grid point. Grid points missing from ``residue_df`` get NaN.
    """
    return pd.merge(residue_df.error, latlon_df, how="right", on="gridpoint")


def get_latlon_df(da):
    """Get the lat/lon coordinates of a DataArray as a pandas dataframe.

    Parameters
    ----------
    da : xarray.DataArray
        DataArray with one-dimensional ``lat`` and ``lon`` coordinates along
        its grid-point dimension.

    Returns
    -------
    pandas.DataFrame
        Columns ``lat`` and ``lon``, indexed by ``gridpoint``. The index is
        positional (0..n-1), matching the gridpoint indices of the transport
        plan, regardless of any ``gridpoint`` coordinate on ``da``.
    """
    return pd.DataFrame({"lat": da.lat, "lon": da.lon}).rename_axis("gridpoint")


def compute_regional_stats(gridded_ds, region_masks, area):
    """Compute location and residual error statistics for several regions.

    Parameters
    ----------
    gridded_ds : xarray.Dataset
        Per-gridpoint summary with ``distance`` (m), ``volume`` (m^3) and
        ``error`` (mm), as returned by
        `PAD_on_sphere.calculate_attributions_from_xarrays` with
        ``gridded_output=True``.
    region_masks : xarray.DataArray
        Boolean masks with a ``gridpoint`` dimension and a region dimension,
        indicating which grid points belong to each region.
    area : xarray.DataArray
        Area of each grid cell along ``gridpoint``. Only used as weights, so
        any unit works.

    Returns
    -------
    xarray.Dataset
        For each region, ``mean_distance`` (volume-weighted mean absolute
        distance, in km) and ``residual_mae`` (area-weighted mean absolute
        residual error, in mm).
    """

    LocationError = (
        np.abs(gridded_ds.distance / 1000)
        .weighted(gridded_ds.volume.fillna(0) * region_masks)
        .mean(dim="gridpoint")
    )
    ResidualError = (
        np.abs(gridded_ds.error.fillna(0))
        .weighted(area * region_masks)
        .mean(dim="gridpoint")
    )
    # Location Error for the region as a Mean Absolute Distance
    # Residual Error for the region as a MAE
    LocationError.name = "mean_distance"
    ResidualError.name = "residual_mae"

    return xr.merge((LocationError, ResidualError))
