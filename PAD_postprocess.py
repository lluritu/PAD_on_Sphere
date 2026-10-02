import pandas as pd
import numpy as np
import xarray as xr


def weighted_average(df, data_col, weight_col, by_col):
    """Fast computation of weighted averages. This is much faster than other groupby/apply methods.

    :param dataframe df: input dataframe containing the columns specified below.
    :param str data_col: df column to be averaged.
    :param str weight_col: df column to be used as a weight.
    :param str by_col: df column to be used as group.

    :return: pandas dataframe with weighted average for each group.

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
    """Compute a single attribution value at each gridpoint, representing total mass displaced and the average displacement.
    Since the transport plan can have multiple displacements starting or ending at the same grid point (Kantorovich relaxation),
    this is useful to produce plots or grid point statistics.

    :param dataframe transportplan_df: a dataframe with a transport plan.
    :param dataframe latlon_df: a dataframe with lat/lon coordinates at each gridpoint.

    :return: a dataframe indicating distance and volume displaced at each grid point. Positive distances represent a water export at origin (fcst > obs) and negative distances represent a water import at destination (fcst < obs).

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


def postprocess_residue_df(residue_df, latlon_df):
    """Postprocess the bias dataframe to ensure that all grid-points are present. Empty points are filled up with nan.

    :param dataframe residue_df: a dataframe with non-attributed precipitation.
    :param dataframe latlon_df: a dataframe with all lat and lon coordinates.

    :return: a pandas dataframe with nan values for the gridpoints without unattributed precipitation.

    """
    return pd.merge(residue_df.error, latlon_df, how="right", on="gridpoint")


def get_latlon_df(da):
    """Get lat lon coordinates from an xarray dataarray and return them as a pandas dataframe.

    :param dataarray da: a dataarray with lat and lon coordinates.

    :return: a pandas dataframe with gridpoint, lat and lon columns.

    """
    return pd.DataFrame({"lat": da.lat, "lon": da.lon}).rename_axis("gridpoint")


def compute_regional_stats(gridded_ds, region_masks, area):
    """Compute location error statistics for several domains.

    :param dataset gridded_ds: an xarray Dataset with distance (in m), volume (in m^3) attributed, and unattributed precipitation in mm at each gridpoint.
    :param dataarray region_masks: a boolean xarray DataArray with the region masks, indication which gridpoints belong to each of the regions.
    :param dataarray area: an xarray DataArray indicating the area of each grid cell.

    :return: an xarray Dataset with volume-weighted mean distance and mean absolute error for each region.

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
