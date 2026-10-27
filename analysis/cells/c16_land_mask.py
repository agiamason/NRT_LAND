# Land mask on the 0.5 deg grid from the RECCAP2 land-region file (1 on land, NaN elsewhere).
R_720_360 = xr.open_dataset(RECCAP_MASK)
R_720_360 = R_720_360.reindex(latitude=R_720_360['latitude'][::-1])
R_720_360 = mask_a(R_720_360.mask)
R_720_360 = R_720_360.rename({'latitude': 'lat', 'longitude': 'lon'})
