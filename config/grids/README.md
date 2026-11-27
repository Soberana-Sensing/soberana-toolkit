# Grids

A named grid fixes the CRS, the pixel size and the alignment rule for every layer a product
contains. Datatype configs name a grid; they do not carry a CRS or a resolution of their own.

```yaml
name: utm28_30m
crs: EPSG:32628
resolution: 30
resolution_policy: native_or_grid
align: upper-left
```

`resolution_policy: native_or_grid` means the effective pixel size is the larger of the grid
resolution and the source's native resolution. Data is coarsened onto the grid when it is finer,
and kept at native resolution when it is coarser. Pixel size never shrinks, so no product claims
detail it does not have.

The processing pipeline records the grid name and a hash of the grid file in
`run_metadata.json`, so an output can always be traced back to the definition that produced it.
