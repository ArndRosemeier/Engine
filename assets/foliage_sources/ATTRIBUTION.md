# Foliage source assets

These files are downloaded for the SpeedTree-inspired foliage prototype. They are
source material, not yet the final runtime atlas.

## Public-domain / CC0 sources

- **paramecij's vegetation base texture pack**
  - Source: https://opengameart.org/content/paramecijs-vegetation-base-texture-pack
  - License: public domain / CC0 as stated on the source pages
  - Author: paramecij
  - Included files: `vegetation_leaf_maple_01.png`, `vegetation_clover_02.png`,
    `vegetation_fern_01.png`, `vegetation_fern_08.png`,
    `vegetation_smallplant_03.png`, all `vegetation_tree_*.png` files
  - Direct source directory: https://opengameart.org/sites/default/files/

- **60 CC0 Vegetation Textures**
  - Source: https://opengameart.org/content/60-cc0-vegetation-textures
  - License: CC0
  - Original attribution in the source listing: textures are from burningwell
  - Included archive: `60 free plants.zip`
  - Direct source URL: https://opengameart.org/sites/default/files/60%20free%20plants.zip

## Usage notes

The assets are retained with their original filenames and hashes for provenance.
Do not remove this file when copying processed images into runtime assets. Before
shipping, review the exact license/source for every generated atlas and retain
this manifest or a generated notice in the distribution.

SHA-256 hashes are recorded in the development notes or can be regenerated with:

```powershell
Get-FileHash assets/foliage_sources/* -Algorithm SHA256
```

## Runtime prototype selection

The first SpeedTree-style showcase slice uses these two cutouts:

- `vegetation_leaf_maple_01.png` — broadleaf prototype
- `vegetation_fern_08.png` — needle/fern prototype

Both are public-domain files from the paramecij pack above. The showcase loads them from this directory and fails loudly if either file is missing.
