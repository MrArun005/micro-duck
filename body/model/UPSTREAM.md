# Where this model comes from

`upstream/` is copied unchanged from **Open Duck Playground**, the MuJoCo training setup that
produced the walking policies that run on real Open Duck Mini v2 robots:

- Repo: https://github.com/apirrone/Open_Duck_Playground
- Commit: `b9be205ac64488c23504ca42e5ec790337adeec3` (2025-08-05)
- Path: `playground/open_duck_mini_v2/xmls/` (`open_duck_mini_v2.xml`, `open_duck_mini_v2_backlash.xml`,
  `scene_flat_terrain.xml`, `joints_properties.xml`, `sensors.xml`, `assets/*.stl`)

It is generated with onshape-to-robot from the same Onshape CAD as the printed parts in
**Open Duck Mini** (https://github.com/apirrone/Open_Duck_Mini, commit
`b23317a485b3cec7d8417f352478778b3475173c`, 2026-01-31), which is Apache-2.0: see `LICENSE`.
The Playground sources carry Apache-2.0 headers too (Antoine Pirrone, Steve Nguyen; DeepMind for
the MuJoCo Playground base). The meshes are the simplified ones (about 0.3 MB in total), so they are
committed here rather than fetched.

The servo parameters in `joints_properties.xml` (kp 13.37, damping 0.56, friction loss 0.068,
armature 0.027, ±3.23 N·m) were identified on real STS3215 servos with the BAM tool
(https://github.com/Rhoban/bam).

Do not edit `upstream/`. Changes go in `../tweaks.yaml`; `../build_model.py` writes `duck.xml`.
The older `mini_bdx/robots/open_duck_mini_v2/robot.xml` in the Open Duck Mini repo is an earlier
export with full-detail meshes (19 MB) and default motor values; it is not used here.
