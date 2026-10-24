# Micro Duck v3 mesh package

Contents
- duck_mesh_v3_head60_jaw22.stl / .glb: scaled (0.7x) visual mesh, head turned 60 deg, jaw open 22 deg
- preview_v3.png: preview image
- duck_v3.xml: MuJoCo model with head yaw +/-180 deg, passive neck pitch and head roll (spring, no servo), jaw servo
- build_v3.py: edits the repo model (body/model/duck.xml) into duck_v3.xml
- export_v3.py: exports the mesh from duck_v3.xml

Status
- Visual mesh only, not print-ready
- Jaw is a placed plate, not a real part; housing and pivot still needed
- Head mass in the repo is 0.59 kg; the report says 0.41 kg. Spring values depend on it.
- Simulation uses full-size mass (2.1 kg)
