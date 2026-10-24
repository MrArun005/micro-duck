import mujoco, numpy as np, trimesh, os
SCALE=0.7; OUT='/home/claude/duck_mesh_v3'
m=mujoco.MjModel.from_xml_path('duck_v3.xml'); d=mujoco.MjData(m); mujoco.mj_resetDataKeyframe(m,d,0)
def setq(n,deg): d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n)]]=np.radians(deg)
setq('head_yaw',60); setq('jaw',22)
mujoco.mj_forward(m,d)
parts=[]; jaw_body=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'jaw')
for g in range(m.ngeom):
    if m.geom_group[g]==3: continue
    T=np.eye(4); T[:3,:3]=d.geom_xmat[g].reshape(3,3); T[:3,3]=d.geom_xpos[g]
    if m.geom_type[g]==mujoco.mjtGeom.mjGEOM_MESH:
        mid=m.geom_dataid[g]; v0,n=m.mesh_vertadr[mid],m.mesh_vertnum[mid]; f0,nf=m.mesh_faceadr[mid],m.mesh_facenum[mid]
        verts=m.mesh_vert[v0:v0+n]@T[:3,:3].T+T[:3,3]
        mesh=trimesh.Trimesh(verts,m.mesh_face[f0:f0+nf],process=False)
    elif m.geom_type[g]==mujoco.mjtGeom.mjGEOM_BOX and m.geom_bodyid[g]==jaw_body:
        mesh=trimesh.creation.box(extents=2*m.geom_size[g]); mesh.apply_transform(T)
        mesh.visual.face_colors=np.array([255,190,0,255],dtype=np.uint8)
    else: continue
    parts.append(mesh)
s=trimesh.util.concatenate(parts); s.apply_scale(SCALE)
lo,hi=s.bounds; s.apply_translation([-(lo[0]+hi[0])/2,-(lo[1]+hi[1])/2,-lo[2]])
print('parts',len(parts),'bounds mm',np.round((s.bounds[1]-s.bounds[0])*1000,1),'faces',len(s.faces))
s.export(f'{OUT}/duck_mesh_v3_head60_jaw22.stl'); s.export(f'{OUT}/duck_mesh_v3_head60_jaw22.glb')
