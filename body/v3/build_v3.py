import mujoco, numpy as np, xml.etree.ElementTree as ET, copy
m = mujoco.MjModel.from_xml_path('duck.xml')
hr_joint = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, 'head_roll')
hr_body = m.jnt_bodyid[hr_joint]
# head mesh vertices in the head_roll body frame
pts = []
for g in range(m.ngeom):
    if m.geom_bodyid[g] != hr_body or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH: continue
    mid = m.geom_dataid[g]; v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid]+m.mesh_vertnum[mid]]
    R = np.zeros(9); mujoco.mju_quat2Mat(R, m.geom_quat[g]); R = R.reshape(3,3)
    pts.append(v @ R.T + m.geom_pos[g])
pts = np.vstack(pts)
front_x, lo_z, mid_y = pts[:,0].max(), pts[:,2].min(), 0.0
jaw_pos = [float(round(front_x - 0.004, 5)), 0.0, float(round(lo_z + 0.012, 5))]
print('head_roll frame: front x %.4f, bottom z %.4f -> jaw pos %s' % (front_x, lo_z, jaw_pos))

tree = ET.parse('duck.xml'); root = tree.getroot()
# 1. head yaw full turn
for j in root.iter('joint'):
    if j.get('name') == 'head_yaw': j.set('range', '-3.14159 3.14159')
    if j.get('name') in ('neck_pitch', 'head_roll'):
        j.set('stiffness', '0.12'); j.set('damping', '0.005'); j.set('springref', '0')
for a in root.iter('general'):
    if a.get('name') == 'head_yaw': a.set('ctrlrange', '-3.14159 3.14159')
# 2. remove servos on passive joints
act = root.find('actuator')
for a in list(act):
    if a.get('name') in ('neck_pitch', 'head_roll'): act.remove(a)
# 3. jaw: hinge + plate on the head_roll body, plus a position servo
hr_el = None
for b in root.iter('body'):
    if b.find("joint[@name='head_roll']") is not None: hr_el = b
jaw = ET.SubElement(hr_el, 'body', {'name': 'jaw', 'pos': ' '.join(map(str, jaw_pos))})
ET.SubElement(jaw, 'joint', {'name': 'jaw', 'axis': '0 1 0', 'range': '0 0.5236', 'damping': '0.05'})
ET.SubElement(jaw, 'geom', {'type': 'box', 'size': '0.030 0.030 0.004', 'pos': '0.030 0 0', 'rgba': '1 0.75 0 1', 'contype': '2', 'conaffinity': '1'})
ET.SubElement(jaw, 'inertial', {'pos': '0.030 0 0', 'mass': '0.01', 'diaginertia': '0.00001 0.00001 0.00002'})
ET.SubElement(act, 'position', {'name': 'jaw', 'joint': 'jaw', 'kp': '4', 'ctrlrange': '0 0.5236', 'forcerange': '-0.25 0.25'})
# keyframe: drop ctrl/qpos entries of removed servos, add jaw
names = [a.get('name') for a in act]  # after removal; need original order for keyframe
orig = ['left_hip_yaw','left_hip_roll','left_hip_pitch','left_knee','left_ankle','right_hip_yaw','right_hip_roll','right_hip_pitch','right_knee','right_ankle','neck_pitch','head_pitch','head_yaw','head_roll']
kf = root.find('keyframe').find('key')
ctrl = kf.get('ctrl').split()
qpos = kf.get('qpos').split()
keep = [i for i, n in enumerate(orig) if n not in ('neck_pitch','head_roll')]
new_ctrl = [ctrl[i] for i in keep] + ['0']
new_qpos = qpos + ['0']
kf.set('ctrl', ' '.join(new_ctrl)); kf.set('qpos', ' '.join(new_qpos))
tree.write('duck_v3.xml')
print('wrote model/duck_v3.xml')
