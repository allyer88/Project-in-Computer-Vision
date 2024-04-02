import numpy as np
from plyfile import PlyData, PlyElement

def write_ply(points, faces, colors, filename):
    vertex = np.array([(x, y, z, r, g, b) for (x, y, z), (r, g, b) in zip(points, colors)],
                      dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    faces = np.array(faces, dtype=[('vertex_indices', 'i4', (3,))])
    
    vertex_el = PlyElement.describe(vertex, 'vertex')
    face_el = PlyElement.describe(faces, 'face')

    PlyData([vertex_el, face_el], text=True).write(filename)