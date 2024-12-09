import os
import glob
import torch
import pickle
import trimesh
import math
import numpy as np
import scipy as sp
import matplotlib
# matplotlib.use('TKAGG')
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from trimesh.ray.ray_triangle import RayMeshIntersector
from trimesh import creation
from mpl_toolkits.mplot3d import Axes3D, art3d  # noqa: F401 unused import
from v2.utils import conf
from v2.data.repr_s2cnn import ToMesh


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class V2Lib:
    def __init__(self, m, n, w, h, d, r, polar, ssm):
        self.m = m  # Rows in view (e.g. latitude lines)
        self.n = n  # Columns in view (e.g. longitude lines)
        self.h = h  # Height of each view, in pixels
        self.w = w  # Width of each view, in pixels
        self.d = d  # The distance between two nearest view points
        self.r = r  # The radius of the ray sphere
        self.polar = polar  # Include the polar point or not

        self.s_view = self.sphere_sample(ssm)  # sphere sampling method
        self.ray_origins, self.ray_edges, self.ray_directions = self._v2_rays()
        self.ray_orig, self.ray_dire = self._np2torch(self.ray_origins, self.ray_directions)  # Torch tensor

        self.mesh_path = None
        self.mesh = None
        self.convex_hull = None

        self.convh_v2_d = self.mesh_v2_d = None  # Ray travelling distance, range [0, 1] as it is inside a unit sphere
        self.convh_v2_a = self.mesh_v2_a = None  # Incident angle, range [0, pi/2]
        self.convh_v2_s = self.mesh_v2_s = None  # Sine of incident angle, range [0, 1]
        self.convh_v2_c = self.mesh_v2_c = None  # Cosine of incident angle, range [0, 1]
        self.convh_v2_p = self.mesh_v2_p = None  # Intersection points of each ray and the mesh

        self.v2 = None

    def sphere_sample(self, method):
        """ Various of sphere sampling methods to initialize the view center of V2

        Args:
            method: str
                Spherical sampling method, could be:
                uniform: http://corysimon.github.io/articles/uniformdistn-on-sphere/
                s2cnn: the method used in the s2cnn paper, Driscoll-Healy grid, https://github.com/jonas-koehler/s2cnn
                trimeshuv: uv sphere sampling from trimesh library
                geodesic: ToDo
                fibonacci: https://stackoverflow.com/a/26127012/5966326
        Returns:

        """
        if method == 'v2_uni':
            return self._sphere_lib('uniform')
        elif method == 'v2_dh':
            return self._sphere_lib('dh')
        elif method == 'v2_soft':
            return self._sphere_lib('soft')
        elif method == 'v2_fibonacci':
            return self._sphere_lib('fibonacci')
        elif method == 's2cnn_dh':
            return self._sphere_s2cnn('Driscoll-Healy')
        elif method == 's2cnn_soft':
            return self._sphere_s2cnn('SOFT')
        elif method == 'trimesh_uv':
            return self._sphere_trimesh_uv()
        else:
            raise ValueError('Unsupported sphere sampling method {}'.format(method))

    def v2repr(self, method):
        if method == 'trimesh':
            v2repr_core = self._v2repr_trimesh
        elif method == 'e2f':
            v2repr_core = self._v2repr_e2f
        elif method == 'mt':
            v2repr_core = self._v2repr_mt
        elif method == 's2cnn':
            v2repr_core = self._v2repr_s2cnn
        else:
            raise ValueError('Unsupported V2 method: {}'.format(method))

        self._v2repr(v2repr_core)

    def load_mesh(self, mesh_path, random_rotations, fix_rotations, random_translation):
        """ Load .obj or .off mesh from the file system. random_rotations and random_translation will randomly rotate and
        translate the object.

        Args:
            mesh_path: str
                the file path for the .obj/.off mesh file
            random_rotations: bool, default True
                the default value is the same as the S2CNN repo
            fix_rotations: list, [a, z, c]
                Fix rotation, specify by rotations on a, z, c axis
            random_translation: float, default 0,1
                the default value is the same as the S2CNN repo

        Returns:

        """
        self.mesh_path = mesh_path
        self.mesh = mesh = ToMesh(random_rotations, fix_rotations, random_translation)(mesh_path)

        # An object typically has 3 dimensions, length, width, and height,
        # but some of them may only have 2, such as curtains, where one dimension will very close to 0.
        # In such case, we can't calculate the convex hull, thus we directly use the mesh itself as convex hull.
        if np.any(np.all(np.abs(self.mesh.vertices) < 1e-12, axis=0)):
            print('=== Object is a 2d object. Directly use mesh as convex hull: {} ==='.format(self.mesh_path))
            self.convex_hull = convex_hull = self.mesh
        else:
            self.convex_hull = convex_hull = self.mesh.convex_hull
        return mesh, convex_hull

    def save_v2(self, dst):
        with open(dst, 'wb+') as f:
            pickle.dump(self.v2, f)

    def plt_v2_repr(self, save_dir=''):
        fig, axes = plt.subplots(2, 3)
        reprs = [self.mesh_v2_d, self.mesh_v2_s, self.mesh_v2_c,
                 self.convh_v2_d, self.convh_v2_s, self.convh_v2_c]
        for i, ax in enumerate(axes.ravel()):
            if i == 0 or i == 3:  # Depth channel
                ax.imshow(reprs[i], vmin=0, vmax=2 * self.r, cmap='gray')
            else:  # Sine, Cosine channel
                ax.imshow(reprs[i], vmin=0, vmax=1, cmap='gray')
            ax.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])

        plt.tight_layout()

        if save_dir:
            plt.savefig(save_dir)
        else:
            plt.show()

    def plt_v2_mesh_d(self, save_dir=''):
        plt.imshow(self.mesh_v2_d, vmin=0, vmax=2 * self.r, cmap='gray')
        plt.axis('off')
        ax = plt.gca()
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        plt.tight_layout()

        if save_dir:
            plt.savefig(save_dir)
        else:
            plt.show()

    def plt_v2_config(self, convh=False, mesh=False, save_dir=''):
        ax = Axes3D(plt.figure(figsize=(10, 10)))
        cube_d = self.r
        ax.set_xlim(-cube_d, cube_d)
        ax.set_ylim(-cube_d, cube_d)
        ax.set_zlim(-cube_d, cube_d)
        ax.scatter(self.ray_origins[:, 0], self.ray_origins[:, 1], self.ray_origins[:, 2])

        mesh_v2_p = self.mesh_v2_p[~(self.mesh_v2_d == 2 * self.r).flatten()]
        ax.scatter(mesh_v2_p[:, 0], mesh_v2_p[:, 1], mesh_v2_p[:, 2])

        if convh:
            convh_v2_p = self.convh_v2_p[~(self.convh_v2_d == 2 * self.r).flatten()]
            ax.scatter(convh_v2_p[:, 0], convh_v2_p[:, 1], convh_v2_p[:, 2])

        if mesh:
            for vtx in self.mesh.triangles:
                tri = art3d.Poly3DCollection([vtx])
                tri.set_color(colors.rgb2hex(sp.rand(3)))
                tri.set_edgecolor('k')
                ax.add_collection3d(tri)

        plt.axis('off')

        if save_dir:
            plt.savefig(save_dir)
        else:
            plt.show()

    def _v2repr(self, v2repr_core):
        self.mesh_v2_d, self.mesh_v2_a, self.mesh_v2_s, self.mesh_v2_c, self.mesh_v2_p = v2repr_core(self.mesh)
        self.convh_v2_d, self.convh_v2_a, self.convh_v2_s, self.convh_v2_c, self.convh_v2_p = v2repr_core(
            self.convex_hull)
        self.v2 = np.dstack([self.mesh_v2_d, self.mesh_v2_s, self.mesh_v2_c,
                             self.convh_v2_d, self.convh_v2_s, self.convh_v2_c])

    def _v2repr_mt(self, mesh):
        """ Möller–Trumbore intersection algorithm
        Ref: https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
        Ref: https://github.com/johnnovak/raytriangle-test
        """

        def ray_triangle_intersect(r_orig, r_dire, v0, v1, v2):
            v0v1 = v1 - v0
            v0v2 = v2 - v0
            pvec = r_dire.cross(v0v2)

            det = torch.sum(v0v1 * pvec, dim=2)

            inv_det = 1.0 / det
            tvec = r_orig - v0
            u = torch.sum(tvec * pvec, dim=2) * inv_det

            qvec = tvec.cross(v0v1)
            v = torch.sum(r_dire * qvec, dim=2) * inv_det

            # No intersection conditions
            # If determinant is near zero, ray lies in plane of triangle
            # Test triangle bounds by u, v
            intersected = ~((det < 1e-12) | (u < 0) | (u > 1) | (v < 0) | (u + v > 1))

            mt_d = torch.sum(v0v2 * qvec, dim=2) * inv_det  # Ray travelling distance from every ray to every triangle
            mt_d = mt_d[intersected]

            mt_tri_i, mt_ray_i = torch.where(intersected)
            return mt_d, mt_tri_i, mt_ray_i

        triangles = torch.tensor(mesh.triangles).to(DEVICE)

        # Break large matrix into small batches to save GPU memory
        ray_size = 512
        tri_size = 16384

        tri_batches = torch.split(triangles, tri_size)
        ray_orig_batches = torch.split(self.ray_orig, ray_size)
        ray_dire_batches = torch.split(self.ray_dire, ray_size)

        all_d = []
        all_tri_i = []
        all_ray_i = []

        # Loop every batch of rays and triangles
        for i in range(len(ray_dire_batches)):
            for j in range(len(tri_batches)):
                tri = tri_batches[j]
                v0 = tri[:, 0, :]
                v1 = tri[:, 1, :]
                v2 = tri[:, 2, :]
                ray_orig = ray_orig_batches[i]
                ray_dire = ray_dire_batches[i]

                v0 = v0.repeat(len(ray_dire), 1, 1).permute(1, 0, 2)
                v1 = v1.repeat(len(ray_dire), 1, 1).permute(1, 0, 2)
                v2 = v2.repeat(len(ray_dire), 1, 1).permute(1, 0, 2)
                ray_orig = ray_orig.repeat(len(v0), 1, 1)
                ray_dire = ray_dire.repeat(len(v0), 1, 1)

                d_, tri_i, ray_i = ray_triangle_intersect(ray_orig, ray_dire, v0, v1, v2)
                tri_i += j * tri_size
                ray_i += i * ray_size
                all_d.extend(d_.tolist())
                all_tri_i.extend(tri_i.tolist())
                all_ray_i.extend(ray_i.tolist())

        # Loop every intersection candidates to find the minimal distance and corresponding triangle index
        v2_d = np.full(len(self.ray_origins), 2 * self.r, dtype=np.float32)
        v2_i = [0] * len(self.ray_orig)
        for i in range(len(all_d)):
            d = all_d[i]
            tri_i = all_tri_i[i]
            ray_i = all_ray_i[i]

            if v2_d[ray_i] > d:
                v2_d[ray_i] = d
                v2_i[ray_i] = tri_i

        v2_p = self.ray_origins + np.expand_dims(v2_d, -1) * self.ray_directions

        normals = mesh.face_normals[v2_i]

        # Force all incident angles in range [0, pi/2]
        # Also, if there is no intersection, we need to set the angle to pi/2
        v2_a = self._angle(self.ray_directions, normals)
        v2_a[v2_a > np.pi / 2] = np.pi - v2_a[v2_a > np.pi / 2]
        v2_a[np.where(v2_d == 2 * self.r)] = np.pi / 2

        v2_s = self._sin(v2_a)
        v2_c = self._cos(v2_a)

        v2_d, v2_a, v2_s, v2_c = self._reshape_v2(v2_d, v2_a, v2_s, v2_c)
        return v2_d, v2_a, v2_s, v2_c, v2_p

    def _v2repr_e2f(self, mesh):
        """ A pure pytorch vectorized implementation from scratch by Tengyu.
        The name e2f is from this CAD addon repo for blender 2.80, cause I first implemented it for blender:
        https://github.com/blender/blender-addons/tree/master/mesh_tiny_cad

        Implementation ref:
            Line-Plane intersection: https://mathworld.wolfram.com/Line-PlaneIntersection.html
            Point is in triangle: https://www.cnblogs.com/graphics/archive/2010/08/05/1793393.html
        The logical is as follows:
            1) Use point-normal form to calculate the intersection points of a plane and a ray
            2) Keep only those intersection points that are in the triangle face
            3) Keep only intersection points that is the closest to their corresponding ray origins

        Args:
            mesh: (Trimesh)

        Returns:
            v2_d: v2 representation depth channel
            v2_a: v2 representation incident angle channel
            v2_s: v2 representation sine of incident angle channel
            v2_c: v2 representation cosine of incident angle channel
        """
        from v2.data.repr_e2f import e2f_stepped

        ray_edges = torch.tensor(self.ray_edges).to(DEVICE)
        mesh_triangles = torch.tensor(mesh.triangles).to(DEVICE)
        face_interval = 16384
        edge_interval = 512

        v2_p, v2_d, v2_s, v2_c = e2f_stepped(ray_edges, mesh_triangles, face_interval, edge_interval)
        v2_a = torch.acos(v2_c)

        v2_d, v2_a, v2_s, v2_c = self._reshape_v2(v2_d, v2_a, v2_s, v2_c)
        v2_d, v2_a, v2_s, v2_c, v2_p = self._torch2np(v2_d, v2_a, v2_s, v2_c, v2_p)

        return v2_d, v2_a, v2_s, v2_c, v2_p

    def _v2repr_trimesh(self, mesh):
        """ Trimesh mesh to generate v2 representation using trimesh RayMeshIntersector class.
        The slowest method to calculate ray triangle intersection.

        Args:
            mesh: (Trimesh)

        Returns:
            v2_d: v2 representation depth channel
            v2_a: v2 representation incident angle channel
            v2_s: v2 representation sine of incident angle channel
            v2_c: v2 representation cosine of incident angle channel
        """

        intersector = RayMeshIntersector(mesh)
        inter_points, index_ray, index_tri = intersector.intersects_location(
            self.ray_origins, self.ray_directions, multiple_hits=False)
        normals = mesh.face_normals[index_tri]
        ray_directions = self.ray_directions[index_ray]

        v2_d = np.full(len(self.ray_origins), 2 * self.r, dtype=np.float32)
        v2_a = np.zeros(len(self.ray_origins))

        d_ = np.linalg.norm(self.ray_origins[index_ray] - inter_points, axis=1)
        a_ = self._angle(ray_directions, normals)
        a_[a_ > np.pi / 2] = np.pi - a_[a_ > np.pi / 2]  # Force all incident angles in range [0, pi/2]

        v2_d[index_ray] = d_
        v2_a[index_ray] = a_
        v2_s = self._sin(v2_a)
        v2_c = self._cos(v2_a)
        v2_p = inter_points

        v2_d, v2_a, v2_s, v2_c = self._reshape_v2(v2_d, v2_a, v2_s, v2_c)
        return v2_d, v2_a, v2_s, v2_c, v2_p

    def _v2repr_s2cnn(self, mesh):
        from v2.data.repr_s2cnn import ProjectOnSphere
        v2 = ProjectOnSphere(bandwidth=self.m // 2)(mesh)
        v2_d = v2[0, :, :]
        v2_c = v2[1, :, :]
        v2_s = v2[2, :, :]

        v2_a = None
        v2_p = None
        v2_d, v2_s, v2_c = self._reshape_v2(v2_d, v2_s, v2_c)
        return v2_d, v2_a, v2_s, v2_c, v2_p

    def _v2_rays(self):
        """ A function to generate views such that:
          V2 representations are generated by normalizing a given 3D object inside a sphere, and sampling points on
          the sphere to form a view plane tangent to the sphere. Then, a view plane will shoot rays parallel to the
           normal towards the object, finally reaching the object (or missing it, becoming a “background” pixel).

        Returns:
            ray_origins: ndarray, (m*n*w*h, 3)
                Origin points to shoot rays.
            ray_edges: ndarray, (m*n*w*h, 2, 3)
                Ray segments.
            ray_directions: ndarray, (m*n*w*h, 3)
                Ray directions starting at origins
        """
        def debug_plt():
            from mpl_toolkits.mplot3d import Axes3D, art3d  # noqa: F401 unused import

            ax = Axes3D(plt.figure(figsize=(10, 10)))
            cube_d = self.r
            ax.set_xlim(-cube_d, cube_d)
            ax.set_ylim(-cube_d, cube_d)
            ax.set_zlim(-cube_d, cube_d)

            ax.scatter(x0, y0, z0)
            c = 0.1
            ax.scatter(x0 + vx[0, :] * c, y0 + vx[1, :] * c, z0 + vx[2, :] * c)
            ax.scatter(x0 + vy[0, :] * c, y0 + vy[1, :] * c, z0 + vy[2, :] * c)
            # ax.scatter(x0 + vz[0, :], y0 + vz[1, :], z0 + vz[2, :])
            plt.show()

        s_d = -2 * self.s_view  # the direction vector shooting to the (0, 0, 0) with length 1, -self.s_view - self.s_view

        # The orthonormal basis on the view surface
        x0 = self.s_view[0, :]
        y0 = self.s_view[1, :]
        z0 = self.s_view[2, :]

        vx = np.vstack((-y0, x0, np.zeros(x0.shape)))  # x basis, the one parallel to the z=0 surface
        vx[:, np.logical_and(x0 == 0, y0 == 0)] = np.vstack((np.ones(x0.shape),  # polar case
                                                             np.zeros(x0.shape),
                                                             np.zeros(x0.shape)))[:, np.logical_and(x0 == 0, y0 == 0)]
        vx = vx / np.linalg.norm(vx, 2, axis=0)

        vy = np.vstack((-z0, -z0 * y0 / x0, (x0 ** 2 + y0 ** 2) / x0))  # y basis
        vy[:, x0 == 0] = np.vstack((np.zeros((x0).shape), -z0, y0))[:, x0 == 0]  # x=0 case
        vy[:, np.logical_and(x0 == 0, y0 == 0)] = np.vstack((np.zeros(x0.shape),  # polar case
                                                             np.ones(x0.shape),
                                                             np.zeros(x0.shape)))[:, np.logical_and(x0 == 0, y0 == 0)]
        vy = vy / np.linalg.norm(vy, 2, axis=0)
        # Since we are uv unwrapping along the z-axis, we need to make sure that all plane towards consistently along z-axis
        vy[:, vy[2, :] > 0] = -vy[:, vy[2, :] > 0]

        vz = -self.s_view  # z basis, the one shooting to the origin
        vz = vz / np.linalg.norm(vz, 2, axis=0)

        # The meshgrid for the new orthonormal basis
        dx = np.linspace(-(self.w // 2) * self.d, self.w // 2 * self.d, self.w)
        dy = np.linspace(-(self.h // 2) * self.d, self.h // 2 * self.d, self.h)
        dx, dy = np.meshgrid(dx, dy)
        dx = dx.flatten()
        dy = dy.flatten()

        dvx = np.tile(vx, (len(dx), 1, 1)).transpose((1, 2, 0)) * dx
        dvy = np.tile(vy, (len(dy), 1, 1)).transpose((1, 2, 0)) * dy

        # Generate all view
        all_view = np.tile(self.s_view, (len(dx), 1, 1)).transpose((1, 2, 0)) + dvx + dvy
        all_ray = all_view + np.tile(s_d, (len(dx), 1, 1)).transpose((1, 2, 0))

        all_view = all_view.reshape((all_view.shape[0], all_view.shape[1] // self.n, self.n, self.h, self.w))
        all_view = all_view.transpose(0, 1, 3, 2, 4)
        all_ray = all_ray.reshape((all_ray.shape[0], all_ray.shape[1] // self.n, self.n, self.h, self.w))
        all_ray = all_ray.transpose(0, 1, 3, 2, 4)

        all_view = all_view.reshape((3, -1))
        all_ray = all_ray.reshape((3, -1))

        all_ray = np.dstack((all_view, all_ray)).transpose((1, 2, 0))

        ray_origins = all_view.T
        ray_edges = all_ray
        ray_directions = all_ray[:, 1, :] - all_ray[:, 0, :]
        return ray_origins, ray_edges, self._unit_vector(ray_directions)

    def _sphere_lib(self, method):
        """ Generate points on a sphere based on the number of sampled points and the radius of the sphere.
        Driscoll Healy and SOFT are reimplemented from the S2CNN paper
        Cohen, Taco S., et al. "Spherical cnns." arXiv preprint arXiv:1801.10130 (2018).
        https://arxiv.org/abs/1801.10130
        The original implementation can be found here:
        https://github.com/AMLab-Amsterdam/lie_learn
        lie_learn/spaces/S2.py - meshgrid(b, grid_type='Driscoll-Healy')

        Uniform is based on http://corysimon.github.io/articles/uniformdistn-on-sphere/, to avoid polar clustering

        Fibonacci is for uniformly packing points on the sphere:
        The Fibonacci sphere algorithm: https://stackoverflow.com/a/26127012/5966326
        """
        def dh():  # Driscoll Healy
            phi_ = np.linspace(0, np.pi, self.m, endpoint=False)
            return phi_

        def soft():  # SOFT
            start = 1 / (2 * self.m)
            end = 1 - start
            phi_ = np.linspace(np.pi * start, np.pi * end, self.m)
            return phi_

        def fibonacci(samples):
            s_view_ = []
            phi_ = math.pi * (3. - math.sqrt(5.))  # golden angle in radians

            if samples == 1:
                s_view_.append((0, 1, 0))
                return s_view_

            for i in range(samples):
                fi_y = 1 - (i / float(samples - 1)) * 2  # fi_y goes from 1 to -1
                radius = math.sqrt(1 - fi_y * fi_y)  # radius at fi_y

                fi_theta_ = phi_ * i  # golden angle increment

                fi_x = math.cos(fi_theta_) * radius
                fi_z = math.sin(fi_theta_) * radius

                s_view_.append((fi_x, fi_y, fi_z))
            return s_view_

        def uniform():  # Uniform
            if not self.polar:
                phi_ = np.linspace(0, 1, self.m + 2)  # +2 because we will remove both polar points later
                phi_ = phi_[~np.logical_or(phi_ == 0, phi_ == 1)]
            else:
                phi_ = np.linspace(0, 1, self.m)
            phi_ = np.arccos(1 - 2 * phi_)
            return phi_

        if method == 'fibonacci':
            s_view = fibonacci(self.m * self.n)
            s_view = np.array(s_view).T
            return s_view

        theta = np.linspace(0, 2 * np.pi, self.n, endpoint=False)

        if method == 'dh':
            phi = dh()
        elif method == 'soft':
            phi = soft()
        elif method == 'uniform':
            phi = uniform()
        else:
            raise ValueError('Invalid method {} for sphere sampling.'.format(method))

        theta, phi = np.meshgrid(theta, phi)

        theta = theta.flatten()
        phi = phi.flatten()

        # When phi is 0 or pi, the point is on the z-axis, and therefore x, y should be zero
        theta[np.logical_or(phi == 0, phi == np.pi)] = 0

        x = self.r * self._sin(phi) * self._cos(theta)
        y = self.r * self._sin(phi) * self._sin(theta)
        z = self.r * self._cos(phi)
        s_view = np.vstack((x, y, z))
        return s_view

    def _sphere_s2cnn(self, method):
        from v2.data.repr_s2cnn import make_sgrid
        bandwidth = self.m // 2
        return make_sgrid(bandwidth, alpha=0, beta=0, gamma=0, grid_type=method).T

    def _sphere_trimesh_uv(self):
        return np.array(creation.uv_sphere(radius=self.r, count=[self.m, (self.n + 1) // 2]).vertices).T

    def _reshape_v2(self, *args):
        return list(map(lambda x: x.reshape((self.m * self.h, self.n * self.w)), args))

    def _angle(self, v1, v2):
        """ Returns the angle in radians between vectors v1 and v2
        """
        v1_u = self._unit_vector(v1)
        v2_u = self._unit_vector(v2)
        return np.arccos(np.clip(np.sum(v1_u * v2_u, axis=1), -1.0, 1.0))

    @staticmethod
    def _torch2np(*args):
        return list(map(lambda x: x.cpu().numpy(), args))

    @staticmethod
    def _np2torch(*args):
        return list(map(lambda x: torch.tensor(x).to(DEVICE), args))

    @staticmethod
    def _unit_vector(vector):
        # Returns the unit vector of the vector
        return vector / np.linalg.norm(vector, axis=1, keepdims=True)

    @staticmethod
    def _sin(x):
        # To correct np.sin(np.pi) isn't equal to 0
        sin = np.sin(x)
        sin[np.abs(sin) < 1e-14] = 0
        return sin

    @staticmethod
    def _cos(x):
        # To correct np.cos(np.pi/2) isn't equal to 0
        cos = np.cos(x)
        cos[np.abs(cos) < 1e-14] = 0
        return cos


