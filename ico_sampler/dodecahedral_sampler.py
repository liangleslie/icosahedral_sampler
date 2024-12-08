import cv2
import numpy as np
from icosahedral_sampler.ico_sampler import utils
from scipy.spatial.transform import Rotation as R

# ######################################################################################################################
#                                               DODECAHEDRAL SAMPLER
# ######################################################################################################################
class DodecahedralSampler:
    def __init__(self, resolution: int = 400):
        """
        Create unwrapped dodecahedral from equirectangular images. This class creates a 3D dodecahedron as internal
        representation and is using it to sample the colors from an equirectangular image.

        Advantages of the representation:
            - faces have very little distorsion
            - can be subdivided (like a mesh) to creates faces (not yet in this repo :( )

        Arguments:s
              resolution: pixel resolution of a face

        References:
            - https://github.com/rdbch/icosahedral_sampler
            - http://www.paulbourke.net/panorama/icosahedral/
            - https://en.wikipedia.org/wiki/Regular_dodecahedron
            - https://mathworld.wolfram.com/RegularDodecahedron.html
        """

        # unit sphere
        radius = 1.0
        self.unit_edge_length = (np.sqrt(5) - 1) / np.sqrt(3) * radius

        self.resolution = resolution

        self.faces = np.array([
            [0,1,2,3,4,],                                                                               # top
            [0,5,11,6,1,],[1,6,10,7,2,],[2,7,14,8,3,],[3,8,13,9,4,],[4,9,12,5,0,],                      # upper hemisphere
            [15,10,7,14,19,],[19,14,8,13,18,],[18,13,9,12,17,],[17,12,5,11,16,],[16,11,6,10,15,],       # lower hemisphere
            [19,18,17,16,15,]                                                                           # bottom
        ])

        # face_map = [                              # maps each face to its neighbours
        #     [0,1],[0,2],[0,3],[0,4],[0,5],
        #     [1,0],[1,2],[1,5],[1,9],[1,10],
        #     [2,0],[2,3],[2,1],[2,10],[2,6],
        #     [3,0],[3,4],[3,2],[3,6],[3,7],
        #     [4,0],[4,5],[4,3],[4,7],[4,8],
        #     [5,0],[5,1],[5,4],[5,8],[5,9],
        #     [2, 6],[3, 6],[7, 6],[10, 6],[11, 6],
        #     [3, 7],[4, 7],[8, 7],[6, 7],[11, 7],
        #     [4, 8],[5, 8],[9, 8],[7, 8],[11, 8],
        #     [5, 9],[1, 9],[10, 9],[8, 9],[11, 9],
        #     [1,10],[2,10],[6,10],[9,10],[11,10],
        #     [10,11],[9,11],[8,11],[7,11],[6,11],
        # ]

        self.vertices = self.get_vertices(radius)

    # =============================================== EDGE LENGTH ======================================================
    @property
    def edge_length(self) -> float:
        """
        Compute the icosahedron edge length in 3D (XYZ). Assumes that all edges have the same length.

        Returns:
            edge length (scalar)
        """
        return np.sqrt(np.sum((self.vertices[0] - self.vertices[1]) ** 2))

    # =============================================== GET ORIENTATION ======================================================
    def get_is_up(self,face_no) -> bool:
        """
        Get whether face is an upright pentagon or not.

        Returns:
            True if face is upright pentagon, false otherwise
        """
        return True if (0<face_no<6) | (face_no == 11) else False

    # =============================================== GET ROTATIONAL OFFSET ======================================================
    def get_rotation(self,face_no) -> float:
        """
        Get rotational offset of the face.

        Returns:
            Rotation offset in radians
        """
        if (face_no == 0) | (face_no == 11) :
             rotation_offset = 0
        elif 0 < face_no < 6:
             rotation_offset = np.pi * 6 / 5 + (face_no-1) * (np.pi * 2 / 5)
        elif 6 <= face_no < 11:
             rotation_offset = -np.pi * 4 / 5 - (face_no-6) * (np.pi * 2 / 5)
        return rotation_offset

    # =============================================== GET VERTICES =====================================================
    def get_vertices(self, radius: float = 1.0) -> np.ndarray:
        """
        Return the list of vertices in 3D for the regular dodecahedron. The dodecahedron has 20 vertices
        and it has been chosen to be aligned norm-south (aka the north and south pole are vertices).

        References:
        - https://en.wikipedia.org/wiki/Regular_dodecahedron - see Spherical Coordinates section

        Args:
            radius: radius of the circumscribed sphere (default: 1.0)

        Returns:
            list of 20 3D vertices (having the length=radius)
        """

        vertices = []

        # for first vertex - solve for y0 = -y4 , which is the absolute y-coordinate of top and bottom face
        top_circumradius = self.unit_edge_length / (2 * np.sin(np.pi / 5))
        y0 = (radius ** 2 - top_circumradius ** 2) ** 0.5
        z0 = (radius ** 2 - y0 ** 2) ** 0.5
        vertices.append(np.array([0, y0, z0]))

        # for vertices 1-4, simply rotate first vertex by (2 * pi / 5) or 72 deg
        for i in range(1, 5):
            vertices.append(utils.rotate_on_axis(vertices[0], "y", i * np.pi * 2 / 5))

        # solve for y5, which is the absolute y-coordinate of the middle vertices
        _ratio = (np.cos(np.pi * 3 / 10) + np.sin(np.pi * 2 / 5)) / (np.sin(np.pi * 2 / 5))
        y5 = (_ratio - 1) / (_ratio + 1) * y0
        z5 = (radius - y5 ** 2) ** 0.5
        vertices.append(np.array([0, y5, z5]))

        # for vertices 6-9, simply rotate first vertex by (2 * pi / 5) or 72 deg
        for i in range(1, 5):
            vertices.append(utils.rotate_on_axis(vertices[5], "y", i * np.pi * 2 / 5))

        # flip northern hemisphere to form southern hemisphere
        for i in range(9,-1,-1):
            vertices.append(np.array([-vertices[i][0], -vertices[i][1], -vertices[i][2]]))

        vertices = np.array(vertices) # shape [12, 3]
        vertices /= np.linalg.norm(vertices, axis=-1, keepdims=True)
        vertices *= radius

        return vertices

    # =============================================== GET PENTAGON COORDS ==============================================
    def get_pentagon_coords(self,
                              base_resolution: int,
                              is_up: bool,
                              center: bool = True,
                              normalize: bool = True,
                              homogeneous: bool = False ) -> np.ndarray:
        """
        Utility function that returns the coordinates of a regular pentagon that is drawn in a rectangular
        image. The pentagon has 10 possible positions: 5 orientations for an upright pentagon (when flat edge is at the bottom), and 5 for an upside down pentagon (flat edge at the top).

        Args:
            base_resolution: edge length in pixels
            is_up: the pentagon is facing up or down
            center: move the origin to be in the pentagon's center of weight
            normalize: return normalized coordinates in interval [0, 1]
            homogeneous: return homogeneous points (add 1s on the last dimension)

        Returns:
            xy coordinates of the points lying inside the pentagon
        """
        scaled_edge_length = base_resolution # set base resolution as edge_length

        # consider a mini right-angle triangle with height=h, length=l, and hypothenuse=edge_length, and angle = 72deg
        h = scaled_edge_length * np.sin(np.pi / 5 * 2)
        l = scaled_edge_length * np.cos(np.pi / 5 * 2)
        pentagon_l = int(2 * l + scaled_edge_length)

        height_to_centre = scaled_edge_length / (2 * np.tan(np.pi / 5))
        pentagon_h = int(height_to_centre + scaled_edge_length / (2 * np.sin(np.pi / 5)))

        # # define triangles in pentagon // not needed, keeping as backup
        # up_pentagon_triangles = np.array([
        #             [[x/2, height_to_centre],[l,0],[x-l,0],],
        #             [[x/2, height_to_centre],[0,h],[l,0],],
        #             [[x/2, height_to_centre],[x/2, y],[0,h],],
        #             [[x/2, height_to_centre],[x,h],[x/2, y],],
        #             [[x/2, height_to_centre],[x-l,0],[x,h],],
        #         ])
        # down_pentagon_triangles = y-up_pentagon_triangles
        # # triangle contains 2 arrays for upright and upside down pentagons, each with 5 vertices, each described by 3 coordinates
        # triangle = np.array([up_pentagon_triangles,down_pentagon_triangles])
        # triangle = triangle.astype(np.int32)

        # define points of the pentagon
        pentagon = np.array([
            [[l,0],[pentagon_l-l,0],[pentagon_l,h],[pentagon_l/2, pentagon_h],[0,h]], # up pentagon
            [[l,pentagon_h],[pentagon_l-l,pentagon_h],[pentagon_l,pentagon_h-h],[pentagon_l/2, 0],[0,pentagon_h-h]], # down pentagon
        ])
        pentagon = pentagon.astype(np.int32)

        # rasterize pentagon (could also be done with analytically, but this is way more elegant)
        canvas = np.zeros([pentagon_h, pentagon_l], dtype=np.uint8)
        canvas = cv2.drawContours(canvas, [*pentagon], int(is_up), color=1, thickness=-1)
        coords = np.argwhere(canvas == 1)[:, ::-1]

        # center coordinates in weight center
        if center:
            coords[..., 0] -= pentagon_l // 2
            if is_up:
                coords[..., 1] -= int(pentagon_h - height_to_centre)
            else:
                coords[..., 1] -= int(height_to_centre)

        # normalize coordinates in interval [0, 1]
        if normalize:
            coords = coords / pentagon_l

        # add homogeneous axis
        if homogeneous:
            ones = np.ones_like(coords[:, 0, None])
            coords = np.concatenate([coords, ones], axis=-1)

        return coords #[N, 2] or [N,3] if homogeneous = True

    # =============================================== GET FACE XYZ =====================================================
    def get_face_xyz(self, face_no: int) -> np.ndarray:
        """
        Method that generates the xyz coordinates of a face.
        These points can be later used to be projected onto the sphere and sample the color from the equirectangular image texture.

        Arguments:
            face_no: face number (0-11)
            res: resolution of the face (number of points of the base)

        Returns:
            coordinates in 3D of a given face of the dodecahedron
        """
        vertex_xyz = self.vertices[self.faces[face_no]]

        # get face center in XYZ
        center = vertex_xyz.mean(axis=0)
        norm   = np.linalg.norm(center)
        norm_center = center / norm


        # generate regular pentagon and scale to edge length
        xyz = self.get_pentagon_coords(self.resolution, self.get_is_up(face_no), normalize=True, homogeneous=True, center=True)

        l = self.resolution * np.cos(np.pi / 5 * 2)
        xy_scale_factor = 2 * l + self.resolution
        xyz[:, :2] *= xy_scale_factor # scale to length of x

        top_circumradius = self.resolution / (2 * np.sin(np.pi / 5))
        sphere_radius = self.resolution / ((np.sqrt(5) - 1) / np.sqrt(3))
        z_scale_factor = (sphere_radius ** 2 - top_circumradius ** 2) ** 0.5
        xyz[:, 2] *= z_scale_factor # scaled to depth

        # rotate triangle to
        phi, theta = utils.xyz_2_polar(norm_center)
        z_rotation_offset = self.get_rotation(face_no)

        pentagon_xyz = xyz @ R.from_euler('yxz', [-phi, theta, z_rotation_offset]).as_matrix()

        return pentagon_xyz

    # =============================================== GET FACE RGB =====================================================
    def get_face_rgb(self, face_no, eq_image):
        """
        Utility method that uses the gnomonic projection to get rgb colors of a face given an equirectangular image.

        Arguments:
            face_no: face number to be returned
            eq_image: equirectangular image

        Returns:
            color sampled from equirectangular images [N, 3]
        """

        utils.check_eq_image_shape(eq_image)
        xyz = self.get_face_xyz(face_no)

        # raycast on sphere
        ray_xyz = xyz /  np.linalg.norm(xyz, axis=1, keepdims=True)

        # rotate to face center
        phi, theta = utils.xyz_2_polar(ray_xyz)
        x, y = utils.polar_2_equi(phi, theta, eq_image.shape)

        rgb = eq_image[y.astype(int), x.astype(int)]

        #TODO add interpolation
        return rgb

    # =============================================== GET FACE IMAGE ===================================================
    def get_face_image(self, face_no, eq_image):
        """
        Project the plane of a face on the sphere and sample the colors.

        Arguments:
            face_no: face number
            eq_image: equirectangular image to sample from

        Returns:
            RGB image of the face
        """

        utils.check_eq_image_shape(eq_image)
        colors = self.get_face_rgb(face_no, eq_image)

        # skew matrix build
        vertex_xyz = self.vertices[self.faces[face_no]]
        xy = self.get_pentagon_coords(self.resolution, self.get_is_up(face_no), normalize=False, homogeneous=False, center=False)

        # consider a mini right-angle triangle with height=h, length=l, and hypothenuse=edge_length, and angle = 72deg
        l = self.resolution * np.cos(np.pi / 5 * 2)
        pentagon_l = int(2 * l + self.resolution)
        height_to_centre = self.resolution / (2 * np.tan(np.pi / 5))
        pentagon_h = int(height_to_centre + self.resolution / (2 * np.sin(np.pi / 5)))

        canvas = np.zeros([pentagon_h, pentagon_l, 4], dtype=np.uint8)
        canvas[xy[:, 1], xy[:, 0],2::-1] = colors
        canvas[xy[:, 1], xy[:, 0], 3] = 255  # Set alpha to opaque where color is present

        return canvas

    # =============================================== UNWRAP ===========================================================
    def unwrap(self, eq_image):
        """
        Project an equirectangular image onto an dodecahedron and unwrapped it onta a plane surface. The resolution of
        the output images will be computed based on the resolution provided at the creation of the object.

        Arguments:
            eq_image: equirectangular image to be samples from

        Returns:
            unwrapped dodecahedron with colors sampled from the equirectangular image.
        """

        # input check
        utils.check_eq_image_shape(eq_image)

        # TODO: implement offset via get_face_xyz; apply (x,y,z) rotation on full image
        colors = [self.get_face_rgb(i, eq_image) for i in range(12)]

        scaled_edge_length = self.resolution # set base resolution as edge_length
        # consider a mini right-angle triangle with height=h, length=l, and hypothenuse=edge_length, and angle = 72deg
        h = scaled_edge_length * np.sin(np.pi / 5 * 2)
        l = scaled_edge_length * np.cos(np.pi / 5 * 2)
        pentagon_l = int(2 * l + scaled_edge_length)
        height_to_centre = scaled_edge_length / (2 * np.tan(np.pi / 5))
        pentagon_h = int(height_to_centre + scaled_edge_length / (2 * np.sin(np.pi / 5)))

        canvas_length = int(3*(pentagon_l+self.resolution)+l)
        canvas_height = int(2*pentagon_h+h)

        canvas = np.zeros([canvas_height, canvas_length, 4], dtype=np.uint8)

        # coordinates for moving the color from faces to canvas
        xy_up   = self.get_pentagon_coords(self.resolution, True, normalize=False, homogeneous=False, center=False)
        xy_down = self.get_pentagon_coords(self.resolution, False, normalize=False, homogeneous=False, center=False)

        # move colors from faces to canvas
        # top face, i.e face 0
        face_0_x_offset = int(pentagon_l-l)
        face_0_y_offset = int(pentagon_h)
        canvas[xy_down[..., 1] + face_0_y_offset, xy_down[..., 0] + face_0_x_offset, 2::-1] = colors[0]
        canvas[xy_down[..., 1] + face_0_y_offset, xy_down[..., 0] + face_0_x_offset, 3] = 255

        # upper faces, i.e. faces 1-5
        for num in range(5):
            face = 1+num
            rotational_offset = -np.pi / 5 - num * (np.pi * 2 / 5)
            upper_x_offset = face_0_x_offset + 2*height_to_centre*np.sin(rotational_offset)
            upper_y_center_offset = height_to_centre - scaled_edge_length / (2 * np.sin(np.pi / 5))
            upper_y_offset = face_0_y_offset + upper_y_center_offset + 2*height_to_centre*np.cos(rotational_offset)
            canvas[xy_up[..., 1] + int(upper_y_offset), xy_up[..., 0] + int(upper_x_offset), 2::-1] = colors[face]
            canvas[xy_up[..., 1] + int(upper_y_offset), xy_up[..., 0] + int(upper_x_offset), 3] = 255

        # bottom face, i.e. face 11
        face_11_x_offset = int(2*(pentagon_l+self.resolution))
        face_11_y_offset = int(h)
        canvas[xy_up[..., 1] + face_11_y_offset, xy_up[..., 0] + face_11_x_offset, 2::-1] = colors[11]
        canvas[xy_up[..., 1] + face_11_y_offset, xy_up[..., 0] + face_11_x_offset, 3] = 255

        # upper faces, i.e. faces 6-10
        for num in range(5):
            face = 6+num
            rotational_offset = np.pi * 1 / 5 - num * (np.pi * 2 / 5)
            lower_x_offset = face_11_x_offset + 2*height_to_centre*np.sin(rotational_offset)
            lower_y_center_offset = height_to_centre - scaled_edge_length / (2 * np.sin(np.pi / 5))
            lower_y_offset = face_11_y_offset - lower_y_center_offset - 2*height_to_centre*np.cos(rotational_offset)
            canvas[xy_down[..., 1] + int(lower_y_offset), xy_down[..., 0] + int(lower_x_offset), 2::-1] = colors[face]
            canvas[xy_down[..., 1] + int(lower_y_offset), xy_down[..., 0] + int(lower_x_offset), 3] = 255

        return canvas
