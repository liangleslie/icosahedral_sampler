import cv2
import numpy as np
import math
from . import utils
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

        self.resolution = resolution

        # unit sphere
        self.unit_edge_length = (np.sqrt(5) - 1) / np.sqrt(3) * self.radius
        
        self.faces = np.array([
            [0,1,2,3,4,],                                                                               # top
            [0,5,11,6,1,],[1,6,10,7,2,],[2,7,14,8,3,],[3,8,13,9,4,],[4,9,12,5,0,],                      # upper hemisphere
            [15,10,7,14,19,],[19,14,8,13,18,],[18,13,9,12,17,],[17,12,5,11,16,],[16,11,6,10,15,],       # lower hemisphere
            [19,18,17,16,15,]                                                                           # bottom
        ])

        self.vertices = self.get_vertices(radius)
        self.centres = self.get_centres(radius)

    # =============================================== EDGE LENGTH ======================================================
    @property
    def edge_length(self) -> float:
        """
        Compute the icosahedron edge length in 3D (XYZ). Assumes that all edges have the same length.

        Returns:
            edge length (scalar)
        """
        return self.unit_edge_length * self.resolution

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
        edge_length = self.unit_edge_length
        # for first vertex - solve for y0 = -y4 , which is the absolute y-coordinate of top and bottom face
        top_circumradius = edge_length / (2 * math.sin(math.pi / 5))
        y0 = (1 - top_circumradius ** 2) ** 0.5
        z0 = (1 - y0 ** 2) ** 0.5
        vertices.append(np.array([0, y0, z0]))

        # for vertices 1-4, simply rotate first vertex by (2 * pi / 5) or 72 deg
        for i in range(1, 5):
            vertices.append(utils.rotate_on_axis(vertices[0], "y", i * math.pi * 2 / 5))

        # solve for y5, which is the absolute y-coordinate of the middle vertices
        _ratio = (math.cos(math.radians(54)) + math.sin(math.radians(72))) / (math.sin(math.radians(72)))
        y5 = (_ratio - 1) / (_ratio + 1) * y0
        z5 = (1 - y5 ** 2) ** 0.5
        vertices.append(np.array([0, y5, z5]))

        # for vertices 6-9, simply rotate first vertex by (2 * pi / 5) or 72 deg
        for i in range(1, 5):
            vertices.append(utils.rotate_on_axis(vertices[5], "y", i * math.pi * 2 / 5))

        # flip northern hemisphere to form southern hemisphere
        for i in range(9,-1,-1):
            vertices.append(np.array([-vertices[i][0], -vertices[i][1], -vertices[i][2]]))

        vertices = np.array(vertices) # shape [12, 3]
        vertices /= np.linalg.norm(vertices, axis=-1, keepdims=True)
        vertices *= radius

        return vertices

    # =============================================== GET CENTRES =====================================================
    def get_centres(self, radius: float = 1.0) -> np.ndarray:
        """
        Return the list of centres in 3D for the regular dodecahedron. The iscosahedron has 12 faces
        and it has been chosen to be aligned norm-south (aka the north and south pole are vertices).

        References:
        - https://en.wikipedia.org/wiki/Regular_dodecahedron - see Spherical Coordinates section

        Args:
            radius: radius of the circumscribed sphere (default: 1.0)

        Returns:
            list of 12 3D centres
        """
        centres = []
        for face in self.faces:
            face_vertices = [self.vertices[vertex] for vertex in face]
            face_centre = np.mean(face_vertices, axis=0)
            centres.append(face_centre)
        
        centres = np.array(centres)
        return centres

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
        h = scaled_edge_length * math.sin(math.pi / 5 * 2)
        l = scaled_edge_length * math.cos(math.pi / 5 * 2)
        x = int(2 * l + scaled_edge_length)

        triangle_height = scaled_edge_length / (2 * math.tan(math.pi / 5))
        y = int(triangle_height + scaled_edge_length / (2 * math.sin(math.pi / 5)))
        
        # # define triangles in pentagon // not needed, keeping as backup
        # up_pentagon_triangles = np.array([
        #             [[x/2, triangle_height],[l,0],[x-l,0],],
        #             [[x/2, triangle_height],[0,h],[l,0],],
        #             [[x/2, triangle_height],[x/2, y],[0,h],],
        #             [[x/2, triangle_height],[x,h],[x/2, y],],
        #             [[x/2, triangle_height],[x-l,0],[x,h],],
        #         ])
        # down_pentagon_triangles = y-up_pentagon_triangles
        # # triangle contains 2 arrays for upright and upside down pentagons, each with 5 vertices, each described by 3 coordinates
        # triangle = np.array([up_pentagon_triangles,down_pentagon_triangles])
        # triangle = triangle.astype(np.int32)

        # define points of the pentagon
        pentagon = np.array([
            [[l,0],[x-l,0],[x,h],[x/2, y],[0,h]], # up pentagon
            [[l,y],[x-l,y],[x,y-h],[x/2, 0],[0,y-h]], # down pentagon
        ])
        pentagon = pentagon.astype(np.int32)

        # rasterize pentagon (could also be done with analytically, but this is way more elegant)
        canvas = np.zeros([y, x], dtype=np.uint8)
        canvas = cv2.drawContours(canvas, [*pentagon], int(is_up), color=1, thickness=-1)
        coords = np.argwhere(canvas == 1)[:, ::-1]

        # center coordinates in weight center
        if center:
            coords[..., 0] -= x // 2
            if is_up:
                coords[..., 1] -= int(triangle_height)
            else:
                coords[..., 1] -= int(y - triangle_height)

        # normalize coordinates in interval [0, 1]
        if normalize:
            coords = coords / x

        # add homogeneous axis
        if homogeneous:
            ones = np.ones_like(coords[:, 0, None])
            coords = np.concatenate([coords, ones], axis=-1)

        return coords #[N, 2]

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
        triangle_map = [[0,1], [1,2], [2,3], [3,4], [4,0]]

        vertex_xyz = self.vertices[self.faces[face_no]]

        # get face center in XYZ
        center = vertex_xyz.mean(axis=0)
        norm   = np.linalg.norm(center)
        center = center / norm

        # generate regular pentagon and scale to edge length
        is_up = True if (0<face_no<6) | (face_no == 11) else False
        xyz = self.get_pentagon_coords(self.resolution, is_up, normalize=True, homogeneous=True, center=True)
        xyz[:, :2] *= self.resolution / self.unit_edge_length  # scale to edge length
        xyz[:, 2]  *= norm

        # rotate triangle to
        phi, theta = utils.xyz_2_polar(center)
        pentagon_xyz = xyz @ R.from_euler('yx', [-phi, theta]).as_matrix()

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

        #TODO add interpolation
        return eq_image[y.astype(int), x.astype(int)]

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
        is_up = True if (0<face_no<6) | (face_no == 11) else False
        xy = self.get_pentagon_coords(self.resolution, is_up, normalize=False, homogeneous=False, center=False)

        # consider a mini right-angle triangle with height=h, length=l, and hypothenuse=edge_length, and angle = 72deg
        l = self.resolution * math.cos(math.pi / 5 * 2)
        x = int(2 * l + self.resolution)
        triangle_height = self.resolution / (2 * math.tan(math.pi / 5))
        y = int(triangle_height + self.resolution / (2 * math.sin(math.pi / 5)))

        canvas = np.zeros([y, x, 4], dtype=np.uint8)
        canvas[xy[:, 1], xy[:, 0],2::-1] = colors
        canvas[xy[:, 1], xy[:, 0], 3] = 255  # Set alpha to opaque where color is present

        return canvas

    # =============================================== UNWRAP ===========================================================
    def unwrap(self, eq_image, face_offset=0):
        """
        Project an equirectangular image onto an dodecahedron and unwrapped it onta a plane surface. The resolution of
        the output images will be computed based on the resolution provided at the creation of the object.

        Arguments:
            eq_image: equirectangular image to be samples from
            face_offset: offset faces when creating the unwrapped image [-2, 2] (default: 0)

        Returns:
            unwrapped dodecahedron with colors sampled from the equirectangular image.
        """

        # input check
        utils.check_eq_image_shape(eq_image)
        assert -2 <= face_offset <= 2, f'The face offset should be in the interval [-2, 2]. Current: {face_offset}'

        colors = [self.get_face_rgb(i, eq_image) for i in range(20)]

        scaled_edge_length = self.resolution # set base resolution as edge_length
        # consider a mini right-angle triangle with height=h, length=l, and hypothenuse=edge_length, and angle = 72deg
        h = scaled_edge_length * math.sin(math.pi / 5 * 2)
        l = scaled_edge_length * math.cos(math.pi / 5 * 2)
        x = int(2 * l + scaled_edge_length)
        triangle_height = scaled_edge_length / (2 * math.tan(math.pi / 5))
        y = int(triangle_height + scaled_edge_length / (2 * math.sin(math.pi / 5)))

        canvas = np.ones([int(2*y+l), int(3*(x+self.resolution)+l), 4], dtype=np.uint8)*255

        ## Still WIP from here on ##
        # # coordinates for moving the color from faces to canvas
        # xy_up   = self.get_pentagon_coords(self.resolution, True, normalize=False, homogeneous=False, center=False)
        # xy_down = self.get_pentagon_coords(self.resolution, False, normalize=False, homogeneous=False, center=False)

        # loc_generator = [[l[0], (face_offset + 2 + l[1]) % 5] for l in enumerate(range(5))]

        # # move colors from faces to canvas
        # for num, loc in loc_generator:
        #     canvas[xy_up[..., 1],  int((loc+0.5)*self.resolution)+xy_up[..., 0]] = colors[num]
        # for num, loc in loc_generator:
        #     canvas[h_res+xy_down[..., 1], int((loc+0.5)*self.resolution)+xy_down[..., 0]] = colors[5+num]
        # for num, loc in loc_generator:
        #     canvas[h_res+xy_up[..., 1], loc*self.resolution+xy_up[..., 0]] = colors[10+num]
        # for num, loc in loc_generator:
        #     canvas[2*h_res+xy_down[..., 1], loc*self.resolution+xy_down[..., 0]] = colors[15+num]

        return canvas
