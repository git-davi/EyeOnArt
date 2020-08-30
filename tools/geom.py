import numpy as np


def segmented_intersections(horz,verz):
    """Finds the intersections between groups of lines."""

    intersections = []
    for i in range(len(horz)):
        for j in range(len(verz)):
            intersections.append(_intersection(horz[i],verz[j]))
    return intersections


def _intersection(line1, line2):
    """Finds the intersection of two lines given in Hesse normal form."""
    rho1, theta1 = line1
    rho2, theta2 = line2
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    x0, y0 = int(x0), int(y0)
    return [x0, y0]


def feature_scaling(features) :
    _min = np.amin(features)
    _max = np.amax(features)

    if _max == _min :
        return features, _min, _max

    features = ( features - _min ) / ( _max - _min )
    return features, _min, _max


def feature_descaling(features, _min, _max) :   
    return (features * (_max - _min)) + _min


def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	# return the ordered coordinates
	return rect


def get_vertices(points):
    points = np.array(points)

    # order by vertical position
    points = points[points[:, :, 1][:, 0].argsort()]
    
    top = points[:2]
    bottom = points[-2:]

    # order horizontal pos
    top = top[top[:, :, 0][:, 0].argsort()]
    bottom = bottom[bottom[:, :, 0][:, 0].argsort()]

    tl = top[0, 0]
    tr = top[1, 0]
    bl = bottom[0, 0]
    br = bottom[1, 0]

    return np.array([tl, tr, br, bl])


def seg_len(p1, p2):
    x = p2[0] - p1[0]
    y = p2[1] - p1[1]
    
    return np.sqrt(x*x + y*y)


def rectify(points):
    # tl tr br bl order
    width = max(int(seg_len(points[0], points[1])), int(seg_len(points[2], points[3])))
    height = max(int(seg_len(points[0], points[3])), int(seg_len(points[1], points[2])))

    tl = [0, 0]
    tr = [width - 1, 0]
    br = [width - 1, height - 1]
    bl = [0, height - 1]

    return np.array([tl, tr, br, bl])