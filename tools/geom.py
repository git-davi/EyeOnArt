import numpy as np


def segmented_intersections(horz,verz):
    """Finds the intersections between groups of lines."""

    intersections = []
    for i in range(len(horz)):
        for j in range(len(verz)):
            intersections.append(_intersection(horz[i],verz[j]))
    return np.array(intersections)


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


def order_points(points):
    # order points : tl tr br bl
    points = np.array(points)

    points = points[points[:, 0].argsort()]
    left = points[0:2]
    right = points[2:4]

    left = left[left[:, 1].argsort()]
    right = right[right[:, 1].argsort()]

    tl = left[0]
    bl = left[1]
    tr = right[0]
    br = right[1]

    return np.array([tl, tr, br, bl])


def order_lines(lines) :
    abs_cos = np.absolute(np.cos(lines[:, 1]))
    h_lines = lines[abs_cos.argsort()][:2]
    v_lines = lines[abs_cos.argsort()][2:]

    top, bottom = h_lines[h_lines[:, 0].argsort()]
    left, right = v_lines[v_lines[:, 0].argsort()]

    return np.array([ top, right, bottom, left ])

def extract_borders(lines) :
    # divide horizontal and vertical lines
    abs_cos = np.absolute(np.cos(lines[:, :, 1])).flatten()
    v_lines = lines[abs_cos > 0.5]
    h_lines = lines[abs_cos <= 0.5]

    h_sort_lines = h_lines[h_lines[:, :, 0].flatten().argsort()]
    v_sort_lines = v_lines[v_lines[:, :, 0].flatten().argsort()]

    if h_sort_lines.size == 0 or v_sort_lines.size == 0 :
        return None

    top = h_sort_lines[0]
    bottom = h_sort_lines[-1]
    left = v_sort_lines[0]
    right = v_sort_lines[-1] 

    borders = np.reshape(np.array([ top, right, bottom, left ]), (4, 2))

    return borders

def seg_len(p1, p2):
    x = p2[0] - p1[0]
    y = p2[1] - p1[1]
    
    return np.sqrt(x*x + y*y)


def rectify_points(points, lines) :
    tl, tr, br, bl = points
    t, r, b, l = lines

    new_tl, new_tr, new_br, new_bl = [None, None, None, None]

    # get new_tl and new_tr
    if tl[1] < tr[1] :
        border = seg_len(tl, tr)
        scale = np.abs(np.cos(t[1] - (np.pi/2)))

        new_x = tl[0] + border/scale

        new_tl = tl
        new_tr = [new_x, tl[1]]
    else :
        border = seg_len(tl, tr)
        scale = np.abs(np.cos(t[1] - (np.pi/2)))

        new_x = tr[0] - border/scale

        new_tl = [new_x, tr[1]]
        new_tr = tr

    
    # get new br
    border = seg_len(new_tr, br)
    scale = np.abs(np.cos(r[1]))

    new_y = new_tr[1] + border/scale

    new_br = [tr[0], new_y]

    # get new_bl
    new_bl = [new_tl[0], new_br[1]]

    return np.array([new_tl, new_tr, new_br, new_bl])
