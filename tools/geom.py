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


def rectify_points(points) :
    tl, tr, br, bl = points
    # angles
    top_a = np.arctan(np.abs(tl[1]-tr[1])/np.abs(tl[0]-tr[0]))
    #left_a = np.arctan(np.abs(tl[0]-bl[0])/np.abs(tl[1]-bl[1]))
    #bottom_a = np.arctan(np.abs(bl[1]-br[1])/np.abs(bl[0]-br[0]))
    right_a = np.arctan(np.abs(tr[0]-br[0])/np.abs(tr[1]-br[1]))

    new_tl, new_tr, new_br, new_bl = [None, None, None, None]

    # get new_tl and new_tr
    if tl[1] < tr[1] :
        border = seg_len(tl, tr)
        scale = np.abs(np.cos(top_a))

        new_x = tl[0] + border/scale

        new_tl = tl
        new_tr = [new_x, tl[1]]
    else :
        border = seg_len(tl, tr)
        scale = np.abs(np.cos(top_a))

        new_x = tr[0] - border/scale

        new_tl = [new_x, tr[1]]
        new_tr = tr

    
    # get new br
    border = seg_len(new_tr, br)
    scale = np.abs(np.cos(right_a))

    new_y = new_tr[1] + border/scale

    new_br = [tr[0], new_y]

    # get new_bl
    new_bl = [new_tl[0], new_br[1]]

    return np.array([new_tl, new_tr, new_br, new_bl])


def rectify_rhombus(points):
        
    tl, tr, br, bl = points
    new_tl, new_tr, new_br, new_bl = [None, None, None, None]
    #calculate middle y of the rhombus
    new_my=np.round((tl[1]+br[1])/2)
    #calulate middle x of the rhomus
    new_mx=np.round((tr[0]+bl[0])/2)
    #we keep the unique x,y that rhombus points don't share
    new_lx=tl[0]
    new_rx=br[0]
    new_ty=tr[1]
    new_by=bl[1]

    new_tl=[new_lx,new_my]
    new_tr=[new_mx,new_ty]
    new_br=[new_rx,new_my]
    new_bl=[new_mx,new_by]

    return np.array([new_tl, new_tr, new_br, new_bl])


import matplotlib.pyplot as plt

def rectify_rhombus_v2(points):
    # rotate points 45 degrees draw a square then rotate the points back (-45 degrees)
    _, t, r, b = points

    side = np.sqrt((t[0]-r[0])*(t[0]-r[0]) + (t[1]-r[1])*(t[1]-r[1]))
    traslation = (t[0], side - t[1])

    # rotation matrix
    rot = np.array([[np.cos(-1*np.pi/4), -1*np.sin(-1*np.pi/4)], [np.sin(-1*np.pi/4), np.cos(-1*np.pi/4)]])

    tl = [-1*side/2, side/2]
    tr = [side/2, side/2]
    br = [side/2, -1*side/2]
    bl = [-1*side/2, -1*side/2]
    
    #print(np.array([new_bl, new_tl, new_tr, new_br]))
    
    t = rot @ tl
    r = rot @ tr
    b = rot @ br
    l = rot @ bl

    t += traslation
    l += traslation
    r += traslation
    b += traslation
    '''
    test = np.round(np.array([l, t, r, b]))
    plt.scatter(test[:, 0], test[:, 1])
    plt.show()
    '''
    # invert top and bottom to respect opencv order l t r b
    return np.round(np.array([l, b, r, t]))
