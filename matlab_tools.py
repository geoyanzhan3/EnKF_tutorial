# Matlab tools create the function useful in Matlab but not in Python
import itertools
import math
import numpy as np
import utm
from datetime import datetime
from pyproj import Proj, transform
import scipy.io

eps = 2.2204e-15


# load matlib matfile
def loadmat(filename):
    """Improved loadmat (replacement for scipy.io.loadmat)
    Ensures correct loading of python dictionaries from mat files.

    Inspired by: https://stackoverflow.com/a/29126361/572908
    """

    def _has_struct(elem):
        """Determine if elem is an array
        and if first array item is a struct
        """
        return isinstance(elem, np.ndarray) and (
            elem.size > 0) and isinstance(
            elem[0], scipy.io.matlab.mio5_params.mat_struct)

    def _check_keys(d):
        """checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        """
        for key in d:
            elem = d[key]
            if isinstance(elem,
                          scipy.io.matlab.mio5_params.mat_struct):
                d[key] = _todict(elem)
            elif _has_struct(elem):
                d[key] = _tolist(elem)
        return d

    def _todict(matobj):
        """A recursive function which constructs from
        matobjects nested dictionaries
        """
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem,
                          scipy.io.matlab.mio5_params.mat_struct):
                d[strg] = _todict(elem)
            elif _has_struct(elem):
                d[strg] = _tolist(elem)
            else:
                d[strg] = elem
        return d

    def _tolist(ndarray):
        """A recursive function which constructs lists from cellarrays
        (which are loaded as numpy ndarrays), recursing into the
        elements if they contain matobjects.
        """
        elem_list = []
        for sub_elem in ndarray:
            if isinstance(sub_elem,
                          scipy.io.matlab.mio5_params.mat_struct):
                elem_list.append(_todict(sub_elem))
            elif _has_struct(sub_elem):
                elem_list.append(_tolist(sub_elem))
            else:
                elem_list.append(sub_elem)
        return elem_list

    data = scipy.io.loadmat(
        filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

# equilivant to matlab datenum
def datenum(datestr, dformat='%Y%m%d-%H%M%S'):
    # if datestr is string create a list for it
    if type(datestr) == str:
        dstr = []
        dstr.append(datestr)
        datestr = dstr
    # initialize datenum array
    dnum = np.zeros(np.array(datestr).shape)
    for i in range(0, len(datestr)):
        # read time with format
        d = datetime.strptime(datestr[i], dformat)
        # calculate the current second
        dsec = d.hour * 3600 + d.minute * 60 + d.second
        # total second in a day
        tsec = 24 * 3600
        # float number of the second
        dsec = dsec / tsec
        # int number of the data
        ddec = d.toordinal()
        # date + time
        ddec_all = ddec + dsec
        # store the datenum
        dnum[i] = ddec_all
    return dnum


# convert day from 0000/01/00 to year string
# dnum must be a number or a numpy array
def datestr(dnum, dformat='%Y%m%d'):
    dformat0 = '{{:{}}}'.format(dformat)
    if isinstance(dnum, np.ndarray) == False:
        dFor = datetime.fromordinal(int(np.floor(dnum)))
        dstr = dformat0.format(dFor)
    else:
        dstr = np.empty(dnum.shape, "U8")
        for i in range(0, dnum.shape[0]):
            dGre = dnum[i]
            dFor = datetime.fromordinal(int(np.floor(dGre)))
            dstr[i] = dformat0.format(dFor)
    return dstr


# convert decimal year to string array
def dec2str(decarr, dformat="{:%Y%m%d}"):
    # begin of the year
    year_beg = np.floor(decarr)
    # end of the year
    year_end = year_beg + 1
    # days of the year_beg from 0000/01/00
    day_beg = datenum(["%d" % x for x in year_beg], '%Y')
    # days of the year_end from 0000/01/00
    day_end = datenum(["%d" % x for x in year_end], '%Y')
    # number of days from day_beg to day_end
    nday = day_end - day_beg
    # proportion of data in a year
    dpro = decarr - year_beg
    # current day from 0000/01/00
    curdate = day_beg + nday * dpro
    # convert date to str
    decstr = datestr(curdate, dformat)
    return decstr


# convert decimal year to day from 0000/01/00
def dec2num(decarr, dformat="{:%Y%m%d}"):
    # begin of the year
    year_beg = np.floor(decarr)
    # end of the year
    year_end = year_beg + 1
    # days of the year_beg from 0000/01/00
    day_beg = datenum(["%d" % x for x in year_beg], '%Y')
    # days of the year_end from 0000/01/00
    day_end = datenum(["%d" % x for x in year_end], '%Y')
    # number of days from day_beg to day_end
    nday = day_end - day_beg
    # proportion of data in a year
    dpro = decarr - year_beg
    # current day from 0000/01/00
    curdate = np.round(day_beg + nday * dpro)
    return curdate


# create a matrix full of nans
def nans(shape, dtype=float):
    a = np.empty(shape, dtype)
    a.fill(np.nan)
    return a


# equilivant to matlab intersect
def intersect(a, b):
    a1, ia = np.unique(a, return_index=True)
    b1, ib = np.unique(b, return_index=True)
    aux = np.concatenate((a1, b1))
    aux.sort()
    c = aux[:-1][aux[1:] == aux[:-1]]
    return c, ia[np.isin(a1, c)], ib[np.isin(b1, c)]


# choose k elements among a
def nchoosek(a, k):
    a = np.array(a)
    all_combos = np.array(list(itertools.combinations(a, k)))
    return all_combos


# I is a matrix storing the index of array(I[i,j])
# calculate mat with the same shape as I,
# its value is found in array
def Imat4arr(array, I):
    mat = nans(I.shape)
    for i in np.arange(0, I.shape[0]):
        for j in np.arange(0, I.shape[1]):
            if np.isnan(I[i, j]) == False:
                Index = int(I[i, j])
                mat[i, j] = array[Index]
    return mat


# MAKEHGTFORM Make a 4x4 transform matrix
# M = MAKEHGTFORM([ax ay az],t)  Rotate around axis
# [ax ay az] by t radians.
def makehgtform(u, t):
    m = np.eye(4)
    u = u / np.linalg.norm(u)
    c, s = computeCosAndSin(t)
    tmp = np.eye(4)
    tmp[0:3, 0:3] = c * np.eye(3) \
                    + (1 - c) * kron(u, u.T) \
                    + s * SkewSymm(u)
    m = m @ tmp
    return m


# used in makehgtform
def fixup(x):
    vals = np.array([0, 1, -1])
    I = np.where(np.abs(x - vals) < 2 * eps)
    if (not I[0]) == False:
        x = vals[I[0]]
        x = x[0]
    return x


# used in makehgtform
def computeCosAndSin(t):
    ct = fixup(np.cos(t))
    st = fixup(np.sin(t))
    return ct, st


# used in makehgtform
def kron(a, b):
    c = np.kron(a, b)
    c = c.reshape(len(b), len(a))
    return c


# used in makehgtform
def SkewSymm(v):
    s = np.array([[0, -v[2], v[1]],
                  [v[2], 0, -v[0]],
                  [-v[1], v[0], 0]]
                 )
    return s


# create surface x, y, z for an ellipsoid with three semiaxises
# a, b, c at the location (x0, y0, z0), the dip and direction of
# the c axis is theta and phi
def ellipsoid(x0, y0, z0,
              a, b, c,
              theta, phi
              ):
    # parameters u and v
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    # ellipsoid equations
    x = a * np.outer(np.cos(u), np.sin(v))
    y = b * np.outer(np.sin(u), np.sin(v))
    z = c * np.outer(np.ones(np.size(u)), np.cos(v))

    # rotation matrix
    m1 = makehgtform([-1, 0, 0], np.pi / 2 - theta)
    m2 = makehgtform([0, 0, 1], phi)

    # prepare for rotations
    xx = x.reshape(-1)
    yy = y.reshape(-1)
    zz = z.reshape(-1)
    # [m * 3] array
    xyz_old = nans([3, len(xx)])
    xyz_old[0, :] = xx
    xyz_old[1, :] = yy
    xyz_old[2, :] = zz
    xyz_old = xyz_old.T
    # rotate x axis
    xyz_new1 = xyz_old @ m1[0:3, 0:3]
    # rotate z axis
    xyz_new2 = xyz_new1 @ m2[0:3, 0:3]
    # 3 1D array
    xx = xyz_new2[:, 0]
    yy = xyz_new2[:, 1]
    zz = xyz_new2[:, 2]
    # reshape to 3 2D matrix
    x = xx.reshape(x.shape)
    y = yy.reshape(y.shape)
    z = zz.reshape(z.shape)

    # translation
    x = x + x0
    y = y + y0
    z = z + z0

    return x, y, z


# convert the lon/lat to UTM
def ll2utm(lat, lon):
    x = np.zeros(lat.shape)
    y = np.zeros(lon.shape)
    for i in np.arange(0, len(lat)):
        utm_res = utm.from_latlon(lat[i], lon[i])
        x[i] = utm_res[0]
        y[i] = utm_res[1]
    return x, y

# convert UTM to lat/lon
def utm2ll(x, y, zone_number, zone_letter):
    if x.shape == y.shape:
        lat = np.zeros(x.shape)
        lon = np.zeros(x.shape)
        for i in np.arange(0, len(x)):
            utm_res = utm.to_latlon(x[i], y[i], zone_number, zone_letter)
            lat[i] = utm_res[0]
            lon[i] = utm_res[1]
        return lat, lon
    else:
        print('[utm2ll error] the size of x does not match with y')



# calculate moving average
def moving_average(a, count=3):
    length = len(a) + count - 1
    A = nans([count, length])
    for i in np.arange(0, count):
        A[i, i:length - count + i + 1] = a
    A_mean = np.nanmean(A, axis=0)
    i = int((count - 1) / 2)
    b = A_mean[i:length - count + i + 1]
    return b


def erf(a):
    b = np.zeros(a.shape)
    for i in np.arange(0, len(a)):
        b[i] = math.erf(a[i])
    return b

# convert the lon/lat to local xyz system
# (if UTM does not work)
# the range of the area cannot be too large
def ll2xy(lat, lon, zone=1):
    outProj = Proj('+proj=utm +zone={}'.format(zone)) # xy
    inProj = Proj('+proj=latlong') # ll
    x, y = transform(inProj, outProj, lon, lat)
    return x, y
