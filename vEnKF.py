# The code is for calculating the Yang et al. (1988) model
# Original code is created by USGS using MATLAB 
# Please cite: Modeling Crustal Deformation near Active Faults and Volcanic Centers—A Catalog of Deformation Models
# By Maurizio Battaglia, Peter F. Cervelli, and Jessica R. Murray
# https://pubs.usgs.gov/tm/13/b1/
# modified to python by Yan Zhan, UIUC (2018)

import numpy as np
import cmath

# pi
pi = np.pi


# % compute the 3D displacement due to a pressurized ellipsoid
# %
# % IN
# % a         semimajor axis [m]
# % b         semiminor axis [m]
# % lambda    Lame's constant [Pa]
# % mu        shear modulus [Pa]
# % nu        Poisson's ratio
# % P         excess pressure (stress intensity on the surface) [pressure units]
# % x,y,x     coordinates of the point(s) where the displacement is computed [m]
# % xs,ys,zs  coordinates of the center of the prolate spheroid (positive downward) [m]
# % theta     plunge angle [rad]
# % phi       trend angle [rad]
# %
# % OUT
# % Ux,Uy,Uz  displacement
# %
# % Note ********************************************************************
# % compute the displacement due to a pressurized ellipsoid
# % using the finite prolate spheroid model by from Yang et al (JGR,1988)
# % and corrections to the model by Newmann et al (JVGR, 2006).
# % The equations by Yang et al (1988) and Newmann et al (2006) are valid for a
# % vertical prolate spheroid only. There is and additional typo at pg 4251 in
# % Yang et al (1988), not reported in Newmann et al. (2006), that gives an error
# % when the spheroid is tilted (plunge different from 90°):
# %           C0 = y0*cos(theta) + zs*sin(theta)
# % The correct equation is
# %           C0 = zs/sin(theta)
# % This error has been corrected in this script.
# % *************************************************************************
# %==========================================================================
# % USGS Software Disclaimer
# % The software and related documentation were developed by the U.S.
# % Geological Survey (USGS) for use by the USGS in fulfilling its mission.
# % The software can be used, copied, modified, and distributed without any
# % fee or cost. Use of appropriate credit is requested.
# %
# % The USGS provides no warranty, expressed or implied, as to the correctness
# % of the furnished software or the suitability for any purpose. The software
# % has been tested, but as with any complex software, there could be undetected
# % errors. Users who find errors are requested to report them to the USGS.
# % The USGS has limited resources to assist non-USGS users; however, we make
# % an attempt to fix reported problems and help whenever possible.
# %==========================================================================
#
#
# % testing parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# % clear all; close all; clc;
# % a = 1000; b = 0.99*a;
# % lambda = 1; mu = lambda; nu = 0.25; P = 0.01;
# % theta = pi*89.99/180; phi = 0;
# % x = linspace(0,2E4,7);
# % y = linspace(0,1E4,7);
# % xs = 0; ys = 0; zs = 5E3;
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def yang_disp(x, y, z,
              xs, ys, zs, a, b,
              lamda, mu, nu, P,
              theta, phi
              ):
    # plunge (dip) angle [deg] [90 = vertical spheroid]
    theta = theta + 1e-5
    # compute the parameters for the spheroid model
    a1, b1, csi, Pdila, Pstar = yangpar(a, b, lamda, mu, nu, P)

    # % translate the coordinates of the points where the displacement is computed
    # in the coordinates systen centered in (xs,0)
    xxn = x - xs
    yyn = y - ys

    # rotate the coordinate system to be coherent with the model coordinate
    # system of Figure 3 (Yang et al., 1988)
    xxp = np.cos(phi) * xxn - np.sin(phi) * yyn
    yyp = np.sin(phi) * xxn + np.cos(phi) * yyn

    # compute displacement for a prolate ellipsoid at csi = c
    U1p, U2p, U3p = yangint(xxp, yyp, z, zs, theta,
                            a1, b1, a, b, csi, mu, nu, Pdila)

    # compute displacement for a prolate ellipsoid at csi = -c
    U1m, U2m, U3m = yangint(xxp, yyp, z, zs, theta,
                            a1, b1, a, b, -csi, mu, nu, Pdila)

    Upx = -U1p - U1m
    Upy = -U2p - U2m
    Upz = U3p + U3m
    # rotate horizontal displacement back (strike)
    Ux = cos(phi) * Upx + sin(phi) * Upy
    Uy = -sin(phi) * Upx + cos(phi) * Upy
    Uz = Upz

    # Yan Zhan correct: Now b can greater than a
    Ux = np.real(Ux)
    Uy = np.real(Uy)
    Uz = np.real(Uz)

    return Ux, Uy, Uz


def yang_ddv(x, y, z, ddv,
             xs, ys, zs, a, b,
             lamda, mu, nu, P,
             theta, phi
             ):
    Ux, Uy, Uz = yang_disp(x, y, z,
                           xs, ys, zs, a, b,
                           lamda, mu, nu, P,
                           theta, phi
                           )

    disp = Ux * ddv[0, :] + Uy * ddv[1, :] + Uz * ddv[2, :]

    return disp


def yangpar(a, b, lamda, mu, nu, P):
    # % compute the parameters for the spheroid model
    # % formulas from [1] Yang et al (JGR,1988)
    # % corrections from [2] Newmann et al (JVGR, 2006), Appendix
    # %
    # % IN
    # % a         semimajor axis [m]
    # % b         semiminor axis [m]
    # % lambda    Lame's constant [Pa]
    # % mu        shear modulus [Pa]
    # % nu        Poisson's ratio
    # % P         excess pressure (stress intensity on the surface) [pressure units]
    # %
    # % OUT
    # % a1, b1    pressure (stress) [units of P] from [1]
    # % c         prolate ellipsoid focus [m]
    # % Pdila     pressure (proportional to double couple forces) [units of P] from [1]
    # % Pstar     pressure [units of P]
    # % (1) Yang et al., 1988; (2) Newman et al., 2006
    ###################################################

    # prolate ellipsoid focus [m]
    c = cmath.sqrt(a ** 2 - b ** 2)
    # for convenience
    a2 = a ** 2
    a3 = a ** 3
    b2 = b ** 2
    c2 = c ** 2
    c3 = c ** 3
    c4 = c ** 4
    c5 = c ** 5

    # [dimensionless]
    ac = (a - c) / (a + c)
    # [m^3]
    coef1 = 2 * pi * a * b2
    # [dimensionless]
    den1 = 8 * pi * (1 - nu)

    # param from (1) [dimensionless]
    Q = 3 / den1
    # param from (1) [dimensionless]
    R = (1 - 2 * nu) / den1
    # param from (1) [dimensionless]
    Ia = -coef1 * (2 / (a * c2) + cmath.log(ac) / c3)
    # param from (1) [1/m^2]
    Iaa = -coef1 * (2 / (3 * a3 * c2) + 2 / (a * c4) + cmath.log(ac) / c5)

    # (A-1) from (2) [dimensionless]
    a11 = 2 * R * (Ia - 4 * pi)
    # (A-2) from (2) [dimensionless]
    a12 = -2 * R * (Ia + 4 * pi)
    # (A-3) from (2) [dimensionless]
    a21 = Q * a2 * Iaa + R * Ia - 1
    # (A-4) from (2) [dimensionless]
    a22 = -Q * a2 * Iaa - Ia * (2 * R - Q)

    # for convenience
    den2 = 3 * lamda + 2 * mu
    num2 = 3 * a22 - a12
    den3 = a11 * a22 - a12 * a21
    num3 = a11 - 3 * a21

    # (A-5) from (2) [units of P]
    Pdila = P * (2 * mu / den2) * (num2 - num3) / den3
    # (A-6) from (2) [units of P]
    Pstar = P * (1 / den2) * (num2 * lamda + 2 * (lamda + mu) * num3) / den3

    # force from (1) [m^2*Pa]
    a1 = - 2 * b2 * Pdila
    # pressure from (1) [Pa]
    b1 = 3 * (b2 / c2) * Pdila + 2 * (1 - 2 * nu) * Pstar

    return a1, b1, c, Pdila, Pstar


def yangint(x, y, z, z0, theta, a1, b1, a, b, csi, mu, nu, Pdila):
    # DModel (USGS)
    # compute the primitive of the displacement for a prolate ellipsoid
    # equation (1)-(8) from Yang et al (JGR, 1988)
    # corrections to some parameters from Newmann et al (JVGR, 2006)
    #
    # IN
    # x,y,x     coordinates of the point(s) where the displacement is computed [m]
    # y0,z0     coordinates of the center of the prolate spheroid (positive downward) [m]
    # theta     plunge angle [rad]
    # a1,b1     pressure (stress) (output from yangpar.m) [units of P]
    # a         semimajor axis [m]
    # b         semiminor axis [m]
    # c         focus of the prolate spheroid (output from yangpar.m) [m]
    # mu        shear modulus [Pa]
    # nu        Poisson's ratio
    # Pdila     pressure (proportional to double couple forces) [units of P]
    #
    # OUT
    # U1,U2,U3 : displacement in local coordinates [m] - see Figure 3 of Yang et al (1988)
    #
    # Notes:
    # The location of the center of the prolate spheroid is (x0,y0,z0)
    #     with x0=0 and y0=0;
    # The free surface is z=0;
    # precalculate parameters that are used often
    ###################################################

    sint = sin(theta) + 1E-15
    cost = cos(theta)  # y0 = 0

    # new coordinates and parameters from Yang et al (JGR, 1988), p. 4251
    # dimensions [m]
    csi2 = csi * cost
    csi3 = csi * sint
    # see Figure 3 of Yang et al (1988)
    x1 = x
    x2 = y
    x3 = z - z0
    xbar3 = z + z0
    # x2 = y - y0
    y1 = x1
    y2 = x2 - csi2
    y3 = x3 - csi3
    ybar3 = xbar3 + csi3
    r2 = x2 * sint - x3 * cost
    q2 = x2 * sint + xbar3 * cost
    r3 = x2 * cost + x3 * sint
    q3 = -x2 * cost + xbar3 * sint
    rbar3 = r3 - csi
    qbar3 = q3 + csi
    R1 = (y1 ** 2 + y2 ** 2 + y3 ** 2) ** 0.5
    R2 = (y1 ** 2 + y2 ** 2 + ybar3 ** 2) ** 0.5

    #########################################################################
    # y0 = 0
    # C0 = y0*cost + z0*sint
    # correction base on test by FEM by P. Tizzani IREA-CNR Napoli
    C0 = z0 / sint
    #########################################################################
    # add 1E-15 to avoid a Divide by Zero warning at the origin
    beta = (q2 * cost + (1 + sint) * (R2 + qbar3)) / (cost * y1 + 1E-15)

    # precalculate parameters that are used often
    drbar3 = R1 + rbar3
    dqbar3 = R2 + qbar3
    dybar3 = R2 + ybar3
    lrbar3 = log(R1 + rbar3)
    lqbar3 = log(R2 + qbar3)
    lybar3 = log(R2 + ybar3)
    atanb = atan(beta)

    # primitive parameters from Yang et al (1988), p. 4252
    Astar1 = a1 / (R1 * drbar3) + b1 * (lrbar3 + (r3 + csi) / drbar3)
    Astarbar1 = -a1 / (R2 * dqbar3) - b1 * (lqbar3 + (q3 - csi) / dqbar3)

    A1 = csi / R1 + lrbar3
    Abar1 = csi / R2 - lqbar3
    A2 = R1 - r3 * lrbar3
    Abar2 = R2 - q3 * lqbar3
    A3 = csi * rbar3 / R1 + R1
    Abar3 = csi * qbar3 / R2 - R2

    Bstar = (a1 / R1 + 2 * b1 * A2) + (3 - 4 * nu) * (a1 / R2 + 2 * b1 * Abar2)
    B = csi * (csi + C0) / R2 - Abar2 - C0 * lqbar3

    # the 4 equations below have been changed to improve the fit to internal deformation
    Fstar1 = 0
    Fstar2 = 0
    F1 = 0
    F2 = 0

    f1 = csi * y1 / dybar3 \
         + (3 / cost ** 2) * (y1 * sint * lybar3 - y1 * lqbar3 + 2 * q2 * atanb) \
         + 2 * y1 * lqbar3 - 4 * xbar3 * atanb / cost
    f2 = csi * y2 / dybar3 \
         + (3 / cost ** 2) * (q2 * sint * lqbar3 - q2 * lybar3 + 2 * y1 * sint * atanb + cost * (R2 - ybar3)) \
         - 2 * cost * Abar2 + (2 / cost) * (xbar3 * lybar3 - q3 * lqbar3)

    # correction after Newmann et al (2006), eq (A-9)
    f3 = (1 / cost) * (q2 * lqbar3 - q2 * sint * lybar3 + 2 * y1 * atanb) \
         + 2 * sint * Abar2 + q3 * lybar3 - csi

    # precalculate coefficients that are used often
    cstar = (a * b ** 2 / csi ** 3) / (16 * mu * (1 - nu))
    cdila = 2 * cstar * Pdila

    # displacement components (2) to (7): primitive of equation (1) from Yang et al (1988)
    # equation (2) from Yang et al (1988)
    Ustar1 = cstar * (Astar1 * y1 + (3 - 4 * nu) * Astarbar1 * y1 + Fstar1 * y1)

    # U2star and U3star changed to improve fit to internal deformation
    # equation (3) from Yang et al (1988)
    Ustar2 = cstar * (sint * (Astar1 * r2 + (3 - 4 * nu) * Astarbar1 * q2 + Fstar1 * q2) + cost * (Bstar - Fstar2))

    # The formula used in the script by Fialko and Andy is different from
    # equation (4) of Yang et al (1988)
    # I use the same to continue to compare the results 2009 07 23
    # Ustar3 = cstar*(-cost*(Astarbar1.*r2 + (3-4*nu)*Astarbar1.*q2 - Fstar1.*q2) + ...
    #         sint*(Bstar+Fstar2) + 2*cost^2*z.*Astarbar1)
    ###################################################################################
    # The equation below is correct - follows equation (4) from Yang et al (1988)
    Ustar3 = cstar * (-cost * (Astar1 * r2 + (3 - 4 * nu) * Astarbar1 * q2 - Fstar1 * q2) + sint * (Bstar + Fstar2))
    # equation (4) from Yang et al (1988)
    ####################################################################################
    # equation (5) from Yang et al (1988)
    Udila1_p1 = A1 * y1 + (3 - 4 * nu) * Abar1 * y1 + F1 * y1
    Udila1_p2 = 4 * (1 - nu) * (1 - 2 * nu) * f1
    Udila1 = cdila * (Udila1_p1 - Udila1_p2)

    # equation (6) from Yang et al (1988)
    Udila2_p1 = sint * (A1 * r2 + (3 - 4 * nu) * Abar1 * q2 + F1 * q2)
    Udila2_p2 = 4 * (1 - nu) * (1 - 2 * nu) * f2
    Udila2_p3 = 4 * (1 - nu) * cost * (A2 + Abar2) + cost * (A3 - (3 - 4 * nu) * Abar3 - F2)
    Udila2 = cdila * (Udila2_p1 - Udila2_p2 + Udila2_p3)

    # equation (7) from Yang et al (1988)
    Udila3_p1 = cost * (-A1 * r2 + (3 - 4 * nu) * Abar1 * q2 + F1 * q2)
    Udila3_p2 = 4 * (1 - nu) * (1 - 2 * nu) * f3
    Udila3_p3 = 4 * (1 - nu) * sint * (A2 + Abar2)
    Udila3_p4 = sint * (A3 + (3 - 4 * nu) * Abar3 + F2 - 2 * (3 - 4 * nu) * B)
    Udila3 = cdila * (Udila3_p1 + Udila3_p2 + Udila3_p3 + Udila3_p4)

    # displacement: equation (8) from Yang et al (1988) - see Figure 3
    U1 = Ustar1 + Udila1  # local x component
    U2 = Ustar2 + Udila2  # local y component
    U3 = Ustar3 + Udila3  # local z component

    return U1, U2, U3

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