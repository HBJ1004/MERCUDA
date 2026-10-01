// CUDA translation of the original SWIFT/MERCURY Kepler drift routines.
// Original authors: Hal Levison and Martin Duncan; mco_sine: John E. Chambers.
// Translated with f2c 20200916 (-a -R), with CUDA annotations and scalar
// intrinsic wrappers. No f2c runtime or translation tool is needed to build.
using doublereal=double;
using integer=int;
__device__ double d_sign(double* a,double* b) { return *b>=0?fabs(*a):-fabs(*a); }
__device__ double d_mod(double* a,double* b) { return fmod(*a,*b); }
__device__ double pow_dd(double* a,double* b) { return pow(*a,*b); }
__device__ int drift_dan__(doublereal *mu, doublereal *x0, doublereal *
	y0, doublereal *z0, doublereal *vx0, doublereal *vy0, doublereal *vz0,
	 doublereal *dt0, integer *iflg);
__device__ int drift_kepmd__(doublereal *dm, doublereal *es, doublereal 
	*ec, doublereal *x, doublereal *s, doublereal *c__);
__device__ int drift_kepu__(doublereal *dt, doublereal *r0, doublereal *
	mu, doublereal *alpha, doublereal *u, doublereal *fp, doublereal *c1, 
	doublereal *c2, doublereal *c3, integer *iflg);
__device__ int drift_kepu_fchk__(doublereal *dt, doublereal *r0, 
	doublereal *mu, doublereal *alpha, doublereal *u, doublereal *s, 
	doublereal *f);
__device__ int drift_kepu_guess__(doublereal *dt, doublereal *r0, 
	doublereal *mu, doublereal *alpha, doublereal *u, doublereal *s);
__device__ int drift_kepu_lag__(doublereal *s, doublereal *dt, 
	doublereal *r0, doublereal *mu, doublereal *alpha, doublereal *u, 
	doublereal *fp, doublereal *c1, doublereal *c2, doublereal *c3, 
	integer *iflg);
__device__ int drift_kepu_new__(doublereal *s, doublereal *dt, 
	doublereal *r0, doublereal *mu, doublereal *alpha, doublereal *u, 
	doublereal *fp, doublereal *c1, doublereal *c2, doublereal *c3, 
	integer *iflgn);
__device__ int drift_kepu_p3solve__(doublereal *dt, doublereal *r0, 
	doublereal *mu, doublereal *alpha, doublereal *u, doublereal *s, 
	integer *iflg);
__device__ int drift_kepu_stumpff__(doublereal *x, doublereal *c0, 
	doublereal *c1, doublereal *c2, doublereal *c3);
__device__ int drift_one__(doublereal *mu, doublereal *x, doublereal *y,
	 doublereal *z__, doublereal *vx, doublereal *vy, doublereal *vz, 
	doublereal *dt, integer *iflg);
__device__ int mco_sine__(doublereal *x, doublereal *sx, doublereal *cx);


__device__ doublereal c_b7 = 1.;
__device__ doublereal c_b12 = 3.;
__device__ doublereal c_b14 = 2.;
__device__ doublereal c_b15 = .33333333333333331;

 __device__ int drift_dan__(doublereal *mu, doublereal *x0, doublereal *
	y0, doublereal *z0, doublereal *vx0, doublereal *vy0, doublereal *vz0,
	 doublereal *dt0, integer *iflg)
{

    doublereal a, c__, f, g, s, u, x, y, z__, c1, c2, c3, r0, ec, dm, en, fp, 
	    dt, es, vx, vy, vz;
    
    doublereal v0s, asq, esq;
    
    doublereal fchk, fdot, gdot, xkep, alpha;

    dt = *dt0;
    *iflg = 0;
    r0 = sqrt(*x0 * *x0 + *y0 * *y0 + *z0 * *z0);
    v0s = *vx0 * *vx0 + *vy0 * *vy0 + *vz0 * *vz0;
    u = *x0 * *vx0 + *y0 * *vy0 + *z0 * *vz0;
    alpha = *mu * 2. / r0 - v0s;
    if (alpha > 0.) {
	a = *mu / alpha;
	asq = a * a;
	en = sqrt(*mu / (a * asq));
	ec = 1. - r0 / a;
	es = u / (en * asq);
	esq = ec * ec + es * es;
	dm = dt * en - (integer) (dt * en / 6.2831853071795862) * 
		6.2831853071795862;
	dt = dm / en;
	if (dm * dm > .16 || esq > .36) {
	    goto L100;
	}
	if (esq * dm * dm < .0016) {
	    drift_kepmd__(&dm, &es, &ec, &xkep, &s, &c__);
	    fchk = xkep - ec * s + es * (1. - c__) - dm;
	    if (fchk * fchk > 1e-13) {
		*iflg = 1;
		return 0;
	    }
	    fp = 1. - ec * c__ + es * s;
	    f = a / r0 * (c__ - 1.) + 1.;
	    g = dt + (s - xkep) / en;
	    fdot = -(a / (r0 * fp)) * en * s;
	    gdot = (c__ - 1.f) / fp + 1.f;
	    x = *x0 * f + *vx0 * g;
	    y = *y0 * f + *vy0 * g;
	    z__ = *z0 * f + *vz0 * g;
	    vx = *x0 * fdot + *vx0 * gdot;
	    vy = *y0 * fdot + *vy0 * gdot;
	    vz = *z0 * fdot + *vz0 * gdot;
	    *x0 = x;
	    *y0 = y;
	    *z0 = z__;
	    *vx0 = vx;
	    *vy0 = vy;
	    *vz0 = vz;
	    *iflg = 0;
	    return 0;
	}
    }
L100:
    drift_kepu__(&dt, &r0, mu, &alpha, &u, &fp, &c1, &c2, &c3, iflg);
    if (*iflg == 0) {
	f = 1. - *mu / r0 * c2;
	g = dt - *mu * c3;
	fdot = -(*mu / (fp * r0)) * c1;
	gdot = 1. - *mu / fp * c2;
	x = *x0 * f + *vx0 * g;
	y = *y0 * f + *vy0 * g;
	z__ = *z0 * f + *vz0 * g;
	vx = *x0 * fdot + *vx0 * gdot;
	vy = *y0 * fdot + *vy0 * gdot;
	vz = *z0 * fdot + *vz0 * gdot;
	*x0 = x;
	*y0 = y;
	*z0 = z__;
	*vx0 = vx;
	*vy0 = vy;
	*vz0 = vz;
    }
    return 0;
} 

 __device__ int drift_kepmd__(doublereal *dm, doublereal *es, doublereal 
	*ec, doublereal *x, doublereal *s, doublereal *c__)
{

    doublereal f, q, y, fp, dx, fpp, fac1, fac2, fppp;

    fac1 = 1. / (1. - *ec);
    q = fac1 * *dm;
    fac2 = *es * *es * fac1 - *ec / 3.;
    *x = q * (1. - fac1 * .5 * q * (*es - q * fac2));

    y = *x * *x;
    *s = *x * (39916800. - y * (6652800. - y * (332640. - y * (7920. - y * (
	    110. - y))))) / 39916800.;
    *c__ = sqrt(1. - *s * *s);

    f = *x - *ec * *s + *es * (1. - *c__) - *dm;
    fp = 1.f - *ec * *c__ + *es * *s;
    fpp = *ec * *s + *es * *c__;
    fppp = *ec * *c__ - *es * *s;
    dx = -f / fp;
    dx = -f / (fp + dx * .5 * fpp);
    dx = -f / (fp + dx * .5 * fpp + dx * .16666666666666666 * dx * fppp);
    *x += dx;

    y = *x * *x;
    *s = *x * (39916800. - y * (6652800. - y * (332640. - y * (7920. - y * (
	    110. - y))))) / 39916800.;
    *c__ = sqrt(1. - *s * *s);
    return 0;
} 

 __device__ int drift_kepu__(doublereal *dt, doublereal *r0, doublereal *
	mu, doublereal *alpha, doublereal *u, doublereal *fp, doublereal *c1, 
	doublereal *c2, doublereal *c3, integer *iflg)
{
    
    doublereal s, fn, fo, st;

    drift_kepu_guess__(dt, r0, mu, alpha, u, &s);
    st = s;

    drift_kepu_new__(&s, dt, r0, mu, alpha, u, fp, c1, c2, c3, iflg);
    if (*iflg != 0) {
	drift_kepu_fchk__(dt, r0, mu, alpha, u, &st, &fo);
	drift_kepu_fchk__(dt, r0, mu, alpha, u, &s, &fn);
	if (fabs(fo) < fabs(fn)) {
	    s = st;
	}
	drift_kepu_lag__(&s, dt, r0, mu, alpha, u, fp, c1, c2, c3, iflg);
    }
    return 0;
} 

 __device__ int drift_kepu_fchk__(doublereal *dt, doublereal *r0, 
	doublereal *mu, doublereal *alpha, doublereal *u, doublereal *s, 
	doublereal *f)
{
    doublereal x, c0, c1, c2, c3;

    x = *s * *s * *alpha;
    drift_kepu_stumpff__(&x, &c0, &c1, &c2, &c3);
    c1 *= *s;
    c2 = c2 * *s * *s;
    c3 = c3 * *s * *s * *s;
    *f = *r0 * c1 + *u * c2 + *mu * c3 - *dt;
    return 0;
} 

 __device__ int drift_kepu_guess__(doublereal *dt, doublereal *r0, 
	doublereal *mu, doublereal *alpha, doublereal *u, doublereal *s)
{
    
    doublereal d__1;

    doublereal a, e, x, y, ec, en, es, cy, sy;
    
    integer iflg;
    doublereal sigma;

    if (*alpha > 0.) {

	if (*dt / *r0 <= .4) {
	    *s = *dt / *r0 - *dt * *dt * *u / (*r0 * 2. * *r0 * *r0);
	    return 0;
	} else {
	    a = *mu / *alpha;
	    en = sqrt(*mu / (a * a * a));
	    ec = 1. - *r0 / a;
	    es = *u / (en * a * a);
	    e = sqrt(ec * ec + es * es);
	    y = en * *dt - es;

	    mco_sine__(&y, &sy, &cy);

	    d__1 = es * cy + ec * sy;
	    sigma = d_sign(&c_b7, &d__1);
	    x = y + sigma * .85 * e;
	    *s = x / sqrt(*alpha);
	}
    } else {

	drift_kepu_p3solve__(dt, r0, mu, alpha, u, s, &iflg);
	if (iflg != 0) {
	    *s = *dt / *r0;
	}
    }
    return 0;
} 

 __device__ int drift_kepu_lag__(doublereal *s, doublereal *dt, 
	doublereal *r0, doublereal *mu, doublereal *alpha, doublereal *u, 
	doublereal *fp, doublereal *c1, doublereal *c2, doublereal *c3, 
	integer *iflg)
{
    
    integer i__1;
    doublereal d__1;

    doublereal f, x, c0;
    integer nc;
    doublereal ds, ln, fdt, fpp;
    
    integer ncmax;

    if (*alpha < 0.) {
	ncmax = 400;
    } else {
	ncmax = 400;
    }
    ln = 5.;

    i__1 = ncmax;
    for (nc = 0; nc <= i__1; ++nc) {
	x = *s * *s * *alpha;
	drift_kepu_stumpff__(&x, &c0, c1, c2, c3);
	*c1 *= *s;
	*c2 = *c2 * *s * *s;
	*c3 = *c3 * *s * *s * *s;
	f = *r0 * *c1 + *u * *c2 + *mu * *c3 - *dt;
	*fp = *r0 * c0 + *u * *c1 + *mu * *c2;
	fpp = (*alpha * -40. + *mu) * *c1 + *u * c0;
	ds = -ln * f / (*fp + d_sign(&c_b7, fp) * sqrt((d__1 = (ln - 1.) * (
		ln - 1.) * *fp * *fp - (ln - 1.) * ln * f * fpp, fabs(d__1))));
	*s += ds;
	fdt = f / *dt;

	if (fdt * fdt < 1e-26) {
	    *iflg = 0;
	    return 0;
	}

    }
    *iflg = 2;
    return 0;
} 

 __device__ int drift_kepu_new__(doublereal *s, doublereal *dt, 
	doublereal *r0, doublereal *mu, doublereal *alpha, doublereal *u, 
	doublereal *fp, doublereal *c1, doublereal *c2, doublereal *c3, 
	integer *iflgn)
{
    doublereal f, x, c0, s2;
    integer nc;
    doublereal ds, fdt, fpp;
    
    doublereal fppp;

    for (nc = 0; nc <= 6; ++nc) {
	s2 = *s * *s;
	x = s2 * *alpha;
	drift_kepu_stumpff__(&x, &c0, c1, c2, c3);
	*c1 *= *s;
	*c2 *= s2;
	*c3 = *c3 * *s * s2;
	f = *r0 * *c1 + *u * *c2 + *mu * *c3 - *dt;
	*fp = *r0 * c0 + *u * *c1 + *mu * *c2;
	fpp = (*mu - *r0 * *alpha) * *c1 + *u * c0;
	fppp = (*mu - *r0 * *alpha) * c0 - *u * *alpha * *c1;
	ds = -f / *fp;
	ds = -f / (*fp + ds * .5 * fpp);
	ds = -f / (*fp + ds * .5 * fpp + ds * ds * fppp * .1666666666666667);
	*s += ds;
	fdt = f / *dt;

	if (fdt * fdt < 1e-26) {
	    *iflgn = 0;
	    return 0;
	}

    }

    *iflgn = 1;
    return 0;
} 

 __device__ int drift_kepu_p3solve__(doublereal *dt, doublereal *r0, 
	doublereal *mu, doublereal *alpha, doublereal *u, doublereal *s, 
	integer *iflg)
{
    
    doublereal d__1;

    doublereal q, r__, a0, a1, a2, p1, p2, sq, sq2, denom;

    denom = (*mu - *alpha * *r0) / 6.;
    a2 = *u * .5 / denom;
    a1 = *r0 / denom;
    a0 = -(*dt) / denom;
    q = (a1 - a2 * a2 / 3.) / 3.;
    r__ = (a1 * a2 - a0 * 3.) / 6. - pow_dd(&a2, &c_b12) / 27.;
    sq2 = pow_dd(&q, &c_b12) + pow_dd(&r__, &c_b14);
    if (sq2 >= 0.) {
	sq = sqrt(sq2);
	if (r__ + sq <= 0.) {
	    d__1 = -(r__ + sq);
	    p1 = -pow_dd(&d__1, &c_b15);
	} else {
	    d__1 = r__ + sq;
	    p1 = pow_dd(&d__1, &c_b15);
	}
	if (r__ - sq <= 0.) {
	    d__1 = -(r__ - sq);
	    p2 = -pow_dd(&d__1, &c_b15);
	} else {
	    d__1 = r__ - sq;
	    p2 = pow_dd(&d__1, &c_b15);
	}
	*iflg = 0;
	*s = p1 + p2 - a2 / 3.;
    } else {
	*iflg = 1;
	*s = 0.;
    }
    return 0;
} 

 __device__ int drift_kepu_stumpff__(doublereal *x, doublereal *c0, 
	doublereal *c1, doublereal *c2, doublereal *c3)
{
    integer i__, n;
    doublereal x2, x3, x4, x5, x6, xm;

    n = 0;
    xm = .1;
    while(fabs(*x) >= xm) {
	++n;
	*x *= .25;
    }

    x2 = *x * *x;
    x3 = *x * x2;
    x4 = x2 * x2;
    x5 = x2 * x3;
    x6 = x3 * x3;

    *c2 = x6 * 1.147074559772972e-11 - x5 * 2.08767569878681e-9 + x4 * 
	    2.755731922398589e-7 - x3 * 2.48015873015873e-5 + x2 * 
	    .001388888888888889 - *x * .04166666666666667 + .5;

    *c3 = x6 * 7.647163731819816e-13 - x5 * 1.605904383682161e-10 + x4 * 
	    2.505210838544172e-8 - x3 * 2.755731922398589e-6 + x2 * 
	    1.984126984126984e-4 - *x * .008333333333333333 + 
	    .1666666666666667;

    *c1 = 1. - *x * *c3;
    *c0 = 1. - *x * *c2;

    if (n != 0) {
	for (i__ = n; i__ >= 1; --i__) {
	    *c3 = (*c2 + *c0 * *c3) * .25;
	    *c2 = *c1 * *c1 * .5;
	    *c1 = *c0 * *c1;
	    *c0 = *c0 * 2. * *c0 - 1.;
	    *x *= 4.;
	}
    }
    return 0;
} 

 __device__ int drift_one__(doublereal *mu, doublereal *x, doublereal *y,
	 doublereal *z__, doublereal *vx, doublereal *vy, doublereal *vz, 
	doublereal *dt, integer *iflg)
{
    integer i__;
    
    doublereal dttmp;

    drift_dan__(mu, x, y, z__, vx, vy, vz, dt, iflg);
    if (*iflg != 0) {
	for (i__ = 1; i__ <= 10; ++i__) {
	    dttmp = *dt / 10.;
	    drift_dan__(mu, x, y, z__, vx, vy, vz, &dttmp, iflg);
	    if (*iflg != 0) {
		return 0;
	    }
	}
    }
    return 0;
} 

 __device__ int mco_sine__(doublereal *x, doublereal *sx, doublereal *cx)
{

    doublereal pi, twopi;

    pi = 3.141592653589793;
    twopi = pi * 2.;

    if (*x > 0.) {
	*x = d_mod(x, &twopi);
    } else {
	*x = d_mod(x, &twopi) + twopi;
    }

    *cx = cos(*x);

    if (*x > pi) {
	*sx = -sqrt(1. - *cx * *cx);
    } else {
	*sx = sqrt(1. - *cx * *cx);
    }

    return 0;
} 

