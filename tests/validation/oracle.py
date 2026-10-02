"""Independent scientific references. Never imports Mercury force routines.

J2/J4/J6 are differentiated from the Legendre potential, rather than copied
from Mercury's acceleration polynomials. PR deliberately implements the agreed
component-wise prescription, not the usual rotationally invariant vector law.
"""
import math
import struct
import mpmath as mp
import numpy as np
import rebound

MU = 2.959122082855911e-4
LIGHT = 299792458.0 * 86400.0 / 149597870700.0
PR_LIGHT = struct.unpack('f', struct.pack('f', 173.1))[0]
mp.mp.dps = 80


def kepler(initial, time, mu=MU):
    """Universal-variable propagation from exact binary64 Cartesian inputs."""
    r0 = list(map(mp.mpf, initial['x']))
    v0 = list(map(mp.mpf, initial['v']))
    mu, time = mp.mpf(mu), mp.mpf(time)
    radius = mp.sqrt(sum(x*x for x in r0))
    rv = sum(x*v for x,v in zip(r0,v0))
    alpha = 2/radius - sum(v*v for v in v0)/mu
    root = mp.sqrt(mu)

    def stumpff(z):
        if abs(z) < mp.mpf('1e-30'):
            return mp.mpf('.5'), mp.mpf(1)/6
        if z > 0:
            q = mp.sqrt(z)
            return (1-mp.cos(q))/z, (q-mp.sin(q))/(q**3)
        q = mp.sqrt(-z)
        return (mp.cosh(q)-1)/(-z), (mp.sinh(q)-q)/(q**3)

    def equation(chi):
        c,s = stumpff(alpha*chi*chi)
        return rv/root*chi*chi*c+(1-alpha*radius)*chi**3*s+radius*chi-root*time

    if time == 0:
        return np.array(r0+v0, dtype=float)
    guess = root*time/radius
    bound = abs(guess)
    while equation(-bound)*equation(bound) > 0: bound *= 2
    chi = mp.findroot(equation,(-bound,bound),solver='bisect',maxsteps=300,tol=mp.mpf('1e-65'))
    c,s = stumpff(alpha*chi*chi)
    f, g = 1-chi*chi*c/radius, time-chi**3*s/root
    r = [f*x+g*v for x,v in zip(r0,v0)]
    final_radius = mp.sqrt(sum(x*x for x in r))
    fdot = root/(radius*final_radius)*(alpha*chi**3*s-chi)
    gdot = 1-chi*chi*c/final_radius
    v = [fdot*x+gdot*y for x,y in zip(r0,v0)]
    return np.array(r+v, dtype=float)


def high_precision_force(masses, positions, velocities, ngf, jcen, nbig, pn):
    """Return full heliocentric force and componentwise absolute term sums."""
    m = list(map(mp.mpf,masses))
    x = [list(map(mp.mpf,p)) for p in positions]
    v = [list(map(mp.mpf,p)) for p in velocities]
    ng = [list(map(mp.mpf,p)) for p in ngf]
    n = len(m)
    terms = [[[] for _ in range(3)] for _ in range(n)]
    radii = [mp.mpf(0)] + [mp.sqrt(sum(z*z for z in p)) for p in x[1:]]
    for i in range(1,n):
        for k in range(3):
            terms[i][k].append(-m[0]*x[i][k]/radii[i]**3)
            for j in range(1,n):
                terms[i][k].append(-m[j]*x[j][k]/radii[j]**3)
                if i != j and (i < nbig or j < nbig):
                    d = [x[j][h]-x[i][h] for h in range(3)]
                    distance = mp.sqrt(sum(z*z for z in d))
                    terms[i][k].append(m[j]*d[k]/distance**3)
        radius = radii[i]
        dot = sum(z*w for z,w in zip(x[i],v[i]))
        vt = [v[i][k]-dot*x[i][k]/radius**2 for k in range(3)]
        tn = mp.sqrt(sum(z*z for z in vt))
        normal = [x[i][1]*v[i][2]-x[i][2]*v[i][1],
                  x[i][2]*v[i][0]-x[i][0]*v[i][2],
                  x[i][0]*v[i][1]-x[i][1]*v[i][0]]
        nn = mp.sqrt(sum(z*z for z in normal))
        q = radius/mp.mpf('2.808')
        g = mp.mpf('.111262')*q**mp.mpf('-2.15')*(1+q**mp.mpf('5.093'))**mp.mpf('-4.6142')
        if radius**2 >= 88 and max(abs(z) for z in ng[i][:3]) <= mp.mpf('1e-7'):
            g = 0
        for k in range(3):
            terms[i][k].append(ng[i][0]*g*x[i][k]/radius)
            if ng[i][1]: terms[i][k].append(ng[i][1]*g*vt[k]/tn)
            if ng[i][2]: terms[i][k].append(ng[i][2]*g*normal[k]/nn)
            if ng[i][4]: terms[i][k].append(ng[i][4]/radius**2*vt[k]/tn)
            if pn:
                speed2 = sum(z*z for z in v[i])
                terms[i][k].append(m[0]/(mp.mpf(LIGHT)**2*radius**3)*
                                     ((4*m[0]/radius-speed2)*x[i][k]+4*dot*v[i][k]))
            if m[i] == 0 and ng[i][3]:
                vr = dot/radius
                # c is rounded as default REAL in the preserved Fortran routine.
                c = mp.mpf(PR_LIGHT)
                terms[i][k].append(mp.mpf(MU)*ng[i][3]/(c*radius**2)*
                    ((c-mp.mpf('2.6')*vr)*x[i][k]/radius-mp.mpf('1.3')*v[i][k]*(1-x[i][k]/radius)))
    obl = [[mp.mpf(0)]*3 for _ in range(n)]
    for i in range(1,n):
        for k in range(3):
            for order,coefficient in zip((2,4,6),jcen):
                if not coefficient: continue
                def potential(component):
                    point = x[i].copy(); point[k] = component
                    r = mp.sqrt(sum(z*z for z in point))
                    return m[0]*mp.mpf(coefficient)/r**(order+1)*mp.legendre(order,point[2]/r)
                obl[i][k] -= mp.diff(potential,x[i][k])
    for i in range(1,n):
        for k in range(3):
            terms[i][k].append(obl[i][k])
            for j in range(1,n): terms[i][k].append(m[j]/m[0]*obl[j][k])
    force = [[sum(t) for t in row] for row in terms]
    scale = [[sum(abs(z) for z in t) for t in row] for row in terms]
    return np.array(force,dtype=float), np.array(scale,dtype=float)


def extras(mu, x, v, params, jcen=(0,0,0), pn=False):
    """Independent double precision callbacks for IAS15 (heliocentric inputs)."""
    r = np.linalg.norm(x); rv = np.dot(x,v)
    t = v-rv*x/r**2; tn = np.linalg.norm(t)
    normal = np.cross(x,v); nn = np.linalg.norm(normal)
    a = np.zeros(3)
    q = r/2.808
    g = .111262*q**(-2.15)*(1+q**5.093)**(-4.6142)
    if r*r >= 88 and max(abs(params.get(key,0)) for key in ('a1','a2','a3')) <= 1e-7: g = 0
    a += params.get('a1',0)*g*x/r
    if params.get('a2',0): a += params['a2']*g*t/tn
    if params.get('a3',0): a += params['a3']*g*normal/nn
    if params.get('yar',0): a += params['yar']/r**2*t/tn
    if pn: a += mu/(LIGHT**2*r**3)*((4*mu/r-np.dot(v,v))*x+4*rv*v)
    if params.get('b',0):
        a += MU*params['b']/(PR_LIGHT*r*r)*((PR_LIGHT-2.6*rv/r)*x/r-1.3*v*(1-x/r))
    return a


def oblateness(mu, x, jcen):
    """Gradient of the zonal-harmonic potential, using Legendre recurrence."""
    r = np.linalg.norm(x); u = x[2]/r; p0,p1 = 1.,u; d0,d1 = 0.,1.
    a = np.zeros(3)
    for order in range(2,7):
        p = ((2*order-1)*u*p1-(order-1)*p0)/order
        derivative = ((2*order-1)*(p1+u*d1)-(order-1)*d0)/order
        if order in (2,4,6):
            coefficient = jcen[order//2-1]
            a += mu*coefficient/r**(order+3)*((order+1)*p+u*derivative)*x
            a[2] -= mu*coefficient/r**(order+2)*derivative
        p0,p1,d0,d1 = p1,p,d1,derivative
    return a


def trajectory(big, small, times, *, central_mass=1., pn=False, jcen=(0,0,0), rcen=.005, epsilon=1e-13, custom=None):
    # param.in takes dimensionless Jn; Mercury scales by the central radius.
    jcen = tuple(value*rcen**order for value,order in zip(jcen,(2,4,6)))
    sim = rebound.Simulation(); sim.G = MU; sim.integrator = 'ias15'
    sim.integrator.epsilon = epsilon
    sim.add(m=central_mass)
    objects = big+small
    for obj in objects:
        sim.add(m=obj['mass'],x=obj['x'][0],y=obj['x'][1],z=obj['x'][2],
                vx=obj['v'][0],vy=obj['v'][1],vz=obj['v'][2])
    sim.N_active = 1+len(big)
    sim.testparticle_type = 1
    sim.force_is_velocity_dependent = 1
    def callback(pointer):
        state = pointer.contents; p = state.particles; sun = p[0]
        origin = np.array([sun.x,sun.y,sun.z]); motion = np.array([sun.vx,sun.vy,sun.vz])
        reaction = np.zeros(3)
        for i,obj in enumerate(objects,1):
            x = np.array([p[i].x,p[i].y,p[i].z])-origin
            v = np.array([p[i].vx,p[i].vy,p[i].vz])-motion
            params = dict(obj.get('params',{}))
            if obj['mass']: params.pop('b',None)
            ob = oblateness(MU*central_mass,x,jcen) if any(jcen) else np.zeros(3)
            a = extras(MU*central_mass,x,v,params,pn=pn)+ob
            if custom is not None: a += custom(state.t,x,v)
            p[i].ax += a[0]; p[i].ay += a[1]; p[i].az += a[2]
            reaction -= obj['mass']/central_mass*ob
        sun.ax += reaction[0]; sun.ay += reaction[1]; sun.az += reaction[2]
    # Keep Python callback alive until all requested epochs have been sampled.
    sim.additional_forces = callback
    states = []
    for time in times:
        sim.integrate(float(time),exact_finish_time=1)
        p = sim.particles; sun = p[0]
        states.append({obj['name']:np.array([p[i].x-sun.x,p[i].y-sun.y,p[i].z-sun.z,
                        p[i].vx-sun.vx,p[i].vy-sun.vy,p[i].vz-sun.vz]) for i,obj in enumerate(objects,1)})
    return states


def errors(actual, expected, distance=1., mu=MU):
    if set(actual) != set(expected): raise AssertionError('Body identities differ')
    velocity = math.sqrt(mu/distance)
    position_error = velocity_error = 0.
    for name,state in expected.items():
        got = np.array(actual[name]['x']+actual[name]['v'])
        if not np.all(np.isfinite(got)): raise AssertionError('Nonfinite saved state')
        position_error = max(position_error,float(np.linalg.norm(got[:3]-state[:3])/distance))
        velocity_error = max(velocity_error,float(np.linalg.norm(got[3:]-state[3:])/velocity))
    return position_error,velocity_error
