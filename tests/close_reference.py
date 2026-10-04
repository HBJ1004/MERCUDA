"""Independent base-224 fixtures and orbital invariants (stdlib, 80 decimal digits).

The encoder uses integer quantization. The oracle uses energy/angular momentum,
not Mercury's Cartesian-to-elements routines. References start from decoded bytes.
"""
from decimal import Decimal as D, localcontext
from pathlib import Path
import math
import shutil
from cases import ROOT, MU

PI=D('3.141592653589793238462643383279502884197169399375105820974944592307816406286208999')
BASE=224

def real_bytes(value, low=0, high=1, n=4):
    with localcontext() as ctx:
        ctx.prec=80
        value=D(str(value)); low=D(str(low)); high=D(str(high))
        count=int((value-low)/(high-low)*BASE**n)
        count=min(BASE**n-1,max(0,count))
        return digits(count,n)

def digits(value,n):
    result=bytearray(n)
    for j in range(n-1,-1,-1):
        value,digit=divmod(value,BASE); result[j]=digit+32
    assert value==0
    return bytes(result)

def float_bytes(value):
    with localcontext() as ctx:
        ctx.prec=80
        value=D(str(value)); exponent=0 if not value else value.copy_abs().adjusted()+1
        assert -112<=exponent<=111
        mantissa=(1+value/(D(10)**exponent))/2
        return real_bytes(mantissa,n=7)+bytes([exponent+144])

def fraction(data):
    assert all(c>=32 for c in data)
    numerator=0
    for c in data: numerator=numerator*BASE+c-32
    return D(numerator)/D(BASE**len(data))

def floating(data): return (2*fraction(data[:7])-1)*D(10)**(data[7]-144)

def header(names=('PLANET','PARTICLE'),masses=None,codes=None,time=2451545,central=1,rcen=.005,rmax=100,precision=3,algorithm=2):
    masses=masses if masses is not None else [.001]+[0]*(len(names)-1)
    codes=codes if codes is not None else range(1,len(names)+1)
    payload=float_bytes(time)+digits(len(names),3)+digits(0,3)+float_bytes(central)
    payload+=float_bytes(0)*3+float_bytes(rcen)+float_bytes(rmax)
    records=[b'\x0c6a'+f'{algorithm:2}'.encode()+payload+str(precision).encode()]
    tail=float_bytes(0)*3+float_bytes(1)
    encoded_masses={}
    for code,name,mass in zip(codes,names,masses):
        if mass not in encoded_masses: encoded_masses[mass]=float_bytes(mass)
        records.append(digits(code,3)+name.encode('ascii').ljust(25,b' ')+encoded_masses[mass]+tail)
    assert all(len(row)==68 for row in records)
    return records

def state(x,v,central=1,rcen=.005,rmax=100):
    radius=math.sqrt(sum(t*t for t in x)); speed=math.sqrt(sum(t*t for t in v))
    theta=math.acos(max(-1,min(1,x[2]/radius))); phi=math.atan2(x[1],x[0])%(2*math.pi)
    vt=0 if speed==0 else math.acos(max(-1,min(1,v[2]/speed)))
    vp=math.atan2(v[1],v[0])%(2*math.pi)
    fv=1/(1+2*(speed*speed*radius/(2*MU*central))**2)
    return (real_bytes(math.log10(radius/rcen),0,math.log10(rmax/rcen))+
            real_bytes(theta,0,PI)+real_bytes(phi,0,2*PI)+real_bytes(fv)+real_bytes(vt,0,PI)+real_bytes(vp,0,2*PI))

def orbit(e=0,inclination=0,phase=0,mass=.001,central=1,q=1):
    angle=math.radians(inclination); speed=math.sqrt(MU*(central+mass)*(1+e)/q)
    return ([q*math.cos(phase),q*math.sin(phase),0],
            [-speed*math.sin(phase)*math.cos(angle),speed*math.cos(phase)*math.cos(angle),speed*math.sin(angle)])

def encounter(first=None,second=None,time=2451545.5,distance=.2,codes=(1,2),central=1,rcen=.005,rmax=100):
    first=first or orbit(central=central); second=second or orbit(mass=0,central=central,q=1.2)
    result=b'\x0c6b'+float_bytes(time)+digits(codes[0],3)+digits(codes[1],3)+float_bytes(distance)
    result+=state(*first,central,rcen,rmax)+state(*second,central,rcen,rmax)
    assert len(result)==73
    return result

def fixture(path,records=None,files=None,names=(),units='days',relative=False,newline=b'\n',final_newline=True):
    path=Path(path); path.mkdir(parents=True,exist_ok=True)
    shutil.copy(ROOT/'message.in.sample',path/'message.in')
    files=files or {'ce.out':records if records is not None else header()+[encounter()]}
    for name,rows in files.items(): (path/name).write_bytes(newline.join(rows)+(newline if final_newline else b''))
    (path/'close.in').write_text(')O+_06\n number of input files = '+str(len(files))+'\n'+'\n'.join(files)+
                               '\n time units = '+units+'\n relative time = '+('yes' if relative else 'no')+'\n'+'\n'.join(names)+'\n')
    return path

def sin_cos(angle):
    # Independent decimal Taylor series, converges on the encoded [0, 2pi) range.
    square=angle*angle; sine=angle; cosine=D(1); st=angle; ct=D(1)
    for n in range(1,160):
        st=-st*square/D((2*n)*(2*n+1)); ct=-ct*square/D((2*n-1)*(2*n))
        sine+=st; cosine+=ct
        if abs(st)+abs(ct)<D('1e-78'): break
    return sine,cosine

def decoded_state(data,rcen,rmax,central):
    r=rcen*((fraction(data[:4])*(rmax/rcen).ln()).exp())
    angles=[PI*fraction(data[4:8]),2*PI*fraction(data[8:12]),PI*fraction(data[16:20]),2*PI*fraction(data[20:24])]
    s,c=sin_cos(angles[0]); sp,cp=sin_cos(angles[1]); sv,cv=sin_cos(angles[2]); svp,cvp=sin_cos(angles[3])
    fv=fraction(data[12:16]); speed=(2*((1/fv-1)/2).sqrt()*central/r).sqrt()
    return [r*s*cp,r*s*sp,r*c],[speed*sv*cvp,speed*sv*svp,speed*cv]

def elements(mu,x,v):
    dot=lambda a,b:sum(i*j for i,j in zip(a,b))
    r=dot(x,x).sqrt(); v2=dot(v,v); energy=v2/2-mu/r
    h=[x[1]*v[2]-x[2]*v[1],x[2]*v[0]-x[0]*v[2],x[0]*v[1]-x[1]*v[0]]
    e=max(D(0),1+2*energy*dot(h,h)/(mu*mu)).sqrt()
    a=D('Infinity') if energy==0 else -mu/(2*energy)
    inclination=math.nan if all(t==0 for t in h) else math.degrees(math.atan2(float((h[0]**2+h[1]**2).sqrt()),float(h[2])))
    return [float(a),float(e),inclination],float(r)

def references(records):
    mapping={}; output=[]
    with localcontext() as ctx:
        ctx.prec=80
        iterator=iter(records)
        for record in iterator:
            if record.startswith(b'\x0c6a'):
                central=floating(record[19:27])*D(str(MU)); rcen=floating(record[51:59]); rmax=floating(record[59:67])
                mapping={}
                count=int(fraction(record[13:16])*BASE**3)+int(fraction(record[16:19])*BASE**3)
                for _ in range(count):
                    meta=next(iterator); code=int(fraction(meta[:3])*BASE**3)
                    mapping[code]=(meta[3:28].decode().strip(),floating(meta[28:36])*D(str(MU)))
            else:
                codes=[int(fraction(record[11:14])*BASE**3),int(fraction(record[14:17])*BASE**3)]
                values=[]; radii=[]
                for j,code in enumerate(codes):
                    x,v=decoded_state(record[25+24*j:49+24*j],rcen,rmax,central)
                    el,r=elements(central+mapping[code][1],x,v); values.append(el); radii.append(r)
                output.append(dict(time=float(floating(record[3:11])),distance=float(floating(record[17:25])),
                                   names=[mapping[code][0] for code in codes],elements=values,radii=radii))
    return output

def rows(path):
    result=[]
    for line in Path(path).read_text().splitlines():
        tokens=line.split()
        if len(tokens) not in (9,11): continue
        try:
            [float(t) for t in tokens[:1 if len(tokens)==9 else 3]+tokens[-7:]]
        except ValueError: continue
        result.append(tokens)
    return result

def validate_rows(path,records,body_name):
    expected=references(records); actual=rows(path); selected=[]
    for ref in expected:
        if body_name in ref['names']: selected.append((ref,ref['names'].index(body_name)))
    assert len(actual)==len(selected),(path,len(actual),len(selected))
    maximum=0.
    for row,(ref,j) in zip(actual,selected):
        assert row[-8]==ref['names'][1-j],row
        target=[ref['distance'],*ref['elements'][j],*ref['elements'][1-j]]
        for k,(token,value) in enumerate(zip(row[-7:],target)):
            observed=float(token)
            if math.isnan(value): assert math.isnan(observed); continue
            if math.isinf(value): assert observed==value; continue
            assert math.isfinite(observed),(token,value)
            decimals=[8,4,6,3,4,6,3][k]
            quantum=(10.**(int(token.split('E')[-1])-8) if 'E' in token else 10.**-decimals)
            tolerance=.5*quantum+64*math.ulp(value)
            if k in (1,4) and abs(ref['radii'][j if k==1 else 1-j]/value)<1e-5:
                r=ref['radii'][j if k==1 else 1-j]
                error=abs(r/observed-r/value)
                assert error<=1e-12+r*tolerance/max(abs(observed*value),1e-300),(token,value,error)
            else:
                error=abs(observed-value); assert error<=tolerance,(token,value,tolerance)
            maximum=max(maximum,error)
    return {'rows':len(actual),'maximum_printed_error':maximum}
