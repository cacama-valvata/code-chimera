xyz = 1
### Skript pro výpočet výparné entalpie chlorobenzenu
from sympy import *
from scipy.misc import derivative
Vm = Symbol('Vm')
Tk = Symbol('Tk')
pk = Symbol('pk')
T = Symbol('T')
R = 8.314

## Nadefinované podmínky
Tp = 273.15 + 25 # K
p = 101325 # Pa

# Kritické veličiny pro chlorobenzen (Young, 1910; NIST, 2018)
Tk_chb = 632.35 # K
pk_chb = 45.191 * 10**5 # Pa
# Parametry Antoinovy rovnice (Brown, 1952; NIST, 2018)
A = 4.11083
B = 1435.675
C = -55.124

## Redlich-Kwong
a = (0.42748*(R**2)*(Tk**2.5))/pk
b = (0.08664*R*Tk)/pk
rovnice_ = ((R*Tp)/(Vm-b)) - (a/(sqrt(Tp)*Vm*(Vm+b))) - p

koreny_chb = solve(rovnice_.subs([(Tk, Tk_chb), (pk, pk_chb)]), Vm)
koreny_chb_re = []
for k in koreny_chb:
    if im(k) < 10**21:
        koreny_chb_re.append(re(k))
Vm_chb = min(koreny_chb_re)

## Antoine
p = 10**(A-(B/(T+C)))
analdiff_p = diff(p, T)
analdiff_p = analdiff_p.subs(T, xyz)
print(N(analdiff_p))
def p(T):
    return 10**(A-(B/(T+C)))
numdiff_p = derivative(p, xyz, dx=1e-6)
print(N(numdiff_p))

# Výsledné výsledky
#print('ro_chb_rk = ' + str(ro_chb_rk) + ' mol/l\n')
#http://docs.sympy.org/latest/tutorial/calculus.html
#https://docs.scipy.org/doc/scipy/reference/generated/scipy.misc.derivative.html
#https://webbook.nist.gov/cgi/cbook.cgi?ID=C108907&Units=SI&Mask=4#Thermo-Phase
#https://stackoverflow.com/questions/9876290/how-do-i-compute-derivative-using-numpy
