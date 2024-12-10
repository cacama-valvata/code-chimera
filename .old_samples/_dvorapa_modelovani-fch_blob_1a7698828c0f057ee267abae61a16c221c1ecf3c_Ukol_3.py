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

# Kritické veličiny pro chlorobenzen (Young, 1910; NIST, 2018)
Tk_chb = 632.35 # K
pk_chb = 45.191 * 10**5 # Pa

# Parametry Antoinovy rovnice (Brown, 1952; NIST, 2018)
A = 4.11083
B = 1435.675
C = -55.124

## Antoine
def p(T):
    return (10**(A-(B/(T+C)))) * 10**5
numdiff_p = derivative(p, Tp, dx=1e-6)

p = (10**(A-(B/(T+C)))) * 10**5
analdiff_p = diff(p, T).subs(T, Tp)

## Redlich-Kwong
a = (0.42748*(R**2)*(Tk**2.5))/pk
b = (0.08664*R*Tk)/pk
rovnice_ = ((R*Tp)/(Vm-b)) - (a/(sqrt(Tp)*Vm*(Vm+b))) - p.subs(T, Tp)

koreny_chb = solve(rovnice_.subs([(Tk, Tk_chb), (pk, pk_chb)]), Vm)
koreny_chb_re = []
for k in koreny_chb:
    if im(k) < 10**21:
        koreny_chb_re.append(re(k))

Vm_g = max(koreny_chb_re)
Vm_l = min(koreny_chb_re)
assert Vm_g != Vm_l
del_Vm = Vm_g - Vm_l

## Clapeyron
del_H_num = numdiff_p * Tp * del_Vm
del_H_anal = analdiff_p * Tp * del_Vm
del_H_num = N(del_H_num/1000)
del_H_anal = N(del_H_anal/1000)

# Výsledné výparné entalpie
print('del_H_vyp_num = ' + str(del_H_num) + ' kJ/mol\n')
print('del_H_vyp_anal = ' + str(del_H_anal) + ' kJ/mol\n')
print('del_H_vyp_tab = 41 ± 4 kJ/mol (průměr více exp. hodnot; NIST 2018)')

#http://docs.sympy.org/latest/tutorial/calculus.html
#https://docs.scipy.org/doc/scipy/reference/generated/scipy.misc.derivative.html
#https://webbook.nist.gov/cgi/cbook.cgi?ID=C108907&Units=SI&Mask=4#Thermo-Phase
#https://stackoverflow.com/questions/9876290/how-do-i-compute-derivative-using-numpy
