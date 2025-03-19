"""
Модифицированная модель Басса:
- Две конкурирующие компании (каждая делит клиентов на 3 категории: d, n, u)
- Один сток Potential (пул потенциальных клиентов)
- Потоки:
    1) Из Potential в компании (прямая реклама + сарафанное радио)
    2) Между компаниями (переманивание)
    3) Обратный поток недовольных в Potential (разочарование)
- Экономические показатели (Cost_Company1, Cost_Company2)

Соответствует заданию: добавлен конкурирующий контейнер,
три категории клиентов, разочарование, переманивание и т.д.
"""

from pathlib import Path
import numpy as np

from pysd.py_backend.statefuls import Integ
from pysd import Component

__pysd_version__ = "3.14.2"
__data = {"scope": None, "time": lambda: 0}
_root = Path(__file__).parent

component = Component()

#######################################################################
#                          CONTROL VARIABLES                          #
#######################################################################

_control_vars = {
    "initial_time": lambda: 0,
    "final_time": lambda: 100,
    "time_step": lambda: 1,
    "saveper": lambda: time_step(),
}

def _init_outer_references(data):
    for key in data:
        __data[key] = data[key]

@component.add(name="Time")
def time():
    """Текущее модельное время"""
    return __data["time"]()

@component.add(name="FINAL TIME")
def final_time():
    return __data["time"].final_time()

@component.add(name="INITIAL TIME")
def initial_time():
    return __data["time"].initial_time()

@component.add(name="SAVEPER")
def saveper():
    return __data["time"].saveper()

@component.add(name="TIME STEP")
def time_step():
    return __data["time"].time_step()

#######################################################################
#                         ПАРАМЕТРЫ МОДЕЛИ                            #
#######################################################################

@component.add(name="Total_Population")
def total_population():
    """
    Общее число людей (потенциальных + клиенты).
    """
    return 100000

@component.add(name="alpha1")
def alpha1():
    """
    Прямая реклама компании 1
    """
    return 0.05

@component.add(name="alpha2")
def alpha2():
    """
    Прямая реклама компании 2
    """
    return 0.05

@component.add(name="beta1")
def beta1():
    """
    Сарафанное радио (компания 1)
    """
    return 0.0005

@component.add(name="beta2")
def beta2():
    """
    Сарафанное радио (компания 2)
    """
    return 0.0005

@component.add(name="gamma1")
def gamma1():
    """
    Агрессивная реклама (переманивание) со стороны компании 1
    """
    return 0.0002

@component.add(name="gamma2")
def gamma2():
    """
    Агрессивная реклама (переманивание) со стороны компании 2
    """
    return 0.0002

@component.add(name="delta1")
def delta1():
    """
    Вероятность разочарования в компании 1 (возврат недовольных в Potential)
    """
    return 0.01

@component.add(name="delta2")
def delta2():
    """
    Вероятность разочарования в компании 2
    """
    return 0.01

@component.add(name="p_d")
def p_d():
    """
    Доля (вероятность), что пришедший клиент станет довольным
    """
    return 0.4

@component.add(name="p_n")
def p_n():
    """
    Доля, что станет нейтральным
    """
    return 0.3

@component.add(name="p_u")
def p_u():
    """
    Доля, что станет недовольным
    """
    return 0.3

#######################################################################
#                        STOCK: Potential                             #
#######################################################################

@component.add(
    name="Potential",
    comp_type="Stateful",
    comp_subtype="Integ",
    depends_on={"_integ_potential": 1},
    other_deps={
        "_integ_potential": {
            "initial": {},
            "step": {"inflow_to_potential": 1, "outflow_from_potential": 1}
        }
    },
)
def potential():
    """
    Пул потенциальных клиентов
    """
    return _integ_potential()

def dpotential_dt():
    return inflow_to_potential() - outflow_from_potential()

_integ_potential = Integ(
    lambda: dpotential_dt(),
    lambda: 90000,  # Изначально 90000 потенциальных
    "_integ_potential"
)

@component.add(
    name="inflow_to_potential",
    comp_type="Auxiliary",
    depends_on={"c1_u": 1, "c2_u": 1, "delta1": 1, "delta2": 1},
)
def inflow_to_potential():
    """
    Приток в Potential: разочарованные (C1_u и C2_u).
    """
    return delta1()*c1_u() + delta2()*c2_u()

@component.add(
    name="outflow_from_potential",
    comp_type="Auxiliary",
    depends_on={
        "flow_potential_to_c1": 1,
        "flow_potential_to_c2": 1
    },
)
def outflow_from_potential():
    """
    Отток из Potential в компании 1 и 2
    """
    return flow_potential_to_c1() + flow_potential_to_c2()

#######################################################################
#          FLOW: Potential -> Company 1 и Potential -> Company 2      #
#######################################################################

@component.add(
    name="flow_potential_to_c1",
    comp_type="Auxiliary",
    depends_on={
        "alpha1": 1, "beta1": 1, "potential": 1,
        "c1_d": 1, "total_population": 1
    },
)
def flow_potential_to_c1():
    """
    Приток из Potential в компанию 1:
    - alpha1 * Potential
    - beta1 * (C1_d / Total_Population) * Potential
    """
    return alpha1()*potential() + beta1()*(c1_d()/total_population())*potential()

@component.add(
    name="flow_potential_to_c2",
    comp_type="Auxiliary",
    depends_on={
        "alpha2": 1, "beta2": 1, "potential": 1,
        "c2_d": 1, "total_population": 1
    },
)
def flow_potential_to_c2():
    return alpha2()*potential() + beta2()*(c2_d()/total_population())*potential()

#######################################################################
#           STOCKS для компании 1: C1_d, C1_n, C1_u                   #
#######################################################################

## 1) C1_d
@component.add(
    name="C1_d",
    comp_type="Stateful",
    comp_subtype="Integ",
    depends_on={"_integ_c1_d": 1},
    other_deps={
        "_integ_c1_d": {
            "initial": {},
            "step": {
                "flow_to_c1_d": 1
            }
        }
    },
)
def c1_d():
    """
    Довольные клиенты компании 1
    """
    return _integ_c1_d()

def dc1_d_dt():
    return flow_to_c1_d()

_integ_c1_d = Integ(
    lambda: dc1_d_dt(),
    lambda: 1000,  # Изначально 1000 довольных в компании 1
    "_integ_c1_d"
)

@component.add(
    name="flow_to_c1_d",
    comp_type="Auxiliary",
    depends_on={
        "p_d": 1,
        "flow_potential_to_c1": 1,
        "flow_c2_to_c1": 1
    },
)
def flow_to_c1_d():
    """
    Приток довольных в C1:
    p_d * (поток из Potential->C1 + поток из C2->C1)
    """
    return p_d() * (flow_potential_to_c1() + flow_c2_to_c1())

## 2) C1_n
@component.add(
    name="C1_n",
    comp_type="Stateful",
    comp_subtype="Integ",
    depends_on={"_integ_c1_n": 1},
    other_deps={
        "_integ_c1_n": {
            "initial": {},
            "step": {
                "flow_to_c1_n": 1,
                "flow_from_c1_n": 1
            }
        }
    },
)
def c1_n():
    """
    Нейтральные клиенты компании 1
    """
    return _integ_c1_n()

def dc1_n_dt():
    return flow_to_c1_n() - flow_from_c1_n()

_integ_c1_n = Integ(
    lambda: dc1_n_dt(),
    lambda: 200,  # Изначально 200 нейтральных
    "_integ_c1_n"
)

@component.add(
    name="flow_to_c1_n",
    comp_type="Auxiliary",
    depends_on={
        "p_n": 1,
        "flow_potential_to_c1": 1,
        "flow_c2_to_c1": 1
    },
)
def flow_to_c1_n():
    return p_n() * (flow_potential_to_c1() + flow_c2_to_c1())

@component.add(
    name="flow_from_c1_n",
    comp_type="Auxiliary",
    depends_on={
        "flow_c1_to_c2": 1,
        "c1_n": 1, "c1_u": 1
    },
)
def flow_from_c1_n():
    """
    Нейтральные не уходят в Potential,
    но могут быть переманены компанией 2.
    """
    return flow_c1_to_c2() * fraction_c1_n()

@component.add(
    name="fraction_c1_n",
    comp_type="Auxiliary",
    depends_on={
        "c1_n": 1,
        "c1_u": 1
    },
)
def fraction_c1_n():
    denom = c1_n() + c1_u() + 1e-9
    return c1_n()/denom

## 3) C1_u
@component.add(
    name="C1_u",
    comp_type="Stateful",
    comp_subtype="Integ",
    depends_on={"_integ_c1_u": 1},
    other_deps={
        "_integ_c1_u": {
            "initial": {},
            "step": {
                "flow_to_c1_u": 1,
                "flow_from_c1_u": 1
            }
        }
    },
)
def c1_u():
    """
    Недовольные клиенты компании 1
    """
    return _integ_c1_u()

def dc1_u_dt():
    return flow_to_c1_u() - flow_from_c1_u()

_integ_c1_u = Integ(
    lambda: dc1_u_dt(),
    lambda: 50,  # Изначально 50 недовольных
    "_integ_c1_u"
)

@component.add(
    name="flow_to_c1_u",
    comp_type="Auxiliary",
    depends_on={
        "p_u": 1,
        "flow_potential_to_c1": 1,
        "flow_c2_to_c1": 1
    },
)
def flow_to_c1_u():
    """
    Приток недовольных в C1:
    p_u * (Potential->C1 + C2->C1)
    """
    return p_u() * (flow_potential_to_c1() + flow_c2_to_c1())

@component.add(
    name="flow_from_c1_u",
    comp_type="Auxiliary",
    depends_on={
        "delta1": 1,
        "c1_u": 1,
        "flow_c1_to_c2": 1,
        "fraction_c1_u": 1
    },
)
def flow_from_c1_u():
    """
    Недовольные:
    - Разочаровываются (delta1*c1_u) и уходят в Potential
    - Переманиваются компанией 2
    """
    return delta1()*c1_u() + flow_c1_to_c2()*fraction_c1_u()

@component.add(
    name="fraction_c1_u",
    comp_type="Auxiliary",
    depends_on={
        "c1_n": 1,
        "c1_u": 1
    },
)
def fraction_c1_u():
    denom = c1_n() + c1_u() + 1e-9
    return c1_u()/denom

#######################################################################
#           STOCKS для компании 2: C2_d, C2_n, C2_u                   #
#######################################################################

## 1) C2_d
@component.add(
    name="C2_d",
    comp_type="Stateful",
    comp_subtype="Integ",
    depends_on={"_integ_c2_d": 1},
    other_deps={
        "_integ_c2_d": {
            "initial": {},
            "step": {
                "flow_to_c2_d": 1
            }
        }
    },
)
def c2_d():
    """
    Довольные клиенты компании 2
    """
    return _integ_c2_d()

def dc2_d_dt():
    return flow_to_c2_d()

_integ_c2_d = Integ(
    lambda: dc2_d_dt(),
    lambda: 700,  # Изначально 700 довольных у компании 2
    "_integ_c2_d"
)

@component.add(
    name="flow_to_c2_d",
    comp_type="Auxiliary",
    depends_on={
        "p_d": 1,
        "flow_potential_to_c2": 1,
        "flow_c1_to_c2": 1
    },
)
def flow_to_c2_d():
    """
    Приток довольных в C2:
    p_d*(Potential->C2 + C1->C2)
    """
    return p_d() * (flow_potential_to_c2() + flow_c1_to_c2())

## 2) C2_n
@component.add(
    name="C2_n",
    comp_type="Stateful",
    comp_subtype="Integ",
    depends_on={"_integ_c2_n": 1},
    other_deps={
        "_integ_c2_n": {
            "initial": {},
            "step": {
                "flow_to_c2_n": 1,
                "flow_from_c2_n": 1
            }
        }
    },
)
def c2_n():
    """
    Нейтральные клиенты компании 2
    """
    return _integ_c2_n()

def dc2_n_dt():
    return flow_to_c2_n() - flow_from_c2_n()

_integ_c2_n = Integ(
    lambda: dc2_n_dt(),
    lambda: 150,
    "_integ_c2_n"
)

@component.add(
    name="flow_to_c2_n",
    comp_type="Auxiliary",
    depends_on={
        "p_n": 1,
        "flow_potential_to_c2": 1,
        "flow_c1_to_c2": 1
    },
)
def flow_to_c2_n():
    return p_n() * (flow_potential_to_c2() + flow_c1_to_c2())

@component.add(
    name="flow_from_c2_n",
    comp_type="Auxiliary",
    depends_on={
        "flow_c2_to_c1": 1,
        "c2_n": 1, "c2_u": 1
    },
)
def flow_from_c2_n():
    """
    Нейтральные 2:
    - Не уходят в Potential
    - Переманиваются компанией 1
    """
    return flow_c2_to_c1() * fraction_c2_n()

@component.add(
    name="fraction_c2_n",
    comp_type="Auxiliary",
    depends_on={
        "c2_n": 1,
        "c2_u": 1
    },
)
def fraction_c2_n():
    denom = c2_n() + c2_u() + 1e-9
    return c2_n()/denom

## 3) C2_u
@component.add(
    name="C2_u",
    comp_type="Stateful",
    comp_subtype="Integ",
    depends_on={"_integ_c2_u": 1},
    other_deps={
        "_integ_c2_u": {
            "initial": {},
            "step": {
                "flow_to_c2_u": 1,
                "flow_from_c2_u": 1
            }
        }
    },
)
def c2_u():
    """
    Недовольные клиенты компании 2
    """
    return _integ_c2_u()

def dc2_u_dt():
    return flow_to_c2_u() - flow_from_c2_u()

_integ_c2_u = Integ(
    lambda: dc2_u_dt(),
    lambda: 50,
    "_integ_c2_u"
)

@component.add(
    name="flow_to_c2_u",
    comp_type="Auxiliary",
    depends_on={
        "p_u": 1,
        "flow_potential_to_c2": 1,
        "flow_c1_to_c2": 1
    },
)
def flow_to_c2_u():
    """
    Приток недовольных в C2
    """
    return p_u()*(flow_potential_to_c2() + flow_c1_to_c2())

@component.add(
    name="flow_from_c2_u",
    comp_type="Auxiliary",
    depends_on={
        "delta2": 1,
        "c2_u": 1,
        "flow_c2_to_c1": 1,
        "fraction_c2_u": 1
    },
)
def flow_from_c2_u():
    """
    Недовольные 2:
    - Могут разочароваться (delta2) и уйти в Potential
    - Могут быть переманены компанией 1
    """
    return delta2()*c2_u() + flow_c2_to_c1()*fraction_c2_u()

@component.add(
    name="fraction_c2_u",
    comp_type="Auxiliary",
    depends_on={
        "c2_n": 1,
        "c2_u": 1
    },
)
def fraction_c2_u():
    denom = c2_n() + c2_u() + 1e-9
    return c2_u()/denom

#######################################################################
#                     FLOWS: Company1 <-> Company2                    #
#######################################################################

@component.add(
    name="flow_c1_to_c2",
    comp_type="Auxiliary",
    depends_on={
        "gamma2": 1, "c2_d": 1,
        "c1_n": 1, "c1_u": 1,
        "total_population": 1
    },
)
def flow_c1_to_c2():
    """
    Переманивание 1->2:
    gamma2 * c2_d * (c1_n + c1_u) / total_population
    """
    return gamma2()*c2_d()*(c1_n()+c1_u())/(total_population()+1e-9)

@component.add(
    name="flow_c2_to_c1",
    comp_type="Auxiliary",
    depends_on={
        "gamma1": 1, "c1_d": 1,
        "c2_n": 1, "c2_u": 1,
        "total_population": 1
    },
)
def flow_c2_to_c1():
    """
    Переманивание 2->1:
    gamma1 * c1_d * (c2_n + c2_u) / total_population
    """
    return gamma1()*c1_d()*(c2_n()+c2_u())/(total_population()+1e-9)

#######################################################################
#                       ЭКОНОМИЧЕСКИЕ ПОКАЗАТЕЛИ                      #
#######################################################################

@component.add(
    name="Cost_Company1",
    comp_type="Auxiliary",
    depends_on={
        "c1_d": 1, "c1_n": 1, "c1_u": 1
    },
)
def cost_company1():
    """
    Довольные: 100 у.е. за каждую сотню + 5 у.е. за каждого
    Нейтральные: 1 у.е.
    Недовольные: 4 у.е.
    """
    dval = c1_d()
    nval = c1_n()
    uval = c1_u()
    hundreds = np.floor(dval/100)
    return 100*hundreds + 5*dval + 1*nval + 4*uval

@component.add(
    name="Cost_Company2",
    comp_type="Auxiliary",
    depends_on={
        "c2_d": 1, "c2_n": 1, "c2_u": 1
    },
)
def cost_company2():
    dval = c2_d()
    nval = c2_n()
    uval = c2_u()
    hundreds = np.floor(dval/100)
    return 100*hundreds + 5*dval + 1*nval + 4*uval
