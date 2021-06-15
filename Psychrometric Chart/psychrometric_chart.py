import autograd.numpy as np
from newton_raphson import function_zero

import matplotlib.pyplot as plt
from matplotlib import collections  as mc
from matplotlib import style

import pandas as pd

from copy import deepcopy


class PsychrometricSolver:
    def __init__(self,
        antoine_params,
        fluid_properties,
        atmosphere_properties,
        Dab # m^2/s
        ):

        self.antoine_params = antoine_params
        self.fluid_properties = fluid_properties
        self.atmosphere_properties = atmosphere_properties
        self.Dab = Dab

    def calc_Ca_inf(self, T_BS, UR):
        '''
        input:
        T_BS - Celsius
        UR - %
        output:
        Ca_inf - kmol / m^3
        '''

        R = 8314 # j/(kmol * K)

        P_sat = self.P_sat(T_BS) * 1000 # Pa
        Pa_inf = P_sat * (UR / 100) # Pa
        Ca_inf = Pa_inf / (R * (T_BS + 273.15)) # kmol / m^3

        return Ca_inf

    def calc_UA(self, T_BS, UR):
        '''
        input:
        T_BS - Celsius
        UR - %
        output:
        UA - g_a/kg_air
        '''

        R = 8314 # j/(kmol * K)
        Ca_inf = self.calc_Ca_inf(T_BS, UR) # kmol / m^3
        m_a = Ca_inf * self.fluid_properties['MM'] * 1000 # g/m^3

        C_t = self.atmosphere_properties['P_atm'] / (R * (T_BS + 273.15)) # kmol / m^3
        C_air = C_t - Ca_inf
        m_air = C_air * self.atmosphere_properties['MM'] # kg / m^3

        UA = m_a / m_air # g_a/kg_air
        return UA

    def calc_T_BU(self, T_BS, UR):
        '''
        input:
        T_BS - Celsius
        UR - %
        output:
        T_BU - Celsius
        '''

        rho = self.atmosphere_properties['rho']
        cp = self.atmosphere_properties['cp']
        mi = self.atmosphere_properties['mi']
        k = self.atmosphere_properties['k']
        MM_fluid = self.fluid_properties['MM']
        vap_heat = self.fluid_properties['vap_heat']

        Sc = mi / (rho * self.Dab)

        alpha = k / (rho * cp)
        Pr = mi / (rho * alpha)

        Ca_inf = self.calc_Ca_inf(T_BS, UR) # kmol / m^3

        def T_func_to_zero(T):
            R = 8315 # j/(kmol * K)
            P_sat = self.P_sat(T) * 1000 # Pa
            Cas = P_sat / (R * (T + 273.15)) # kmol / m^3
            Ca_inf_func = Cas - rho * cp * ((Sc/Pr)**(2/3)) * (T_BS - T) / (MM_fluid * vap_heat) # kmol / m^3
            return Ca_inf_func - Ca_inf

        T_BU = function_zero(T_func_to_zero, initial_guess=T_BS)

        return T_BU

    def calc_T_BS(self, T_BU, UR):
        '''
        input:
        T_BU - Celsius
        UR - %
        output:
        T_BS - Celsius
        '''

        rho = self.atmosphere_properties['rho']
        cp = self.atmosphere_properties['cp']
        mi = self.atmosphere_properties['mi']
        k = self.atmosphere_properties['k']
        MM_fluid = self.fluid_properties['MM']
        vap_heat = self.fluid_properties['vap_heat']

        Sc = mi / (rho * self.Dab)

        alpha = k / (rho * cp)
        Pr = mi / (rho * alpha)

        R = 8315 # j/(kmol * K)
        P_sat = self.P_sat(T_BU) * 1000 # Pa
        Cas = P_sat / (R * (T_BU + 273.15)) # kmol / m^3

        def T_func_to_zero(T):


            Ca_inf_antoine = self.calc_Ca_inf(T, UR) # kmol / m^3
            Ca_inf_func = Cas - rho * cp * ((Sc/Pr)**(2/3)) * (T - T_BU) / (MM_fluid * vap_heat) # kmol / m^3
            return Ca_inf_func - Ca_inf_antoine

        T_BS = function_zero(T_func_to_zero, initial_guess=T_BU)

        return T_BS

    def calc_UR(self, T_BS, T_BU):
        '''
        input:
        T_BS - Celsius
        T_BU - Celsius
        output:
        UA - g_a/kg_air
        '''

        R = 8314 # j/(kmol * K)
        rho = self.atmosphere_properties['rho']
        cp = self.atmosphere_properties['cp']
        mi = self.atmosphere_properties['mi']
        k = self.atmosphere_properties['k']
        MM_fluid = self.fluid_properties['MM']
        vap_heat = self.fluid_properties['vap_heat']

        Sc = mi / (rho * self.Dab)

        alpha = k / (rho * cp)
        Pr = mi / (rho * alpha)

        P_sat_T_BU = self.P_sat(T_BU) * 1000 # Pa
        Cas = P_sat_T_BU / (R * (T_BU + 273.15)) # kmol / m^3
        Ca_inf = Cas - rho * cp * ((Sc/Pr)**(2/3)) * (T_BS - T_BU) / (MM_fluid * vap_heat) # kmol / m^3
        Pa_inf = Ca_inf * R * (T_BS + 273.15) # Pa

        P_sat_T_BS = self.P_sat(T_BS) * 1000 # Pa

        UR = 100* (Pa_inf / P_sat_T_BS)

        return UR

    def P_sat(self, T):
        '''
        input:
        T - Celsius
        output:
        P - kPa
        '''

        A = self.antoine_params['A']
        B = self.antoine_params['B']
        C = self.antoine_params['C']
        P = np.exp(A - B / (T + C))

        return P


class PsychrometricChart:
    def __init__(self,
        psychrometric_solver,
        title,
        filename,
        background_color='w',
        T_BS_range=(0, 50),
        num_T_BS_points=51,
        UR_pct_range=(10, 100),
        num_UR_points=19,
        UA_display_range=False,
        UA_display_step=100,
        ):

        self.psychrometric_solver = psychrometric_solver
        self.title = title
        self.filename = filename
        self.background_color = background_color

        self.T_BS_range = T_BS_range
        self.num_T_BS_points = num_T_BS_points
        self.UR_pct_range = UR_pct_range
        self.num_UR_points = num_UR_points

        self.UA_display_range = UA_display_range
        self.UA_display_step = UA_display_step

        self.T_BS_values = np.linspace(self.T_BS_range[0], self.T_BS_range[1], num=self.num_T_BS_points)
        self.UR_values = np.linspace(self.UR_pct_range[0], self.UR_pct_range[1], num=self.num_UR_points)

        self.T_BU_array = np.zeros((self.num_T_BS_points, self.num_UR_points))
        self.UA_array = np.zeros((self.num_T_BS_points, self.num_UR_points))

    def generate_arrays(self):
        for i, T_BS in enumerate(self.T_BS_values):
            for j, UR in enumerate(self.UR_values):
                self.T_BU_array[i][j] = self.psychrometric_solver.calc_T_BU(T_BS, UR)
                self.UA_array[i][j] = self.psychrometric_solver.calc_UA(T_BS, UR)

        return self.T_BU_array, self.UA_array

    def generate_spreadsheet(self):
        T_BU_array, UA_array = self.generate_arrays()

        T_BU_df = pd.DataFrame(T_BU_array, columns=self.UR_values)
        T_BU_df.index = self.T_BS_values

        UA_df = pd.DataFrame(UA_array, columns=self.UR_values)
        UA_df.index = self.T_BS_values

        T_BU_df.to_excel('Temperaturas de Bulbo Umido.xlsx')
        UA_df.to_excel('Umidades Absolutas.xlsx')

        print('spreadsheet generated')

    def calc_isoenthalpic_lines(self, T_BS_values):
        T_BU_values = deepcopy(T_BS_values)

        isoenthalpic_lines = []
        for T_BU in T_BU_values:
            line = []
            for T_BS in T_BS_values:
                UR = self.psychrometric_solver.calc_UR(T_BS, T_BU)
                line.append(self.psychrometric_solver.calc_UA(T_BS, UR))

            isoenthalpic_lines.append(deepcopy(line))

        return isoenthalpic_lines

    def get_isoenthalpic_lines_labels(self, T_BS_values):
        T_BU_values = deepcopy(T_BS_values)

        labels = []
        for i, T_BU in enumerate(T_BU_values):
            if i % 5 == 0 and i != 0:
                label_coord = (T_BU_values[i], self.psychrometric_solver.calc_UA(T_BU, 100))
                labels.append((int(T_BU), label_coord))

        return labels

    def define_base_ax(self, fig, ax, T_BS_values, UA_array, UA_display_range, UA_display_step):
        for UA_column_for_UR, UR_value in zip(UA_array.transpose(), self.UR_values):
            ax.plot(
                self.T_BS_values, UA_column_for_UR,
                c='r',
                linewidth=0.3,
                alpha=0.65,
                zorder=3
                )

            label_T_BS = 0.9 * self.T_BS_range[1]
            label_UA = self.psychrometric_solver.calc_UA(label_T_BS, UR_value) #feio
            ax.annotate(
                f'{int(UR_value)}%',
                xy=(label_T_BS, label_UA),
                bbox=dict(boxstyle="round", fc="w", linewidth=0.2),
                size=3
                )

        isoenthalpic_lines = self.calc_isoenthalpic_lines(T_BS_values)

        for line in isoenthalpic_lines:
            ax.plot(
                self.T_BS_values, line,
                c='g',
                linewidth=0.3,
                alpha=0.65,
                zorder=1
                )

        isoenthalpic_lines_labels = self.get_isoenthalpic_lines_labels(self.T_BS_values)
        for label in isoenthalpic_lines_labels:
            ax.annotate(
                label[0],
                xy=label[1],
                xytext=(
                    label[1][0]-(self.T_BS_range[1]-self.T_BS_range[0])/100,
                    label[1][1]+(UA_display_range[1]-UA_display_range[0])/33,
                    ),
                size=6,
                rotation=66,
                arrowprops=dict(arrowstyle="-", linewidth=0.3)
                )

        ax.set_xlim(self.T_BS_range[0], self.T_BS_range[1])
        ax.set_xticks(self.T_BS_values, minor=True)

        ax.set_ylim(UA_display_range[0], UA_display_range[1])
        UA_axis = np.arange(UA_display_range[0], UA_display_range[1], UA_display_step)
        ax.set_yticks(UA_axis, minor=True)
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()

        ax.fill_between(
            self.T_BS_values, UA_array[:,-1], UA_display_range[1],
            color=self.background_color,
            alpha=1,
            zorder=2,
            linewidth=0.0
            )

        ax.grid(color='k', linestyle='-', linewidth=0.3)
        ax.grid(which='minor', color='k', linestyle='-', linewidth=.2, alpha=0.6)

    def plot_chart(self):
        T_BU_array, UA_array = self.generate_arrays()
        if not self.UA_display_range:
            self.UA_display_range = (0, UA_array[-1][-1] * 0.5)

        fig = plt.figure(facecolor=self.background_color)
        ax = fig.add_subplot(1, 1, 1, facecolor=self.background_color)
        ax.set_title(self.title)
        ax.set_xlabel('Temperatura de Bulbo Seco (°C)', size=6)
        ax.set_ylabel('Umidade Absoluta (g/kg)', size=6)

        self.define_base_ax(fig, ax, self.T_BS_values, UA_array, self.UA_display_range, self.UA_display_step)

        ax.annotate(
            'Temperatura de Bulbo Úmido (°C)',
            xy=(
                (self.T_BS_range[1]-self.T_BS_range[0]) * 0.1,
                (self.UA_display_range[1]-self.UA_display_range[0]) * 0.3,
                ),
            size=6,
            rotation=30,
            )
        ax.annotate(
            f"P = {self.psychrometric_solver.atmosphere_properties['P_atm']} Pa",
            xy=(
                (self.T_BS_range[1]-self.T_BS_range[0]) * 0.05,
                (self.UA_display_range[1]-self.UA_display_range[0]) * 0.9,
                ),
            size=6,
            )
        fig.savefig(f'{self.filename}.jpeg', dpi=2400)

        print('chart generated')


antoine_params = {
    'A': 13.8193,
    'B': 2696.04,
    'C': 224.317,
}

fluid_properties = {
    'MM': 86.177,
    'vap_heat': 334.944e3 # J/kg
}

atmosphere_properties = {
    'MM': 28.6,
    'P_atm': 101325, # Pa
    'cp': 1008,     # J/(kg*K)
    'rho': 1.14,     # kg/m^3
    'mi': 206.7e-7,  # Pa*s
    'k': 29.5e-3,    # W/(m*K)
}

psychrometric_solver = PsychrometricSolver(
    antoine_params,
    fluid_properties,
    atmosphere_properties,
    Dab=8e-6 # m^2/s
    )

psychrometric_chart = PsychrometricChart(
    psychrometric_solver,
    title='Carta Psicrométrica Hexano',
    filename='Carta Psicrométrica Hexano',
    background_color='w',
    T_BS_range=(0, 50),
    num_T_BS_points=51,
    UR_pct_range=(10, 100),
    num_UR_points=19,
    UA_display_range=(0, 1700),
    UA_display_step=100,
    )

psychrometric_chart.generate_spreadsheet()
psychrometric_chart.plot_chart()
