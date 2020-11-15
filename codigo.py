import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
# import matplotlib.pyplot as plt

st.sidebar.title('Projeto Final')
#st.title('Projeto Final')

# Calculo de zero de funções:
# • Método Bissecção
#  - Função
#  - a
#  - b
#  - max
# • Método Newton
#  - funcao
#  - x0
#  - max

# Sistemas Lineares:
# • Gauss (direto)
# • LU (direto)
# • Jacobi (iterativo)
# • Gauss-Seidel (iterativo)

# Interpolação Polinomial:
# • Forma de Lagrange
# • Forma de Newton

opcao = st.sidebar.selectbox(   #Caixa de seleção
    'Escolha um cálculo',
    (
        'Selecione uma opção',
        'Zero de Funções',
        'Sistemas Lineares',
        'Interpolação Polinomial',
    )
)

if(opcao == 'Zero de Funções'):
    metodo_zero = st.sidebar.radio(   #Alternativas
     'Escolha o método',
     ('Bissecção', 'Newton'))
    
    if(metodo_zero == 'Bissecção'):
        # ...
        st.title('Zero de Funções - Bisseção')
        funcao = st.sidebar.text_input("Equação:")
        a = st.sidebar.text_input("Valor de a:")
        b = st.sidebar.text_input("Valor de b:")
        if (funcao is not None) and ((a is not None)and (b is not None)):
            None
            
    elif(metodo_zero == 'Newton'):
        st.title('Zero de Funções - Newton')
        funcao = st.sidebar.text_input("Equação:")
        x0 = st.sidebar.text_input("Aproximação inicial (x0):")
        max_it = st.sidebar.text_input("Qntd máxima de iterações:")

elif(opcao == 'Sistemas Lineares'):
    metodo_linear = st.sidebar.radio(
     'Escolha o método',
     ('Gauss (direto)', 'LU (direto)', 'Jacobi (iterativo)', 'Gauss-Seidel (iterativo)'))

    if(metodo_linear == 'Gauss (direto)'):
        st.title('Sistema Linear Direto - Gauss')

    elif(metodo_linear == 'LU (direto)'):
        st.title('Sistema Linear Direto - LU')

    elif(metodo_linear == 'Jacobi (iterativo)'):
        st.title('Sistema Linear Iterativo - Jacobi')

    elif(metodo_linear == 'Gauss-Seidel (iterativo)'):
        st.title('Sistema Linear Iterativo - Gauss-Seidel')
        name_dict = {"Anord":"", "Bernald":""}

        for k, v in name_dict.items():
            name_dict[k] = st.text_input(k, v)
            st.write(name_dict[k])


elif(opcao == 'Interpolação Polinomial'):
    metodo_interpol = st.sidebar.radio(
     'Escolha a forma',
     ('Lagrange', 'Newton'))

    if(metodo_interpol == 'Lagrange'):
        st.title('Interpolação Polinomial - Lagrange')
        # Pegar e converter as coordenadas em uma lista 2d
        pontos = st.sidebar.text_input("Conjunto de dados (Formato: x0 y0;x1 y1;x2 y2):")    #"-1 4;0 1;2 -1"
        pontos = pontos.split(";")                  #['-1 4', '0 1', '2 -1']
        pontos = [i.split(" ") for i in pontos]     #[['-1', '4'], ['0', '1'], ['2', '-1']]
        dados = [list(map(int,i)) for i in pontos] #[[-1, 4], [0, 1], [2, -1]]
        
        x = st.sidebar.text_input("Valor de x:")    #"1,5"
        x = x.replace(",", ".")     #"1,5" -> "1.5"

        if pontos is not None:
            st.write(f"Dados = {dados}")
            st.write(f"x = {x}")

            formula_l = '(({x}-{x1})*({x}-{x2})) / (({x0}-{x1})*({x0}-{x2}))'.format(x=x, x0=dados[0][0], x1=dados[1][0], x2=dados[2][0])

            formula = 'L0*{y0} + L1*{y1} + L2*{y2}'.format(y0=dados[0][1], y1=dados[1][1], y2=dados[2][1])
            p = lambda x: eval(formula)

    elif(metodo_interpol == 'Newton'):
        st.title('Interpolação Polinomial - Newton')
