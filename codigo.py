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


elif(opcao == 'Interpolação Polinomial'):
    metodo_interpol = st.sidebar.radio(
     'Escolha a forma',
     ('Lagrange', 'Newton'))

    if(metodo_interpol == 'Lagrange'):
        st.title('Interpolação Polinomial - Lagrange')

    elif(metodo_interpol == 'Newton'):
        st.title('Interpolação Polinomial - Newton')
