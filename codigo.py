import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import re
import math
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# Calculo de zero de funções:
# ○ Método Bissecção
#  - Função, a, b, max
# ○ Método Newton
#  - funcao, x0, max

# Sistemas Lineares:
# ○ Gauss (direto)
# • LU (direto)
# ○ Jacobi (iterativo)
# ○ Gauss-Seidel (iterativo)

# Interpolação Polinomial:
# • Forma de Lagrange
#   - Coordenadas, x
# ○ Forma de Newton
#   - Coordenadas, x

def bisseccao(f, interv_a, interv_b, N):
    # -- Exceção: f(a)*f(b) >= 0 --
    if f(interv_a)*f(interv_b) >= 0:
        st.write(f"Falha no método da Bissecção: o produto de f(a) e f(b) é maior ou igual a zero.")
        return None

    # -- Processo padrão --
    a, b = interv_a, interv_b
    dados = [[], [], [], [], [], []]

    for K in range(0, N+1):
        # Adicionando a primeira linha
        xk = (a+b)/2    #Calculo da média do intervalo
        fxk = f(xk)     #E a função dessa média
        if (K == 0):
            dados[0].append(0)
            dados[1].append(a)
            dados[2].append(b)
            dados[3].append(xk)
            dados[4].append(fxk)
            if (fxk == 0 or abs(fxk) <= 10e-3):
                dados[5].append("True")
            else:
                dados[5].append("False")

        # Adicionando o resto das linhas
        else:
            # -- Substituições --
            if (f(a)*fxk < 0):
                a, b = a, xk    #b é substituído por xk
            elif (f(b)*fxk < 0):
                a, b = xk, b    #a é substituído

            xk = (a+b)/2
            fxk = f(xk)

            dados[0].append(K)
            dados[1].append(a)
            dados[2].append(b)
            dados[3].append(xk)
            dados[4].append(fxk)

            # -- Caso ache uma raíz real --
            if (fxk == 0):
                st.write(f"Solução exata encontrada!")
                dados[5].append("True")
                return xk, dados

            # -- Caso atinja condição de parada --
            elif (abs(fxk) <= 10e-3):
                st.write(f"Método interrompido pela condição de parada!")
                dados[5].append("True")
                return xk, dados
            
            else:
                dados[5].append("False")
    return xk, dados

def newton_zero(f, aprox_inic, max):
    dados = [[], [], [], []]
    xk = aprox_inic

    for K in range(0, max+1):
        # -- Adicionando a primeira iteração --
        if (K == 0):
            fxk = f(xk)
            dados[0].append(K)
            dados[1].append(xk)
            dados[2].append(f(xk))

            if (abs(fxk) <= 10e-3 or fxk == 0):
                st.write(f"Condição de parada atingida!")
                dados[3].append("True")
                return xk, dados
            else:
                dados[3].append("False")
    
        # -- Adicionando as outras iterações --
        else:
            f_deriv = derivada(f)
            xk = xk - (f(xk)/f_deriv(xk))
            fxk = f(xk)
            dados[0].append(K)
            dados[1].append(xk)
            dados[2].append(fxk)

            # Caso encontre a solução
            if (fxk == 0):
                st.write(f"\nSolução exata encontrada!")
                dados[3].append("True")
                return xk, dados

            # Caso atinja condição de parada
            elif (abs(fxk) <= 10e-3):
                st.write(f"Método interrompido pela condição de parada!")
                dados[3].append("True")
                return xk, dados

            # Nenhum dos outros
            else:
                dados[3].append("False")
    return xk, dados

def derivada(f):
  f_deriv = lambda x: math.exp(x)-8*x
  return f_deriv

def plotarTabela(dados, metodo):
    #chart = st.line_chart(dados)
    if(metodo == "bisseccao"):   #Plotar tabela da Bisseccao
        tabela = go.Figure(
            data=[go.Table(columnwidth = [100,100,100,100,200,100],
                      header = dict(values=['K', 'a', 'b', 'xk', 'f(xk)', 'Parar?']),
                      cells = dict(values=dados)) ])
        st.write(tabela)

    elif(metodo == "newton"):   #Plotar tabela de Newton
        tabela = go.Figure(
            data=[go.Table(columnwidth = [100,250,250,100],
                      header = dict(values=['K', 'xk', 'f(xk)', 'Parar?']),
                      cells = dict(values=dados)) ])
        st.write(tabela)

    elif(metodo == "lu"):   #Plotar matrix LU
        #for linha in dados: #Matriz transposta apenas pra printar
            #dados = [[dados[j][i] for j in range(len(dados))] for i in range(len(dados[0]))] 
        #tabela = go.Figure(
        #    data=[go.Table(columnwidth = [250,250,250],
        #              header = dict(fill_color='white'),
        #              cells = dict(values=dados)) ])
        #st.write(tabela)
        tabela = pd.DataFrame({ '0': [dados[0][0], dados[1][0], dados[2][0]],
                                '1': [dados[0][1], dados[1][1], dados[2][1]],
                                '2': [dados[0][2], dados[1][2], dados[2][2]] })
        st.write(tabela)
    else:
        tabela = pd.DataFrame({ 'x': [dados[0][0], dados[1][0], dados[2][0]],
                                'y': [dados[0][1], dados[1][1], dados[2][1]] })
        st.write(tabela)

def lu(eq1, eq2, eq3):
    st.write(f"**-- Etapa 1: Equações**")
    eq1 = lista_equacao(equacao1)
    eq2 = lista_equacao(equacao2)
    eq3 = lista_equacao(equacao3)
    st.write(f"Equação 1 = {eq1}")
    st.write(f"Equação 2 = {eq2}")
    st.write(f"Equação 3 = {eq3}")

    st.write(f"**-- Etapa 2: Matriz A**")
    a = [[eq1[0][0], eq1[1][0], eq1[2][0]], [eq2[0][0], eq2[1][0], eq2[2][0]], [eq3[0][0], eq3[1][0], eq3[2][0]]]
    st.write(f"Matriz A = {a}")
    plotarTabela(a,"lu")

    st.write(f"**-- Etapa 3: l21 e l31**")
    u11, u12, u13 = float(a[0][0]), float(a[0][1]), float(a[0][2])
    l21 = float(a[1][0]) / u11
    st.write(f"l21 = {float(a[1][0])} / {u11} = {l21}")
    l31 = float(a[2][0]) / u11
    st.write(f"l31 = {float(a[2][0])} / {u11} = {l31}")

    st.write(f"**-- Etapa 4: u22 e u23**")
    u22 = float(a[1][1])-l21*u12
    st.write(f"u22 = {float(a[1][1])} - {l21} * {u12} = {u22}")
    u23 = float(a[1][2])-l21*u13
    st.write(f"u23 = {float(a[1][2])} - {l21} * {u13} = {u23}")

    st.write(f"**-- Etapa 5: l32**")
    l32 = (float(a[2][1])-l31*u12) / u22
    st.write(f"l32 = {float(a[2][1])} - {l31} * {u12} = {l32}")

    st.write(f"**-- Etapa 6: u33**")
    u33 = float(a[2][2])-l31*u13-l32*u23
    st.write(f"u33 = {float(a[2][2])} - {l31}\*{u13} - {l32}\*{u23} = {u33}")

    st.write(f"**-- Etapa 7: Matrizes L e U**")
    l = [[1,0,0], [l21,1,0], [l31,l32,1]]
    u = [[a[0][0],a[0][1],a[0][2]], [0,u22,u23], [0,0,u33]]

    st.write(f"Matriz L = {l}")
    plotarTabela(l,"lu")
    st.write(f"Matriz U = {u}")
    plotarTabela(u,"lu")

    #st.write(f"-- Etapa 8: Multiplicadores --")
    #equacao_1 = [[u[0][0],"x1"], [u[0][1],"x2"], [u[0][2],"x3"], [eq1[4][0]]]
    #equacao_2 = [[u[1][1],"x2"], [u[1][2],"x3"], [eq2[4][0]]]
    #equacao_3 = [[u[2][2],"x3"], [eq3[4][0]]]
    #st.write(f"{equacao_1}")
    #st.write(f"{equacao_2}")
    #st.write(f"{equacao_3}")

def lista_equacao(eq):
    eq = eq.replace(",", ".")
    eq = re.split(" \+ | | = ", eq)
    eq = [i.split("*") for i in eq]
    return eq

def lagrange(dados, x):
    st.write(f"**-- Etapa 1: Coletar os dados**")
    st.write(f"x = {x}")
    st.write(f"Dados = {dados}")
    plotarTabela(dados, "lagrange")

    x0, x1, x2 = dados[0][0], dados[1][0], dados[2][0]
    y0, y1, y2 = dados[0][1], dados[1][1], dados[2][1]

    st.write(f"**-- Etapa 2: Calcular os L's**")
    formula_l0 = f'((x-{x1})*(x-{x2})) / (({x0}-{x1})*({x0}-{x2}))'
    formula_l1 = f'((x-{x0})*(x-{x2})) / (({x1}-{x0})*({x1}-{x2}))'
    formula_l2 = f'((x-{x0})*(x-{x1})) / (({x2}-{x0})*({x2}-{x1}))'
    #formula_l = '(({x}-{x1})*({x}-{x2})) / (({x0}-{x1})*({x0}-{x2}))'.format(x=x, x0=dados[0][0], x1=dados[1][0], x2=dados[2][0])

    l0 = lambda x: eval(formula_l0)
    l1 = lambda x: eval(formula_l1)
    l2 = lambda x: eval(formula_l2)

    novo_l0 = formula_l0.replace("*", "\*").replace("x", str(x))   #Muda o asterisco por causa da sintax Markdown e substitui o x
    st.write(f"L0 = ((x-x1)\*(x-x2)) / ((x0-x1)\*(x0-x2)) = {novo_l0} = {l0(x)}")

    novo_l1 = formula_l1.replace("*", "\*").replace("x", str(x))
    st.write(f"L1 = ((x-x0)\*(x-x2)) / ((x1-x0)\*(x1-x2)) = {novo_l1} = {l1(x)}")

    novo_l2 = formula_l2.replace("*", "\*").replace("x", str(x))
    st.write(f"L2 = ((x-x0)\*(x-x1)) / ((x2-x0)\*(x2-x1)) = {novo_l2} = {l2(x)}")

    st.write(f"**-- Etapa 3: Calcular o p(x)**")
    px = f'{l0(x)}*{y0} + {l1(x)}*{y1} + {l2(x)}*{y2}'
    p = lambda x: eval(px)

    px_aux = px.replace("*", "\*")  #Só pra trocar o * e printar
    st.write(f"p({x}) = l0\*y0 + l1\*y1 + l2\*y2 = {px_aux}")

    return p(x)

def newton_inter(dados, x):
    x1, x2, x3 = dados[0][0], dados[1][0], dados[2][0]
    y1, y2, y3 = dados[0][1], dados[1][1], dados[2][1]

    px1 = b0 = y1

    b1 = (y2-y1) / (x2-x1)
    px2 = b0 + b1*(x2-x1)

    b2 = (((y3-y2)/(x3-x2)) - ((y2-y1)/(x2-x1))) / (x3-x1)
    px3 = b0 + b1*(x2-x1) + b2*(x3-x1)*(x3-x2)

    return px3





st.sidebar.title('Projeto Final')

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
        st.title('Zero de Funções - Bisseção')
        funcao = st.sidebar.text_input("Equação (Obs: math.exp(x) para e^x; x**y para potência; x*y para multiplicação; x/y para divisão):")
        a = st.sidebar.text_input("Valor de a:")
        b = st.sidebar.text_input("Valor de b:")
        max_it = st.sidebar.text_input("Qntd máxima de iterações:")
        
        f = lambda x: eval(funcao)
        raiz, dados = bisseccao(f, int(a), int(b), int(max_it))
        st.write(f"Valor de x resultante: {raiz}")
        plotarTabela(dados, "bisseccao")
            
    elif(metodo_zero == 'Newton'):
        st.title('Zero de Funções - Newton')
        funcao = st.sidebar.text_input("Equação (Obs: math.exp(x) para e^x; x**y para potência; x*y para multiplicação; x/y para divisão):")
        x0 = st.sidebar.text_input("Aproximação inicial (x0):")
        max_it = st.sidebar.text_input("Qntd máxima de iterações:")

        f = lambda x: eval(funcao)
        raiz, dados = newton_zero(f, int(x0), int(max_it))
        st.write(f"Valor de x resultante: {raiz}")
        plotarTabela(dados, "newton")


elif(opcao == 'Sistemas Lineares'):
    metodo_linear = st.sidebar.radio(
     'Escolha o método',
     ('Gauss (direto)', 'LU (direto)', 'Jacobi (iterativo)', 'Gauss-Seidel (iterativo)'))

    if(metodo_linear == 'Gauss (direto)'):
        st.title('Sistema Linear Direto - Gauss')

    elif(metodo_linear == 'LU (direto)'):
        st.title('Sistema Linear Direto - LU')
        equacao1 = st.sidebar.text_input("1° equação (Formato ex: 5*x1 +1*x2 -2*x3 = 10):")
        equacao2 = st.sidebar.text_input("2° equação:")
        equacao3 = st.sidebar.text_input("3° equação:")

        lu(equacao1, equacao2, equacao3)

    elif(metodo_linear == 'Jacobi (iterativo)'):
        st.title('Sistema Linear Iterativo - Jacobi')

    elif(metodo_linear == 'Gauss-Seidel (iterativo)'):
        st.title('Sistema Linear Iterativo - Gauss-Seidel')
        
        #name_dict = {"1° input":"", "2° input":""}
        #for k, v in name_dict.items():
        #    name_dict[k] = st.sidebar.text_input(k, v)
        #    st.write(name_dict[k])


elif(opcao == 'Interpolação Polinomial'):
    metodo_interpol = st.sidebar.radio(
     'Escolha a forma',
     ('Lagrange', 'Newton'))

    if(metodo_interpol == 'Lagrange'):
        st.title('Interpolação Polinomial - Lagrange')
        # Pega e converte as coordenadas em uma lista 2d
        pontos = st.sidebar.text_input("Conjunto de dados (Formato: x0 y0;x1 y1;x2 y2):")    #"-1 4;0 1;2 -1"
        pontos = pontos.split(";")                  #['-1 4', '0 1', '2 -1']
        pontos = [i.split(" ") for i in pontos]     #[['-1', '4'], ['0', '1'], ['2', '-1']]
        dados = [list(map(int,i)) for i in pontos]  #[[-1, 4], [0, 1], [2, -1]]
        
        # Pega o input e muda a vírgula por ponto, se tiver
        x = st.sidebar.text_input("Valor de x:")    #"1,5"
        x = x.replace(",", ".")     #"1,5" -> "1.5"
        x = float(x)

        px = lagrange(dados, x)
        st.write(f"p({x}) = {px}")

    elif(metodo_interpol == 'Newton'):
        st.title('Interpolação Polinomial - Newton')
        # Pega e converte as coordenadas em uma lista 2d
        pontos = st.sidebar.text_input("Conjunto de dados (Formato: x0 y0;x1 y1;x2 y2):")
        pontos = pontos.split(";")
        pontos = [i.split(" ") for i in pontos]
        dados = [list(map(int,i)) for i in pontos]

        # Pega o input e muda a vírgula por ponto, se tiver
        x = st.sidebar.text_input("Valor de x:")
        x = x.replace(",", ".")
        x = float(x)

        st.write(f"Dados = {dados}")
        st.write(f"x = {x}")
        px = newton_inter(dados, x)
        st.write(f"p({x}) = {px}")


