import streamlit as st
import numpy as np
from scipy.integrate import odeint
import plotly.graph_objects as go

st.set_page_config(
    page_title="Subcaso 1.2: Equilibrio Latente", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- NAVEGACIÓN RÁPIDA ---
st.sidebar.subheader("🧭 Navegación")
st.sidebar.page_link("Home.py", label="🏠 Inicio")
st.sidebar.page_link("pages/1_Caso_1_Libre_de_Enfermedad.py", label="1️⃣ Caso 1 (DFE)")
st.sidebar.page_link("pages/2_Caso_2_Equilibrio_Endemico.py", label="2️⃣ Caso 2 (Endémico)")
st.sidebar.divider()

# --- TÍTULO ---
st.title("Subcaso 1.2: Equilibrio con Latencia ($P_1$)")
st.markdown(r"Análisis cuando $\beta X - \mu = 0$ con $E \neq 0$ y $I=0$.")

# --- MOSTRAR MODELO ---
with st.expander("📖 Ver Ecuaciones del Modelo", expanded=False):
    st.latex(r"""
    \begin{aligned}
    \frac{dX}{dt} &= \lambda^{*} - \mu X - \beta X E \\
    \frac{dE}{dt} &= \beta X E - \epsilon E I - \mu E \\
    \frac{dI}{dt} &= \epsilon E I - (\mu + \omega + \gamma) I \\
    \frac{dH}{dt} &= \omega I - (\mu + \alpha + \delta) H \\
    \frac{dR}{dt} &= \gamma I + \delta H - \mu R
    \end{aligned}
    """)

# --- PARÁMETROS ---
st.sidebar.header("⚙️ Configuración P₁")

# Demografía
st.sidebar.subheader("📊 Demografía")
lam = st.sidebar.number_input(r"Tasa de entrada (λ*)", value=10.0, step=0.1, help="Tasa de nacimientos o inmigración")
mu = st.sidebar.number_input(r"Mortalidad natural (μ)", value=0.1, step=0.01, help="Tasa de mortalidad natural")

# Transmisión
st.sidebar.subheader("🦠 Transmisión")
st.sidebar.info("ℹ️ Ajusta Beta para que E* sea positivo.")
beta = st.sidebar.slider(r"Tasa de contacto (β)", min_value=0.001, max_value=0.1, value=0.02, step=0.001, format="%.3f", help="Tasa de transmisión efectiva")
epsilon = st.sidebar.slider(r"Progresión E→I (ε)", min_value=0.001, max_value=0.1, value=0.02, step=0.001, format="%.3f", help="Tasa de progresión a infeccioso")

# Clínicos
st.sidebar.subheader("🏥 Clínicos")
omega = st.sidebar.slider(r"Hospitalización (ω)", min_value=0.01, max_value=0.5, value=0.2, step=0.01, help="Tasa de hospitalización")
gamma = st.sidebar.slider(r"Recuperación directa (γ)", min_value=0.01, max_value=0.5, value=0.1, step=0.01, help="Tasa de recuperación sin hospitalización")
alpha = st.sidebar.slider(r"Mortalidad enfermedad (α)", min_value=0.01, max_value=0.2, value=0.05, step=0.01, help="Tasa de mortalidad por la enfermedad")
delta = st.sidebar.slider(r"Alta hospitalaria (δ)", min_value=0.01, max_value=0.5, value=0.15, step=0.01, help="Tasa de recuperación hospitalaria")

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Configuración Simulación")
t_max = st.sidebar.slider("Tiempo de Simulación", 50, 3000, 500, help="Duración total de la simulación")

# Condiciones iniciales ajustadas automáticamente para estar cerca de P1
st.sidebar.subheader("🧪 Condiciones Iniciales")
manual_ci = st.sidebar.checkbox("¿Ingresar condiciones manualmente?", help="Desmarca para usar valores automáticos cerca del equilibrio P₁")

# --- CÁLCULOS MATEMÁTICOS P₁ ---
try:
    # P₁ = (mu/beta, lambda/mu - mu/beta, 0, 0, 0)
    X_p1 = mu / beta
    E_p1 = (lam / mu) - (mu / beta)
    
    # Valores Propios
    ev1 = -mu
    ev2 = -beta * E_p1 if E_p1 > 0 else -mu  # Evitar valores negativos en sqrt
    ev3 = (epsilon * E_p1) - (mu + omega + gamma) if E_p1 > 0 else -(mu + omega + gamma)
    ev4 = -(mu + alpha + delta)
    ev5 = -mu
    
    # Determinante del Jacobiano (simplificado)
    det_J = (mu**2) * beta * max(E_p1, 0) * (mu + alpha + delta) * ev3

except (ZeroDivisionError, ValueError):
    X_p1, E_p1, ev1, ev2, ev3, ev4, ev5, det_J = 0, 0, 0, 0, 0, 0, 0, 0

# Configurar condiciones iniciales
if manual_ci:
    X0 = st.sidebar.number_input("Población Susceptible Inicial (X₀)", value=float(X_p1) if X_p1 > 0 else 50.0, min_value=0.0)
    E0 = st.sidebar.number_input("Población Expuesta Inicial (E₀)", value=float(E_p1) if E_p1 > 0 else 5.0, min_value=0.0)
    I0 = st.sidebar.number_input("Perturbación Infecciosa (I₀)", value=0.1, min_value=0.0, help="Pequeña perturbación para probar estabilidad")
else:
    X0 = X_p1 if X_p1 > 0 else 50.0
    E0 = E_p1 if E_p1 > 0 else 0.0
    I0 = 0.1  # Pequeña perturbación para ver si I crece
    st.sidebar.info(f"**CI Automáticas:**\n- X₀ = {X0:.2f}\n- E₀ = {E0:.2f}\n- I₀ = {I0}")

# --- SIMULACIÓN ---
def model(y, t, lam, mu, beta, epsilon, omega, gamma, alpha, delta):
    X, E, I, H, R = y
    dXdt = lam - mu * X - beta * X * E
    dEdt = beta * X * E - epsilon * E * I - mu * E
    dIdt = epsilon * E * I - (mu + omega + gamma) * I
    dHdt = omega * I - (mu + alpha + delta) * H
    dRdt = gamma * I + delta * H - mu * R
    return [dXdt, dEdt, dIdt, dHdt, dRdt]

t = np.linspace(0, t_max, t_max * 2)
y0 = [X0, E0, I0, 0, 0]
ret = odeint(model, y0, t, args=(lam, mu, beta, epsilon, omega, gamma, alpha, delta))
X, E, I, H, R = ret.T

# --- VISUALIZACIÓN ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📈 Dinámica Temporal cerca de $P_1$")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=X, name="Susceptibles (X)", line=dict(color='blue', width=2)))
    fig.add_trace(go.Scatter(x=t, y=E, name="Expuestos (E)", line=dict(color='orange', width=2)))
    fig.add_trace(go.Scatter(x=t, y=I, name="Infecciosos (I)", line=dict(color='red', width=3)))
    fig.add_trace(go.Scatter(x=t, y=H, name="Hospitalizados (H)", line=dict(color='purple', width=2, dash='dash')))
    fig.add_trace(go.Scatter(x=t, y=R, name="Recuperados (R)", line=dict(color='green', width=2, dash='dash')))
    
    # Agregar líneas horizontales de equilibrio teórico
    if E_p1 > 0:
        fig.add_hline(y=X_p1, line_dash="dot", line_color="blue", annotation_text="X* (P₁)", annotation_position="bottom right")
        fig.add_hline(y=E_p1, line_dash="dot", line_color="orange", annotation_text="E* (P₁)", annotation_position="top right")
    
    fig.update_layout(
        template="plotly_white", 
        xaxis_title="Tiempo", 
        yaxis_title="Población",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Análisis de resultados
    st.markdown("---")
    if E_p1 <= 0:
        st.error("""
        ⚠️ **Equilibrio Matemáticamente Imposible:** 
        
        $E_{P1} \\leq 0$. Este equilibrio no existe biológicamente con los parámetros actuales.
        
        **Solución:** Aumenta λ* o disminuye β para hacer $E_{P1} > 0$.
        """)
    elif ev3 < 0:
        st.success("""
        ✅ **Estable ante Infección:** 
        
        $\\lambda_3 < 0$. Aunque haya expuestos, los infecciosos (I) tienden a 0.
        El sistema se mantiene en el equilibrio latente P₁.
        """)
    else:
        st.error("""
        🔥 **Inestable:** 
        
        $\\lambda_3 > 0$. La pequeña perturbación en I crecerá exponencialmente, 
        llevando al sistema al equilibrio endémico P* (Caso 2).
        """)

with col2:
    st.subheader("🔍 Análisis del Equilibrio $P_1$")
    
    # Valores del equilibrio
    st.markdown("#### 📊 Valores Teóricos")
    st.latex(r"P_1 = \left( \frac{\mu}{\beta}, \frac{\lambda^*}{\mu} - \frac{\mu}{\beta}, 0, 0, 0 \right)")
    
    col_met1, col_met2 = st.columns(2)
    with col_met1:
        st.metric("X*", f"{X_p1:.2f}")
    with col_met2:
        if E_p1 > 0:
            st.metric("E*", f"{E_p1:.2f}")
        else:
            st.metric("E*", "Inválido", delta="< 0", delta_color="off")
    
    st.markdown("---")
    
    # Análisis de estabilidad
    st.markdown("#### 🎯 Análisis de Estabilidad")
    st.markdown("**Valores Propios del Sistema:**")
    
    # Valor propio crítico λ₃
    st.markdown("**λ₃ (Crítico para invasión):**")
    if ev3 < 0:
        st.success(f"$\\lambda_3 = {ev3:.3f}$")
        st.caption("ESTABLE: I no puede invadir el sistema")
    else:
        st.error(f"$\\lambda_3 = {ev3:.3f}$")
        st.caption("INESTABLE: I crecerá exponencialmente")
    
    st.markdown("**Otros valores propios:**")
    col_ev1, col_ev2 = st.columns(2)
    with col_ev1:
        st.metric("λ₁", f"{ev1:.3f}")
        st.metric("λ₄", f"{ev4:.3f}")
    with col_ev2:
        st.metric("λ₂", f"{ev2:.3f}")
        st.metric("λ₅", f"{ev5:.3f}")
    
    st.markdown("---")
    
    # Información adicional
    st.markdown("#### 📈 Métricas Adicionales")
    R0_basic = (beta * (lam / mu)) / mu
    st.metric("Número Reproductivo Básico (R₀)", f"{R0_basic:.3f}")
    
    if E_p1 > 0:
        st.metric("Determinante |J(P₁)|", f"{det_J:.2e}")

# --- INFORMACIÓN ADICIONAL ---
with st.expander("📋 Ver Datos Numéricos y Explicación", expanded=False):
    col_exp1, col_exp2 = st.columns(2)
    
    with col_exp1:
        st.markdown("""
        ### 🤔 ¿Qué representa P₁?
        
        El equilibrio P₁ representa un estado donde:
        - Hay **susceptibles y expuestos** pero **no hay infecciosos**
        - La tasa de crecimiento de expuestos es cero ($\\beta X - \\mu = 0$)
        - Es un equilibrio **inestable** que puede evolucionar hacia P₀ o P*
        """)
        
        st.markdown("""
        ### 🔍 Condición de Existencia
        
        Para que P₁ exista biológicamente:
        """)
        st.latex(r"E_{P1} = \frac{\lambda^*}{\mu} - \frac{\mu}{\beta} > 0")
        st.latex(r"\Rightarrow \beta > \frac{\mu^2}{\lambda^*}")
        
    with col_exp2:
        st.markdown("### 📊 Datos de Simulación")
        st.markdown("**Poblaciones Finales:**")
        st.write(f"- Susceptibles (X): {X[-1]:.2f}")
        st.write(f"- Expuestos (E): {E[-1]:.2f}")
        st.write(f"- Infecciosos (I): {I[-1]:.2f}")
        st.write(f"- Hospitalizados (H): {H[-1]:.2f}")
        st.write(f"- Recuperados (R): {R[-1]:.2f}")
        st.write(f"- **Total:** {X[-1] + E[-1] + I[-1] + H[-1] + R[-1]:.2f}")
        
        st.markdown("**Condiciones Iniciales Usadas:**")
        st.write(f"- X₀ = {X0:.2f}")
        st.write(f"- E₀ = {E0:.2f}")
        st.write(f"- I₀ = {I0:.2f}")

# --- PIE DE PÁGINA ---
st.markdown("---")
st.caption("""
**Nota:** Este equilibrio P₁ es teóricamente interesante pero raramente observable en la práctica 
debido a su inestabilidad. Pequeñas perturbaciones llevan al sistema hacia P₀ (enfermedad eliminada) 
o P* (enfermedad endémica).
""")