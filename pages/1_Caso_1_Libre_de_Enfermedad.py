import streamlit as st
import numpy as np
import os 
from scipy.integrate import odeint
import plotly.graph_objects as go

st.set_page_config(
    page_title="Caso 1: Libre de Enfermedad", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CARGAR CSS ELEGANTE ---
# Ruta correcta al CSS
css_path = os.path.join(os.path.dirname(__file__), "..", "stylo", "csscaso1.css")

# Cargar el CSS
with open(css_path, "r") as f:
    css = f.read()

st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

# --- NAVEGACIÓN RÁPIDA SIDEBAR ---
st.sidebar.subheader("Navegación")
st.sidebar.page_link("Home.py", label="🏠 Inicio")
st.sidebar.page_link("pages/2_Caso_2_Equilibrio_Endemico.py", label="➡️ Ir al Caso 2 (Endémico)")
st.sidebar.divider()

# --- CONTENIDO PRINCIPAL ---
st.title("Caso 1: Equilibrio Libre de Infecciosos ($P_0$)")
st.markdown("Análisis de estabilidad del sistema cuando no hay infecciosos ($I=0$) ni expuestos ($E=0$).")

# --- MOSTRAR MODELO Y PARAMETROS ---
with st.expander("📖 Ver Ecuaciones del Modelo y Definiciones", expanded=False):
    col_eq, col_def = st.columns(2)
    
    with col_eq:
        st.markdown("#### Ecuaciones Diferenciales")
        st.latex(r"""
        \begin{aligned}
        \frac{dX}{dt} &= \lambda^{*} - \mu X - \beta X E \\
        \frac{dE}{dt} &= \beta X E - \epsilon E I - \mu E \\
        \frac{dI}{dt} &= \epsilon E I - (\mu + \omega + \gamma) I \\
        \frac{dH}{dt} &= \omega I - (\mu + \alpha + \delta) H \\
        \frac{dR}{dt} &= \gamma I + \delta H - \mu R
        \end{aligned}
        """)
    
    with col_def:
        st.markdown("#### Definición de Parámetros")
        st.markdown("""
        * $\lambda^*$: Tasa de entrada (nacimientos/inmigración)
        * $\mu$: Tasa de mortalidad natural
        * $\beta$: Tasa de contacto efectivo (transmisión)
        * $\epsilon$: Tasa de progresión de Expuesto a Infeccioso
        * $\omega$: Tasa de hospitalización
        * $\gamma$: Tasa de recuperación directa (sin hospital)
        * $\alpha$: Mortalidad inducida por la enfermedad
        * $\delta$: Tasa de alta hospitalaria (recuperación)
        """)

# --- PARAMETROS (SIDEBAR) ---
st.sidebar.header("⚙️ Configuración Caso 1")

# Demografía
st.sidebar.subheader("📊 Demografía")
lam = st.sidebar.number_input(r"Tasa de entrada (λ*)", value=10.0, step=0.1, help="Tasa de nacimientos o inmigración")
mu = st.sidebar.number_input(r"Mortalidad natural (μ)", value=0.1, step=0.01, help="Tasa de mortalidad natural")

# Transmisión
st.sidebar.subheader("🦠 Transmisión")
st.sidebar.info("ℹ️ Aumenta Beta para desestabilizar el equilibrio.")
beta = st.sidebar.slider(r"Tasa de contacto (β)", min_value=0.001, max_value=0.1, value=0.005, step=0.001, format="%.3f", help="Tasa de transmisión efectiva")
epsilon = st.sidebar.slider(r"Progresión E→I (ε)", min_value=0.001, max_value=0.1, value=0.02, step=0.001, format="%.3f", help="Tasa de progresión a infeccioso")

# Clínicos
st.sidebar.subheader("🏥 Clínicos")
omega = st.sidebar.slider(r"Hospitalización (ω)", min_value=0.01, max_value=0.5, value=0.2, step=0.01, help="Tasa de hospitalización")
gamma = st.sidebar.slider(r"Recuperación directa (γ)", min_value=0.01, max_value=0.5, value=0.1, step=0.01, help="Tasa de recuperación sin hospitalización")
alpha = st.sidebar.slider(r"Mortalidad enfermedad (α)", min_value=0.01, max_value=0.2, value=0.05, step=0.01, help="Tasa de mortalidad por la enfermedad")
delta = st.sidebar.slider(r"Alta hospitalaria (δ)", min_value=0.01, max_value=0.5, value=0.15, step=0.01, help="Tasa de recuperación hospitalaria")

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Condiciones Iniciales")
t_max = st.sidebar.slider("Tiempo de Simulación", 50, 1000, 300, help="Duración total de la simulación")
X0 = st.sidebar.number_input("Población Susceptible Inicial (X₀)", value=90.0, min_value=0.0, help="Población susceptible inicial")
E0 = st.sidebar.number_input("Perturbación Inicial (E₀)", value=1.0, min_value=0.0, help="Pequeña perturbación inicial de expuestos")

# --- CÁLCULOS ---
X_p0 = lam / mu
ev1 = -mu
ev2 = beta * X_p0 - mu
ev3 = -(mu + omega + gamma)
ev4 = -(mu + alpha + delta)
ev5 = -mu

# --- SIMULACIÓN ---
def model(y, t, lam, mu, beta, epsilon, omega, gamma, alpha, delta):
    X, E, I, H, R = y
    dXdt = lam - mu*X - beta*X*E
    dEdt = beta*X*E - epsilon*E*I - mu*E
    dIdt = epsilon*E*I - (mu + omega + gamma)*I
    dHdt = omega*I - (mu + alpha + delta)*H
    dRdt = gamma*I + delta*H - mu*R
    return [dXdt, dEdt, dIdt, dHdt, dRdt]

t = np.linspace(0, t_max, t_max*2)
y0 = [X0, E0, 0, 0, 0]
ret = odeint(model, y0, t, args=(lam, mu, beta, epsilon, omega, gamma, alpha, delta))
X, E, I, H, R = ret.T

# --- VISUALIZACIÓN ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📈 Dinámica Temporal")
    
    # Gráfico principal
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=X, name="Susceptibles (X)", line=dict(color='blue', width=2)))
    fig.add_trace(go.Scatter(x=t, y=E, name="Expuestos (E)", line=dict(color='orange', width=2, dash='dot')))
    fig.add_trace(go.Scatter(x=t, y=I, name="Infecciosos (I)", line=dict(color='red', width=3)))
    fig.add_trace(go.Scatter(x=t, y=H, name="Hospitalizados (H)", line=dict(color='purple', width=2, dash='dash')))
    fig.add_trace(go.Scatter(x=t, y=R, name="Recuperados (R)", line=dict(color='green', width=2, dash='dash')))
    
    fig.update_layout(
        template="plotly_white", 
        xaxis_title="Tiempo",
        yaxis_title="Población",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Análisis de resultados
    if ev2 < 0:
        st.success("✅ **El sistema es estable**: La perturbación inicial de expuestos desaparece y la enfermedad se extingue.")
    else:
        st.error("⚠️ **El sistema es inestable**: Los expuestos crecen, buscando un nuevo equilibrio endémico.")

with col2:
    st.subheader("🔍 Análisis de Estabilidad ($P_0$)")
    
    # Equilibrio
    st.markdown("#### Punto de Equilibrio")
    st.latex(fr"P_0 = \left( \frac{{\lambda^*}}{{\mu}}, 0, 0, 0, 0 \right) = ({X_p0:.2f}, 0, 0, 0, 0)")
    
    st.markdown("#### Valores Propios (Eigenvalues)")
    st.markdown("Determinan si pequeñas perturbaciones crecen o desaparecen:")
    
    # Estado de estabilidad
    if ev2 < 0:
        st.success("**ESTABLE**")
        st.caption("La enfermedad se extingue naturalmente ($\lambda_2 < 0$)")
    else:
        st.error("**INESTABLE**")
        st.caption("Cualquier caso introducido provocará un brote ($\lambda_2 > 0$)")
    
    st.divider()
    
    # Detalle de valores propios
    st.markdown("**Detalle de Valores Propios:**")
    
    col_ev1, col_ev2 = st.columns(2)
    with col_ev1:
        st.metric("λ₁", f"{ev1:.3f}")
        st.metric("λ₃", f"{ev3:.3f}")
        st.metric("λ₅", f"{ev5:.3f}")
    
    with col_ev2:
        if ev2 < 0:
            st.metric("λ₂", f"{ev2:.3f}", delta="Estable", delta_color="normal")
        else:
            st.metric("λ₂", f"{ev2:.3f}", delta="Inestable", delta_color="inverse")
        st.metric("λ₄", f"{ev4:.3f}")
    
    # Número reproductivo básico
    R0 = (beta * X_p0) / mu
    st.divider()
    st.metric("Número Reproductivo Básico (R₀)", f"{R0:.3f}")
    if R0 < 1:
        st.caption("R₀ < 1: La enfermedad desaparece")
    else:
        st.caption("R₀ ≥ 1: La enfermedad persiste")

# --- INFORMACIÓN ADICIONAL ---
with st.expander("📊 Ver Datos Numéricos Finales"):
    col_data1, col_data2 = st.columns(2)
    with col_data1:
        st.markdown("**Poblaciones Finales:**")
        st.write(f"- Susceptibles (X): {X[-1]:.2f}")
        st.write(f"- Expuestos (E): {E[-1]:.2f}")
        st.write(f"- Infecciosos (I): {I[-1]:.2f}")
    with col_data2:
        st.write(f"- Hospitalizados (H): {H[-1]:.2f}")
        st.write(f"- Recuperados (R): {R[-1]:.2f}")
        st.write(f"- Total: {X[-1] + E[-1] + I[-1] + H[-1] + R[-1]:.2f}")