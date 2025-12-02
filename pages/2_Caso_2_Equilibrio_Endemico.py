import streamlit as st
import numpy as np
import os 

from scipy.integrate import odeint
import plotly.graph_objects as go

st.set_page_config(
    page_title="Caso 2: Equilibrio Endémico", 
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
st.sidebar.page_link("pages/1_Caso_1_Libre_de_Enfermedad.py", label="⬅️ Ir al Caso 1 (Libre de Enfermedad)")
st.sidebar.divider()

# --- CONTENIDO PRINCIPAL ---
st.title("Caso 2: Equilibrio Endémico ($P^*$)")
st.markdown("Análisis cuando la infección persiste en la población ($I \\neq 0$).")

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

# --- PARAMETROS (SIDEBAR) ---
st.sidebar.header("⚙️ Configuración Caso 2")

# Demografía
st.sidebar.subheader("📊 Demografía")
lam = st.sidebar.number_input(r"Tasa de entrada (λ*)", value=10.0, step=0.1, key="lam2")
mu = st.sidebar.number_input(r"Mortalidad natural (μ)", value=0.1, step=0.01, key="mu2")

# Transmisión
st.sidebar.subheader("🦠 Transmisión")
beta = st.sidebar.slider(r"Tasa de contacto (β)", min_value=0.01, max_value=0.2, value=0.05, step=0.001, format="%.3f", key="beta2")
epsilon = st.sidebar.slider(r"Progresión E→I (ε)", min_value=0.001, max_value=0.1, value=0.02, step=0.001, format="%.3f", key="epsilon2")

# Clínicos
st.sidebar.subheader("🏥 Clínicos")
omega = st.sidebar.slider(r"Hospitalización (ω)", min_value=0.01, max_value=0.5, value=0.2, step=0.01, key="omega2")
gamma = st.sidebar.slider(r"Recuperación directa (γ)", min_value=0.01, max_value=0.5, value=0.1, step=0.01, key="gamma2")
alpha = st.sidebar.slider(r"Mortalidad enfermedad (α)", min_value=0.01, max_value=0.2, value=0.05, step=0.01, key="alpha2")
delta = st.sidebar.slider(r"Alta hospitalaria (δ)", min_value=0.01, max_value=0.5, value=0.15, step=0.01, key="delta2")

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Condiciones Iniciales")
t_max = st.sidebar.slider("Tiempo de Simulación", 50, 3000, 500, key="tmax2")
X0 = st.sidebar.number_input("Población Susceptible Inicial (X₀)", value=50.0, min_value=0.0, key="X02")
E0 = st.sidebar.number_input("Población Expuesta Inicial (E₀)", value=5.0, min_value=0.0, key="E02")
I0 = st.sidebar.number_input("Población Infecciosa Inicial (I₀)", value=5.0, min_value=0.0, key="I02")

# --- CÁLCULOS P* ---
try:
    E_star = (mu + omega + gamma) / epsilon
    X_star = lam / (mu + beta * E_star)
    I_star = (beta * X_star * E_star) / (epsilon * E_star + mu)  # Corregido
    H_star = (omega * I_star) / (mu + alpha + delta)
    R_star = (gamma * I_star + delta * H_star) / mu
    
    # Validar que todas las poblaciones sean no negativas
    valid_equilibrium = all(x >= 0 for x in [X_star, E_star, I_star, H_star, R_star])
    
except (ZeroDivisionError, ValueError):
    E_star = X_star = I_star = H_star = R_star = 0
    valid_equilibrium = False

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
y0 = [X0, E0, I0, 0, 0]
ret = odeint(model, y0, t, args=(lam, mu, beta, epsilon, omega, gamma, alpha, delta))
X, E, I, H, R = ret.T

# --- VISUALIZACIÓN ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📈 Dinámica hacia el Equilibrio Endémico")
    
    fig = go.Figure()
    colors = ['blue', 'orange', 'red', 'purple', 'green']
    names = ['Susceptibles (X)', 'Expuestos (E)', 'Infecciosos (I)', 'Hospitalizados (H)', 'Recuperados (R)']
    data = [X, E, I, H, R]
    
    for i, (name, color, values) in enumerate(zip(names, colors, data)):
        fig.add_trace(go.Scatter(x=t, y=values, name=name, line=dict(color=color, width=2 if i != 2 else 3)))
    
    # Agregar líneas de equilibrio teórico si es válido
    if valid_equilibrium and I_star > 0:
        fig.add_hline(y=X_star, line_dash="dash", line_color="blue", annotation_text="X*")
        fig.add_hline(y=E_star, line_dash="dash", line_color="orange", annotation_text="E*")
        fig.add_hline(y=I_star, line_dash="dash", line_color="red", annotation_text="I*")
    
    fig.update_layout(
        template="plotly_white", 
        xaxis_title="Tiempo", 
        yaxis_title="Población",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("🎯 Valores del Equilibrio ($P^*$)")
    
    # Validación de existencia
    if valid_equilibrium and I_star > 0:
        st.success("✅ **Equilibrio Endémico Válido**")
        st.markdown("El sistema converge a un estado constante de infección donde $I > 0$.")
        
        # Mostrar valores de equilibrio
        st.markdown("---")
        st.metric("X* (Susceptibles)", f"{X_star:.2f}")
        st.metric("E* (Expuestos)", f"{E_star:.2f}")
        st.metric("I* (Infecciosos)", f"{I_star:.2f}")
        st.metric("H* (Hospitalizados)", f"{H_star:.2f}")
        st.metric("R* (Recuperados)", f"{R_star:.2f}")
        
        # Población total
        N_total = X_star + E_star + I_star + H_star + R_star
        st.metric("Población Total", f"{N_total:.2f}")
        
    else:
        st.error("❌ **Equilibrio Endémico Inválido**")
        st.markdown("""
        **Atención:** El equilibrio endémico no es biológicamente válido con los parámetros actuales.
        
        Posibles causas:
        - Parámetros de transmisión muy bajos
        - Mortalidad muy alta
        - Condiciones que no permiten persistencia de la infección
        """)
        
        st.info("💡 **Sugerencia:** Aumenta la tasa de contacto (β) o disminuye la mortalidad (μ) para alcanzar un equilibrio endémico.")

# --- ANÁLISIS ADICIONAL ---
col_ana1, col_ana2 = st.columns(2)

with col_ana1:
    st.subheader("📊 Prevalencia de la Enfermedad")
    if valid_equilibrium and I_star > 0:
        prevalence = (E_star + I_star + H_star) / (X_star + E_star + I_star + H_star + R_star) * 100
        st.metric("Prevalencia Total", f"{prevalence:.1f}%")
        
        infectious_prevalence = I_star / (X_star + E_star + I_star + H_star + R_star) * 100
        st.metric("Prevalencia Infecciosa", f"{infectious_prevalence:.1f}%")
    else:
        st.info("No hay enfermedad endémica con los parámetros actuales")

with col_ana2:
    st.subheader("🔍 Número Reproductivo Efectivo")
    if valid_equilibrium and I_star > 0:
        R_eff = (beta * X_star) / mu
        st.metric("R efectivo", f"{R_eff:.3f}")
        if R_eff > 1:
            st.caption("R > 1: La enfermedad persiste")
        else:
            st.caption("R ≤ 1: La enfermedad debería desaparecer")
    else:
        R0 = (beta * (lam/mu)) / mu
        st.metric("R₀ básico", f"{R0:.3f}")

# --- DATOS FINALES ---
with st.expander("📋 Ver Datos Numéricos Completos"):
    st.markdown("**Poblaciones Finales de la Simulación:**")
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        st.write(f"- Susceptibles (X): {X[-1]:.2f}")
        st.write(f"- Expuestos (E): {E[-1]:.2f}")
        st.write(f"- Infecciosos (I): {I[-1]:.2f}")
    with col_f2:
        st.write(f"- Hospitalizados (H): {H[-1]:.2f}")
        st.write(f"- Recuperados (R): {R[-1]:.2f}")
        st.write(f"- **Total:** {X[-1] + E[-1] + I[-1] + H[-1] + R[-1]:.2f}")