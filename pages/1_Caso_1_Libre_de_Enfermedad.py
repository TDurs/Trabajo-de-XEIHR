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

# --- SECCIÓN: VALIDACIÓN ESPECTRAL (GERSHGORIN) ---
st.divider()
st.subheader("🎯 Validación Espectral: Círculos de Gershgorin")

st.markdown("""
El **Teorema de Gershgorin** establece que todos los autovalores de una matriz están contenidos 
en la unión de discos de Gershgorin. Cada disco está centrado en un elemento diagonal y su radio 
es la suma de valores absolutos de los elementos no-diagonales en esa fila.
""")

# Mostrar matriz Jacobiana
st.markdown("#### Matriz Jacobiana en $P_0$:")
st.latex(r"""
J(P_0) = \begin{pmatrix}
-\mu & -\beta X_p^0 & 0 & 0 & 0 \\
0 & \beta X_p^0 - \mu & 0 & 0 & 0 \\
0 & 0 & -(\mu+\omega+\gamma) & 0 & 0 \\
0 & 0 & \omega & -(\mu+\alpha+\delta) & 0 \\
0 & 0 & 0 & \delta & -\mu
\end{pmatrix}
""")

# Cálculos de Gershgorin
col_gersh1, col_gersh2 = st.columns(2)

with col_gersh1:
    st.markdown("#### Discos de Gershgorin")
    
    # Centro y radio para cada fila
    # Fila 1 (X): Centro = -μ, Radio = |−βX_p0|
    c1 = -mu
    r1 = abs(-beta * X_p0)
    
    # Fila 2 (E): Centro = βX_p0 - μ, Radio = 0
    c2 = beta * X_p0 - mu
    r2 = 0
    
    # Fila 3 (I): Centro = -(μ+ω+γ), Radio = 0
    c3 = -(mu + omega + gamma)
    r3 = 0
    
    # Fila 4 (H): Centro = -(μ+α+δ), Radio = |ω|
    c4 = -(mu + alpha + delta)
    r4 = abs(omega)
    
    # Fila 5 (R): Centro = -μ, Radio = |δ|
    c5 = -mu
    r5 = abs(delta)
    
    st.write(f"**Fila 1 (X):** Centro = {c1:.3f}, Radio = {r1:.3f}")
    st.write(f"**Fila 2 (E):** Centro = {c2:.3f}, Radio = {r2:.3f} **← Crítico**")
    st.write(f"**Fila 3 (I):** Centro = {c3:.3f}, Radio = {r3:.3f}")
    st.write(f"**Fila 4 (H):** Centro = {c4:.3f}, Radio = {r4:.3f}")
    st.write(f"**Fila 5 (R):** Centro = {c5:.3f}, Radio = {r5:.3f}")

with col_gersh2:
    st.markdown("#### Interpretación")
    st.info("""
    El disco **crítico (Fila 2)** determina la estabilidad.
    
    - Si su centro $c_2 = \\beta X_p^0 - \\mu$ está en la **zona roja** (lado positivo del plano), 
      el sistema es **inestable**.
    
    - Si está en la **zona azul** (lado negativo), el sistema es **estable**.
    """)

# Gráfico de Gershgorin
fig_gersh = go.Figure()

# Colores de los discos según posición
disk_colors = ['lightblue', 'red' if c2 > 0 else 'lightgreen', 'lightblue', 'lightblue', 'lightblue']
disk_centers = [c1, c2, c3, c4, c5]
disk_radii = [r1, r2, r3, r4, r5]
disk_labels = ['X (Fila 1)', 'E (Fila 2) - Crítico', 'I (Fila 3)', 'H (Fila 4)', 'R (Fila 5)']

# Agregar discos como círculos en el plano complejo
for i, (center, radius, color, label) in enumerate(zip(disk_centers, disk_radii, disk_colors, disk_labels)):
    if radius > 0:
        # Crear puntos del círculo
        theta = np.linspace(0, 2*np.pi, 100)
        circle_x = center + radius * np.cos(theta)
        circle_y = radius * np.sin(theta)
        fig_gersh.add_trace(go.Scatter(
            x=circle_x, y=circle_y,
            fill='toself',
            name=label,
            line_color=color,
            fillcolor=color,
            opacity=0.3,
            mode='lines'
        ))
    
    # Agregar marcador del centro
    fig_gersh.add_trace(go.Scatter(
        x=[center], y=[0],
        mode='markers',
        marker=dict(size=10, color=color, symbol='circle'),
        name=f'{label} (Centro)',
        showlegend=False
    ))

# Agregar eje imaginario (línea vertical en x=0)
fig_gersh.add_vline(x=0, line_dash="dash", line_color="black", line_width=2, annotation_text="Re(λ)=0")

# Agregar zona de estabilidad (izquierda) y inestabilidad (derecha)
fig_gersh.add_vrect(x0=min(disk_centers)-1, x1=0, fillcolor="green", opacity=0.05, 
                    line_width=0, layer="below", annotation_text="Estable", annotation_position="top left")
fig_gersh.add_vrect(x0=0, x1=max(disk_centers)+1, fillcolor="red", opacity=0.05, 
                    line_width=0, layer="below", annotation_text="Inestable", annotation_position="top right")

# Actualizar layout
fig_gersh.update_layout(
    title="Espectro de Autovalores: Círculos de Gershgorin en P₀",
    xaxis_title="Re(λ)",
    yaxis_title="Im(λ)",
    template="plotly_white",
    height=500,
    xaxis=dict(zeroline=True, showgrid=True),
    yaxis=dict(zeroline=True, showgrid=True),
    hovermode='closest'
)

st.plotly_chart(fig_gersh, use_container_width=True)

# Conclusión
st.markdown("---")
if c2 < 0:
    st.success("""
    ✅ **Conclusión:** El disco crítico (Fila 2) está completamente en el semiplano izquierdo.
    Todos los autovalores tienen parte real negativa → **Sistema Estable**.
    """)
else:
    st.error("""
    ⚠️ **Conclusión:** El disco crítico (Fila 2) se extiende al semiplano derecho.
    Al menos uno de los autovalores puede ser positivo → **Posible Inestabilidad**.
    """)

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