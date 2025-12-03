import streamlit as st
import numpy as np
import os 
from scipy.integrate import odeint
import plotly.graph_objects as go

# Importar configuración de tema
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from plotly_theme import COLORS, get_transparent_layout

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

# --- VISUALIZACIÓN PROFESIONAL COMPLETA ---
st.markdown("---")
st.markdown("## 📊 ANÁLISIS DE ESTABILIDAD - EQUILIBRIO P₀")

# SECCIÓN 1: GRÁFICO TEMPORAL COMPLETO (FULL WIDTH)
st.markdown("### 📈 Dinámica Temporal del Sistema XEIHR")
st.markdown("*Evolución de las 5 compartimentos epidemiológicos desde la condición inicial hacia el equilibrio*")

fig = go.Figure()
fig.add_trace(go.Scatter(x=t, y=X, name="Susceptibles (X)", 
                        line=dict(color=COLORS['susceptibles'], width=3)))
fig.add_trace(go.Scatter(x=t, y=E, name="Expuestos (E)", 
                        line=dict(color=COLORS['expuestos'], width=2.5, dash='dot')))
fig.add_trace(go.Scatter(x=t, y=I, name="Infecciosos (I)", 
                        line=dict(color=COLORS['infecciosos'], width=3.5)))
fig.add_trace(go.Scatter(x=t, y=H, name="Hospitalizados (H)", 
                        line=dict(color=COLORS['hospitalizados'], width=2.5, dash='dash')))
fig.add_trace(go.Scatter(x=t, y=R, name="Recuperados (R)", 
                        line=dict(color=COLORS['recuperados'], width=3)))

fig.update_layout(
    **get_transparent_layout(
        height=550,
        xaxis_title="Tiempo (t)",
        yaxis_title="Población (individuos)",
        hovermode='x unified'
    ),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, font=dict(size=11)),
    title=None
)
st.plotly_chart(fig, use_container_width=True)

# SECCIÓN 2: PANEL DE ESTADO Y MÉTRICAS PRINCIPALES
st.markdown("### 🔍 Estado Dinámico del Sistema")

col_status1, col_status2, col_status3 = st.columns(3)

with col_status1:
    st.markdown("**Estado de Estabilidad**")
    if ev2 < 0:
        st.success("✅ SISTEMA ESTABLE", icon="✅")
        st.markdown("*La enfermedad se extingue naturalmente*")
    else:
        st.error("⚠️ SISTEMA INESTABLE", icon="⚠️")
        st.markdown("*Hay riesgo de brote epidémico*")

with col_status2:
    st.markdown("**Número Reproductivo Básico**")
    R0 = (beta * X_p0) / mu
    col_r0_num, col_r0_status = st.columns([2, 1])
    with col_r0_num:
        st.metric("R₀", f"{R0:.4f}", label_visibility="collapsed")
    with col_r0_status:
        if R0 < 1:
            st.markdown("🟢 **R₀ < 1**\n*Controlado*")
        else:
            st.markdown("🔴 **R₀ > 1**\n*Riesgo*")

with col_status3:
    st.markdown("**Punto de Equilibrio**")
    st.info(f"P₀ = ({X_p0:.1f}, 0, 0, 0, 0)", icon="ℹ️")
    st.markdown(f"*Susceptibles: {X_p0:.2f} individuos*")

# SECCIÓN 3: AUTOVALORES Y ANÁLISIS ESPECTRAL
st.markdown("---")
st.markdown("### 🔬 Análisis de Autovalores (Análisis Espectral)")
st.markdown("*Los autovalores determinan la estabilidad local del equilibrio. Si todos tienen parte real negativa, el sistema es estable.*")

col_ev1, col_ev2, col_ev3, col_ev4, col_ev5 = st.columns(5)

with col_ev1:
    st.metric("λ₁", f"{ev1:.4f}")
    if ev1 < 0:
        st.caption("🟢 Negativo")
    else:
        st.caption("🔴 Positivo")

with col_ev2:
    st.metric("λ₂", f"{ev2:.4f}")
    if ev2 < 0:
        st.caption("🟢 Negativo\n(CRÍTICO)")
    else:
        st.caption("🔴 Positivo\n(CRÍTICO)")

with col_ev3:
    st.metric("λ₃", f"{ev3:.4f}")
    if ev3 < 0:
        st.caption("🟢 Negativo")
    else:
        st.caption("🔴 Positivo")

with col_ev4:
    st.metric("λ₄", f"{ev4:.4f}")
    if ev4 < 0:
        st.caption("🟢 Negativo")
    else:
        st.caption("🔴 Positivo")

with col_ev5:
    st.metric("λ₅", f"{ev5:.4f}")
    if ev5 < 0:
        st.caption("🟢 Negativo")
    else:
        st.caption("🔴 Positivo")

st.info("""
**Interpretación:** El autovalor **λ₂** es el más crítico para la estabilidad del equilibrio libre de enfermedad. 
Si λ₂ < 0, la enfermedad desaparece; si λ₂ > 0, puede haber brote epidémico.
""")

# SECCIÓN 4: CÍRCULOS DE GERSHGORIN - GRÁFICO COMPLETO
st.markdown("---")
st.markdown("### 🎯 Validación Espectral: Teorema de Gershgorin")
st.markdown("""
El **Teorema de Gershgorin** establece que todos los autovalores están contenidos en la unión de discos en el plano complejo.
Cada disco está centrado en un elemento diagonal de la matriz y su radio es la suma de valores absolutos de los elementos no-diagonales.
""")

# Cálculos de Gershgorin
c1 = -mu
r1 = abs(-beta * X_p0)
c2 = beta * X_p0 - mu
r2 = 0
c3 = -(mu + omega + gamma)
r3 = 0
c4 = -(mu + alpha + delta)
r4 = abs(omega)
c5 = -mu
r5 = abs(delta)

fig_gersh = go.Figure()

color_critical = COLORS['error'] if c2 > 0 else COLORS['success']

disk_centers = [c1, c2, c3, c4, c5]
disk_radii = [r1, r2, r3, r4, r5]
disk_labels = ['X (Fila 1)', 'E (Fila 2) - CRÍTICO', 'I (Fila 3)', 'H (Fila 4)', 'R (Fila 5)']
disk_colors = [COLORS['primary'], color_critical, COLORS['primary'], COLORS['primary'], COLORS['primary']]

for center, radius, color, label in zip(disk_centers, disk_radii, disk_colors, disk_labels):
    if radius > 0.001:
        fig_gersh.add_shape(
            type="circle",
            x0=center - radius, y0=-radius,
            x1=center + radius, y1=radius,
            line=dict(color=color, width=2.5),
            fillcolor=color,
            opacity=0.15,
            name=label
        )
    
    fig_gersh.add_trace(go.Scatter(
        x=[center], y=[0],
        mode='markers',
        marker=dict(size=12, color=color, symbol='circle', line=dict(color=COLORS['dark'], width=2)),
        name=label,
        hovertemplate=f'<b>{label}</b><br>Centro: {center:.3f}<br>Radio: {radius:.3f}<extra></extra>'
    ))

fig_gersh.add_vline(x=0, line_dash="solid", line_color='#0f172a', line_width=3.5, 
                    annotation_text="Re(λ)=0", annotation_position="top left", annotation_font_color='#0f172a', annotation_font_size=13, annotation_font_family="Arial")
fig_gersh.add_hline(y=0, line_dash="solid", line_color='#64748b', line_width=2)

fig_gersh.add_trace(go.Scatter(
    x=[0], y=[0],
    mode='markers',
    marker=dict(size=11, color='#0f172a', symbol='x', line=dict(color='#0f172a', width=3)),
    name='Origen (0,0)',
    hovertemplate='<b>Origen</b><br>Re(λ)=0, Im(λ)=0<extra></extra>'
))

min_center = min(disk_centers)
max_center = max(disk_centers)
max_radius = max(disk_radii) if disk_radii else 0.5

fig_gersh.add_vrect(x0=min_center-max_radius-1, x1=0, fillcolor=COLORS['success'], opacity=0.06, line_width=0, layer="below")
fig_gersh.add_vrect(x0=0, x1=max_center+max_radius+1, fillcolor=COLORS['error'], opacity=0.06, line_width=0, layer="below")

fig_gersh.update_layout(
    **get_transparent_layout(
        height=600,
        xaxis_title="Parte Real: Re(λ)",
        yaxis_title="Parte Imaginaria: Im(λ)"
    ),
    title=None,
    showlegend=True,
    legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.95)', bordercolor='#0f172a', borderwidth=2, font=dict(size=11, color='#0f172a'))
)

fig_gersh.update_xaxes(zeroline=False)
fig_gersh.update_yaxes(zeroline=False)

st.plotly_chart(fig_gersh, use_container_width=True)

col_gersh_interp1, col_gersh_interp2 = st.columns(2)

with col_gersh_interp1:
    st.markdown("**Significado de las Zonas**")
    st.markdown("""
    - **Zona Azul (Izquierda):** Semiplano izquierdo
      - Todos los autovalores tienen parte real negativa
      - ✅ Implica ESTABILIDAD
    
    - **Zona Roja (Derecha):** Semiplano derecho
      - Al menos un autovalor tiene parte real positiva
      - ⚠️ Implica INESTABILIDAD
    """)

with col_gersh_interp2:
    st.markdown("**Análisis del Disco Crítico (Fila 2 - E)**")
    if c2 < 0:
        st.success(f"""
        ✅ El disco crítico está completamente en el semiplano izquierdo.
        - Centro: {c2:.4f}
        - Radio: {r2:.4f}
        - **Conclusión: Sistema ESTABLE**
        """)
    else:
        st.error(f"""
        ⚠️ El disco crítico se extiende al semiplano derecho.
        - Centro: {c2:.4f}
        - Radio: {r2:.4f}
        - **Conclusión: Riesgo de INESTABILIDAD**
        """)
# SECCIÓN 5: MAPA DE CALOR R₀
st.markdown("---")
st.markdown("### 🔥 Análisis de Sensibilidad: Mapa de Calor de R₀")
st.markdown("""
Visualización del número reproductivo básico en el espacio bidimensional de parámetros (β, μ).
La **línea blanca crítica** marca donde R₀ = 1, separando la zona segura (azul, R₀ < 1) de la zona de riesgo (rojo, R₀ > 1).
""")

beta_range = np.linspace(0.001, 0.15, 120)
mu_range = np.linspace(0.01, 0.5, 120)
beta_mesh, mu_mesh = np.meshgrid(beta_range, mu_range)

R0_mesh = (beta_mesh * lam) / (mu_mesh ** 2)

fig_heatmap = go.Figure(data=go.Contour(
    z=R0_mesh,
    x=beta_range,
    y=mu_range,
    colorscale=[
        [0.0, COLORS['primary']],
        [0.3, '#60a5fa'],
        [0.5, 'white'],
        [0.7, '#fca5a5'],
        [1.0, COLORS['error']]
    ],
    contours=dict(
        showlabels=True,
        labelfont=dict(size=11, color='black'),
        labelformat='.1f'
    ),
    colorbar=dict(title='$R_0$', thickness=20, len=0.8, tickfont=dict(size=10)),
    hovertemplate='β: %{x:.4f}<br>μ: %{y:.4f}<br>$R_0$: %{z:.3f}<extra></extra>'
))

fig_heatmap.add_trace(go.Contour(
    z=R0_mesh,
    x=beta_range,
    y=mu_range,
    contours=dict(start=1, end=1, size=0),
    line=dict(color='#0f172a', width=5),
    showscale=False,
    hoverinfo='skip',
    name='R₀ = 1 (Frontera Crítica)'
))

current_R0 = (beta * lam) / (mu ** 2)
fig_heatmap.add_trace(go.Scatter(
    x=[beta], y=[mu],
    mode='markers',
    marker=dict(size=16, color='gold', symbol='diamond', line=dict(color='#0f172a', width=3)),
    name=f'Configuración Actual\n(R₀={current_R0:.3f})',
    hovertemplate='<b>Parámetros Actuales</b><br>β: %{x:.4f}<br>μ: %{y:.4f}<br>R₀: ' + f'{current_R0:.3f}<extra></extra>'
))

fig_heatmap.update_layout(
    **get_transparent_layout(
        height=600,
        xaxis_title="Tasa de Transmisión (β)",
        yaxis_title="Tasa de Mortalidad Natural (μ)"
    ),
    title=None
)

st.plotly_chart(fig_heatmap, use_container_width=True)

col_heat_interp1, col_heat_interp2 = st.columns(2)

with col_heat_interp1:
    st.markdown("**Interpretación de Zonas**")
    st.markdown(f"""
    - **Zona Azul (R₀ < 1):** 
      - Enfermedad controlable
      - Desaparece naturalmente
      - ✅ Seguro epidemiológico
    
    - **Zona Roja (R₀ > 1):**
      - Enfermedad se propaga
      - Requiere intervención
      - ⚠️ Zona de Riesgo
    """)

with col_heat_interp2:
    st.markdown("**Configuración Actual**")
    st.metric("R₀ Actual", f"{current_R0:.4f}", label_visibility="collapsed")
    if current_R0 < 1:
        st.success("✅ Escenario Seguro - Enfermedad Controlable")
    else:
        st.warning("⚠️ Escenario de Riesgo - Requiere Intervención")
    
    st.markdown("**Sugerencia:**")
    if current_R0 < 1:
        st.info("Mantén β bajo o μ alto para preservar estabilidad")
    else:
        st.info("Aumenta μ (mortalidad) o disminuye β (transmisión) para estabilizar")

# SECCIÓN FINAL: DATOS NUMÉRICOS
st.markdown("---")
st.markdown("### 📋 Datos Numéricos y Poblaciones Finales")

col_data1, col_data2, col_data3 = st.columns(3)

with col_data1:
    st.markdown("**Poblaciones Finales**")
    st.write(f"• **X (Susceptibles):** {X[-1]:.2f}")
    st.write(f"• **E (Expuestos):** {E[-1]:.6f}")
    st.write(f"• **I (Infecciosos):** {I[-1]:.6f}")

with col_data2:
    st.markdown("**Continuación**")
    st.write(f"• **H (Hospitalizados):** {H[-1]:.2f}")
    st.write(f"• **R (Recuperados):** {R[-1]:.2f}")
    st.write(f"• **Población Total:** {X[-1] + E[-1] + I[-1] + H[-1] + R[-1]:.2f}")

with col_data3:
    st.markdown("**Parámetros de Simulación**")
    st.write(f"• **Tiempo Total:** {t_max} unidades")
    st.write(f"• **Puntos Simulados:** {len(t)}")
    st.write(f"• **β (Transmisión):** {beta:.4f}")

st.success("✅ **Análisis completado exitosamente**")