import streamlit as st
import numpy as np
from scipy.integrate import odeint
import os
import plotly.graph_objects as go

st.set_page_config(
    page_title="Subcaso 1.2: Equilibrio Latente", 
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

# --- SECCIÓN 1: DINÁMICA TEMPORAL (FULL WIDTH) ---
st.markdown("---")
st.subheader("📈 Dinámica Temporal cerca de $P_1$")
st.markdown("*Evolución de las 5 poblaciones desde condiciones iniciales hacia el equilibrio P₁*")

fig = go.Figure()
fig.add_trace(go.Scatter(x=t, y=X, name="Susceptibles (X)", line=dict(color='#2563eb', width=2.5)))
fig.add_trace(go.Scatter(x=t, y=E, name="Expuestos (E)", line=dict(color='#f59e0b', width=2.5)))
fig.add_trace(go.Scatter(x=t, y=I, name="Infecciosos (I)", line=dict(color='#ef4444', width=3)))
fig.add_trace(go.Scatter(x=t, y=H, name="Hospitalizados (H)", line=dict(color='#a855f7', width=2, dash='dash')))
fig.add_trace(go.Scatter(x=t, y=R, name="Recuperados (R)", line=dict(color='#10b981', width=2, dash='dash')))

# Agregar líneas horizontales de equilibrio teórico
if E_p1 > 0:
    fig.add_hline(y=X_p1, line_dash="dot", line_color="#2563eb", annotation_text="X* (P₁)", annotation_position="bottom right", annotation_font_color='#0f172a', annotation_font_size=11)
    fig.add_hline(y=E_p1, line_dash="dot", line_color="#f59e0b", annotation_text="E* (P₁)", annotation_position="top right", annotation_font_color='#0f172a', annotation_font_size=11)

fig.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    xaxis_title="Tiempo",
    yaxis_title="Población",
    height=550,
    font=dict(family="'Inter', sans-serif", size=11, color='#0f172a'),
    legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5, bgcolor='rgba(255,255,255,0.95)', bordercolor='#0f172a', borderwidth=2),
    hovermode='x unified'
)
st.plotly_chart(fig, use_container_width=True)

# --- SECCIÓN 2: ESTADO DEL EQUILIBRIO (3 COLUMNAS) ---
st.markdown("---")
st.subheader("🔍 Estado del Equilibrio $P_1$")

col_p1_1, col_p1_2, col_p1_3 = st.columns(3)

with col_p1_1:
    st.markdown("""
    <div style="background: #f8fafc; padding: 16px; border-radius: 8px; border-left: 4px solid #2563eb;">
        <h4 style="color: #1e293b; margin-top: 0;">📊 Valores de Equilibrio</h4>
        <p style="color: #475569; font-size: 13px;">Coordenadas del punto P₁</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.latex(r"P_1 = \left( \frac{\mu}{\beta}, \frac{\lambda^*}{\mu} - \frac{\mu}{\beta}, 0, 0, 0 \right)")
    
    col_x, col_e = st.columns(2)
    with col_x:
        st.metric("X*", f"{X_p1:.3f}", help="Susceptibles en equilibrio")
    with col_e:
        if E_p1 > 0:
            st.metric("E*", f"{E_p1:.3f}", help="Expuestos en equilibrio")
        else:
            st.metric("E*", "N/A", help="Equilibrio no válido biológicamente")

with col_p1_2:
    st.markdown("""
    <div style="background: #f8fafc; padding: 16px; border-radius: 8px; border-left: 4px solid #f59e0b;">
        <h4 style="color: #1e293b; margin-top: 0;">🎯 Análisis de Estabilidad</h4>
        <p style="color: #475569; font-size: 13px;">Criterio: Valor propio λ₃</p>
    </div>
    """, unsafe_allow_html=True)
    
    if E_p1 > 0:
        st.markdown("**Eigenvalor Crítico λ₃:**")
        if ev3 < 0:
            st.success(f"λ₃ = {ev3:.4f}", icon="✅")
            st.caption("Sistema estable ante invasión de I")
        else:
            st.error(f"λ₃ = {ev3:.4f}", icon="⚠️")
            st.caption("Sistema inestable ante perturbación en I")
    else:
        st.error("Equilibrio no válido", icon="❌")
        st.caption("E* ≤ 0: No existe biológicamente")

with col_p1_3:
    st.markdown("""
    <div style="background: #f8fafc; padding: 16px; border-radius: 8px; border-left: 4px solid #10b981;">
        <h4 style="color: #1e293b; margin-top: 0;">📈 Métricas del Sistema</h4>
        <p style="color: #475569; font-size: 13px;">Indicadores reproductivos</p>
    </div>
    """, unsafe_allow_html=True)
    
    R0_basic = (beta * (lam / mu)) / mu
    st.metric("R₀ (Reproductivo)", f"{R0_basic:.3f}", help="Número reproductivo básico")
    
    if E_p1 > 0:
        st.metric("|J(P₁)|", f"{det_J:.2e}", help="Determinante Jacobiano")

# --- SECCIÓN 3: ANÁLISIS DETALLADO DE AUTOVALORES (5 COLUMNAS) ---
st.markdown("---")
st.subheader("🔬 Análisis de Autovalores del Sistema")

col_ev1, col_ev2, col_ev3, col_ev4, col_ev5 = st.columns(5)

with col_ev1:
    st.markdown("""
    <div style="background: #dbeafe; padding: 12px; border-radius: 6px; border: 1px solid #2563eb;">
        <p style="color: #1e293b; font-weight: 600; font-size: 13px; margin: 0 0 8px 0;">λ₁</p>
        <p style="color: #1e293b; font-weight: 700; font-size: 16px; margin: 0;">{:.4f}</p>
        <p style="color: #64748b; font-size: 11px; margin: 4px 0 0 0;">Estabilidad</p>
    </div>
    """.format(ev1), unsafe_allow_html=True)

with col_ev2:
    st.markdown("""
    <div style="background: #fed7aa; padding: 12px; border-radius: 6px; border: 1px solid #f59e0b;">
        <p style="color: #1e293b; font-weight: 600; font-size: 13px; margin: 0 0 8px 0;">λ₂</p>
        <p style="color: #1e293b; font-weight: 700; font-size: 16px; margin: 0;">{:.4f}</p>
        <p style="color: #64748b; font-size: 11px; margin: 4px 0 0 0;">Transmisión</p>
    </div>
    """.format(ev2), unsafe_allow_html=True)

with col_ev3:
    color_bg = "#fee2e2" if ev3 > 0 else "#dcfce7"
    color_border = "#ef4444" if ev3 > 0 else "#10b981"
    st.markdown("""
    <div style="background: {bg}; padding: 12px; border-radius: 6px; border: 2px solid {border};">
        <p style="color: #1e293b; font-weight: 600; font-size: 13px; margin: 0 0 8px 0;">λ₃ (CRÍTICO)</p>
        <p style="color: #1e293b; font-weight: 700; font-size: 16px; margin: 0;">{value:.4f}</p>
        <p style="color: #64748b; font-size: 11px; margin: 4px 0 0 0;">Invasión I</p>
    </div>
    """.format(bg=color_bg, border=color_border, value=ev3), unsafe_allow_html=True)

with col_ev4:
    st.markdown("""
    <div style="background: #f3e8ff; padding: 12px; border-radius: 6px; border: 1px solid #a855f7;">
        <p style="color: #1e293b; font-weight: 600; font-size: 13px; margin: 0 0 8px 0;">λ₄</p>
        <p style="color: #1e293b; font-weight: 700; font-size: 16px; margin: 0;">{:.4f}</p>
        <p style="color: #64748b; font-size: 11px; margin: 4px 0 0 0;">Recuperación</p>
    </div>
    """.format(ev4), unsafe_allow_html=True)

with col_ev5:
    st.markdown("""
    <div style="background: #d1fae5; padding: 12px; border-radius: 6px; border: 1px solid #10b981;">
        <p style="color: #1e293b; font-weight: 600; font-size: 13px; margin: 0 0 8px 0;">λ₅</p>
        <p style="color: #1e293b; font-weight: 700; font-size: 16px; margin: 0;">{:.4f}</p>
        <p style="color: #64748b; font-size: 11px; margin: 4px 0 0 0;">Mortalidad</p>
    </div>
    """.format(ev5), unsafe_allow_html=True)

# --- SECCIÓN 4: INTERPRETACIÓN DE RESULTADOS ---
st.markdown("---")
st.subheader("📋 Interpretación de Resultados")

if E_p1 <= 0:
    st.error("""
    ### ⚠️ Equilibrio Matemáticamente Imposible
    
    **Problema:** $E_{P1} \\leq 0$
    
    Este equilibrio **no existe biológicamente** con los parámetros actuales, ya que la población expuesta 
    debe ser positiva. Esto significa que el parámetro β (tasa de contacto) es demasiado bajo.
    
    **Soluciones:**
    - Aumenta **λ*** (tasa de entrada)
    - Disminuye **β** (tasa de contacto)
    - Verifica que β > μ²/λ*
    """)
elif ev3 < 0:
    st.success("""
    ### ✅ Sistema Estable ante Invasión de Infecciosos
    
    **Análisis:** $\\lambda_3 < 0$
    
    Aunque existan susceptibles y expuestos en equilibrio, **una pequeña perturbación en la población 
    infecciosa (I) decaerá exponencialmente**. El sistema permanecerá en P₁.
    
    **Interpretación:**
    - El equilibrio latente P₁ es **localmente estable**
    - Los infecciosos no pueden mantener la enfermedad
    - El sistema vuelve al equilibrio libre de enfermedad P₀
    """)
else:
    st.error("""
    ### 🔥 Sistema Inestable ante Invasión de Infecciosos
    
    **Análisis:** $\\lambda_3 > 0$
    
    Una pequeña perturbación en la población infecciosa (I) **crecerá exponencialmente**, 
    llevando el sistema desde P₁ hacia el **equilibrio endémico P*** (Caso 2).
    
    **Interpretación:**
    - El equilibrio latente P₁ es **inestable**
    - Los infecciosos pueden mantener la enfermedad
    - El sistema evolucionará a la enfermedad endémica
    """)

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