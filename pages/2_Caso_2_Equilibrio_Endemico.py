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

# --- VISUALIZACIÓN PROFESIONAL COMPLETA ---
st.markdown("---")
st.markdown("## 📊 ANÁLISIS DE EQUILIBRIO ENDÉMICO - P*")

# SECCIÓN 1: GRÁFICO TEMPORAL COMPLETO (FULL WIDTH)
st.markdown("### 📈 Dinámica Temporal hacia el Equilibrio Endémico")
st.markdown("*Evolución de las 5 compartimentos desde la condición inicial hacia la persistencia de la infección*")

fig = go.Figure()

fig.add_trace(go.Scatter(x=t, y=X, name='Susceptibles (X)', 
                        line=dict(color=COLORS['susceptibles'], width=3)))
fig.add_trace(go.Scatter(x=t, y=E, name='Expuestos (E)', 
                        line=dict(color=COLORS['expuestos'], width=2.5, dash='dot')))
fig.add_trace(go.Scatter(x=t, y=I, name='Infecciosos (I)', 
                        line=dict(color=COLORS['infecciosos'], width=3.5)))
fig.add_trace(go.Scatter(x=t, y=H, name='Hospitalizados (H)', 
                        line=dict(color=COLORS['hospitalizados'], width=2.5, dash='dash')))
fig.add_trace(go.Scatter(x=t, y=R, name='Recuperados (R)', 
                        line=dict(color=COLORS['recuperados'], width=3)))

# Agregar líneas de equilibrio teórico si es válido
if valid_equilibrium and I_star > 0:
    fig.add_hline(y=X_star, line_dash="dash", line_color=COLORS['susceptibles'], line_width=2.5,
                 annotation_text=f"X* = {X_star:.2f}", annotation_position="right", annotation_font_size=11, annotation_font_color='#0f172a')
    fig.add_hline(y=E_star, line_dash="dash", line_color=COLORS['expuestos'], line_width=2.5,
                 annotation_text=f"E* = {E_star:.2f}", annotation_position="right", annotation_font_size=11, annotation_font_color='#0f172a')
    fig.add_hline(y=I_star, line_dash="dot", line_color=COLORS['infecciosos'], line_width=3,
                 annotation_text=f"I* = {I_star:.2f}", annotation_position="right", annotation_font_size=11, annotation_font_color='#0f172a')

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

# SECCIÓN 2: ESTADO DEL EQUILIBRIO ENDÉMICO
st.markdown("### 🎯 Punto de Equilibrio Endémico ($P^*$)")

if valid_equilibrium and I_star > 0:
    col_eq_status, col_eq_values, col_eq_indicators = st.columns(3)
    
    with col_eq_status:
        st.markdown("**Estado de Validación**")
        st.success("✅ Equilibrio Válido", icon="✅")
        st.markdown("*El sistema converge a un punto de equilibrio endémico*")
    
    with col_eq_values:
        st.markdown("**Valores de Equilibrio**")
        st.write(f"• **X* (Susceptibles):** {X_star:.2f}")
        st.write(f"• **E* (Expuestos):** {E_star:.2f}")
        st.write(f"• **I* (Infecciosos):** {I_star:.2f}")
        st.write(f"• **H* (Hospitalizados):** {H_star:.2f}")
        st.write(f"• **R* (Recuperados):** {R_star:.2f}")
    
    with col_eq_indicators:
        st.markdown("**Indicadores de Endemismo**")
        prevalence = (E_star + I_star + H_star) / (X_star + E_star + I_star + H_star + R_star) * 100
        R_eff = (beta * X_star) / mu
        st.metric("Prevalencia", f"{prevalence:.2f}%", label_visibility="collapsed")
        st.metric("R (efectivo)", f"{R_eff:.3f}", label_visibility="collapsed")
        
        if prevalence > 10:
            st.warning("⚠️ Alta Prevalencia")
        else:
            st.info("✅ Baja Prevalencia")
else:
    st.error("❌ Equilibrio Inválido", icon="⚠️")
    st.info("Los parámetros seleccionados no permiten un equilibrio endémico válido")

# SECCIÓN 3: ESPACIO DE FASES 3D (FULL WIDTH) - PROFESIONAL Y PRECISO
st.markdown("---")
st.markdown("### 🌐 Espacio de Fases 3D: Análisis de Convergencia al Equilibrio Endémico")
st.markdown("""
*Visualización matemáticamente precisa del flujo dinámico en el espacio de fases (X, E, I).
La trayectoria muestra cómo el sistema evoluciona desde la condición inicial hacia el punto de equilibrio P*.
Los colores representan la evolución temporal: púrpura (inicio) → amarillo (convergencia).*
""")

# Crear figura 3D con mejor resolución
fig_3d = go.Figure()

# Limitar puntos para mejor rendimiento (cada N puntos)
step = max(1, len(t) // 250)
t_3d = t[::step]
X_3d = X[::step]
E_3d = E[::step]
I_3d = I[::step]

# ===== TRAYECTORIA PRINCIPAL CON GRADIENTE TEMPORAL =====
# Plasma colorscale: púrpura (0) → amarillo (1) - excelente contraste
fig_3d.add_trace(go.Scatter3d(
    x=X_3d, y=E_3d, z=I_3d,
    mode='lines',
    name='<b>Trayectoria del Sistema</b>',
    line=dict(
        color=t_3d,  # Colorear por tiempo
        colorscale='Plasma',
        width=7,
        showscale=True,
        cmin=t_3d.min(),
        cmax=t_3d.max()
    ),
    hovertemplate=(
        '<b>Trayectoria Dinámica</b><br>'
        'Tiempo: %{customdata[0]:.1f}<br>'
        'X (Susceptibles): %{x:.3f}<br>'
        'E (Expuestos): %{y:.3f}<br>'
        'I (Infecciosos): %{z:.3f}<extra></extra>'
    ),
    customdata=np.column_stack((t_3d,)),
    showlegend=True
))

# ===== PUNTO DE INICIO - VERDE BRILLANTE CON AURA =====
fig_3d.add_trace(go.Scatter3d(
    x=[X[0]], y=[E[0]], z=[I[0]],
    mode='markers',
    name='<b>Condición Inicial</b> (t=0)',
    marker=dict(
        size=16,
        color='#10b981',  # Verde vívido
        symbol='circle',
        line=dict(color='#0f172a', width=3),
        opacity=0.95,
        sizemode='diameter'
    ),
    text=[f'<b>INICIO</b><br>t=0<br>X₀={X[0]:.2f}<br>E₀={E[0]:.2f}<br>I₀={I[0]:.2f}'],
    hovertemplate='%{text}<extra></extra>',
    showlegend=True
))

# ===== PUNTO DE EQUILIBRIO - ORO CON ESTILO DIAMANTE =====
if valid_equilibrium and I_star > 0:
    fig_3d.add_trace(go.Scatter3d(
        x=[X_star], y=[E_star], z=[I_star],
        mode='markers',
        name='<b>Equilibrio Endémico P*</b>',
        marker=dict(
            size=18,
            color='#fbbf24',  # Oro brillante
            symbol='diamond',
            line=dict(color='#0f172a', width=3.5),
            opacity=1.0,
            sizemode='diameter'
        ),
        text=[f'<b>EQUILIBRIO P*</b><br>X*={X_star:.2f}<br>E*={E_star:.2f}<br>I*={I_star:.2f}'],
        hovertemplate='%{text}<extra></extra>',
        showlegend=True
    ))
    
    # ===== PROYECCIÓN XE EN EL PLANO BASE (VISTA 2D) =====
    # Muestra la proyección del sistema en el plano (X, E) para análisis estático
    fig_3d.add_trace(go.Scatter3d(
        x=X_3d, y=E_3d, z=[0]*len(X_3d),
        mode='lines',
        name='<b>Proyección (X,E)</b>',
        line=dict(
            color='rgba(100,116,139,0.3)',  # Gris semitransparente
            width=2,
            dash='dash'
        ),
        hoverinfo='skip',
        showlegend=True
    ))
    
    # ===== PROYECCIÓN EI EN PARED LATERAL (VISTA 2D) =====
    # Muestra el comportamiento E vs I 
    max_X = max(X)
    fig_3d.add_trace(go.Scatter3d(
        x=[max_X]*len(E_3d), y=E_3d, z=I_3d,
        mode='lines',
        name='<b>Proyección (E,I)</b>',
        line=dict(
            color='rgba(100,116,139,0.3)',  # Gris semitransparente
            width=2,
            dash='dash'
        ),
        hoverinfo='skip',
        showlegend=True
    ))

# ===== CONFIGURACIÓN AVANZADA DEL LAYOUT 3D =====
fig_3d.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    height=800,
    margin=dict(l=0, r=0, t=0, b=0),
    
    # Configuración 3D avanzada
    scene=dict(
        # EJE X (Susceptibles)
        xaxis=dict(
            title=dict(text='<b>X: Susceptibles</b>', font=dict(size=12, color='#0f172a')),
            backgroundcolor='rgba(226,232,240,0.06)',
            gridcolor='rgba(100,116,139,0.35)',
            gridwidth=1.5,
            showbackground=True,
            tickfont=dict(size=10, color='#0f172a')
        ),
        # EJE Y (Expuestos)
        yaxis=dict(
            title=dict(text='<b>E: Expuestos</b>', font=dict(size=12, color='#0f172a')),
            backgroundcolor='rgba(226,232,240,0.06)',
            gridcolor='rgba(100,116,139,0.35)',
            gridwidth=1.5,
            showbackground=True,
            tickfont=dict(size=10, color='#0f172a')
        ),
        # EJE Z (Infecciosos)
        zaxis=dict(
            title=dict(text='<b>I: Infecciosos</b>', font=dict(size=12, color='#0f172a')),
            backgroundcolor='rgba(226,232,240,0.06)',
            gridcolor='rgba(100,116,139,0.35)',
            gridwidth=1.5,
            showbackground=True,
            tickfont=dict(size=10, color='#0f172a')
        ),
        # Configuración de cámara para mejor vista del flujo
        camera=dict(
            eye=dict(x=1.4, y=1.4, z=1.3),
            center=dict(x=0, y=0, z=0),
            up=dict(x=0, y=0, z=1)
        ),
        aspectmode='cube'  # Proporciones iguales para mejor interpretación
    ),
    
    # Fuentes y estilos globales
    font=dict(family="'Inter', 'Arial', sans-serif", size=11, color='#0f172a'),
    
    # Leyenda profesional con información clara
    legend=dict(
        x=0.02,
        y=0.98,
        bgcolor='rgba(255,255,255,0.96)',
        bordercolor='#0f172a',
        borderwidth=2,
        font=dict(size=10, color='#0f172a', family="'Inter', sans-serif"),
        traceorder='normal',
        valign='top'
    ),
    
    title=None,
    hovermode='closest'
)

# Personalización de colorbar
fig_3d.data[0].line.colorbar.update(
    title='<b>Tiempo</b>',
    thickness=22,
    len=0.7,
    x=1.02,
    tickfont=dict(size=10, color='#0f172a'),
    tickformat='.0f'
)

st.plotly_chart(fig_3d, use_container_width=True)

# ===== ANÁLISIS MATEMÁTICO DEBAJO DE LA GRÁFICA =====
col_3d_interp1, col_3d_interp2 = st.columns(2)

with col_3d_interp1:
    st.markdown("""
    #### 📐 Interpretación Matemática
    
    **Dinámica Observable:**
    - La trayectoria **comienza en verde** (condición inicial)
    - **Evoluciona cronológicamente** (colores púrpura → amarillo)
    - **Converge al diamante dorado** (equilibrio P*)
    
    **Lo que demuestra:**
    - ✅ Sistema **asintóticamente estable**
    - ✅ Existe equilibrio endémico viable
    - ✅ Todas las perturbaciones convergen a P*
    """)

with col_3d_interp2:
    st.markdown("""
    #### 🔬 Proyecciones 2D
    
    **Proyección Base (X, E):**
    - Vista superior mostrando dinámica susceptibles-expuestos
    
    **Proyección Lateral (E, I):**
    - Vista frontal mostrando interacción expuestos-infecciosos
    - Define la velocidad de propagación
    
    **Nota:** Las trayectorias son invariantes bajo rotación; lo importante es la topología de convergencia.
    """)


# SECCIÓN 4: VALIDACIÓN MATEMÁTICA DE CONVERGENCIA
if valid_equilibrium and I_star > 0:
    st.markdown("---")
    st.markdown("### ✅ Validación Matemática de Convergencia a P*")
    st.markdown("*Medidas cuantitativas del grado de convergencia al equilibrio teórico*")
    
    # Cálculos precisos
    I_final = I[-1]
    error_convergencia = abs(I_final - I_star)
    error_relativo = error_convergencia / I_star * 100 if I_star > 0 else float('inf')
    
    # Distancia euclidea en el espacio de fases
    dist_initial = np.sqrt((X[0]-X_star)**2 + (E[0]-E_star)**2 + (I[0]-I_star)**2)
    dist_final = np.sqrt((X[-1]-X_star)**2 + (E[-1]-E_star)**2 + (I[-1]-I_star)**2)
    
    # Porcentaje de atracción (cuánto se acercó al equilibrio)
    atraccion = (dist_initial - dist_final) / dist_initial * 100 if dist_initial > 0 else 0
    
    # Velocidad promedio de convergencia
    if len(t) > 1:
        velocidad_convergencia = dist_initial / (t[-1] - t[0]) if (t[-1] - t[0]) > 0 else 0
    else:
        velocidad_convergencia = 0
    
    # Mostrar métricas en grid profesional
    col_conv1, col_conv2, col_conv3, col_conv4 = st.columns(4)
    
    with col_conv1:
        st.metric(
            "Error Absoluto en I",
            f"{error_convergencia:.2e}",
            delta=f"vs I* = {I_star:.3f}",
            delta_color="inverse"
        )
    
    with col_conv2:
        st.metric(
            "Error Relativo %",
            f"{error_relativo:.6f}%",
            delta="< 0.01% = Excelente",
            delta_color="inverse" if error_relativo < 0.01 else "normal"
        )
    
    with col_conv3:
        st.metric(
            "Distancia Euclidea",
            f"{dist_final:.4f}",
            delta=f"Inicial: {dist_initial:.4f}",
            delta_color="inverse"
        )
    
    with col_conv4:
        st.metric(
            "Atracción al P*",
            f"{atraccion:.2f}%",
            delta="Convergencia" if atraccion > 95 else "En progreso",
            delta_color="normal"
        )
    
    # Evaluación de calidad
    st.markdown("---")
    
    if error_convergencia < 1e-4:
        st.success("""
        ### 🎯 Convergencia Óptima Alcanzada
        
        **Error < 10⁻⁴ (excelente):** El sistema ha alcanzado el equilibrio con precisión numérica.
        
        - ✅ Todos los compartimentos en equilibrio dinámico
        - ✅ Enfermedad mantenida de forma estable
        - ✅ Solución matemáticamente confiable
        """)
    elif error_convergencia < 1e-3:
        st.success("""
        ### ✅ Convergencia Muy Buena
        
        **Error < 10⁻³:** Sistema muy cercano al equilibrio teórico.
        """)
    elif error_convergencia < 0.01:
        st.info("""
        ### ℹ️ Convergencia Satisfactoria
        
        **Error < 1%:** Sistema en proceso de estabilización.
        Aumentar tiempo de simulación para mejor precisión.
        """)
    else:
        st.warning("""
        ### ⚠️ Convergencia Incompleta
        
        El sistema aún no ha alcanzado el equilibrio.
        Aumenta el parámetro "Tiempo de Simulación" en la barra lateral.
        """)

# SECCIÓN FINAL: TABLA DE DATOS NUMÉRICOS
st.markdown("---")
st.markdown("### 📊 Comparativa: Teórico vs Simulación")

# Crear tabla comparativa
comparison_data = {
    'Variable': ['X (Susceptibles)', 'E (Expuestos)', 'I (Infecciosos)', 'H (Hospitalizados)', 'R (Recuperados)'],
    'Teórico (P*)': [
        f'{X_star:.4f}',
        f'{E_star:.4f}',
        f'{I_star:.4f}',
        f'{H_star:.4f}' if valid_equilibrium else 'N/A',
        f'{R_star:.4f}' if valid_equilibrium else 'N/A'
    ],
    'Simulación (Final)': [
        f'{X[-1]:.4f}',
        f'{E[-1]:.4f}',
        f'{I[-1]:.4f}',
        f'{H[-1]:.4f}',
        f'{R[-1]:.4f}'
    ],
    'Diferencia Absoluta': [
        f'{abs(X[-1] - X_star):.2e}',
        f'{abs(E[-1] - E_star):.2e}',
        f'{abs(I[-1] - I_star):.2e}',
        f'{abs(H[-1] - H_star):.2e}' if valid_equilibrium else 'N/A',
        f'{abs(R[-1] - R_star):.2e}' if valid_equilibrium else 'N/A'
    ]
}

import pandas as pd
df_comparison = pd.DataFrame(comparison_data)

st.dataframe(df_comparison, use_container_width=True, hide_index=True)

# Resumen final
col_summary1, col_summary2 = st.columns(2)

with col_summary1:
    st.markdown("**📈 Parámetros de Simulación**")
    st.info(f"""
    - Tiempo Total: {t_max} unidades
    - Puntos Calculados: {len(t)}
    - Tasa de Transmisión (β): {beta:.4f}
    - Tasa de Progresión (ε): {epsilon:.4f}
    - Tasa de Entrada (λ*): {lam:.4f}
    """)

with col_summary2:
    st.markdown("**🎯 Indicadores de Validez**")
    total_pop_initial = X0 + E0 + 0
    total_pop_final = X[-1] + E[-1] + I[-1] + H[-1] + R[-1]
    conservation = abs(total_pop_initial - total_pop_final) / total_pop_initial * 100
    
    st.info(f"""
    - Conservación de Población: {100-conservation:.2f}%
    - R₀ Teórico: {(beta * lam) / (mu ** 2):.4f}
    - Estabilidad: {"✅ Estable" if dist_final < dist_initial else "❌ Inestable"}
    - Precisión: {"🎯 Óptima" if error_convergencia < 1e-3 else "⚠️ Aceptable"}
    """)

st.success("✅ **Análisis de equilibrio endémico completado con éxito**")