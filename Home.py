import streamlit as st

# Configuración de la página
st.set_page_config(
    page_title="Modelo XEIHR - Inicio",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 Simulación del Modelo Epidemiológico XEIHR")

st.markdown("""
### Panel de Control Central
Seleccione el escenario que desea analizar haciendo clic en los botones a continuación o usando el menú lateral.
""")

# --- BOTONES DE NAVEGACIÓN RÁPIDA ---
col1, col2 = st.columns(2)

with col1:
    st.info("📉 **Caso 1: Enfermedad Controlada**")
    st.markdown("Análisis del equilibrio cuando la infección desaparece ($P_0$).")
    st.page_link("pages/1_Caso_1_Libre_de_Enfermedad.py", label="Ir al Gráfico del Caso 1", icon="1️⃣", use_container_width=True)

with col2:
    st.warning("📈 **Caso 2: Enfermedad Endémica**")
    st.markdown("Análisis del equilibrio cuando la infección persiste ($P^*$).")
    st.page_link("pages/2_Caso_2_Equilibrio_Endemico.py", label="Ir al Gráfico del Caso 2", icon="2️⃣", use_container_width=True)

st.divider()

st.subheader("📚 Definición del Modelo Matemático")
st.latex(r"""
\begin{aligned}
\frac{dX}{dt} &= \lambda^{*} - \mu X - \beta X E \\
\frac{dE}{dt} &= \beta X E - \epsilon E I - \mu E \\
\frac{dI}{dt} &= \epsilon E I - (\mu + \omega + \gamma) I \\
\frac{dH}{dt} &= \omega I - (\mu + \alpha + \delta) H \\
\frac{dR}{dt} &= \gamma I + \delta H - \mu R
\end{aligned}
""")

st.markdown("""
#### Variables:
* **X:** Susceptibles | **E:** Expuestos | **I:** Infecciosos | **H:** Hospitalizados | **R:** Recuperados

#### Parámetros:
* **λ***: Tasa de entrada (nacimientos/inmigración)
* **μ**: Tasa de mortalidad natural
* **β**: Tasa de contacto efectivo
* **ε**: Tasa de progresión de Expuesto a Infeccioso
* **ω**: Tasa de hospitalización
* **γ**: Tasa de recuperación directa
* **α**: Mortalidad inducida por la enfermedad
* **δ**: Tasa de alta hospitalaria
""")