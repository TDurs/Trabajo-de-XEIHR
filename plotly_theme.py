"""
Configuración global de tema Plotly para aplicación XEIHR.
Define estilos consistentes con fondo transparente para integración con CSS.
"""

# Colores institucionales del CSS
COLORS = {
    'primary': '#2563eb',      # Azul
    'primary_dark': '#1d4ed8', # Azul oscuro
    'secondary': '#7c3aed',    # Púrpura
    'success': '#10b981',      # Verde
    'warning': '#f59e0b',      # Naranja/Amarillo
    'error': '#ef4444',        # Rojo
    'dark': '#1e293b',         # Gris oscuro
    'light': '#f8fafc',        # Gris muy claro
    'gray': '#64748b',         # Gris medio
    'border': '#e2e8f0',       # Gris borde
    'susceptibles': '#2563eb',
    'expuestos': '#f59e0b',
    'infecciosos': '#ef4444',
    'hospitalizados': '#7c3aed',
    'recuperados': '#10b981'
}

def hex_to_rgb(hex_color):
    """Convierte color hex a RGB."""
    hex_color = hex_color.lstrip('#')
    return f"{int(hex_color[0:2], 16)}, {int(hex_color[2:4], 16)}, {int(hex_color[4:6], 16)}"

# Configuración base para todas las figuras
BASE_LAYOUT = {
    'template': 'plotly_white',
    'paper_bgcolor': 'rgba(0,0,0,0)',  # Transparente
    'plot_bgcolor': 'rgba(0,0,0,0)',   # Transparente
    'font': {
        'family': "'Inter', 'Arial', sans-serif",
        'size': 12,
        'color': '#0f172a'  # Casi negro para máximo contraste
    },
    'xaxis': {
        'showgrid': True,
        'gridwidth': 1.2,
        'gridcolor': 'rgba(100, 116, 139, 0.4)',  # Gris más visible
        'zeroline': False,
        'linecolor': '#64748b',  # Gris medio oscuro
        'linewidth': 2,
        'title': {
            'font': {'color': '#0f172a', 'size': 13}
        }
    },
    'yaxis': {
        'showgrid': True,
        'gridwidth': 1.2,
        'gridcolor': 'rgba(100, 116, 139, 0.4)',  # Gris más visible
        'zeroline': False,
        'linecolor': '#64748b',  # Gris medio oscuro
        'linewidth': 2,
        'title': {
            'font': {'color': '#0f172a', 'size': 13}
        }
    },
    'margin': {'l': 60, 'r': 40, 't': 80, 'b': 60},
    'hovermode': 'closest'
}

def get_transparent_layout(title="", height=500, **kwargs):
    """
    Retorna configuración de layout con fondo transparente.
    
    Args:
        title: Título del gráfico
        height: Altura en píxeles
        **kwargs: Argumentos adicionales para actualizar
    """
    layout = BASE_LAYOUT.copy()
    layout['height'] = height
    if title:
        layout['title'] = {
            'text': title,
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 14, 'color': COLORS['dark']}
        }
    layout.update(kwargs)
    return layout
