import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────
#  TRADUCCIONES
# ─────────────────────────────────────────────
TRANSLATIONS = {
    "🇪🇸 Español": {
        # Sidebar / nav
        "menu_title": "🏁 Menú Principal",
        "go_to": "Ir a:",
        "analyzer": "📊 Analizador de Telemetría",
        "faq": "❓ FAQ & Metodología",
        "settings": "⚙️ Configuración",
        "year": "Año",
        "sampling": "Muestreo (Cada N puntos)",
        "min_speed": "Velocidad Mínima",
        "clear_data": "🗑️ Borrar Datos Guardados",
        "grand_prix": "Gran Premio",
        "session": "Sesión",
        "driver": "Piloto",
        "language": "🌐 Idioma",
        # Main view
        "page_title": "🏎️ F1 2026 IA: Analizador de Telemetría",
        "load_laps": "🚀 Cargar Historial de Vueltas",
        "laps_loaded": "Cargadas {n} vueltas",
        "select_lap": "Selecciona la vuelta:",
        "analyze_lap": "📊 Analizar Vuelta Seleccionada",
        "analyzing": "Analizando...",
        "api_error": "Error en API ({endpoint}): {e}",
        # Metrics
        "lap_spend": "Gasto Vuelta",
        "recovery": "Recuperación",
        "net_balance": "Balance Neto",
        # Cluster labels
        "harvesting": "Harvesting (Carga)",
        "neutral": "Neutral/Coast",
        "deployment": "Deployment (Gasto)",
        "clipping": "Clipping (Fin de Energía)",
        # Hover
        "rpm": "RPM",
        "vel": "Vel",
        # Circuit type
        "circuit_type": "Tipo de Circuito (Límite 2026)",
        "circuit_normal": "🏁 Normal — 8.5 MJ",
        "circuit_limited": "⚠️ Recuperación limitada — 8.0 MJ",
        "circuit_highspeed": "🚀 Alta velocidad — 5.0 MJ",
        # FAQ
        "faq_title": "❓ Preguntas Frecuentes y Metodología",
        "faq_content": """
### 📖 Guía de Operación
Esta herramienta utiliza datos en tiempo real de la API OpenF1 para simular el comportamiento de los futuros monoplazas de **2026**.

1. **Filtro de Muestreo:** Si el mapa se ve con puntos muy separados, baja el muestreo a 1.
2. **Velocidad Mínima:** Úsalo para limpiar el mapa de paradas en boxes o trompos.

---
### 🧠 El Motor de Inteligencia Artificial
#### ¿Cómo clasificamos los datos?
Utilizamos un modelo de **Clustering K-Means**. Este algoritmo analiza cuatro variables simultáneamente:
* **Velocidad** ($v$)
* **Throttle** (Acelerador %)
* **Brake** (Freno %)
* **Aceleración Longitudinal** ($a$)

A partir de estos datos, la IA distingue automáticamente cuándo el coche está entregando potencia, cuándo está regenerando y cuándo está en fase neutra.

#### ¿Qué es el "Clipping" Morado?
El Clipping es el fenómeno donde el motor eléctrico deja de empujar en una recta porque se ha agotado el límite de energía por vuelta. En nuestro modelo, se detecta cuando:
$$ (Throttle > 95\\%) \\land (Aceleración \\le 0) \\land (Velocidad > 250 km/h) $$

---
### ⚡ Simulación de Energía 2026
El reglamento de 2026 establece **tres límites de recuperación de energía según el tipo de circuito**:
- **8.5 MJ** en circuitos convencionales con frenadas medianas/altas
- **8.0 MJ** en circuitos donde la recuperación no alcanza ese tope
- **5.0 MJ** en circuitos de alta velocidad sin grandes frenadas (Monza, Silverstone, Jeddah)

Con un MGU-K de **350 kW**, los 8.5 MJ permiten usar la potencia eléctrica máxima durante **~24 segundos** por vuelta. En circuitos rápidos (5 MJ), apenas **~14 segundos**.

* **Overdrive:** Por encima de los **290 km/h** la potencia eléctrica disponible decrece, lo que dificulta adelantar en recta en tramos de más de 290 km/h.
* **Recuperación:** La simulación calcula la energía recuperada por el MGU-K basándose en la presión del freno y la duración de la frenada. Sin recuperación no hay deployment.
""",
        "faq_tip": "💡 **Dato curioso:** En circuitos como Spa o Monza, el Clipping será el factor determinante para los adelantamientos en 2026.",
    },

    "🇬🇧 English": {
        "menu_title": "🏁 Main Menu",
        "go_to": "Go to:",
        "analyzer": "📊 Telemetry Analyzer",
        "faq": "❓ FAQ & Methodology",
        "settings": "⚙️ Settings",
        "year": "Year",
        "sampling": "Sampling (Every N points)",
        "min_speed": "Minimum Speed",
        "clear_data": "🗑️ Clear Saved Data",
        "grand_prix": "Grand Prix",
        "session": "Session",
        "driver": "Driver",
        "language": "🌐 Language",
        "page_title": "🏎️ F1 2026 AI: Telemetry Analyzer",
        "load_laps": "🚀 Load Lap History",
        "laps_loaded": "{n} laps loaded",
        "select_lap": "Select lap:",
        "analyze_lap": "📊 Analyze Selected Lap",
        "analyzing": "Analyzing...",
        "api_error": "API error ({endpoint}): {e}",
        "lap_spend": "Lap Deployment",
        "recovery": "Recovery",
        "net_balance": "Net Balance",
        "harvesting": "Harvesting (Charge)",
        "neutral": "Neutral/Coast",
        "deployment": "Deployment (Spend)",
        "clipping": "Clipping (Energy Depleted)",
        "rpm": "RPM",
        "vel": "Speed",
        # Circuit type
        "circuit_type": "Circuit Type (2026 Limit)",
        "circuit_normal": "🏁 Normal — 8.5 MJ",
        "circuit_limited": "⚠️ Limited recovery — 8.0 MJ",
        "circuit_highspeed": "🚀 High speed — 5.0 MJ",
        "faq_title": "❓ Frequently Asked Questions & Methodology",
        "faq_content": """
### 📖 Operation Guide
This tool uses real-time data from the OpenF1 API to simulate the behavior of the upcoming **2026** single-seaters.

1. **Sampling Filter:** If the map shows widely spaced points, lower the sampling to 1.
2. **Minimum Speed:** Use it to clean up the map from pit stops or spins.

---
### 🧠 The Artificial Intelligence Engine
#### How do we classify the data?
We use a **K-Means Clustering** model. This algorithm simultaneously analyzes four variables:
* **Speed** ($v$)
* **Throttle** (Accelerator %)
* **Brake** (Brake %)
* **Longitudinal Acceleration** ($a$)

From this data, the AI automatically distinguishes when the car is delivering power, when it is regenerating, and when it is in a neutral phase.

#### What is "Clipping" (Purple)?
Clipping is the phenomenon where the electric motor stops pushing on a straight because the lap energy limit has been exhausted. In our model, it is detected when:
$$ (Throttle > 95\\%) \\land (Acceleration \\le 0) \\land (Speed > 250 km/h) $$

---
### ⚡ 2026 Energy Simulation
The 2026 regulations set **three recovery energy limits depending on circuit type**:
- **8.5 MJ** on conventional circuits with medium/high braking zones
- **8.0 MJ** on circuits where recovery cannot reach that cap
- **5.0 MJ** on high-speed circuits with minimal braking (Monza, Silverstone, Jeddah)

With a **350 kW** MGU-K, 8.5 MJ allows maximum electric power for **~24 seconds** per lap. On fast circuits (5 MJ), only **~14 seconds**.

* **Overdrive:** Above **290 km/h** available electric power decreases, making overtaking on long straights more difficult.
* **Recovery:** The simulation calculates energy recovered by the MGU-K based on brake pressure and braking duration. Without recovery, there is no deployment.
""",
        "faq_tip": "💡 **Fun fact:** On circuits like Spa or Monza, Clipping will be the determining factor for overtakes in 2026.",
    },

    "🇧🇷 Português": {
        "menu_title": "🏁 Menu Principal",
        "go_to": "Ir para:",
        "analyzer": "📊 Analisador de Telemetria",
        "faq": "❓ FAQ & Metodologia",
        "settings": "⚙️ Configuração",
        "year": "Ano",
        "sampling": "Amostragem (A cada N pontos)",
        "min_speed": "Velocidade Mínima",
        "clear_data": "🗑️ Limpar Dados Salvos",
        "grand_prix": "Grande Prêmio",
        "session": "Sessão",
        "driver": "Piloto",
        "language": "🌐 Idioma",
        "page_title": "🏎️ F1 IA: Analisador de Telemetria",
        "load_laps": "🚀 Carregar Histórico de Voltas",
        "laps_loaded": "{n} voltas carregadas",
        "select_lap": "Selecione a volta:",
        "analyze_lap": "📊 Analisar Volta Selecionada",
        "analyzing": "Analisando...",
        "api_error": "Erro na API ({endpoint}): {e}",
        "lap_spend": "Gasto na Volta",
        "recovery": "Recuperação",
        "net_balance": "Saldo Líquido",
        "harvesting": "Harvesting (Carga)",
        "neutral": "Neutro/Planagem",
        "deployment": "Deployment (Gasto)",
        "clipping": "Clipping (Energia Esgotada)",
        "rpm": "RPM",
        "vel": "Vel",
        # Circuit type
        "circuit_type": "Tipo de Circuito (Limite 2026)",
        "circuit_normal": "🏁 Normal — 8,5 MJ",
        "circuit_limited": "⚠️ Recuperação limitada — 8,0 MJ",
        "circuit_highspeed": "🚀 Alta velocidade — 5,0 MJ",
        "faq_title": "❓ Perguntas Frequentes e Metodologia",
        "faq_content": """
### 📖 Guia de Operação
Esta ferramenta usa dados em tempo real da API OpenF1 para simular o comportamento dos futuros monolugares de **2026**.

1. **Filtro de Amostragem:** Se o mapa aparecer com pontos muito espaçados, reduza a amostragem para 1.
2. **Velocidade Mínima:** Use para limpar o mapa de paradas nos boxes ou trompos.

---
### 🧠 O Motor de Inteligência Artificial
#### Como classificamos os dados?
Utilizamos um modelo de **Clustering K-Means**. Este algoritmo analisa quatro variáveis simultaneamente:
* **Velocidade** ($v$)
* **Throttle** (Acelerador %)
* **Brake** (Freio %)
* **Aceleração Longitudinal** ($a$)

A partir desses dados, a IA distingue automaticamente quando o carro está entregando potência, quando está regenerando e quando está em fase neutra.

#### O que é o "Clipping" Roxo?
O Clipping é o fenômeno em que o motor elétrico para de empurrar numa reta porque o limite de energia por volta foi esgotado. No nosso modelo, é detectado quando:
$$ (Throttle > 95\\%) \\land (Aceleração \\le 0) \\land (Velocidade > 250 km/h) $$

---
### ⚡ Simulação de Energia 2026
O regulamento de 2026 estabelece **três limites de recuperação de energia conforme o tipo de circuito**:
- **8,5 MJ** em circuitos convencionais com zonas de frenagem médias/altas
- **8,0 MJ** em circuitos onde a recuperação não atinge esse teto
- **5,0 MJ** em circuitos de alta velocidade sem grandes frenagens (Monza, Silverstone, Jeddah)

Com um MGU-K de **350 kW**, os 8,5 MJ permitem usar a potência elétrica máxima por **~24 segundos** por volta. Em circuitos rápidos (5 MJ), apenas **~14 segundos**.

* **Overdrive:** Acima de **290 km/h** a potência elétrica disponível diminui, dificultando ultrapassagens em retas longas.
* **Recuperação:** A simulação calcula a energia recuperada pelo MGU-K com base na pressão do freio e na duração da frenagem. Sem recuperação, não há desdobramento.
""",
        "faq_tip": "💡 **Curiosidade:** Em circuitos como Spa ou Monza, o Clipping será o fator determinante para as ultrapassagens em 2026.",
    },
}

# ─────────────────────────────────────────────
#  CONFIGURACIÓN DE PÁGINA
# ─────────────────────────────────────────────
st.set_page_config(page_title="F1-Explained", layout="wide", page_icon="🏎️")
BASE_URL = "https://api.openf1.org/v1"

# ─────────────────────────────────────────────
#  TOPBAR F1-EXPLAINED
# ─────────────────────────────────────────────
st.html("""
<link href="https://fonts.googleapis.com/css2?family=Titillium+Web:wght@200;400;600;700;900&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  header[data-testid="stHeader"] {
    background: transparent !important;
    height: 0px !important;
  }
  .main .block-container {
    padding-top: 72px !important;
  }
  section[data-testid="stSidebar"] > div:first-child {
    padding-top: 72px !important;
  }
  #f1-topbar {
    position: fixed;
    top: 0; left: 0; right: 0;
    z-index: 9999;
    height: 52px;
    background: #E8002D;
    display: flex;
    align-items: center;
    padding: 0 24px;
    box-shadow: 0 2px 24px rgba(232,0,45,0.45);
    border-bottom: 2px solid #C0001E;
  }
  #f1-topbar::after {
    content: "";
    position: absolute;
    right: 0; top: 0; bottom: 0;
    width: 320px;
    background: linear-gradient(105deg, transparent 30px, rgba(0,0,0,0.12) 30px);
    pointer-events: none;
  }
  /* Logo centrado en el viewport completo, ignora el sidebar */
  #f1-logo {
    position: absolute;
    left: 50%;
    transform: translateX(-50%);
    font-family: "Titillium Web", sans-serif;
    font-weight: 900;
    color: #FFFFFF;
    text-transform: uppercase;
    display: flex;
    align-items: baseline;
    gap: 0;
    line-height: 1;
    user-select: none;
    white-space: nowrap;
  }
  #f1-logo .f1-part {
    font-size: 28px;
    letter-spacing: -0.04em;
  }
  #f1-logo .dash {
    color: rgba(255,255,255,0.45);
    margin: 0 6px;
    font-weight: 200;
    font-size: 20px;
  }
  #f1-logo .explained-part {
    font-weight: 600;
    letter-spacing: 0.14em;
    font-size: 14px;
    color: rgba(255,255,255,0.90);
  }
  #f1-tagline {
    margin-left: auto;
    font-family: "DM Mono", monospace;
    font-size: 10px;
    letter-spacing: 0.22em;
    color: rgba(255,255,255,0.50);
    text-transform: uppercase;
    z-index: 1;
  }
</style>
<div id="f1-topbar">
  <div id="f1-logo">
    <span class="f1-part">F1</span>
    <span class="dash">&#8212;</span>
    <span class="explained-part">EXPLAINED</span>
  </div>
  <div id="f1-tagline">Telemetry &amp; Energy Analysis</div>
</div>
""")

# ─────────────────────────────────────────────
#  SELECTOR DE IDIOMA (primer widget del sidebar)
# ─────────────────────────────────────────────
with st.sidebar:
    lang_options = list(TRANSLATIONS.keys())

    # Inicializar idioma en session state si no existe
    if "selected_lang" not in st.session_state:
        st.session_state.selected_lang = lang_options[0]  # Español por defecto

    selected_lang = st.selectbox(
        "🌐 Language / Idioma / Idioma",
        lang_options,
        index=lang_options.index(st.session_state.selected_lang),
        key="lang_selector"
    )
    st.session_state.selected_lang = selected_lang
    st.divider()

# Atajo para obtener texto traducido
T = TRANSLATIONS[selected_lang]

# ─────────────────────────────────────────────
#  CONSULTA API
# ─────────────────────────────────────────────
def get_data_api(endpoint, params=None):
    try:
        req = requests.Request('GET', f"{BASE_URL}/{endpoint}", params=params)
        prepared = req.prepare()
        r = requests.get(prepared.url, timeout=30)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        st.error(T["api_error"].format(endpoint=endpoint, e=e))
    return []

# ─────────────────────────────────────────────
#  MOTOR IA Y ENERGÍA
#  Claves internas FIJAS (no traducidas) para
#  que la lógica no dependa del idioma activo.
# ─────────────────────────────────────────────
IA_HARVESTING  = "harvesting"
IA_NEUTRAL     = "neutral"
IA_DEPLOYMENT  = "deployment"
IA_CLIPPING    = "clipping"

def ia_label(key):
    """Devuelve el string traducido para mostrar en pantalla."""
    return T[key]

def aplicar_ia_f1(df):
    if len(df) < 10:
        return df
    df['accel'] = df['speed'].diff().fillna(0)
    features = ['speed', 'throttle', 'brake', 'accel']
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df[features].fillna(0))
    model = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['cluster'] = model.fit_predict(data_scaled)
    c_means = df.groupby('cluster')['throttle'].mean().sort_values()
    # ia_status_key: clave interna fija
    mapping = {
        c_means.index[0]: IA_HARVESTING,
        c_means.index[1]: IA_NEUTRAL,
        c_means.index[2]: IA_DEPLOYMENT,
    }
    df['ia_status_key'] = df['cluster'].map(mapping)
    clipping_mask = (df['throttle'] > 95) & (df['accel'] <= 0) & (df['speed'] > 250)
    df.loc[clipping_mask, 'ia_status_key'] = IA_CLIPPING
    # ia_status: string traducido solo para mostrar
    df['ia_status'] = df['ia_status_key'].map(ia_label)
    return df


def calcular_energia_2026(df):
    # Cap dt a 0.5s — gaps mayores son pausas de telemetría, no tiempo real de motor
    df['dt'] = df['date'].diff().dt.total_seconds().fillna(0).clip(upper=0.12)
    df['racha_id'] = (df['ia_status_key'] != df['ia_status_key'].shift()).cumsum()
    
    # Constantes físicas
    MASA_F1 = 800  # kg (peso mínimo reglamentario ~798 kg con piloto)
    EF_MGU_K = 0.75  # Eficiencia de conversión del MGU-K (~70-80%)
    P_MAX_DEPLOYMENT = 350000  # W
    P_MAX_HARVESTING = 350000  # W (límite de potencia regenerativa)

    def estimar_potencia(row, idx):
        v, key = row['speed'], row['ia_status_key']
        
        # ── DEPLOYMENT ──────────────────────────────────
        if key == IA_DEPLOYMENT:
            p_max = P_MAX_DEPLOYMENT
            # Derating por encima de 290 km/h
            if v > 290:
                factor = max(0.3, 1 - (v - 290) / 100)
                p_max *= factor
            return p_max * (row['throttle'] / 100)
        
        # ── CLIPPING  ──────────────────────────────
        elif key == IA_CLIPPING:
            return 0
        
        # ── HARVESTING ───────────────────────────
        elif key == IA_HARVESTING:
            # Obtener velocidad siguiente para calcular ΔE_cinética
            if idx < len(df) - 1:
                v_actual = row['speed'] / 3.6  # km/h -> m/s
                v_next = df.iloc[idx + 1]['speed'] / 3.6
                dt = row['dt']
                
                # ΔE_cinética = 0.5 * m * (v1² - v2²)
                delta_E_k = 0.5 * MASA_F1 * (v_actual**2 - v_next**2)
                
                # Factor de regeneración: a más freno, más va a mecánicos
                # Con brake bajo el MGU-K captura casi todo
                # Con brake alto (>80%) los frenos mecánicos disipan mucho
                brake_pct = min(row['brake'], 100)
                if brake_pct > 5:
                    # Curva: 90% regen con brake bajo, 20% con brake muy alto
                    regen_factor = max(0.2, 1 - brake_pct / 150)
                else:
                    # Sin freno activo, lift & coast recupera menos
                    regen_factor = 0.3
                
                # Energía recuperable = ΔE_k * eficiencia * factor_regen
                E_recuperable = delta_E_k * EF_MGU_K * regen_factor
                
                # Potencia = Energía / tiempo, limitada a 350 kW
                if dt > 0:
                    potencia = -min(E_recuperable / dt, P_MAX_HARVESTING)
                else:
                    potencia = 0
                
                return potencia
            else:
                # Último punto, asumir potencia mínima
                return -40000
        
        # ── NEUTRAL ─────────────────────────────────────────────
        return 0

    # Aplicar cálculo punto por punto con índice
    df['power_w'] = [estimar_potencia(row, idx) for idx, row in df.iterrows()]
    df['energy_j'] = df['power_w'] * df['dt']
    return df

# ─────────────────────────────────────────────
#  NAVEGACIÓN
# ─────────────────────────────────────────────
with st.sidebar:
    st.title(T["menu_title"])
    v_seleccionada = st.radio(
        T["go_to"],
        [T["analyzer"], T["faq"]]
    )
    st.divider()

    # ── BOTÓN DONAR ─────────────────────────────
    import streamlit.components.v1 as components
    components.html("""
<!DOCTYPE html>
<html>
<head>
<link href="https://fonts.googleapis.com/css2?family=Titillium+Web:wght@700;900&display=swap" rel="stylesheet">
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  body { background:transparent; padding: 4px 0; }
  a#donate-btn {
    display: block;
    width: 100%;
    padding: 10px 0;
    background: transparent;
    border: 1.5px solid #E8002D;
    border-radius: 3px;
    color: #E8002D;
    font-family: "Titillium Web", sans-serif;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: .2em;
    text-transform: uppercase;
    cursor: pointer;
    transition: background .2s, color .2s;
    text-align: center;
    text-decoration: none;
  }
  a#donate-btn:hover { background: #E8002D; color: #fff; }
</style>
</head>
<body>
  <a id="donate-btn" href="https://link.mercadopago.com.ar/lucaprestia" target="_blank" rel="noopener">
    ❤ &nbsp;Apoyar el proyecto
  </a>
</body>
</html>
""", height=46, scrolling=False)
    st.divider()

# ─────────────────────────────────────────────
#  VISTA 1: ANALIZADOR
# ─────────────────────────────────────────────
if v_seleccionada == T["analyzer"]:
    # ── SIDEBAR: solo ajustes técnicos ──────────────────────
    with st.sidebar:
        st.header(T["settings"])
        year = st.selectbox(T["year"], [2026], index=0)
        muestreo = st.slider(T["sampling"], 1, 10, 1)
        v_min = st.slider(T["min_speed"], 0, 100, 0)
        circuit_options = {
            T["circuit_normal"]:    8.5,
            T["circuit_limited"]:   8.0,
            T["circuit_highspeed"]: 5.0,
        }
        sel_circuit = st.selectbox(T["circuit_type"], list(circuit_options.keys()))
        ENERGY_LIMIT = circuit_options[sel_circuit]
        st.divider()
        if st.button(T["clear_data"], use_container_width=True):
            st.session_state.laps_data = None
            st.session_state.telemetry_data = None
            st.rerun()

    # ── SESSION STATE ────────────────────────────────────────
    if "laps_data" not in st.session_state:
        st.session_state.laps_data = None
    if "telemetry_data" not in st.session_state:
        st.session_state.telemetry_data = None

    # ── PÁGINA PRINCIPAL ─────────────────────────────────────
    st.title(T["page_title"])

    # ── PASO 1: Selección GP / Sesión / Piloto ───────────────
    st.html("""<div style="display:flex;align-items:center;gap:12px;margin-bottom:4px">
      <div style="font-family:'Titillium Web',sans-serif;font-size:11px;font-weight:700;
                  letter-spacing:.25em;color:#E8002D;text-transform:uppercase">
        <span style="display:inline-flex;align-items:center;justify-content:center;width:18px;height:18px;border:1.5px solid #E8002D;border-radius:50%;font-size:10px;font-weight:700;margin-right:8px;flex-shrink:0">1</span>Selección
      </div>
      <div style="flex:1;height:1px;background:#222230"></div>
    </div>""")

    meetings = get_data_api("meetings", {"year": year})
    if meetings:
        m_map = {m['meeting_official_name']: m['meeting_key'] for m in meetings}
        c1, c2, c3 = st.columns(3)
        with c1:
            sel_gp = st.selectbox(T["grand_prix"], list(m_map.keys()))
        s_raw = get_data_api("sessions", {"meeting_key": m_map[sel_gp]})
        s_map = {s['session_name']: s['session_key'] for s in s_raw}
        with c2:
            sel_session = st.selectbox(T["session"], list(s_map.keys()))
        s_key = s_map[sel_session]
        d_raw = get_data_api("drivers", {"session_key": s_key})
        d_map = {f"{d['last_name']} (#{d['driver_number']})": d['driver_number'] for d in d_raw}
        with c3:
            sel_driver_name = st.selectbox(T["driver"], list(d_map.keys()))
        d_num = d_map[sel_driver_name]

        if st.button(T["load_laps"], type="primary"):
            laps_raw = get_data_api("laps", {"session_key": s_key, "driver_number": d_num})
            if laps_raw:
                df_l = pd.DataFrame(laps_raw).dropna(subset=['date_start', 'lap_duration'])
                df_l['date_start'] = pd.to_datetime(df_l['date_start'], format='mixed')
                st.session_state.laps_data = df_l.sort_values('lap_number')
                st.session_state.telemetry_data = None
                st.success(T["laps_loaded"].format(n=len(df_l)))

    # ── PASO 2: Selección de vuelta ──────────────────────────
    if st.session_state.laps_data is not None:
        st.html("""<div style="display:flex;align-items:center;gap:12px;margin:16px 0 4px">
          <div style="font-family:'Titillium Web',sans-serif;font-size:11px;font-weight:700;
                      letter-spacing:.25em;color:#E8002D;text-transform:uppercase">
            <span style="display:inline-flex;align-items:center;justify-content:center;width:18px;height:18px;border:1.5px solid #E8002D;border-radius:50%;font-size:10px;font-weight:700;margin-right:8px;flex-shrink:0">2</span>Vuelta
          </div>
          <div style="flex:1;height:1px;background:#222230"></div>
        </div>""")

        n_laps = len(st.session_state.laps_data)
        st.markdown(
            f"**{sel_driver_name}** · {sel_session} · {sel_gp} · "
            f"<span style='color:#6060A0'>{n_laps} vueltas</span>",
            unsafe_allow_html=True
        )

        def fmt_lap(row):
            dur = row['lap_duration']
            try:
                mins = int(dur) // 60
                secs = dur - mins * 60
                t_str = f"{mins}:{secs:06.3f}"
            except Exception:
                t_str = "–:–––"
            return f"Vuelta {int(row['lap_number'])}  —  {t_str}"

        laps_df = st.session_state.laps_data
        lap_options = {fmt_lap(row): row['lap_number'] for _, row in laps_df.iterrows()}
        col_sel, col_btn = st.columns([3, 1])
        with col_sel:
            sel_label = st.selectbox(T["select_lap"], list(lap_options.keys()),
                                     index=len(lap_options) - 1, label_visibility="collapsed")
        sel_lap = lap_options[sel_label]
        v_info = laps_df[laps_df['lap_number'] == sel_lap].iloc[0]
        with col_btn:
            do_analyze = st.button(T["analyze_lap"], type="primary", use_container_width=True)

        if do_analyze:
            with st.spinner(T["analyzing"]):
                t_start = v_info['date_start']
                t_end = t_start + pd.Timedelta(seconds=v_info['lap_duration'] + 0.8)
                params = {
                    "session_key": s_key,
                    "driver_number": d_num,
                    "date>": t_start.isoformat(),
                    "date<": t_end.isoformat()
                }
                c_raw = get_data_api("car_data", params)
                l_raw = get_data_api("location", params)
                if c_raw and l_raw:
                    df = pd.merge_asof(
                        pd.DataFrame(c_raw)
                          .assign(date=lambda x: pd.to_datetime(x['date'], format='mixed'))
                          .sort_values('date'),
                        pd.DataFrame(l_raw)
                          .assign(date=lambda x: pd.to_datetime(x['date'], format='mixed'))
                          .sort_values('date')[['date', 'x', 'y']],
                        on='date', direction='nearest', tolerance=pd.Timedelta(seconds=1)
                    )
                    if v_min > 0:
                        df = df[df['speed'] >= v_min]
                    if muestreo > 1:
                        df = df.iloc[::muestreo]
                    st.session_state.telemetry_data = (
                        df.dropna(subset=['x', 'y'])
                          .pipe(aplicar_ia_f1)
                          .pipe(calcular_energia_2026)
                    )

    if st.session_state.telemetry_data is not None:
        st.html("""<div style="display:flex;align-items:center;gap:12px;margin:16px 0 4px">
          <div style="font-family:'Titillium Web',sans-serif;font-size:11px;font-weight:700;
                      letter-spacing:.25em;color:#E8002D;text-transform:uppercase">
            <span style="display:inline-flex;align-items:center;justify-content:center;width:18px;height:18px;border:1.5px solid #E8002D;border-radius:50%;font-size:10px;font-weight:700;margin-right:8px;flex-shrink:0">3</span>Análisis
          </div>
          <div style="flex:1;height:1px;background:#222230"></div>
        </div>""")
        df_p = st.session_state.telemetry_data
        # Usar ia_status_key (clave fija) para la lógica de energía
        rachas = df_p.groupby('racha_id').agg({'ia_status_key': 'first', 'energy_j': 'sum'}).reset_index()
        gasto = rachas[rachas['ia_status_key'] == IA_DEPLOYMENT]['energy_j'].sum() / 1e6
        carga = abs(rachas[rachas['ia_status_key'] == IA_HARVESTING]['energy_j'].sum() / 1e6)

        LIMIT = ENERGY_LIMIT
        balance = gasto - carga
        exceso  = max(0, gasto - LIMIT)
        limit_pct   = min(LIMIT / max(gasto, 0.01), 1.0) * 100
        over_width  = max(0, 100 - limit_pct)
        regen_pct   = min(carga / LIMIT, 1.0) * 100
        balance_pct_red   = min(gasto  / max(gasto + carga, 0.01), 1.0) * 100
        balance_pct_green = 100 - balance_pct_red

        # Textos dinámicos por idioma
        t_deploy_label = T.get("w_deploy_label",   {"🇪🇸 Español":"Gasto (Deployment)","🇬🇧 English":"Deployment","🇧🇷 Português":"Gasto (Deployment)"}[selected_lang])
        t_over_note    = T.get("w_over_note",       {"🇪🇸 Español":"sobre el límite","🇬🇧 English":"over limit","🇧🇷 Português":"acima do limite"}[selected_lang])
        t_within_note  = T.get("w_within_note",     {"🇪🇸 Español":"dentro del límite","🇬🇧 English":"within limit","🇧🇷 Português":"dentro do limite"}[selected_lang])
        t_clip_note    = T.get("w_clip_note",       {"🇪🇸 Español":"⚠ El motor eléctrico se <strong style='color:#CC00FF'>clipea</strong> — deja de empujar porque superó el límite de 4.0 MJ por vuelta.",
                                                       "🇬🇧 English":"⚠ The electric motor <strong style='color:#CC00FF'>clips</strong> — stops pushing because it exceeded the 4.0 MJ per-lap limit.",
                                                       "🇧🇷 Português":"⚠ O motor elétrico <strong style='color:#CC00FF'>clipeia</strong> — para de empurrar pois ultrapassou o limite de 4,0 MJ por volta."}[selected_lang])
        t_noClip_note  = T.get("w_noClip_note",     {"🇪🇸 Español":"✓ Sin clipping en esta vuelta. El motor eléctrico entregó potencia de forma continua.",
                                                       "🇬🇧 English":"✓ No clipping this lap. The electric motor delivered power continuously.",
                                                       "🇧🇷 Português":"✓ Sem clipping nesta volta. O motor elétrico entregou potência continuamente."}[selected_lang])
        t_regen_label  = T.get("w_regen_label",     {"🇪🇸 Español":"Recuperación (MGU-K)","🇬🇧 English":"Recovery (MGU-K)","🇧🇷 Português":"Recuperação (MGU-K)"}[selected_lang])
        t_regen_desc   = T.get("w_regen_desc",      {"🇪🇸 Español":"Energía capturada en frenadas reutilizada en rectas.","🇬🇧 English":"Energy captured in braking, reused on straights.","🇧🇷 Português":"Energia capturada nas frenagens e reutilizada nas retas."}[selected_lang])
        t_regen_active = T.get("w_regen_active",    {"🇪🇸 Español":"Regen activo","🇬🇧 English":"Regen active","🇧🇷 Português":"Regen ativo"}[selected_lang])
        t_balance_lbl  = T.get("w_balance_lbl",     {"🇪🇸 Español":"Balance Neto","🇬🇧 English":"Net Balance","🇧🇷 Português":"Saldo Líquido"}[selected_lang])
        t_bal_deficit  = T.get("w_bal_deficit",     {"🇪🇸 Español":"El <strong style='color:#FF1801'>gasto supera</strong> lo recuperado. Clipping inevitable en rectas.","🇬🇧 English":"<strong style='color:#FF1801'>Deployment exceeds</strong> recovery. Clipping inevitable on straights.","🇧🇷 Português":"O <strong style='color:#FF1801'>gasto supera</strong> a recuperação. Clipping inevitável nas retas."}[selected_lang])
        t_bal_ok       = T.get("w_bal_ok",          {"🇪🇸 Español":"Buena eficiencia energética. La recuperación cubre gran parte del gasto.","🇬🇧 English":"Good energy efficiency. Recovery covers most of the deployment.","🇧🇷 Português":"Boa eficiência energética. A recuperação cobre grande parte do gasto."}[selected_lang])
        t_deficit_pill = T.get("w_deficit_pill",    {"🇪🇸 Español":"Balance deficitario","🇬🇧 English":"Deficit balance","🇧🇷 Português":"Saldo deficitário"}[selected_lang])
        t_ok_pill      = T.get("w_ok_pill",         {"🇪🇸 Español":"Balance positivo","🇬🇧 English":"Positive balance","🇧🇷 Português":"Saldo positivo"}[selected_lang])
        t_legend_dep   = T.get("w_legend_dep",      {"🇪🇸 Español":"Energía eléctrica enviada a las ruedas para acelerar.","🇬🇧 English":"Electric energy sent to wheels for acceleration.","🇧🇷 Português":"Energia elétrica enviada às rodas para acelerar."}[selected_lang])
        t_legend_hrv   = T.get("w_legend_hrv",      {"🇪🇸 Español":"Energía recuperada por el MGU-K en frenadas.","🇬🇧 English":"Energy recovered by MGU-K under braking.","🇧🇷 Português":"Energia recuperada pelo MGU-K nas frenagens."}[selected_lang])
        t_legend_clip  = T.get("w_legend_clip",     {"🇪🇸 Español":"Motor eléctrico sin energía — sin empuje extra en recta.","🇬🇧 English":"Electric motor out of energy — no extra push on straight.","🇧🇷 Português":"Motor elétrico sem energia — sem impulso extra na reta."}[selected_lang])

        t_methodology_note = {
            "🇪🇸 Español": "Esta estimación usa una fórmula genérica de 350 kW para todos los motores. No refleja la eficiencia particular de cada unidad de potencia.",
            "🇬🇧 English": "This estimate uses a generic 350 kW formula for all power units. It does not reflect the specific efficiency of each manufacturer.",
            "🇧🇷 Português": "Esta estimativa usa uma fórmula genérica de 350 kW para todas as unidades de potência. Não reflete a eficiência específica de cada fabricante.",
        }[selected_lang]
        over_label    = f"+{exceso:.2f} {t_over_note}"   if exceso > 0 else t_within_note
        bar_note      = t_clip_note if exceso > 0 else t_noClip_note
        balance_class = "critical" if balance > 0 else "ok"
        balance_color = "#FF1801" if balance > 0 else "#00E5A0"
        balance_desc  = t_bal_deficit if balance > 0 else t_bal_ok
        balance_pill  = t_deficit_pill if balance > 0 else t_ok_pill
        pill_class    = "danger" if balance > 0 else "ok"

        # ── Donut: arco verde y rojo según proporción
        CIRCUM = 175.9
        green_arc = balance_pct_green / 100 * CIRCUM
        red_arc   = balance_pct_red   / 100 * CIRCUM
        red_offset = CIRCUM - green_arc

        html_widget = f"""
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=DM+Mono:wght@300;400;500&display=swap" rel="stylesheet">
<style>
.ef-wrap {{ font-family:'DM Mono',monospace; color:#E8E8F0; }}
.ef-header {{ font-family:'Orbitron',sans-serif; font-size:10px; letter-spacing:.3em; color:#6060A0; text-transform:uppercase; margin-bottom:16px; display:flex; align-items:center; gap:10px; }}
.ef-header::after {{ content:''; flex:1; height:1px; background:#222230; }}
.ef-deploy {{ background:#111118; border:1px solid #222230; border-radius:4px; padding:20px; margin-bottom:14px; position:relative; overflow:hidden; }}
.ef-deploy::before {{ content:''; position:absolute; top:0;left:0;right:0; height:2px; background:linear-gradient(90deg,#FF1801,#CC00FF 60%,transparent); }}
.ef-row {{ display:flex; justify-content:space-between; align-items:baseline; margin-bottom:10px; flex-wrap:wrap; gap:8px; }}
.ef-label {{ font-family:'Orbitron',sans-serif; font-size:9px; letter-spacing:.2em; color:#6060A0; text-transform:uppercase; }}
.ef-val {{ font-family:'Orbitron',sans-serif; font-size:26px; font-weight:900; color:#FF1801; }}
.ef-val .unit {{ font-size:13px; font-weight:400; color:#6060A0; margin-left:3px; }}
.ef-val .sub {{ font-size:11px; font-weight:400; color:#CC00FF; margin-left:10px; }}
.ef-track {{ height:14px; background:#1a1a28; border-radius:2px; position:relative; margin-bottom:8px; overflow:visible; }}
.ef-fill-deploy {{ height:100%; border-radius:2px; background:linear-gradient(90deg,#FF1801,#FF6B00); width:{limit_pct:.1f}%; }}
.ef-fill-over {{ position:absolute; top:0; height:100%; left:{limit_pct:.1f}%; width:{over_width:.1f}%;
  background:repeating-linear-gradient(60deg,#FF00FF22 0px,#FF00FF22 4px,transparent 4px,transparent 8px);
  border-left:2px solid #CC00FF; }}
.ef-limit {{ position:absolute; top:-6px; bottom:-6px; width:2px; background:#FFD600; left:{limit_pct:.1f}%; z-index:10; }}
.ef-limit-tag {{ position:absolute; top:-22px; left:{limit_pct:.1f}%; transform:translateX(-50%);
  font-family:'Orbitron',sans-serif; font-size:9px; color:#FFD600; letter-spacing:.1em; white-space:nowrap; }}
.ef-note {{ font-size:10px; color:#6060A0; margin-top:4px; }}
.ef-bottom {{ display:grid; grid-template-columns:1fr 1fr; gap:14px; margin-bottom:14px; }}
.ef-card {{ background:#111118; border:1px solid #222230; border-radius:4px; padding:18px; position:relative; overflow:hidden; }}
.ef-card::before {{ content:''; position:absolute; top:0;left:0;right:0; height:2px; }}
.ef-card.recovery::before {{ background:#00E5A0; }}
.ef-card.balance.critical::before {{ background:#FF1801; }}
.ef-card.balance.ok::before {{ background:#00E5A0; }}
.ef-clabel {{ font-family:'Orbitron',sans-serif; font-size:9px; letter-spacing:.2em; color:#6060A0; text-transform:uppercase; margin-bottom:10px; }}
.ef-cval {{ font-family:'Orbitron',sans-serif; font-size:24px; font-weight:900; line-height:1; }}
.ef-cval .unit {{ font-size:11px; font-weight:400; color:#6060A0; margin-left:3px; }}
.ef-cdesc {{ font-size:10px; color:#6060A0; margin-top:8px; line-height:1.5; }}
.ef-regen-track {{ height:8px; background:#1a1a28; border-radius:2px; margin-top:10px; overflow:hidden; }}
.ef-regen-fill {{ height:100%; background:linear-gradient(90deg,#00994d,#00E5A0); width:{regen_pct:.1f}%; border-radius:2px; }}
.ef-regen-labels {{ display:flex; justify-content:space-between; font-size:9px; color:#6060A0; margin-top:3px; }}
.ef-pill {{ display:inline-flex; align-items:center; gap:5px; padding:3px 9px; border-radius:2px;
  font-family:'Orbitron',sans-serif; font-size:8px; font-weight:700; letter-spacing:.15em; text-transform:uppercase; margin-top:8px; }}
.ef-pill.danger {{ background:#FF180115; border:1px solid #FF180155; color:#FF1801; }}
.ef-pill.ok     {{ background:#00E5A015; border:1px solid #00E5A055; color:#00E5A0; }}
.ef-pill .dot   {{ width:5px; height:5px; border-radius:50%; background:currentColor; animation:efpulse 1.4s infinite; }}
@keyframes efpulse {{ 0%,100%{{opacity:1}} 50%{{opacity:.3}} }}
.ef-fill-deploy {{ animation:efgrow 1.2s cubic-bezier(0.16,1,0.3,1) forwards; transform-origin:left; }}
.ef-fill-over   {{ animation:efgrow 1.2s 0.1s cubic-bezier(0.16,1,0.3,1) forwards; transform-origin:left; }}
.ef-regen-fill  {{ animation:efgrow 1.4s 0.3s cubic-bezier(0.16,1,0.3,1) both; transform-origin:left; }}
@keyframes efgrow {{ from{{transform:scaleX(0)}} to{{transform:scaleX(1)}} }}
@keyframes efdonutg {{ from{{stroke-dasharray:0 175.9}} to{{stroke-dasharray:{green_arc:.1f} {CIRCUM:.1f}}} }}
@keyframes efdonutr {{ from{{stroke-dashoffset:0}} to{{stroke-dashoffset:-{red_offset:.1f}}} }}
.ef-donut-wrap {{ display:flex; align-items:center; gap:14px; }}
.ef-legend {{ background:#111118; border:1px solid #222230; border-radius:4px; padding:14px 18px;
  display:grid; grid-template-columns:repeat(3,1fr); gap:10px; }}
.ef-exp {{ display:flex; align-items:flex-start; gap:7px; }}
.ef-exp-dot {{ width:9px; height:9px; border-radius:50%; flex-shrink:0; margin-top:3px; }}
.ef-exp-key {{ font-family:'Orbitron',sans-serif; font-size:8px; letter-spacing:.15em; color:#E8E8F0; text-transform:uppercase; }}
.ef-exp-def {{ font-size:9px; color:#C8C8D8; margin-top:2px; line-height:1.4; }}
</style>
<div class="ef-wrap">
  <div class="ef-header">⚡ energy analysis</div>

  <div class="ef-deploy">
    <div class="ef-row">
      <span class="ef-label">{t_deploy_label}</span>
      <span class="ef-val">{gasto:.2f}<span class="unit">MJ</span><span class="sub">{over_label}</span></span>
    </div>
    <div class="ef-track">
      <div class="ef-limit-tag">LIMIT {LIMIT:.1f} MJ</div>
      <div class="ef-limit"></div>
      <div class="ef-fill-deploy"></div>
      <div class="ef-fill-over"></div>
    </div>
    <div class="ef-note">{bar_note}</div>
  </div>

  <div class="ef-bottom">
    <div class="ef-card recovery">
      <div class="ef-clabel">{t_regen_label}</div>
      <div class="ef-cval" style="color:#00E5A0">{carga:.2f}<span class="unit">MJ</span></div>
      <div class="ef-regen-track"><div class="ef-regen-fill"></div></div>
      <div class="ef-regen-labels"><span>0 MJ</span><span style="color:#00E5A0">{carga:.2f} MJ</span><span>8 MJ</span></div>
      <div class="ef-cdesc">{t_regen_desc}</div>
      <div class="ef-pill ok"><span class="dot"></span>{t_regen_active}</div>
    </div>

    <div class="ef-card balance {balance_class}">
      <div class="ef-clabel">{t_balance_lbl}</div>
      <div class="ef-donut-wrap">
        <svg width="68" height="68" viewBox="0 0 72 72" style="flex-shrink:0">
          <circle cx="36" cy="36" r="28" fill="none" stroke="#1a1a28" stroke-width="10"/>
          <circle cx="36" cy="36" r="28" fill="none" stroke="#00E5A0" stroke-width="10"
            stroke-dasharray="0 175.9" transform="rotate(-90 36 36)"
            style="animation:efdonutg 1.5s 0.2s ease forwards"/>
          <circle cx="36" cy="36" r="28" fill="none" stroke="#FF1801" stroke-width="10"
            stroke-dasharray="{red_arc:.1f} {CIRCUM:.1f}" stroke-dashoffset="0" transform="rotate(-90 36 36)"
            style="animation:efdonutr 1.5s 0.2s ease forwards"/>
          <text x="36" y="40" text-anchor="middle" font-family="Orbitron" font-size="10" font-weight="900" fill="{balance_color}">{abs(balance):.1f}</text>
        </svg>
        <div>
          <div class="ef-cval" style="color:{balance_color}">{abs(balance):.2f}<span class="unit">MJ</span></div>
          <div class="ef-cdesc" style="margin-top:6px">{balance_desc}</div>
        </div>
      </div>
      <div class="ef-pill {pill_class}"><span class="dot"></span>{balance_pill}</div>
    </div>
  </div>

  <div class="ef-legend">
    <div class="ef-exp">
      <div class="ef-exp-dot" style="background:#FF1801"></div>
      <div><div class="ef-exp-key">Deployment</div><div class="ef-exp-def">{t_legend_dep}</div></div>
    </div>
    <div class="ef-exp">
      <div class="ef-exp-dot" style="background:#00E5A0"></div>
      <div><div class="ef-exp-key">Harvesting</div><div class="ef-exp-def">{t_legend_hrv}</div></div>
    </div>
    <div class="ef-exp">
      <div class="ef-exp-dot" style="background:#CC00FF"></div>
      <div><div class="ef-exp-key">Clipping</div><div class="ef-exp-def">{t_legend_clip}</div></div>
    </div>
  </div>

  <div style="margin-top:12px; padding:10px 14px; background:#0f0f1a; border:1px solid #222230;
              border-left:3px solid #FFD600; border-radius:4px; font-size:10px; color:#6060A0; line-height:1.6;">
    <span style="font-family:Orbitron,sans-serif; font-size:8px; letter-spacing:.2em; color:#FFD600; text-transform:uppercase;">
      ⚠ Metodología
    </span><br>
    {t_methodology_note}
  </div>
</div>
"""
        # ── Calcular punto de inicio y dirección de recorrido
        df_sorted = df_p.sort_values('date')
        start_row = df_sorted.iloc[0]
        # Dirección: vector entre primer y quinto punto para la flecha
        arrow_row = df_sorted.iloc[min(5, len(df_sorted)-1)]
        dx = arrow_row['x'] - start_row['x']
        dy = arrow_row['y'] - start_row['y']
        # Normalizar para tamaño consistente
        mag = max((dx**2 + dy**2)**0.5, 1)
        arrow_len = (df_p['x'].max() - df_p['x'].min()) * 0.06
        ax = start_row['x'] + dx / mag * arrow_len
        ay = start_row['y'] + dy / mag * arrow_len

        # Colores por estado
        clrs = {
            IA_DEPLOYMENT: '#FF2200',
            IA_CLIPPING:   '#DD00FF',
            IA_HARVESTING: '#00FF88',
            IA_NEUTRAL:    '#CCCCCC',
        }

        fig = go.Figure()

        # ── Trazado base del circuito (línea gris muy tenue)
        fig.add_trace(go.Scatter(
            x=df_sorted['x'], y=df_sorted['y'],
            mode='lines',
            line=dict(color='rgba(255,255,255,0.07)', width=8),
            hoverinfo='skip', showlegend=False, name='_track'
        ))

        # ── Puntos por estado con hover rico
        for key, color in clrs.items():
            stts = T[key]
            m = df_p['ia_status_key'] == key
            if not m.any():
                continue
            sub = df_p.loc[m]
            gear_col = sub['n_gear'].astype(str) if 'n_gear' in sub.columns else None
            drs_col  = sub['drs'].astype(str)    if 'drs'    in sub.columns else None

            hover_lines = (
                "<b style='color:" + color + "'>" + stts + "</b><br>" +
                "🕐 " + sub['date'].dt.strftime('%H:%M:%S.%f').str[:-3] + "<br>" +
                "⚡ " + T["rpm"] + ": " + sub['rpm'].astype(str) + "<br>" +
                "🏎 " + T["vel"] + ": <b>" + sub['speed'].astype(str) + " km/h</b>" +
                ("<br>⚙ Gear: " + gear_col if gear_col is not None else "") +
                ("<br>📡 DRS: "  + drs_col  if drs_col  is not None else "")
            )

            fig.add_trace(go.Scatter(
                x=sub['x'], y=sub['y'],
                mode='markers', name=stts,
                hovertext=hover_lines, hoverinfo='text',
                marker=dict(
                    color=color, size=7,
                    line=dict(width=0),
                    opacity=0.92,
                )
            ))

        # ── Bandera a cuadros: marcador especial en el punto de inicio
        fig.add_trace(go.Scatter(
            x=[start_row['x']], y=[start_row['y']],
            mode='markers+text',
            marker=dict(
                symbol='star', color='#FFD600', size=22,
                line=dict(color='#000000', width=1.5)
            ),
            text=['🏁'], textposition='top center',
            textfont=dict(size=20),
            hovertext='<b>🏁 Inicio de Vuelta</b><br>' + start_row['date'].strftime('%H:%M:%S.%f')[:-3],
            hoverinfo='text',
            name='🏁 Inicio', showlegend=True
        ))

        # ── Flecha de dirección de recorrido
        fig.add_annotation(
            x=ax, y=ay,
            ax=start_row['x'], ay=start_row['y'],
            xref='x', yref='y', axref='x', ayref='y',
            arrowhead=3, arrowsize=2.5, arrowwidth=2.5,
            arrowcolor='#FFD600',
            showarrow=True, text=''
        )

        # ── Mini-etiqueta "DIRECCIÓN" junto a la flecha
        fig.add_annotation(
            x=(start_row['x'] + ax) / 2,
            y=(start_row['y'] + ay) / 2,
            text='<b>DIR</b>',
            showarrow=False,
            font=dict(size=9, color='#FFD600', family='monospace'),
            bgcolor='rgba(0,0,0,0.6)',
            bordercolor='#FFD600', borderwidth=1,
            xref='x', yref='y'
        )

        fig.update_layout(
            plot_bgcolor='#05050D',
            paper_bgcolor='#05050D',
            height=780,
            margin=dict(l=10, r=10, t=50, b=10),
            font=dict(color='white', family='monospace'),
            xaxis=dict(visible=False, scaleanchor='y'),
            yaxis=dict(visible=False),
            legend=dict(
                orientation='h', y=1.04, x=0.5, xanchor='center',
                font=dict(size=13, color='white', family='monospace'),
                bgcolor='rgba(0,0,0,0)', borderwidth=0,
            ),
            hoverlabel=dict(
                bgcolor='#0A0A18',
                bordercolor='#333355',
                font=dict(size=12, color='white', family='monospace'),
            ),
        )
        st.plotly_chart(fig, use_container_width=True)

        # ─────────────────────────────────────────────────────────
        # GRÁFICOS DE TELEMETRÍA TEMPORAL
        # ─────────────────────────────────────────────────────────
        st.html("""<div style="display:flex;align-items:center;gap:12px;margin:32px 0 16px">
          <div style="font-family:'Titillium Web',sans-serif;font-size:11px;font-weight:700;
                      letter-spacing:.25em;color:#E8002D;text-transform:uppercase">
            Telemetría Temporal
          </div>
          <div style="flex:1;height:1px;background:#222230"></div>
        </div>""")

        # Preparar datos temporales
        df_sorted = df_p.sort_values('date').reset_index(drop=True)
        df_sorted['time_delta'] = (df_sorted['date'] - df_sorted['date'].iloc[0]).dt.total_seconds()

        # Layout común para todos los gráficos
        common_layout = dict(
            plot_bgcolor='#05050D',
            paper_bgcolor='#05050D',
            height=220,
            margin=dict(l=50, r=20, t=30, b=40),
            font=dict(color='white', family='monospace', size=11),
            xaxis=dict(
                title='Tiempo (s)',
                gridcolor='#1a1a28',
                showgrid=True,
                zeroline=False,
            ),
            hovermode='x unified',
        )

        # ── VELOCIDAD ────────────────────────────────────────────
        fig_speed = go.Figure()
        fig_speed.add_trace(go.Scatter(
            x=df_sorted['time_delta'],
            y=df_sorted['speed'],
            mode='lines',
            name='Velocidad',
            line=dict(color='#00D4FF', width=2),
            fill='tozeroy',
            fillcolor='rgba(0,212,255,0.1)',
        ))
        fig_speed.update_layout(
            **common_layout,
            title='Velocidad',
            yaxis=dict(title='km/h', gridcolor='#1a1a28'),
            showlegend=False,
        )

        # ── RPM ──────────────────────────────────────────────────
        fig_rpm = go.Figure()
        fig_rpm.add_trace(go.Scatter(
            x=df_sorted['time_delta'],
            y=df_sorted['rpm'],
            mode='lines',
            name='RPM',
            line=dict(color='#FF6B00', width=2),
            fill='tozeroy',
            fillcolor='rgba(255,107,0,0.1)',
        ))
        fig_rpm.update_layout(
            **common_layout,
            title='RPM',
            yaxis=dict(title='RPM', gridcolor='#1a1a28'),
            showlegend=False,
        )

        # ── THROTTLE / BRAKE ─────────────────────────────────────
        fig_pedals = go.Figure()
        # Throttle en verde
        fig_pedals.add_trace(go.Scatter(
            x=df_sorted['time_delta'],
            y=df_sorted['throttle'],
            mode='lines',
            name='Acelerador',
            line=dict(color='#00FF88', width=2),
            fill='tozeroy',
            fillcolor='rgba(0,255,136,0.15)',
        ))
        # Brake en rojo (negativo)
        fig_pedals.add_trace(go.Scatter(
            x=df_sorted['time_delta'],
            y=-df_sorted['brake'],
            mode='lines',
            name='Freno',
            line=dict(color='#FF2200', width=2),
            fill='tozeroy',
            fillcolor='rgba(255,34,0,0.15)',
        ))
        fig_pedals.update_layout(
            **common_layout,
            title='Acelerador / Freno',
            yaxis=dict(title='% (↑Accel ↓Freno)', gridcolor='#1a1a28'),
            legend=dict(orientation='h', y=1.1, x=0.5, xanchor='center', font=dict(size=10)),
        )

        # ── MARCHAS ──────────────────────────────────────────────
        if 'n_gear' in df_sorted.columns:
            fig_gear = go.Figure()
            fig_gear.add_trace(go.Scatter(
                x=df_sorted['time_delta'],
                y=df_sorted['n_gear'],
                mode='lines',
                name='Marcha',
                line=dict(color='#FFD600', width=3, shape='hv'),
            ))
            fig_gear.update_layout(
                **common_layout,
                title='Marcha',
                yaxis=dict(
                    title='Gear',
                    gridcolor='#1a1a28',
                    dtick=1,
                    range=[0, 9],
                ),
                showlegend=False,
            )

        # ── RENDERIZAR GRÁFICOS ──────────────────────────────────
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_speed, use_container_width=True)
            st.plotly_chart(fig_pedals, use_container_width=True)
        with col2:
            st.plotly_chart(fig_rpm, use_container_width=True)
            if 'n_gear' in df_sorted.columns:
                st.plotly_chart(fig_gear, use_container_width=True)


        # Widget de energía DEBAJO del mapa
        st.html(html_widget)

# ─────────────────────────────────────────────
#  VISTA 2: FAQ & METODOLOGÍA
# ─────────────────────────────────────────────
elif v_seleccionada == T["faq"]:
    st.title(T["faq_title"])
    st.markdown(T["faq_content"])
    st.info(T["faq_tip"])
