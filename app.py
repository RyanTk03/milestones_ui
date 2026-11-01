import streamlit as st
import streamlit.components.v1 as components
from tabs import jalon1, jalon2, jalon3, jalon4

st.set_page_config(
    page_title="CardioPatch",
    page_icon=":material/monitor_heart:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Icons (inline SVG, Lucide-style) ----------
def svg(path: str, size: int = 20, stroke: str = "currentColor", sw: float = 1.8) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 24 24" fill="none" stroke="{stroke}" stroke-width="{sw}" '
        f'stroke-linecap="round" stroke-linejoin="round">{path}</svg>'
    )

HEART_PULSE = (
    '<path d="M19 14c1.49-1.46 3-3.21 3-5.5A5.5 5.5 0 0 0 16.5 3c-1.76 0-3 .5-4.5 2-1.5-1.5-2.74-2-4.5-2A5.5 5.5 0 0 0 2 8.5c0 2.3 1.5 4.05 3 5.5l7 7Z"/>'
    '<path d="M3.22 12H9.5l.5-1 2 4.5 2-7 1.5 3.5h5.27"/>'
)
GRAD_CAP = '<path d="M22 10v6M2 10l10-5 10 5-10 5z"/><path d="M6 12v5c3 3 9 3 12 0v-5"/>'

# ---------- Scroll keeper: stay exactly where the user changed a widget ----------
def keep_scroll_position():
    components.html(
        """
        <script>
        (function () {
          const P = window.parent, D = P.document;
          if (P.__cpScrollKeeper) return;          // install once per browser session
          P.__cpScrollKeeper = true;

          const WIDGETS = [
            'stSelectbox','stRadio','stSlider','stCheckbox','stToggle','stNumberInput',
            'stTextInput','stTextArea','stMultiSelect','stButton','stSegmentedControl',
            'stPills','stSelectSlider','stDateInput','stTimeInput','stFileUploader'
          ].map(t => '[data-testid="' + t + '"]').join(',');

          const getMain = () =>
            D.querySelector('[data-testid="stMain"]') ||
            D.querySelector('section.main') || D.querySelector('.main');

          let timer = null, inner = null;

          function release() {
            clearInterval(timer); timer = null;
            if (inner) { inner.style.minHeight = ''; inner = null; }
          }

          function lock(e) {
            const t = e.target;
            if (!t || !t.closest) return;
            // widgets themselves, or the option list of a dropdown (rendered in a portal)
            if (!t.closest(WIDGETS) && !t.closest('[data-baseweb="popover"]')) return;

            const main = getMain();
            if (!main) return;
            release();

            const saved = main.scrollTop;
            inner = main.querySelector('[data-testid="stMainBlockContainer"]') || main.firstElementChild;
            if (inner) inner.style.minHeight = inner.scrollHeight + 'px';  // page can't shrink

            let ticks = 0;
            timer = setInterval(function () {
              if (Math.abs(main.scrollTop - saved) > 2) main.scrollTop = saved;
              if (++ticks > 160) release();        // safety stop after ~8 s
            }, 50);
          }

          // user takes over -> stop interfering
          ['wheel', 'touchmove', 'keydown'].forEach(function (ev) {
            D.addEventListener(ev, function (e) {
              if (ev === 'keydown' && !['ArrowUp','ArrowDown','PageUp','PageDown','Home','End',' '].includes(e.key)) return;
              release();
            }, { passive: true, capture: true });
          });

          D.addEventListener('pointerdown', lock, true);
          D.addEventListener('change', lock, true);
        })();
        </script>
        """,
        height=0,
    )

# ---------- CSS ----------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --accent: #ff3d68;
        --accent-2: #8b5cf6;
        --surface: rgba(255,255,255,0.035);
        --border: rgba(255,255,255,0.08);
        --muted: rgba(255,255,255,0.55);
    }
    html, body, [class*="css"], .stMarkdown, button { font-family: 'Inter', sans-serif !important; }
    code, pre { font-family: 'JetBrains Mono', monospace !important; }

    #MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }
    .block-container { padding-top: 2.2rem !important; max-width: 1280px; }

    /* hide the 0-height helper iframe container */
    div[data-testid="stElementContainer"]:has(iframe[height="0"]) {
        position: absolute; height: 0; margin: 0; overflow: hidden;
    }

    /* ---------- Hero ---------- */
    .hero {
        position: relative; overflow: hidden;
        border-radius: 24px; padding: 2.6rem 2.8rem;
        background:
            radial-gradient(900px 300px at 85% -20%, rgba(139,92,246,0.35), transparent 60%),
            radial-gradient(700px 300px at 0% 120%, rgba(255,61,104,0.30), transparent 60%),
            linear-gradient(160deg, #1a1a26 0%, #12121a 100%);
        border: 1px solid var(--border);
        box-shadow: 0 20px 60px -20px rgba(0,0,0,0.6), inset 0 1px 0 rgba(255,255,255,0.06);
        margin-bottom: 1.8rem;
    }
    .hero svg.ecg { position: absolute; left: 0; right: 0; bottom: 18px; width: 100%; height: 90px; opacity: .55; }
    .hero-top { display: flex; align-items: center; gap: 1rem; position: relative; z-index: 1; }
    .logo {
        width: 54px; height: 54px; border-radius: 16px; display: grid; place-items: center;
        background: linear-gradient(135deg, var(--accent), var(--accent-2)); color: #fff;
        box-shadow: 0 8px 24px -6px rgba(255,61,104,0.6);
    }
    .hero h1 { margin: 0; padding: 0; font-size: 2.3rem; font-weight: 800; letter-spacing: -0.03em; color: #fff; line-height: 1.1; }
    .hero h1 span { background: linear-gradient(90deg, var(--accent), #c084fc); -webkit-background-clip: text; background-clip: text; color: transparent; }
    .hero p { position: relative; z-index: 1; margin: 1rem 0 1.3rem 0; color: var(--muted); font-size: 1.05rem; max-width: 640px; }
    .chips { position: relative; z-index: 1; display: flex; gap: .5rem; flex-wrap: wrap; padding-bottom: 3.2rem; }
    .chip {
        padding: .28rem .8rem; border-radius: 999px; font-size: .78rem; font-weight: 500;
        color: rgba(255,255,255,.85); background: rgba(255,255,255,.06); border: 1px solid var(--border);
        letter-spacing: .02em;
    }

    /* ---------- Tabs (segmented control) ---------- */
    .stTabs [role="tablist"] {
        gap: .35rem !important; padding: .35rem !important;
        background: var(--surface) !important; border: 1px solid var(--border) !important;
        border-radius: 16px !important; width: fit-content; max-width: 100%;
    }
    .stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none !important; }
    .stTabs button[role="tab"] {
        height: 42px; padding: 0 1.1rem !important; border-radius: 12px !important;
        background: transparent !important; transition: all .18s ease;
    }
    .stTabs button[role="tab"] p { font-size: .9rem; font-weight: 500; color: var(--muted) !important; margin: 0; }
    .stTabs button[role="tab"]:hover { background: rgba(255,255,255,.05) !important; }
    .stTabs button[role="tab"]:hover p { color: #fff !important; }
    .stTabs button[role="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, var(--accent), #d6285a 55%, var(--accent-2)) !important;
        box-shadow: 0 8px 22px -8px rgba(255,61,104,.7);
    }
    .stTabs button[role="tab"][aria-selected="true"] p { color: #fff !important; font-weight: 600; }

    /* ---------- Section header inside each tab ---------- */
    .sec { display: flex; align-items: center; gap: 1rem; margin: 1.6rem 0 1.4rem 0; }
    .sec .num {
        font-family: 'JetBrains Mono', monospace; font-size: .8rem; color: var(--accent);
        padding: .35rem .6rem; border-radius: 10px; background: rgba(255,61,104,.1); border: 1px solid rgba(255,61,104,.25);
    }
    .sec h3 { margin: 0; padding: 0; font-size: 1.35rem; font-weight: 700; letter-spacing: -0.02em; }
    .sec p  { margin: .15rem 0 0 0; color: var(--muted); font-size: .92rem; }

    /* ---------- Widgets ---------- */
    div[data-testid="stMetric"] {
        background: var(--surface); border: 1px solid var(--border);
        border-radius: 16px; padding: 1.1rem 1.2rem;
    }
    div[data-testid="stMetricLabel"] p { color: var(--muted); font-size: .8rem; text-transform: uppercase; letter-spacing: .06em; }
    div[data-testid="stMetricValue"] { font-weight: 700; letter-spacing: -0.02em; }

    div[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 16px !important; border-color: var(--border) !important; background: var(--surface); }
    div[data-testid="stExpander"] { border-radius: 14px; border: 1px solid var(--border); background: var(--surface); }
    div[data-testid="stAlert"] { border-radius: 14px; border: 1px solid var(--border); }
    div[data-testid="stDataFrame"] { border-radius: 14px; overflow: hidden; border: 1px solid var(--border); }

    .stButton > button, .stDownloadButton > button {
        border-radius: 12px; border: 1px solid var(--border); background: var(--surface);
        color: #fff; font-weight: 500; padding: .55rem 1.2rem; transition: all .18s ease;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        border-color: var(--accent); background: rgba(255,61,104,.1); color: #fff; transform: translateY(-1px);
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, var(--accent), var(--accent-2)); border: none;
        box-shadow: 0 8px 22px -8px rgba(255,61,104,.7);
    }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] { background: #0e0e15; border-right: 1px solid var(--border); }
    section[data-testid="stSidebar"] > div { padding-top: 1.4rem; }
    .sb-brand { display: flex; align-items: center; gap: .7rem; }
    .sb-brand .logo { width: 38px; height: 38px; border-radius: 11px; }
    .sb-brand b { font-size: 1.1rem; letter-spacing: -0.02em; display: block; line-height: 1.1; }
    .sb-brand small { color: var(--muted); font-size: .75rem; }
    .sb-label { margin: 1.8rem 0 .8rem; font-size: .68rem; font-weight: 600; letter-spacing: .14em; text-transform: uppercase; color: var(--muted); }
    .sb-course { display: flex; gap: .7rem; align-items: flex-start; color: rgba(255,255,255,.8); font-size: .88rem; line-height: 1.35; }
    .sb-course svg { color: var(--accent); flex-shrink: 0; margin-top: 2px; }
    .step { display: flex; align-items: center; gap: .8rem; padding: .5rem 0; position: relative; font-size: .88rem; color: rgba(255,255,255,.8); }
    .step .dot {
        width: 26px; height: 26px; border-radius: 50%; display: grid; place-items: center; flex-shrink: 0;
        font-family: 'JetBrains Mono', monospace; font-size: .72rem; color: var(--accent);
        background: rgba(255,61,104,.1); border: 1px solid rgba(255,61,104,.3);
    }
    .step:not(:last-child)::after {
        content: ""; position: absolute; left: 12.5px; top: 34px; width: 1px; height: 18px; background: var(--border);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Scroll fix (all jalons) ----------
keep_scroll_position()

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown(
        f"""
        <div class="sb-brand">
            <div class="logo">{svg(HEART_PULSE, 20)}</div>
            <div><b>CardioPatch</b><small>Télésurveillance cardiaque</small></div>
        </div>

        <div class="sb-label">Cours</div>
        <div class="sb-course">{svg(GRAD_CAP, 18)}
            <div><b>M2 SDIA</b><br>Advanced Artificial Intelligence<br>Étude de cas 3</div>
        </div>

        <div class="sb-label">Parcours</div>
        <div class="step"><div class="dot">01</div>Descripteurs → décision</div>
        <div class="step"><div class="dot">02</div>Signal brut</div>
        <div class="step"><div class="dot">03</div>Holter → patch</div>
        <div class="step"><div class="dot">04</div>Séquences</div>
        """,
        unsafe_allow_html=True,
    )

# ---------- Hero ----------
ECG = (
    "M0 45 H140 L158 45 L170 20 L184 72 L198 4 L214 62 L226 45 H420 L438 45 L450 20 L464 72 L478 4 "
    "L494 62 L506 45 H700 L718 45 L730 20 L744 72 L758 4 L774 62 L786 45 H1000 L1018 45 L1030 20 "
    "L1044 72 L1058 4 L1074 62 L1086 45 H1300"
)
st.markdown(
    f"""
    <div class="hero">
        <svg class="ecg" viewBox="0 0 1300 90" preserveAspectRatio="none" fill="none">
            <defs><linearGradient id="g" x1="0" x2="1">
                <stop offset="0" stop-color="#ff3d68" stop-opacity="0"/>
                <stop offset=".5" stop-color="#ff3d68"/>
                <stop offset="1" stop-color="#8b5cf6" stop-opacity="0"/>
            </linearGradient></defs>
            <path d="{ECG}" stroke="url(#g)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
        <div class="hero-top">
            <div class="logo">{svg(HEART_PULSE, 28)}</div>
            <h1>Cardio<span>Patch</span></h1>
        </div>
        <p>Télésurveillance cardiaque par deep learning : du descripteur clinique au modèle de séquences.</p>
        <div class="chips">
            <span class="chip">M2 SDIA</span>
            <span class="chip">Advanced AI</span>
            <span class="chip">Étude de cas 3</span>
            <span class="chip">4 jalons</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

def section(n: int, title: str, desc: str):
    st.markdown(
        f'<div class="sec"><div class="num">0{n} / 04</div>'
        f'<div><h3>{title}</h3><p>{desc}</p></div></div>',
        unsafe_allow_html=True,
    )

# ---------- Tabs ----------
t1, t2, t3, t4 = st.tabs([
    ":material/tune:  Descripteurs",
    ":material/show_chart:  Signal brut",
    ":material/monitor_heart:  Holter → patch",
    ":material/timeline:  Séquences",
])

with t1:
    section(1, "Descripteurs → décision", "Des descripteurs cliniques à la décision.")
    jalon1.render()
with t2:
    section(2, "Signal brut", "Apprentissage directement sur le signal ECG brut.")
    jalon2.render()
with t3:
    section(3, "Holter → patch", "Transfert et adaptation du Holter vers le patch.")
    jalon3.render()
with t4:
    section(4, "Séquences", "Modélisation de séquences temporelles.")
    jalon4.render()