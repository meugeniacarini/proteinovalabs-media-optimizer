import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import qmc

st.set_page_config(page_title="ProteinovaLabs | Media Optimizer", page_icon="🧬", layout="wide")

VARS = [
    "glucose_gL", "pyruvate_gL", "aa_essential_gL", "aa_nonessential_gL",
    "FGF2_ugL", "TGFb_ugL", "IGF1_ugL", "plant_hydrolysate_gL",
    "minerals_A_gL", "minerals_B_gL", "vitamins_A_gL", "vitamins_B_gL"
]
LABELS = {
    "glucose_gL":"Glucosa (g/L)", "pyruvate_gL":"Piruvato (g/L)",
    "aa_essential_gL":"Aminoácidos esenciales (g/L)",
    "aa_nonessential_gL":"Aminoácidos no esenciales (g/L)",
    "FGF2_ugL":"FGF2 (µg/L)", "TGFb_ugL":"TGF-β (µg/L)",
    "IGF1_ugL":"IGF-1 (µg/L)", "plant_hydrolysate_gL":"Hidrolizado vegetal (g/L)",
    "minerals_A_gL":"Minerales A (g/L)", "minerals_B_gL":"Minerales B (g/L)",
    "vitamins_A_gL":"Vitaminas A (g/L)", "vitamins_B_gL":"Vitaminas B (g/L)"
}
BOUNDS = np.array([
    [0.5,6.0],[0.0,0.5],[0.1,3.0],[0.1,3.0],[0.0,0.05],[0.0,0.05],
    [0.0,0.1],[0.0,10.0],[0.0,10.0],[0.0,10.0],[0.0,1.0],[0.0,1.0]
], dtype=float)
COSTS = np.array([1.0,3.0,10.0,8.0,500.0,400.0,300.0,5.0,2.0,2.0,15.0,12.0])
PH_TARGET, PH_RANGE = 7.2, (6.9,7.5)
OSM_TARGET, OSM_RANGE = 300.0, (260.0,340.0)
OSM_COEFF = np.array([1.0,1.2,2.0,1.7,0.1,0.1,0.1,1.0,0.9,1.1,0.5,0.5])

def cost(x):
    return float(np.dot(x, COSTS))

def proxy_osm(x):
    return 250.0 + np.dot(x, OSM_COEFF)

def proxy_ph(x):
    return 7.2 + 0.02*(x[9]-0.5*x[10]) - 0.015*x[8] + 0.005*x[7]

def penalty_ranges(x, k_ph=50.0, k_osm=0.05):
    ph, osm = proxy_ph(x), proxy_osm(x)
    pph = k_ph * (PH_RANGE[0]-ph)**2 if ph < PH_RANGE[0] else (
        k_ph * (ph-PH_RANGE[1])**2 if ph > PH_RANGE[1] else 0.0)
    posm = k_osm * (OSM_RANGE[0]-osm)**2 if osm < OSM_RANGE[0] else (
        k_osm * (osm-OSM_RANGE[1])**2 if osm > OSM_RANGE[1] else 0.0)
    return float(pph + posm)

def simulated_growth(x):
    x = np.asarray(x, dtype=float)
    base = (
        0.25*np.tanh(0.4*(x[0]-2.5)) +
        0.18*np.tanh(2.0*(x[1]-0.15)) +
        0.30*np.tanh(0.7*(x[2]+x[3]-1.5)) +
        0.20*np.tanh(8.0*(x[4]+0.8*x[6]-0.02)) +
        0.10*np.tanh(6.0*(0.03-x[5])) +
        0.18*np.tanh(0.15*x[7]) +
        0.15*np.tanh(0.08*(x[8]+x[9])) +
        0.12*np.tanh(0.3*(x[10]+x[11]))
    )
    gph = np.exp(-((proxy_ph(x)-PH_TARGET)**2)/0.05)
    gosm = np.exp(-((proxy_osm(x)-OSM_TARGET)**2)/800)
    g = np.clip((1.0+base)*0.8*gph*gosm, 0.0, 1.2)
    return float(max(0.0, g-0.001*penalty_ranges(x,100.0,0.12)))

def sample(n, seed=42):
    sampler = qmc.Sobol(d=len(VARS), scramble=True, seed=seed)
    m = int(np.ceil(np.log2(n)))
    return qmc.scale(sampler.random_base2(m)[:n], BOUNDS[:,0], BOUNDS[:,1])

def fit_ensemble(X, y, n_models=5, seed=42):
    rng = np.random.RandomState(seed)
    models = []
    for _ in range(n_models):
        rs = int(rng.randint(0, 1_000_000_000))
        model = make_pipeline(
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(128,64), activation="relu",
                         solver="adam", alpha=1e-4, batch_size=32,
                         learning_rate_init=1e-3, max_iter=800,
                         early_stopping=True, n_iter_no_change=30,
                         random_state=rs)
        )
        model.fit(X, y)
        models.append(model)
    return models

def predict(models, X):
    arr = np.column_stack([m.predict(X) for m in models])
    return arr.mean(axis=1), arr.std(axis=1)

def run_optimizer(n0, pop_size, generations, alpha, beta, lam, seed=42):
    np.random.seed(seed)
    X0 = sample(n0, seed)
    y0 = np.array([simulated_growth(x) for x in X0])
    Xtr, Xte, ytr, yte = train_test_split(X0, y0, test_size=.2, random_state=seed)
    models = fit_ensemble(Xtr, ytr)
    mu_test, _ = predict(models, Xte)
    mape = float(np.mean(np.abs((mu_test-yte)/np.maximum(1e-6,yte))))

    def fitness(x):
        x = np.clip(np.asarray(x), BOUNDS[:,0], BOUNDS[:,1])
        mu, sig = predict(models, x.reshape(1,-1))
        c, pen = cost(x), penalty_ranges(x)
        score = alpha*float(mu[0]) - beta*c - lam*float(sig[0]) - pen
        return score, float(mu[0]), c, pen, float(sig[0])

    pop = sample(pop_size, seed+1)
    hist = []
    for _ in range(generations):
        vals = [fitness(x) for x in pop]
        scores = np.array([v[0] for v in vals])
        hist.append((pop.copy(), vals))
        new = []
        while len(new) < pop_size:
            ids = np.random.choice(len(pop), 3, replace=False)
            p1 = pop[ids[np.argmax(scores[ids])]].copy()
            ids = np.random.choice(len(pop), 3, replace=False)
            p2 = pop[ids[np.argmax(scores[ids])]].copy()
            child = p1.copy()
            if np.random.rand() < .9:
                lo = np.minimum(p1,p2)-.4*np.abs(p1-p2)
                hi = np.maximum(p1,p2)+.4*np.abs(p1-p2)
                child = np.random.uniform(lo,hi)
            if np.random.rand() < .3:
                child += np.random.randn(len(VARS))*.08*(BOUNDS[:,1]-BOUNDS[:,0])
            new.append(np.clip(child,BOUNDS[:,0],BOUNDS[:,1]))
        pop = np.array(new)
    final_pop, final_vals = hist[-1]
    result = pd.DataFrame(final_pop, columns=VARS)
    result["growth_pred"] = [v[1] for v in final_vals]
    result["cost_index"] = [v[2] for v in final_vals]
    result["penalty"] = [v[3] for v in final_vals]
    result["uncertainty"] = [v[4] for v in final_vals]
    result["fitness"] = [v[0] for v in final_vals]
    result = result.sort_values("fitness", ascending=False).drop_duplicates(subset=VARS).head(10).reset_index(drop=True)
    return result, mape, hist

st.markdown("""
<style>
.stApp {background:#F5F7FA;}
h1,h2,h3 {color:#0B1730;}
div[data-testid="stMetric"] {background:white;padding:14px;border-radius:12px;border:1px solid #E3E8EF;}
.hero {background:#0B1730;color:white;padding:24px 28px;border-radius:16px;margin-bottom:18px;}
.hero h1 {color:white;margin:0;font-size:2rem;}
.hero p {color:#D7E2EF;margin:8px 0 0 0;}
.smallnote {color:#526174;font-size:.88rem;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="hero"><h1>ProteinovaLabs | AI Media Optimizer</h1><p>Exploración computacional de formulaciones de medios de cultivo sin suero · GA + modelo sustituto</p></div>', unsafe_allow_html=True)
st.warning("DEMO CON DATOS SIMULADOS. El crecimiento se calcula con una función matemática de demostración, no con resultados experimentales ni datos biológicos validados. Los costos, pH y osmolalidad son índices/proxies ilustrativos.")
st.markdown("Esta interfaz reproduce el flujo del notebook: muestreo inicial Sobol → ensemble de redes MLP → optimización con algoritmo genético → ranking de candidatos.")

with st.sidebar:
    st.header("Configuración")
    n0 = st.select_slider("Diseño inicial (D0)", options=[100,200,300,500], value=200)
    pop_size = st.select_slider("Población del algoritmo genético", options=[40,60,80,120], value=80)
    generations = st.slider("Generaciones", 5, 40, 20, 5)
    st.caption("Más muestras y generaciones aumentan el tiempo de cálculo.")
    st.divider()
    st.subheader("Pesos de optimización")
    alpha = st.slider("ALPHA · crecimiento", 0.1, 2.0, 1.0, 0.1)
    beta = st.slider("BETA · costo", 0.0, 2.0, 0.8, 0.1)
    lam = st.slider("LAMBDA · incertidumbre", 0.0, 1.0, 0.3, 0.05)
    run = st.button("Ejecutar optimización", type="primary", use_container_width=True)

tab1, tab2, tab3 = st.tabs(["1 · Formulación", "2 · Optimización", "3 · Metodología"])

with tab1:
    st.subheader("Variables de formulación")
    st.write("Los rangos iniciales y los coeficientes provienen del notebook suministrado. Modificá las concentraciones para definir una formulación de referencia; la búsqueda genética explora el espacio completo de rangos.")
    cols = st.columns(3)
    values = []
    for i, var in enumerate(VARS):
        lo, hi = BOUNDS[i]
        with cols[i%3]:
            val = st.slider(LABELS[var], float(lo), float(hi), float((lo+hi)/2), step=float((hi-lo)/100))
        values.append(val)
    x = np.array(values)
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Costo estimado (índice)", f"{cost(x):,.1f}")
    m2.metric("pH proxy", f"{proxy_ph(x):.2f}")
    m3.metric("Osmolalidad proxy", f"{proxy_osm(x):.1f}")
    m4.metric("Penalización", f"{penalty_ranges(x):.3f}")
    st.caption("La formulación seleccionada es una referencia visual. No representa una receta recomendada ni una formulación validada.")

with tab2:
    if run:
        with st.spinner("Entrenando el ensemble y ejecutando el algoritmo genético…"):
            results, mape, history = run_optimizer(n0,pop_size,generations,alpha,beta,lam)
        st.session_state["results"] = results
        st.session_state["mape"] = mape
        st.session_state["history"] = history
    if "results" in st.session_state:
        results = st.session_state["results"]
        mape = st.session_state["mape"]
        best = results.iloc[0]
        a,b,c,d = st.columns(4)
        a.metric("Candidatos", len(results))
        b.metric("Crecimiento predicho (μ)", f"{best.growth_pred:.3f}")
        c.metric("Costo estimado (índice)", f"{best.cost_index:.2f}")
        d.metric("MAPE aprox. test", f"{mape*100:.1f}%")
        st.subheader("Relación costo–crecimiento")
        fig, ax = plt.subplots(figsize=(8,4))
        ax.scatter(results["cost_index"],results["growth_pred"],alpha=.85)
        ax.set_xlabel("Costo estimado (índice del modelo)")
        ax.set_ylabel("Crecimiento predicho μ(x)")
        ax.set_title("Candidatos destacados · última generación")
        ax.grid(alpha=.2)
        st.pyplot(fig, use_container_width=True)
        st.subheader("Top candidatos")
        display = results.copy()
        display.insert(0,"Ranking",np.arange(1,len(display)+1))
        st.dataframe(display, use_container_width=True, hide_index=True)
        st.download_button("Descargar Top 10 (CSV)", results.to_csv(index=False).encode("utf-8"), "proteinovalabs_top_candidates_demo.csv", "text/csv")
    else:
        st.info("Ajustá los parámetros en el panel lateral y seleccioná «Ejecutar optimización» para generar los candidatos.")
with tab3:
    st.subheader("Cómo funciona el prototipo")
    steps = [
        ("01 · Diseño inicial","Muestreo Sobol de formulaciones dentro de los 12 rangos definidos."),
        ("02 · Datos de entrenamiento","La función simulada genera el dataset D0 para esta demostración."),
        ("03 · Modelo sustituto","Ensemble de cinco redes MLP; se reporta incertidumbre entre predicciones."),
        ("04 · Algoritmo genético","Selección por torneo, cruce y mutación para optimizar crecimiento, costo e incertidumbre."),
        ("05 · Resultados","Ranking de candidatos y visualización de la relación costo–crecimiento.")
    ]
    for title,desc in steps:
        st.markdown(f"**{title}**  \n{desc}")
    st.divider()
    st.markdown("**Para una versión de investigación:** sustituir la función `true_growth` por datos experimentales, recalibrar rangos y costos con datos reales, validar el modelo sustituto y añadir restricciones de solubilidad e incompatibilidad.")
    st.markdown('<p class="smallnote">Prototipo conceptual de ProteinovaLabs. No usar para preparar medios, tomar decisiones experimentales ni realizar afirmaciones de desempeño sin validación de laboratorio.</p>', unsafe_allow_html=True)

st.markdown("---")
st.caption("ProteinovaLabs · AI-assisted formulation exploration · MVP conceptual")
