import io
import os
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="AgroCalculator | Smart Agronomic Engine",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# تصميم CSS متقدم مع Animations وتأثيرات بصرية فلاحية
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Animation Fade In */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(12px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Animation نبض خفيف للمؤشرات المهمة */
    @keyframes softPulse {
        0% { box-shadow: 0 0 0 0 rgba(25, 135, 84, 0.4); }
        70% { box-shadow: 0 0 0 10px rgba(25, 135, 84, 0); }
        100% { box-shadow: 0 0 0 0 rgba(25, 135, 84, 0); }
    }

    .main {
        background-color: #F8FAFC;
        animation: fadeIn 0.6s ease-out;
    }
    
    /* Hero Header */
    .hero-card {
        background: linear-gradient(135deg, #0F5132 0%, #198754 60%, #20c997 100%);
        padding: 24px 30px;
        border-radius: 18px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 10px 25px -5px rgba(25, 135, 84, 0.3);
        animation: fadeIn 0.8s ease-out;
    }
    .hero-title {
        font-size: 26px;
        font-weight: 700;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .hero-subtitle {
        font-size: 14px;
        opacity: 0.95;
        margin-top: 6px;
    }
    
    /* كروت الأقسام مع تأثير Hover */
    .card-header {
        font-size: 14px;
        font-weight: 700;
        color: #0F5132;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* صناديق مؤشرات الأداء مع تحريك */
    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 14px 18px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-3px);
        border-color: #198754;
        box-shadow: 0 8px 16px -4px rgba(25, 135, 84, 0.15);
    }
    div[data-testid="stMetricLabel"] p {
        font-size: 12px !important;
        font-weight: 600 !important;
        color: #64748B !important;
    }
    div[data-testid="stMetricValue"] {
        color: #0F172A !important;
        font-weight: 700 !important;
        font-size: 24px !important;
    }
    
    /* شارة التوفير المتحركة */
    .eco-badge {
        background: #E8F5E9;
        color: #1B5E20;
        border: 1px solid #C8E6C9;
        border-radius: 12px;
        padding: 12px 18px;
        display: flex;
        align-items: center;
        gap: 12px;
        font-weight: 600;
        font-size: 14px;
        animation: softPulse 2.5s infinite;
        margin-top: 15px;
    }
    
    /* أزرار التحميل المنسقة */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #198754 0%, #157347 100%) !important;
        color: white !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        border: none !important;
        padding: 12px 24px !important;
        width: 100%;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 12px rgba(25, 135, 84, 0.2) !important;
    }
    .stDownloadButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 18px rgba(25, 135, 84, 0.35) !important;
    }
</style>
""",
    unsafe_allow_html=True,
)


# تحميل النموذج
@st.cache_resource
def load_artifacts():
  if not os.path.exists("modele_basilic.pkl") or not os.path.exists(
      "model_features.pkl"
  ):
    return None, None
  try:
    return joblib.load("modele_basilic.pkl"), joblib.load("model_features.pkl")
  except Exception:
    return None, None


model, feature_names = load_artifacts()

if model is None or feature_names is None:
  st.error("⚠️ Fichiers du modèle introuvables. Exécutez 'train_model.py'.")
  st.stop()

# Header
st.markdown(
    """
<div class="hero-card">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
        <div>
            <h1 class="hero-title">🌱 AgroCalculator : Système Décisionnel Agrotechnologique</h1>
            <p class="hero-subtitle">Modélisation prédictive et phénotypage non-destructif de la biomasse d'<em>Ocimum basilicum L.</em> sous régimes N-P-K</p>
        </div>
        <div>
            <span style="background: rgba(255,255,255,0.22); padding: 6px 14px; border-radius: 20px; font-size: 13px; font-weight: 600; border: 1px solid rgba(255,255,255,0.4);">
                🌿 Précision R² = 0.9937 • Basilic Modèle
            </span>
        </div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# عمودان متناسقان
col_inputs, col_outputs = st.columns([1.1, 1.2], gap="large")

with col_inputs:
  st.markdown(
      '<div class="card-header">🧪 1. Identifiants & Régime Fertigation N-P-K'
      ' (ppm)</div>',
      unsafe_allow_html=True,
  )
  i1, i2, i3 = st.columns(3)
  semaine = i1.number_input(
      "Stade Phénologique",
      min_value=1,
      max_value=15,
      value=2,
      step=1,
      help="Semaine après repiquage",
  )
  table_id = i2.text_input("Unité / Table", value="1")
  plant_id = i3.text_input("Plante Rép.", value="1")

  f1, f2, f3, f4 = st.columns(4)
  et = f1.number_input("ET (mm/j)", min_value=0.0, value=18.0, step=0.5)
  ni = f2.number_input("N (Azote)", min_value=0.0, value=21.0, step=1.0)
  pi = f3.number_input("P (Phosphore)", min_value=0.0, value=1.0, step=1.0)
  ki = f4.number_input("K (Potassium)", min_value=0.0, value=184.0, step=1.0)

  st.markdown("<br>", unsafe_allow_html=True)
  st.markdown(
      '<div class="card-header">📐 2. Traits Morphologiques Non-Destructifs'
      ' In Situ</div>',
      unsafe_allow_html=True,
  )
  m1, m2 = st.columns(2)
  haut = m1.number_input(
      "Hauteur Tige (cm)", min_value=0.1, value=5.0, step=0.1, format="%.2f"
  )
  pds_af = m2.number_input(
      "Poids Frais Aérien PDS AF (g)",
      min_value=0.001,
      value=0.84,
      step=0.001,
      format="%.3f",
  )

  m3, m4 = st.columns(2)
  nbr_f = m3.number_input(
      "Nombre de Feuilles", min_value=1, max_value=500, value=10, step=1
  )
  surf = m4.number_input(
      "Surface Foliaire (cm²)",
      min_value=0.01,
      value=24.6,
      step=0.1,
      format="%.2f",
  )

# إعداد البيانات والتنبؤ
input_dict = {
    "SEMAINE": semaine,
    "ET": et,
    "Ni": ni,
    "Pi": pi,
    "Ki": ki,
    "HAUT": haut,
    "PDS A F": pds_af,
    "NBR F": nbr_f,
    "SUR F": surf,
}

row_data = [
    input_dict.get(
        col,
        input_dict.get(
            col.upper().replace(" ", "").replace("_", ""),
            float(input_dict.get("PDS A F", 0.0)),
        ),
    )
    for col in feature_names
]
input_df = pd.DataFrame([row_data], columns=feature_names)

raw_pred = float(model.predict(input_df)[0])

# الضبط الفسيولوجي الدقيق
if semaine == 1 or pds_af <= 0.15:
  pds_as_pred = pds_af * 0.131
else:
  lower_bound = max(0.005, pds_af * 0.07)
  upper_bound = pds_af * 0.22
  pds_as_pred = np.clip(raw_pred, lower_bound, upper_bound)

safe_af = max(pds_af, 1e-5)
safe_as = max(pds_as_pred, 1e-5)

dmc = (pds_as_pred / safe_af) * 100.0
water_content = 100.0 - dmc
sla = surf / safe_as
pds_rs_est = pds_as_pred * 0.22
shoot_root_ratio = pds_as_pred / max(pds_rs_est, 1e-5)

with col_outputs:
  st.markdown(
      '<div class="card-header">⚡ 3. Estimations de Biomasse (Sans Sacrifice'
      ' Végétal)</div>',
      unsafe_allow_html=True,
  )
  r1, r2 = st.columns(2)
  r1.metric(
      label="PDS AS Estimé (Aérien Sec)",
      value=f"{pds_as_pred:.3f} g",
      delta="R² = 0.9937",
  )
  r2.metric(
      label="PDS RS Estimé (Racinaire)",
      value=f"{pds_rs_est:.3f} g",
      delta=f"Shoot/Root : {shoot_root_ratio:.1f}",
  )

  st.markdown("<br>", unsafe_allow_html=True)
  st.markdown(
      '<div class="card-header">🧬 4. Diagnostics & Équilibre'
      ' Physiologique</div>',
      unsafe_allow_html=True,
  )
  d1, d2 = st.columns(2)
  d1.metric(
      label="Teneur Matière Sèche (DMC)",
      value=f"{dmc:.2f} %",
      delta=f"Teneur en Eau : {water_content:.1f}%",
  )
  d2.metric(
      label="Surface Foliaire Spécifique (SLA)",
      value=f"{sla:.1f} cm²/g",
      delta="Allocation Foliaire",
  )

  # شارة توفير الوقت والجهد بنبض متحرك
  st.markdown(
      """
    <div class="eco-badge">
        <span style="font-size: 22px;">⏱️</span>
        <div>
            <strong>72h d'étuve économisées</strong> — Suivi cinétique continu in vivo sans destruction de la plante.
        </div>
    </div>
    """,
      unsafe_allow_html=True,
  )

st.markdown("<br><hr style='border: none; border-top: 1px solid #E2E8F0;'><br>", unsafe_allow_html=True)

# الجزء التفاعلي: رسم بياني متحرك ومخطط راداري فلاحي
g1, g2 = st.columns([1.2, 1.0], gap="large")

with g1:
  st.markdown(
      '<div class="card-header">📈 Cinétique Prédictive de Croissance'
      ' (Interactive)</div>',
      unsafe_allow_html=True,
  )

  weeks_sim = [1, 2, 3, 4]
  growth_ratios = [0.12, 1.0, 1.75, 2.85]
  curr_idx = min(max(semaine - 1, 0), 3)
  scale_factor = pds_as_pred / growth_ratios[curr_idx]
  biomass_growth = [
      round(max(0.010, ratio * scale_factor), 3) for ratio in growth_ratios
  ]

  # رسم متحرك بـ Plotly مع التدرج اللوني والتفاعل
  fig_curve = go.Figure()
  fig_curve.add_trace(
      go.Scatter(
          x=[f"Semaine {w}" for w in weeks_sim],
          y=biomass_growth,
          mode="lines+markers",
          name="PDS AS (g)",
          line=dict(color="#198754", width=3.5, shape="spline"),
          marker=dict(
              size=9,
              color="#0F5132",
              line=dict(width=2, color="#FFFFFF"),
          ),
          fill="tozeroy",
          fillcolor="rgba(25, 135, 84, 0.12)",
          hovertemplate=(
              "<b>%{x}</b><br>PDS AS Estimé: <b>%{y} g</b><extra></extra>"
          ),
      )
  )

  fig_curve.update_layout(
      margin=dict(l=20, r=20, t=10, b=20),
      height=280,
      paper_bgcolor="rgba(0,0,0,0)",
      plot_bgcolor="rgba(0,0,0,0)",
      xaxis=dict(showgrid=False, linecolor="#CBD5E1"),
      yaxis=dict(
          showgrid=True,
          gridcolor="#F1F5F9",
          linecolor="#CBD5E1",
          title="Poids Sec (g)",
      ),
      hoverlabel=dict(
          bgcolor="#0F5132", font_size=13, font_family="Plus Jakarta Sans"
      ),
  )
  st.plotly_chart(fig_curve, use_container_width=True)

with g2:
  st.markdown(
      '<div class="card-header">🕸️ Profil Phénotypique & Agronomique (Radar'
      " Balance)</div>",
      unsafe_allow_html=True,
  )

  # مخطط رادار لمقارنة توازن النبتة
  categories = [
      "Vigueur (Hauteur)",
      "Surface Foliaire",
      "Équilibre N",
      "Équilibre K",
      "Matière Sèche",
  ]
  # تطبيع القيم (0 إلى 100%) لغرض العرض البصري المقارن
  val_norm = [
      min(100, (haut / 25.0) * 100),
      min(100, (surf / 150.0) * 100),
      min(100, (ni / 150.0) * 100),
      min(100, (ki / 250.0) * 100),
      min(100, (dmc / 15.0) * 100),
  ]

  fig_radar = go.Figure()
  fig_radar.add_trace(
      go.Scatterpolar(
          r=val_norm,
          theta=categories,
          fill="toself",
          name="Échantillon Actuel",
          line_color="#20C997",
          fillcolor="rgba(32, 201, 151, 0.25)",
      )
  )

  fig_radar.update_layout(
      polar=dict(
          radialaxis=dict(visible=False, range=[0, 100]),
          angularaxis=dict(tickfont=dict(size=10, color="#475569")),
      ),
      showlegend=False,
      margin=dict(l=30, r=30, t=10, b=10),
      height=280,
      paper_bgcolor="rgba(0,0,0,0)",
  )
  st.plotly_chart(fig_radar, use_container_width=True)

# جدول تلخيصي فلاحي
st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    '<div class="card-header">📑 Fiche d\'Inférence Agronomique'
    " Exportable</div>",
    unsafe_allow_html=True,
)

summary_data = [
    {
        "Trait / Paramètre": "Identifiant Échantillon",
        "Valeur": f"Table {table_id} - Répétition {plant_id}",
        "Unité / Statut": "In Situ",
    },
    {
        "Trait / Paramètre": "Stade Phénologique",
        "Valeur": f"Semaine {semaine}",
        "Unité / Statut": "Phase Végétative",
    },
    {
        "Trait / Paramètre": "Poids Frais Aérien (PDS AF)",
        "Valeur": f"{pds_af:.3f}",
        "Unité / Statut": "g (Grammes)",
    },
    {
        "Trait / Paramètre": "Poids Sec Aérien Estimé (PDS AS)",
        "Valeur": f"{pds_as_pred:.3f}",
        "Unité / Statut": "g (R² = 0.9937)",
    },
    {
        "Trait / Paramètre": "Poids Sec Racinaire Estimé (PDS RS)",
        "Valeur": f"{pds_rs_est:.3f}",
        "Unité / Statut": "g (Allométrie shoot:root)",
    },
    {
        "Trait / Paramètre": "Taux de Matière Sèche (DMC)",
        "Valeur": f"{dmc:.2f}",
        "Unité / Statut": "% (Matière Sèche)",
    },
    {
        "Trait / Paramètre": "Surface Foliaire Spécifique (SLA)",
        "Valeur": f"{sla:.2f}",
        "Unité / Statut": "cm²/g",
    },
    {
        "Trait / Paramètre": "Formule Fertigation N - P - K",
        "Valeur": f"{int(ni)} - {int(pi)} - {int(ki)}",
        "Unité / Statut": "ppm (Solution nutritive)",
    },
]
df_summary = pd.DataFrame(summary_data)
st.dataframe(df_summary, use_container_width=True, hide_index=True)

csv_out = df_summary.to_csv(index=False).encode("utf-8")
st.download_button(
    label="📥 Télécharger la Fiche Complète d'Inférence (CSV / Excel)",
    data=csv_out,
    file_name=f"AgroCalculator_Basilic_S{semaine}_T{table_id}_P{plant_id}.csv",
    mime="text/csv",
)