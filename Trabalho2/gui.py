import streamlit as st
import pandas as pd
from model_interface import ModelInterface


st.set_page_config(page_title="Predição Imobiliária", layout="wide")

st.markdown("""
<style>
/* Deixa tudo mais bonito e consistente */
.reportview-container { background: #F8F9FA; }
.sidebar .sidebar-content { background: #FFFFFF; }



/* Títulos */
h1 {
    text-align: center;
    font-weight: 800;
}
</style>
""", unsafe_allow_html=True)


st.markdown("<h1>🏡 Predição de Valores Imobiliários</h1>", unsafe_allow_html=True)
st.write("Insira abaixo os dados do imóvel para obter estimativas de preço usando três modelos estatísticos.")

@st.cache_resource
def load_api():
    return ModelInterface()

api = load_api()



st.header("📋 Preencha os dados do imóvel")

col1, col2 = st.columns(2)

with col1:
    property_type = st.selectbox("Tipo de imóvel", ["Casa", "Apartamento", "Cobertura"])
    property_subtype = st.text_input("Subtipo de imóvel", "Padrão")
    bedrooms = st.number_input("Quartos", value=2)
    bathrooms = st.number_input("Banheiros", value=1)
    parking_spaces = st.number_input("Vagas de estacionamento", value=2)
    size_category = st.selectbox("Categoria de Tamanho", ["small", "medium", "large"])

with col2:
    area_util = st.number_input("Área útil (m²)", value=100.0)
    area_total = st.number_input("Área total (m²)", value=242.0)
    condominium_fee = st.number_input("Taxa de condomínio (R$)", value=0.0)

st.write("### 🏢 Amenidades")

ac1, ac2, ac3 = st.columns(3)

with ac1:
    has_pool = st.checkbox("Piscina")
    has_bbq = st.checkbox("Área de churrasco")
    has_playground = st.checkbox("Playground")
    has_sauna = st.checkbox("Sauna")

with ac2:
    has_party_room = st.checkbox("Salão de festas")
    has_sports_court = st.checkbox("Quadra esportiva")
    has_24h_security = st.checkbox("Segurança 24h")

with ac3:
    has_laundry = st.checkbox("Lavanderia")
    has_closet = st.checkbox("Closet")
    has_office = st.checkbox("Escritório")
    has_pantry = st.checkbox("Despensa")





predict_button = st.button("🔍 **Calcular Predições**", use_container_width=True)



if predict_button:
    record = pd.DataFrame([{
        "property_code": 0,
        "property_type": property_type,
        "property_subtype": property_subtype,
        "sale_price": 0,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "parking_spaces": parking_spaces,
        "area_util": area_util,
        "area_total": area_total,
        "condominium_fee": condominium_fee,
        "has_pool": has_pool,
        "has_bbq": has_bbq,
        "has_playground": has_playground,
        "has_sauna": has_sauna,
        "has_party_room": has_party_room,
        "has_sports_court": has_sports_court,
        "has_24h_security": has_24h_security,
        "has_laundry": has_laundry,
        "has_closet": has_closet,
        "has_office": has_office,
        "has_pantry": has_pantry,
        "size_category": size_category,
        "amenity_score": 0
    }])

    result = api.get_predictions(record)

    st.markdown("<h2>📊 Resultados da Predição</h2>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)


    with c1:
        st.subheader("📘 Linear Model (OLS)")
        lp = float(result["ols"]["mean"]) * 100000
        lp_up = float(result["ols"]["mean_ci_upper"]) * 100000
        lp_low = float(result["ols"]["mean_ci_lower"]) * 100000
        st.write(f"**Preço estimado:** R${lp:,.2f}")
        st.write(f"Alta: R${lp_up:,.2f}")
        st.write(f"Baixa: R${lp_low:,.2f}")

    with c2:
        st.subheader("📙 GLM Gamma")
        g = float(result["glm"]["mean"]) * 10000
        g_up = float(result["glm"]["mean_ci_upper"]) * 10000
        g_low = float(result["glm"]["mean_ci_lower"]) * 10000
        st.write(f"**Preço estimado:** R${g:,.2f}")
        st.write(f"Alta: R${g_up:,.2f}")
        st.write(f"Baixa: R${g_low:,.2f}")

    with c3:
        st.subheader("📗 XGBoost")
        xp = float(result["xgb"][0]) * 100000
        st.write(f"**Preço estimado:** R${xp:,.2f}")

