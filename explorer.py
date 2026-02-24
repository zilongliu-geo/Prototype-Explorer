import os
import re
import csv
import pandas as pd
import numpy as np
import streamlit as st
import networkx as nx
import altair as alt
import requests
from itertools import combinations
import folium
from streamlit_folium import st_folium
from streamlit_agraph import agraph, Node, Edge, Config

BASE_DIR = 'responses'
COLORBLIND_PALETTE = [
    '#4477AA', '#EE6677', '#228833', '#CCBB44', '#66CCEE',
    '#AA3377', '#BBBBBB', '#EE99AA', '#77AADD', '#99DDFF',
    '#44BB99', '#FFAABB', '#DDDD77', '#88CCAA', '#AADDCC',
    '#FFCC88', '#CC99BB', '#DDAA77', '#88AACC', '#CCAA88'
]

def filename_to_temp(fname):
    try:
        return float(fname.replace('.csv', '').replace('_', '.'))
    except ValueError:
        return None

def get_valid_categories(path):
    return sorted([
        d for d in os.listdir(path)
        if os.path.isdir(os.path.join(path, d)) and re.match(r'^\w+$', d)
    ])

@st.cache_data
def load_data(model, category, fname):
    return pd.read_csv(
        os.path.join(BASE_DIR, model, category, fname),
        engine='python', quoting=csv.QUOTE_NONE, on_bad_lines='skip'
    )

def geocode_place(place_name, username):
    url = "http://api.geonames.org/searchJSON"
    params = {
        "q": place_name,
        "maxRows": 1,
        "username": username
    }
    try:
        response = requests.get(url, params=params, timeout=5)
        data = response.json()
        if data.get("geonames"):
            result = data["geonames"][0]
            return {
                "name": result["name"],
                "latitude": float(result["lat"]),
                "longitude": float(result["lng"]),
                "fcodeName": result["fcodeName"]
            }
    except Exception as e:
        st.warning(f"Geocoding error for '{place_name}': {e}")
    return None

def geocode_all_places(names, username):
    geocoded_results = []
    for name in names:
        result = geocode_place(name, username)
        if result:
            result['input_name'] = name  # store original input exemplar
            geocoded_results.append(result)
        else:
            st.warning(f"Failed to geocode: {name}")
    return geocoded_results

PERSPECTIVE_HELP = (
    "Each perspective offers a unique lens on the geographic-category norms produced by the selected large language models."
)
MODEL_HELP = (
    "The selected models are autoregressive large language models that support chat completions.\n"
    "They emcompass GPT, DeepSeek, and Llama models.\n"
    "There are 11 models in total.\n"
)
CATEGORY_HELP = (
    "The majority of the selected geographic categories come from [GeoNames](https://www.geonames.org).\n"
    "GeoNames category codes are refined to maximally informative categories from a linguistic perspective. \n"
    "In addition, the superordinate category *place* and basic-level categories introduced by [Lloyd et al. (1996)](https://www.tandfonline.com/doi/abs/10.1111/j.0033-0124.1996.00181.x) are added.\n"
    "There are 297 categories in total.\n"
)
TEMPERATURE_HELP = "The sampling temperature controls the uncertainty of model outputs."
DIVERSITY_HELP = (
    "The order controls sensitivity of geographic diversity to rare and common exemplars:\n"
    "- 0 favors rare exemplars,\n"
    "- 1 balances rare and common exemplars,\n"
    "- 2 favors common exemplars.\n\n"
    "The measures of geographic diversity follow Liu et al. (2025):\n"
    "Liu, Z., Janowicz, K., Majic, I., Shi, M., Fortacz, A., Karimi, M., Mai, G., & Currier, K. (2025).\n"
    "\"Operationalizing geographic diversity for the evaluation of AI-generated content\".\n"
    "Transactions in GIS, 29(3), e70057. https://doi.org/10.1111/tgis.70057"
)

USERNAME_HELP = (
    "The GeoNames username is required for geocoding."
)
st.title("Prototype Explorer")

perspective = st.sidebar.radio(
    "Select Perspective",
    ["Main", "Sampling Temperature", "Geographic Category", "Large Language Model"],
    help=PERSPECTIVE_HELP
)

models = sorted([
    m for m in os.listdir(BASE_DIR)
    if os.path.isdir(os.path.join(BASE_DIR, m))
])

if perspective == "Main":
    st.markdown("""
    **Prototype Explorer** is an observatory for examining category norms produced by large language models.
    Use the sidebar to navigate between analytical perspectives.

    The **Main** perspective showcases *raw* LLM-generated exemplars, produced under specific experimental settings of category production.
    """)

    model = st.sidebar.selectbox("Select Model", models, help=MODEL_HELP)
    categories = get_valid_categories(os.path.join(BASE_DIR, model))
    category = st.sidebar.selectbox("Select Category", categories, help=CATEGORY_HELP)

    folder = os.path.join(BASE_DIR, model, category)
    temp_map = {}
    for f in os.listdir(folder):
        t = filename_to_temp(f)
        if t is not None:
            temp_map[t] = f
    temps = sorted(temp_map.keys())
    selected_temp = st.sidebar.select_slider("Select Temperature", options=temps, help=TEMPERATURE_HELP)

    df = load_data(model, category, temp_map[selected_temp])
    st.dataframe(df)

elif perspective == "Sampling Temperature":
    st.markdown("""
    The **Sampling Temperature** perspective showcases exemplar frequency and geographic diversity across temperatures.
    """)

    model = st.sidebar.selectbox("Select Model", models, help=MODEL_HELP)
    categories = get_valid_categories(os.path.join(BASE_DIR, model))
    category = st.sidebar.selectbox("Select Category", categories, help=CATEGORY_HELP)
    order = st.sidebar.slider(
        "Select Order of Geographic Diversity",
        min_value=0,
        max_value=2,
        value=1,
        step=1,
        help=DIVERSITY_HELP
    )

    folder = os.path.join(BASE_DIR, model, category)
    temp_map = {filename_to_temp(f): f for f in os.listdir(folder) if filename_to_temp(f) is not None}
    temps = sorted(temp_map.keys())

    with st.spinner('Loading data and rendering chart...'):
        data = []
        for temp in temps:
            df_temp = load_data(model, category, temp_map[temp])
            if 'name' in df_temp.columns:
                for name in df_temp['name'].dropna().astype(str).str.strip():
                    data.append({'temperature': temp, 'name': name})

        if not data:
            st.info("No data available for this selection.")
        else:
            df_temp_all = pd.DataFrame(data)
            freq_df = df_temp_all.pivot_table(
                index='temperature', columns='name', aggfunc='size', fill_value=0
            ).sort_index()

            absolute_zero = None
            for temp in sorted(freq_df.index):
                row = freq_df.loc[temp]
                unique_exemplars = (row > 0).sum()
                if unique_exemplars == 1:
                    absolute_zero = temp
                else:
                    break

            if absolute_zero is not None:
                st.markdown(f"Is there an *absolute zero*? From temperature 0 to **{absolute_zero}**, the model always outputs one single exemplar.")

            freq_df_reset = freq_df.reset_index().melt(id_vars='temperature', var_name='name', value_name='count')
            name_counts = freq_df_reset.groupby('name')['count'].sum()
            frequent_names = name_counts[name_counts > 1].index
            filtered_df = freq_df_reset[freq_df_reset['name'].isin(frequent_names)].copy()

            diversity_records = []
            for temp, row in freq_df.iterrows():
                counts = row.values
                total = counts.sum()
                if total == 0:
                    diversity = np.nan
                else:
                    probs = counts / total
                    if order == 0:
                        diversity = np.sum(probs > 0)
                    elif order == 1:
                        entropy = -np.sum(probs[probs > 0] * np.log(probs[probs > 0]))
                        diversity = np.exp(entropy)
                    elif order == 2:
                        diversity = 1 / np.sum(probs**2)
                    else:
                        diversity = np.nan
                diversity_records.append({"temperature": temp, "diversity": diversity})

            df_diversity = pd.DataFrame(diversity_records)

            base = alt.Chart(filtered_df).encode(x=alt.X('temperature:O', title='Temperature'))

            bar = base.mark_bar().encode(
                y=alt.Y('count:Q', stack='zero', title='Frequency', scale=alt.Scale(domain=[0, 50])),
                color=alt.Color('name:N', title='Exemplar'),
                tooltip=['temperature', 'name', 'count']
            )

            line = alt.Chart(df_diversity).mark_line(point=True, strokeWidth=2, color='black').encode(
                x=alt.X('temperature:O'),
                y=alt.Y('diversity:Q', title=f'Geographic Diversity (Order {order})'),
                tooltip=['temperature', 'diversity']
            ).interactive()

            combined = alt.layer(bar, line).resolve_scale(y='independent').properties(width=750, height=450)
            st.altair_chart(combined, use_container_width=False, key=f"{model}_{category}_sampling_temp_chart")

elif perspective == "Geographic Category":
    st.markdown("""
    The **Geographic Category** perspective showcases semantic clusters of categories. Categories are connected if they share an identical set of LLM-generated exemplars.
    """)

    selected_model = st.sidebar.selectbox("Select Model", models, help=MODEL_HELP)
    model_path = os.path.join(BASE_DIR, selected_model)

    temperatures = set()
    for category_name in os.listdir(model_path):
        category_path = os.path.join(model_path, category_name)
        if os.path.isdir(category_path):
            for filename in os.listdir(category_path):
                temp = filename_to_temp(filename)
                if temp is not None:
                    temperatures.add(temp)
    temperatures = sorted(temperatures)
    selected_temp = st.sidebar.select_slider("Select Temperature", options=temperatures, help=TEMPERATURE_HELP)

    category_to_names = {}
    for category_name in os.listdir(model_path):
        category_path = os.path.join(model_path, category_name)
        if os.path.isdir(category_path):
            for filename in os.listdir(category_path):
                if filename_to_temp(filename) == selected_temp:
                    file_path = os.path.join(category_path, filename)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            lines = [line.strip() for line in f.readlines() if line.strip()]
                        lines = lines[1:]  # skip header
                        names = set(lines)
                        if names:
                            category_to_names[category_name] = names
                    except Exception as e:
                        st.error(f"Error reading {file_path}: {e}")

    if not category_to_names:
        st.warning("No data for selected model and temperature.")
        st.stop()

    G = nx.Graph()
    G.add_nodes_from(category_to_names.keys())
    for a, b in combinations(category_to_names.keys(), 2):
        set_a, set_b = category_to_names[a], category_to_names[b]
        jaccard = len(set_a & set_b) / len(set_a | set_b)
        if jaccard == 1.0:
            G.add_edge(a, b)

    G.remove_nodes_from(list(nx.isolates(G)))
    if G.number_of_nodes() == 0:
        st.warning("No semantic links found.")
        st.stop()

    clusters = list(nx.connected_components(G))

    clusters = [c for c in clusters if len(c) >= 3]

    nodes_to_keep = set().union(*clusters)
    G = G.subgraph(nodes_to_keep).copy()

    st.write(f"The graph contains **{len(clusters)}** cluster(s).")

    if G.number_of_nodes() == 0:
        st.warning("No clusters found.")
        st.stop()

    edges = []
    for u, v in G.edges():
        shared = category_to_names[u] & category_to_names[v]
        label = f"{len(shared)}"
        title = ", ".join(sorted(shared)[:5])
        edges.append(Edge(source=u, target=v, label=label, title=title))

    nodes = [Node(id=node, label=node) for node in G.nodes()]

    config = Config(
        directed=False,
        nodeHighlightBehavior=True,
        node={'labelProperty': 'label', 'size': 500, 'renderLabel': True},
        link={'highlightColor': '#f00'},
        height=600,
        width=800,
        bgcolor="#FFFFFF",
    )

    agraph(nodes=nodes, edges=edges, config=config)


elif perspective == "Large Language Model":
    st.markdown("""
    The **Large Language Model** perspective shows exemplars shared across the majority of models.
    """)

    # Collect all available temperatures across models and categories
    all_temps = set()
    for model in models:
        model_path = os.path.join(BASE_DIR, model)
        for cat in get_valid_categories(model_path):
            cat_path = os.path.join(model_path, cat)
            for fname in os.listdir(cat_path):
                t = filename_to_temp(fname)
                if t is not None:
                    all_temps.add(t)
    all_temps = sorted(all_temps)

    # Sidebar inputs
    selected_temp = st.sidebar.select_slider("Select Temperature", options=all_temps, help=TEMPERATURE_HELP)
    
    # Find categories shared across all models
    category_sets = [set(get_valid_categories(os.path.join(BASE_DIR, m))) for m in models]
    shared_categories = sorted(set.intersection(*category_sets))
    
    selected_category = st.sidebar.selectbox("Select Category", shared_categories, help=CATEGORY_HELP)
    
    username = st.sidebar.text_input("Enter GeoNames username", help = USERNAME_HELP)
    
    # Require all selections
    if selected_temp is None:
        st.warning("Please select a temperature to proceed.")
        st.stop()
    if not selected_category:
        st.warning("Please select a category to proceed.")
        st.stop()
    if not username.strip():
        st.warning("Please enter a GeoNames username to proceed.")
        st.stop()

    # Load exemplars per model for selected category and temperature
    model_exemplars = {}
    for model in models:
        folder = os.path.join(BASE_DIR, model, selected_category)
        temp_map = {filename_to_temp(f): f for f in os.listdir(folder) if filename_to_temp(f) is not None}
        fname = temp_map.get(selected_temp)
        if fname:
            df = load_data(model, selected_category, fname)
            if 'name' in df.columns:
                names = df['name'].dropna().astype(str).str.strip()
                model_exemplars[model] = set(names)

    if not model_exemplars:
        st.warning("No data found for this temperature and category across models.")
        st.stop()

    # Count how many models share each exemplar
    all_names = set().union(*model_exemplars.values())
    name_counts = {name: sum(name in names for names in model_exemplars.values()) for name in all_names}

    # Filter for exemplars shared by more than 5 models
    exemplars_to_geocode = {name for name, count in name_counts.items() if count > 5}

    if not exemplars_to_geocode:
        st.warning("No exemplars are found.")
        st.stop()

    # Geocode filtered exemplars and keep track of input names
    with st.spinner("Geocoding exemplars..."):
        geo_results = []
        for name in exemplars_to_geocode:
            result = geocode_place(name, username)
            if result:
                result['input_name'] = name
                geo_results.append(result)
            else:
                st.warning(f"Failed to geocode: {name}")

    if not geo_results:
        st.warning("No exemplars are successfully geocoded.")
        st.stop()

    geo_df = pd.DataFrame(geo_results)
    geo_df['shared_by_models'] = geo_df['input_name'].map(name_counts).fillna(0).astype(int)

    if {'latitude', 'longitude'}.issubset(geo_df.columns):
        center_lat = geo_df['latitude'].mean()
        center_lon = geo_df['longitude'].mean()
        m = folium.Map(location=[center_lat, center_lon], zoom_start=3)

        for _, row in geo_df.iterrows():
            popup_html = (
                f"<b>Exemplar:</b> {row['input_name']}<br>"
                f"<b>Geocoded Name:</b> {row['name']}<br>"
                f"<b>Geocoded Category:</b> {row['fcodeName']}<br>"
                f"<b>Latitude:</b> {row['latitude']:.4f}<br>"
                f"<b>Longitude:</b> {row['longitude']:.4f}<br>"
                f"<b>Number of Shared Models:</b> {row['shared_by_models']}"
            )
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=6,
                color='#4477AA',
                fill=True,
                fill_color='#4477AA',
                fill_opacity=0.7,
                popup=folium.Popup(popup_html, max_width=300)
            ).add_to(m)

        st.write(f"Number of exemplars: {len(geo_df)}")
        st_folium(m, width=800, height=600)

    else:
        st.warning("Missing coordinate data, unable to display map.")
