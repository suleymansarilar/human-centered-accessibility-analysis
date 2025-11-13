import pandas as pd
import plotly.express as px
import streamlit as st

from llm_helper import generate_llm_summary

DATA_PATH = "sample_accessibility.csv"


def get_rule_summary(row):
    lines = []

    score = row["accessibility_score"]
    if score >= 75:
        lines.append("Overall access is strong.")
    elif score >= 60:
        lines.append("Access is okay but can improve.")
    else:
        lines.append("Access is weak and needs attention.")

    distance = row["avg_path_distance"]
    if distance > 1000:
        lines.append("Walking distance is very high. Try shortcuts or shuttles.")
    elif distance > 800:
        lines.append("Walking distance is a bit high. Add more pedestrian options.")
    else:
        lines.append("Walking distance looks comfortable.")

    transit = row["transit_nodes"]
    if transit <= 4:
        lines.append("Transit stops are rare. Add more stops if you can.")
    elif transit <= 7:
        lines.append("Transit coverage is fine but could grow.")
    else:
        lines.append("Transit coverage is solid.")

    notes = row.get("notes", "")
    if isinstance(notes, str) and notes.strip():
        lines.append("Local context: " + notes.strip())

    return "\n".join(lines)


st.set_page_config(page_title="GenAI Accessibility Dashboard", layout="wide")
st.title("GenAI-Guided Accessibility Dashboard (Prototype)")
st.caption("Simple dashboard with an optional AI summary.")

with st.sidebar:
    st.header("About the prototype")
    st.write("1. Load sample accessibility metrics.")
    st.write("2. Pick a neighborhood.")
    st.write("3. Review the charts and the short narrative.")
    st.write("Turn on the LLM narrative if openai is installed and OPENAI_API_KEY is set.")
    use_llm = st.checkbox("Use LLM narrative (needs API key)")


data = pd.read_csv(DATA_PATH)
choice = st.selectbox("Choose a neighborhood", data["neighborhood"])
selected = data.set_index("neighborhood").loc[choice]

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Accessibility Overview")
    bar_chart = px.bar(
        data,
        x="neighborhood",
        y="accessibility_score",
        color="accessibility_score",
        color_continuous_scale="Viridis",
        labels={"accessibility_score": "Accessibility Score"},
        title="Accessibility Scores by Neighborhood",
    )
    st.plotly_chart(bar_chart, use_container_width=True)

    st.subheader("Walking Distance vs Transit Coverage")
    scatter_chart = px.scatter(
        data,
        x="avg_path_distance",
        y="transit_nodes",
        size="population",
        color="accessibility_score",
        hover_name="neighborhood",
        labels={
            "avg_path_distance": "Average Path Distance (m)",
            "transit_nodes": "Transit Nodes",
            "population": "Population",
        },
        title="Assessing walking burden and transit availability",
    )
    st.plotly_chart(scatter_chart, use_container_width=True)

with col2:
    st.subheader(f"Narrative: {choice}")

    context = {
        "neighborhood": choice,
        "accessibility_score": selected["accessibility_score"],
        "avg_path_distance": selected["avg_path_distance"],
        "transit_nodes": selected["transit_nodes"],
        "population": selected["population"],
        "notes": selected.get("notes", ""),
    }

    if use_llm:
        summary = generate_llm_summary(context)
    else:
        summary = get_rule_summary(selected)

    st.text(summary)

    st.markdown("---")
    st.markdown("### Additional metrics")
    st.metric("Accessibility Score", f"{selected['accessibility_score']}")
    st.metric("Avg. Path Distance", f"{selected['avg_path_distance']} m")
    st.metric("Transit Nodes", int(selected["transit_nodes"]))
    st.metric("Population", f"{selected['population']:,}")
