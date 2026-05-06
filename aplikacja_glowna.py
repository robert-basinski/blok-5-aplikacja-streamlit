from pathlib import Path

import streamlit as st


st.set_page_config(
    page_title="System relokacji rowerów",
    layout="wide",
)


APP_DIR = Path(__file__).resolve().parent


st.markdown(
    """
    <div style="margin-bottom:1.4rem;">
        <div style="font-size:2.4rem; font-weight:800; line-height:1.15; color:#111827; margin-bottom:0.45rem;">
            System relokacji rowerów
        </div>
        <div style="font-size:1rem; color:#6b7280; line-height:1.45;">
            Wspólna aplikacja: panel operacyjny dyspozytora oraz techniczny widok dzień–stacja.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


main_tab_operational, main_tab_technical = st.tabs(
    [
        "Panel operacyjny",
        "Panel techniczny dzień–stacja",
    ]
)


with main_tab_operational:
    st.subheader("Panel operacyjny")

    st.info(
        "Tutaj zostanie przeniesiona część operacyjna: plan dnia, lista działań, mapa, karta kierowcy i feedback."
    )


with main_tab_technical:
    st.subheader("Panel techniczny dzień–stacja")

    st.info(
        "Tutaj zostanie przeniesiona część techniczna: scoring, dane modelu i diagnostyka dzień–stacja."
    )