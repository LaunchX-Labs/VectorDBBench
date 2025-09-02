import streamlit as st


def NavToRunTest(container):
    container.subheader("Run your test")
    container.write("You can set the configs and run your own test.")
    navClick = container.button("Run Your Test &nbsp;&nbsp;>")
    if navClick:
        st.switch_page("pages/run_test.py")


def NavToQuriesPerDollar(container):
    container.subheader("Compare qps with price.")
    navClick = container.button("QP$ (Quries per Dollar) &nbsp;&nbsp;>")
    if navClick:
        st.switch_page("pages/quries_per_dollar.py")


def NavToResults(container, key="nav-to-results"):
    navClick = container.button("< &nbsp;&nbsp;Back to Results", key=key)
    if navClick:
        switch_page("results")


def NavToPages(st):
    options = [
        {"name": "Run Test", "link": "run_test"},
        {"name": "Results", "link": "results"},
        {"name": "Quries Per Dollar", "link": "quries_per_dollar"},
        {"name": "Concurrent", "link": "concurrent"},
        {"name": "Label Filter", "link": "label_filter"},
        {"name": "Streaming", "link": "streaming"},
        {"name": "Tables", "link": "tables"},
        {"name": "Custom Dataset", "link": "custom"},
    ]

    html = ""
    for i, option in enumerate(options):
        html += f'<a href="/{option["link"]}" target="_self" style="text-decoration: none; padding: 0.1px 0.2px;">{option["name"]}</a>'
        if i < len(options) - 1:
            html += '<span style="color: #888; margin: 0 5px;">|</span>'
    st.markdown(html, unsafe_allow_html=True)
