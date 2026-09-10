import streamlit as st
from dotenv import load_dotenv
import os

from SRC.agents.agents import (
    build_search_agent,
    build_read_agent,
    writer_chain,
    critic_chain,
)

load_dotenv()


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Multi-Agent Research System",
    page_icon="🔬",
    layout="wide",
)


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown(
    """
    <style>
        .main-title {
            font-size: 42px;
            font-weight: 700;
            margin-bottom: 0;
        }

        .subtitle {
            font-size: 18px;
            color: #888;
            margin-top: 0;
        }

        .step-title {
            font-size: 22px;
            font-weight: 600;
        }

        .status-box {
            padding: 12px;
            border-radius: 8px;
            margin: 10px 0;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    '<p class="main-title">🔬 Multi-Agent Research System</p>',
    unsafe_allow_html=True,
)

st.markdown(
    '<p class="subtitle">'
    "Search the web → Read sources → Write a report → Critique the report"
    "</p>",
    unsafe_allow_html=True,
)

st.divider()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.header("⚙️ Configuration")

    st.markdown("### Model")

    st.info("Groq • GPT-OSS-20B")

    st.markdown("### Pipeline")

    st.markdown(
        """
        1. 🔎 Search Agent
        2. 📖 Reader Agent
        3. ✍️ Writer Chain
        4. 🧐 Critic Chain
        """
    )

    st.divider()

    if os.getenv("GROQ_API_KEY"):
        st.success("Groq API key loaded")
    else:
        st.error("GROQ_API_KEY not found")


# --------------------------------------------------
# USER INPUT
# --------------------------------------------------

st.subheader("Research Topic")

topic = st.text_area(
    "What would you like to research?",
    placeholder=(
        "Example: The impact of AI on the job market in 2026"
    ),
    height=100,
)


run_button = st.button(
    "🚀 Run Research Pipeline",
    type="primary",
    use_container_width=True,
)


# --------------------------------------------------
# PIPELINE
# --------------------------------------------------

if run_button:

    if not topic.strip():
        st.warning("Please enter a research topic.")
        st.stop()

    # ----------------------------------------------
    # INITIAL STATE
    # ----------------------------------------------

    state = {}

    # ----------------------------------------------
    # STEP 1 — SEARCH
    # ----------------------------------------------

    with st.status(
        "🔎 Search Agent is researching...",
        expanded=True,
    ) as status:

        st.write("Searching for recent and reliable information...")

        try:
            search_agent = build_search_agent()

            search_result = search_agent.invoke(
                {
                    "messages": [
                        (
                            "user",
                            f"Find recent, reliable and detailed information "
                            f"about: {topic}",
                        )
                    ]
                }
            )

            state["search_results"] = (
                search_result["messages"][-1].content
            )

            st.success("Search completed.")

        except Exception as e:

            status.update(
                label="❌ Search Agent failed",
                state="error",
            )

            st.error(str(e))
            st.stop()

        status.update(
            label="✅ Search Agent completed",
            state="complete",
        )


    # ----------------------------------------------
    # DISPLAY SEARCH RESULTS
    # ----------------------------------------------

    with st.expander("🔎 View Search Results"):

        st.markdown(state["search_results"])


    # ----------------------------------------------
    # STEP 2 — READER
    # ----------------------------------------------

    with st.status(
        "📖 Reader Agent is scraping sources...",
        expanded=True,
    ) as status:

        try:

            reader_agent = build_read_agent()

            reader_result = reader_agent.invoke(
                {
                    "messages": [
                        (
                            "user",
                            f"""
Based on the following search results about
"{topic}", select the most relevant URL
and scrape it for deeper content.

Search Results:

{state["search_results"][:5000]}
""",
                        )
                    ]
                }
            )

            state["scraped_content"] = (
                reader_result["messages"][-1].content
            )

            if not state["scraped_content"].strip():

                st.warning(
                    "The Reader Agent returned empty content."
                )

            else:

                st.success("Source scraping completed.")

        except Exception as e:

            status.update(
                label="❌ Reader Agent failed",
                state="error",
            )

            st.error(str(e))

            # Don't necessarily stop the entire pipeline.
            state["scraped_content"] = ""

        status.update(
            label="✅ Reader Agent completed",
            state="complete",
        )


    # ----------------------------------------------
    # DISPLAY SCRAPED CONTENT
    # ----------------------------------------------

    with st.expander("📖 View Scraped Content"):

        if state["scraped_content"]:

            st.markdown(state["scraped_content"])

        else:

            st.info(
                "No scraped content was returned by the Reader Agent."
            )


    # ----------------------------------------------
    # COMBINE RESEARCH
    # ----------------------------------------------

    research_combined = (
        f"SEARCHED RESULTS:\n\n"
        f"{state['search_results']}\n\n"
        f"DETAILED SCRAPED CONTENT:\n\n"
        f"{state['scraped_content']}"
    )


    # ----------------------------------------------
    # STEP 3 — WRITER
    # ----------------------------------------------

    with st.status(
        "✍️ Writer is drafting the report...",
        expanded=True,
    ) as status:

        try:

            state["report"] = writer_chain.invoke(
                {
                    "topic": topic,
                    "research": research_combined,
                }
            )

            st.success("Report generated.")

        except Exception as e:

            status.update(
                label="❌ Writer failed",
                state="error",
            )

            st.error(str(e))
            st.stop()

        status.update(
            label="✅ Writer completed",
            state="complete",
        )


    # ----------------------------------------------
    # DISPLAY REPORT
    # ----------------------------------------------

    st.divider()

    st.subheader("📄 Research Report")

    st.markdown(state["report"])


    # ----------------------------------------------
    # STEP 4 — CRITIC
    # ----------------------------------------------

    with st.status(
        "🧐 Critic is reviewing the report...",
        expanded=True,
    ) as status:

        try:

            state["feedback"] = critic_chain.invoke(
                {
                    "topic": topic,
                    "research": research_combined,
                    "report": state["report"],
                }
            )

            st.success("Critic review completed.")

        except Exception as e:

            status.update(
                label="❌ Critic failed",
                state="error",
            )

            st.error(str(e))

            state["feedback"] = ""

        status.update(
            label="✅ Critic completed",
            state="complete",
        )


    # ----------------------------------------------
    # CRITIC OUTPUT
    # ----------------------------------------------

    st.divider()

    st.subheader("🧐 Critic Feedback")

    if state["feedback"]:

        st.markdown(state["feedback"])

    else:

        st.warning("No critic feedback was returned.")


    # ----------------------------------------------
    # DOWNLOAD REPORT
    # ----------------------------------------------

    st.divider()

    st.subheader("📥 Export")

    st.download_button(
        label="⬇️ Download Report",
        data=state["report"],
        file_name="research_report.md",
        mime="text/markdown",
        use_container_width=True,
    )