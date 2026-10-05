import json
import os

import requests
import streamlit as st


API_URL = os.getenv("RFP_API_URL", "http://localhost:8000")


st.set_page_config(
    page_title="RFP Intelligence Platform",
    page_icon="📄",
    layout="wide",
)

st.title("RFP Intelligence Platform")
st.caption("Search, ask questions, and extract cited bid information.")

search_tab, ask_tab, index_tab = st.tabs(
    ["Search", "Ask a question / Extract", "Index documents"]
)


with search_tab:
    st.subheader("Search bid documents")

    query = st.text_input("Search query")
    bid_filter = st.text_input("Bid ID filter (optional)", key="search_bid")
    doc_type = st.selectbox(
        "Document type",
        ["Any", "rfp", "addendum", "specs", "affidavit", "bid_page"],
    )
    top_k = st.slider("Number of results", min_value=1, max_value=10, value=5)

    if st.button("Search", key="search_button"):
        if not query.strip():
            st.warning("Enter a search query.")
        else:
            params = {"q": query, "top_k": top_k}
            if bid_filter.strip():
                params["bid_id"] = bid_filter.strip()
            if doc_type != "Any":
                params["doc_type"] = doc_type

            try:
                response = requests.get(
                    f"{API_URL}/search",
                    params=params,
                    timeout=120,
                )
                response.raise_for_status()
                results = response.json()["results"]

                if not results:
                    st.info("No matching passages found.")

                for index, result in enumerate(results, start=1):
                    citation = result.get("citation", {})
                    page = citation.get("page")
                    page_text = f", page {page}" if page else ""

                    with st.expander(
                        f"{index}. {citation.get('file', 'Unknown source')}{page_text}"
                    ):
                        st.write(result.get("text", ""))
                        st.caption(
                            f"Bid: {result.get('bid_id')} · "
                            f"Type: {result.get('doc_type')} · "
                            f"Score: {result.get('rerank_score', result.get('score'))}"
                        )

            except requests.RequestException as e:
                st.error(f"Search request failed: {e}")


with ask_tab:
    st.subheader("Ask about a bid or extract its fields")

    mode = st.radio(
        "Task",
        ["Answer a question", "Extract bid JSON"],
        horizontal=True,
    )
    ask_bid_id = st.text_input("Bid ID (required for extraction)")
    question = st.text_area(
        "Question",
        placeholder="Example: What is the final submission deadline?",
    )

    if st.button("Submit", key="ask_button"):
        if mode == "Extract bid JSON" and not ask_bid_id.strip():
            st.warning("Enter a bid ID for extraction.")
        elif mode == "Answer a question" and not question.strip():
            st.warning("Enter a question.")
        else:
            payload = {
                "bid_id": ask_bid_id.strip() or None,
                "question": question.strip(),
                "mode": (
                    "extract"
                    if mode == "Extract bid JSON"
                    else "question_answering"
                ),
            }

            try:
                response = requests.post(
                    f"{API_URL}/ask",
                    json=payload,
                    timeout=300,
                )
                response.raise_for_status()
                result = response.json()

                if payload["mode"] == "extract":
                    output_json = json.dumps(result, indent=2, ensure_ascii=False)
                    st.json(result)
                    st.download_button(
                        "Download bid JSON",
                        data=output_json,
                        file_name=f"{result['bid_id']}.json",
                        mime="application/json",
                    )
                else:
                    st.markdown(result.get("answer", ""))
                    sources = result.get("sources", [])
                    if sources:
                        st.markdown("**Sources**")
                        for source in sources:
                            page = source.get("page")
                            suffix = f", page {page}" if page else ""
                            st.write(f"- {source['file']}{suffix}")

            except requests.RequestException as e:
                st.error(f"Ask request failed: {e}")


with index_tab:
    st.subheader("Index bid documents")
    index_bid_id = st.text_input(
        "Bid ID to index (leave blank to index all DATA folders)"
    )

    if st.button("Start indexing"):
        payload = {"bid_id": index_bid_id.strip() or None}

        try:
            response = requests.post(
                f"{API_URL}/index",
                json=payload,
                timeout=600,
            )
            response.raise_for_status()
            st.success("Indexing finished.")
            st.json(response.json())

        except requests.RequestException as e:
            st.error(f"Index request failed: {e}")