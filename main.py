# app.py
import os
import json
import pandas as pd
import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="Integration Test Scenario Extractor", page_icon="🧪", layout="centered")

st.title("🧪 Integration Test Scenarios from Document")
st.caption("Upload a .txt or .md file. The app extracts **integration** flows (2+ modules/functions) only.")

# --- UI: File upload (AC1) ---
uploaded = st.file_uploader(
    "Upload your requirements / spec document (.txt, .md)",
    type=["txt", "md", "markdown"],
    accept_multiple_files=False
)

# Display small instructions matching your rules
with st.expander("What will be extracted?"):
    st.markdown("""
- **Only** scenarios that involve **two or more** modules/functions/systems (i.e., **integration** flows).  
- **No inventions**. If the source does not specify something, it will be **'Not specified in document'**.  
- Covers **main (happy)**, **alternate**, and **exception** flows.  
- Output columns:
  1) **Requirement Location (as per document)**
  2) **Integration Flow Summary**
  3) **Related Modules/Functions/Systems**
  4) **Test Scenario (Integration)**
  5) **Main/Alternate/Exception Flow**
""")

def read_text(file):
    try:
        return file.read().decode("utf-8", errors="replace")
    except Exception:
        return file.read().decode(errors="replace")

# --- OpenAI client ---
def get_client():
    # Requires OPENAI_API_KEY in environment
    return OpenAI()

# --- Call OpenAI with function calling to get structured scenarios (AC2) ---
def extract_scenarios_with_function_call(doc_text: str):
    client = get_client()

    # Define function (tool) schema for structured extraction
    tools = [
        {
            "type": "function",
            "function": {
                "name": "submit_integration_scenarios",
                "description": "Return extracted integration test scenarios as a structured list.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "scenarios": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "requirement_location": {
                                        "type": "string",
                                        "description": "Exact chapter/section/heading, or short quoted snippet that clearly locates the requirement in the document."
                                    },
                                    "integration_flow_summary": {
                                        "type": "string",
                                        "description": "Brief summary of the multi-module integration/business flow."
                                    },
                                    "related_modules_functions_systems": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": "List of all modules/functions/systems participating in the flow."
                                    },
                                    "test_scenario_integration": {
                                        "type": "string",
                                        "description": "Concise integration scenario demonstrating cross-module coordination."
                                    },
                                    "flow_type": {
                                        "type": "string",
                                        "enum": ["Main Flow", "Alternate", "Exception", "Variant", "Not specified in document"],
                                        "description": "Classify as Main Flow, Alternate, Exception, Variant, or Not specified in document."
                                    }
                                },
                                "required": [
                                    "requirement_location",
                                    "integration_flow_summary",
                                    "related_modules_functions_systems",
                                    "test_scenario_integration",
                                    "flow_type"
                                ]
                            }
                        }
                    },
                    "required": ["scenarios"]
                }
            }
        }
    ]

    system_prompt = (
        "You are a senior test analyst specializing in INTEGRATION scenario extraction with a big-picture, end-to-end perspective.\n"
        "Goal: From the provided document, extract ONLY integration flows that span TWO OR MORE modules/functions/systems, emphasizing high-level business journeys and cross-system interactions.\n\n"
        "BIGGER-PICTURE RULES:\n"
        "1) Prioritize end-to-end outcomes and system-to-system interactions. Collapse granular UI/internal steps into higher-level actions.\n"
        "2) Do NOT invent or assume anything not in the document. If something is missing or ambiguous, use 'Not specified in document'.\n"
        "3) Include main (happy) flows, and only alternates/exceptions that materially change cross-system interactions, data contracts, or outcomes.\n"
        "4) Skip single-module/unit-level items and minor step variations that do not affect integrations.\n"
        "5) Each scenario must be clear for stakeholders and highlight participating systems and the core integration touchpoints; when mentioned in the source, note APIs/events/queues/identity propagation/data mapping; otherwise use 'Not specified in document'.\n"
        "6) 'Requirement Location' should clearly help the reader find the source: section/heading names or a short quoted phrase if no headings.\n"
        "7) Output MUST be a single function call to submit_integration_scenarios.\n"
    )

    user_prompt = (
        "Extract integration test scenarios based ONLY on the document below.\n"
        "Focus on the bigger picture: end-to-end flows with 2+ modules/functions/systems and core cross-system interactions. Merge micro-steps; keep a minimal, comprehensive set of scenarios. No inventions—if unknown, write 'Not specified in document'.\n"
        "Return as an array of objects via the function call.\n"
        "\n--- DOCUMENT START ---\n"
        f"{doc_text}\n"
        "--- DOCUMENT END ---\n"
    )

    # Use Chat Completions with function calling (tools)
    resp = client.chat.completions.create(
        model="gpt-4.1",
        temperature=0.1,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        tools=tools,
        tool_choice={"type": "function", "function": {"name": "submit_integration_scenarios"}}
    )

    msg = resp.choices[0].message
    tool_calls = getattr(msg, "tool_calls", None)

    if not tool_calls:
        raise RuntimeError("Model did not return a function call. Try a smaller document or adjust content.")

    # Parse the first tool call arguments as JSON
    args_str = tool_calls[0].function.arguments
    parsed = json.loads(args_str)
    scenarios = parsed.get("scenarios", [])

    # Normalize data
    norm = []
    for s in scenarios:
        loc = s.get("requirement_location") or "Not specified in document"
        summary = s.get("integration_flow_summary") or "Not specified in document"
        modules = s.get("related_modules_functions_systems") or []
        # Ensure at least two modules to satisfy integration-only constraint
        if isinstance(modules, list) and len(modules) < 2:
            # Skip any scenario that ends up with <2 modules (safety net)
            continue
        scenario = s.get("test_scenario_integration") or "Not specified in document"
        ftype = s.get("flow_type") or "Not specified in document"

        norm.append({
            "Requirement Location (as per document)": loc,
            "Integration Flow Summary": summary,
            "Related Modules/Functions/Systems": ", ".join([m.strip() for m in modules if str(m).strip()]),
            "Test Scenario (Integration)": scenario,
            "Main/Alternate/Exception Flow": ftype
        })

    return norm

# --- Main action ---
if uploaded:
    text = read_text(uploaded)

    if st.button("Generate Integration Scenarios"):
        if not os.getenv("OPENAI_API_KEY"):
            st.error("OPENAI_API_KEY is not set. Please set it in your environment and retry.")
        else:
            with st.spinner("Analyzing document and extracting integration flows..."):
                try:
                    rows = extract_scenarios_with_function_call(text)

                    if not rows:
                        st.warning("No integration flows found (2+ modules). Check the document content.")
                    else:
                        df = pd.DataFrame(rows, columns=[
                            "Requirement Location (as per document)",
                            "Integration Flow Summary",
                            "Related Modules/Functions/Systems",
                            "Test Scenario (Integration)",
                            "Main/Alternate/Exception Flow"
                        ])
                        st.success("Extraction complete.")
                        st.dataframe(df, use_container_width=True, hide_index=True)

                        # Optional: download CSV
                        csv = df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "Download CSV",
                            data=csv,
                            file_name="integration_test_scenarios.csv",
                            mime="text/csv",
                        )

                except Exception as e:
                    st.error(f"Something went wrong: {e}")
else:
    st.info("Please upload a .txt or .md file to begin.")
