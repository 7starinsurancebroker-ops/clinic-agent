"""
Website version of the clinic agent for Streamlit Community Cloud.
Needs clinic_agent.py in the same folder.
The API key goes in Streamlit's Secrets settings, never in code.
"""
from datetime import datetime

import anthropic
import streamlit as st

from clinic_agent import CLINIC, MODEL, SYSTEM, TOOLS, run_tool

st.set_page_config(page_title=CLINIC["name"], page_icon="🦷")
st.title(f"{CLINIC['name']} - Virtual Receptionist")
st.caption("Ask about hours, services, or book an appointment. I'm an AI assistant.")

client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
system_prompt = f"Right now it is {datetime.now():%A, %Y-%m-%d %H:%M}. Use this as today's date.\n\n" + SYSTEM

if "messages" not in st.session_state:
    st.session_state.messages = []  # full API history
    st.session_state.display = []   # what the patient sees

for m in st.session_state.display:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if prompt := st.chat_input("Type your message..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.display.append({"role": "user", "content": prompt})
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Typing..."):
            while True:
                resp = client.messages.create(
                    model=MODEL, max_tokens=1024, system=system_prompt,
                    tools=TOOLS, messages=st.session_state.messages,
                )
                st.session_state.messages.append({"role": "assistant", "content": resp.content})
                if resp.stop_reason != "tool_use":
                    break
                results = [
                    {"type": "tool_result", "tool_use_id": b.id, "content": run_tool(b.name, b.input)}
                    for b in resp.content if b.type == "tool_use"
                ]
                st.session_state.messages.append({"role": "user", "content": results})
        reply = "".join(b.text for b in resp.content if b.type == "text")
        st.markdown(reply)
    st.session_state.display.append({"role": "assistant", "content": reply})
