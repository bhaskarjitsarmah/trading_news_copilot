import os
import dotenv
from dotenv import load_dotenv

from langchain.callbacks import StreamlitCallbackHandler
from langchain.agents import AgentType, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.tools.yahoo_finance_news import YahooFinanceNewsTool
from langchain.callbacks import StdOutCallbackHandler

import streamlit as st
import openai

load_dotenv()


api_base = os.getenv("AZURE_OPENAI_BASE")
api_key = os.getenv("AZURE_OPENAI_KEY")

api_type = os.getenv("AZURE_OPENAI_APITYPE")
api_version = os.getenv("AZURE_OPENAI_APIVERSION")
model = os.getenv("GPT_MODEL")

env = dotenv.dotenv_values(".env")

openai.api_type = api_type
openai.api_base = api_base
openai.api_version = api_version
openai.api_key = api_key

chat_model_id = model


handler = StdOutCallbackHandler()
llm = ChatOpenAI(temperature=0.0,engine=chat_model_id,openai_api_key=st.secrets['path'])


tools = [YahooFinanceNewsTool()]
agent_chain = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        st.write("🧠 thinking...")
        st_callback = StreamlitCallbackHandler(st.container())
        response = agent_chain.run(prompt, callbacks=[st_callback])
        st.write(response)