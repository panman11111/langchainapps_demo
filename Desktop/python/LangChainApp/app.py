import streamlit as st
from settings import agent
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain.schema import (SystemMessage, HumanMessage, AIMessage)

def main():
    # ページトップ文言
    st.title("LangChain demo")

    # サイドバー(パラメータ調整用)
    st.sidebar.title("Parameters")
    model = st.sidebar.selectbox("Model:", ["gpt-3.5-turbo", "text-davinci-002", "text-curie-003"])
    temperature = st.sidebar.slider("Temperature:", min_value=0.0, max_value=2.0, value=1.0, step=0.1)
    max_tokens = st.sidebar.slider("Max_tokens:", min_value=0, max_value=255, value=0, step=1)
    top_p = st.sidebar.slider("Top_p:", min_value=0.0, max_value=1.0, value=1.0, step=0.1)
    frequency_penalty = st.sidebar.slider("Frenquery_penalty:", min_value=-2.0, max_value=2.0, value=0.0, step=0.1)
    presence_penalty = st.sidebar.slider("Presence_penalty:", min_value=-2.0, max_value=2.0, value=0.0, step=0.1)

    # チャット
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    if prompt := st.chat_input(): 
        st.session_state.messages.append(HumanMessage(content=prompt))
        # st.session_state.messages.append(AIMessage(content="test")) # APIを使用しないテスト会話用
        with st.chat_message("assistant"): 
            st_callback = StreamlitCallbackHandler(st.container()) 
            response = agent.run(
                input = prompt, 
                callbacks=[st_callback], 
                model=model,
                temperature=temperature, 
                max_tokens=max_tokens, 
                top_p=top_p, 
                frequency_penalty=frequency_penalty, 
                presence_penalty=presence_penalty
                ) 
            st.session_state.messages.append(AIMessage(content=response))

    # 履歴管理
    messages = st.session_state.get('messages', [])
    for message in messages:
        if isinstance(message, AIMessage):
            with st.chat_message('assistant'):
                st.markdown(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message('user'):
                st.markdown(message.content)
        else:  # isinstance(message, SystemMessage):
            st.write(f"System message: {message.content}")

if __name__ == "__main__":
    main()
