import streamlit as st

st.title("Streamlit Test App")
st.write("Halo! Kalau kamu bisa lihat tulisan ini, berarti Streamlit kamu jalan 😄")

name = st.text_input("Masukkan nama kamu:")
if name:
    st.success(f"Halo, {name}! 👋")
