import streamlit as st
import pandas as pd
import numpy as np
from evaluate import User

#Title
u1 = User()

st.title("📰 Fake News Detector")

link = st.text_input("Enter a news article URL:")

if st.button("Check News"):
    if link.strip():
        try:
            result = u1.news(link)
            st.success(result)
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.warning("Please enter a valid URL.")