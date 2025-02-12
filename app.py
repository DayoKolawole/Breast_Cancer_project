import seaborn as sns
import matplotlib.pyplot as plt

# Mock demographic data for visualization
demographics = pd.DataFrame({
    "Age": np.random.randint(30, 80, 100),
    "Gender": np.random.choice(["Male", "Female"], 100),
    "Diagnosis": np.random.choice(["Benign", "Malignant"], 100, p=[0.7, 0.3])
})

st.subheader("📊 Demographic Distribution")

# Gender Distribution
st.write("### 🧑‍🤝‍🧑 Gender Distribution")
gender_count = demographics["Gender"].value_counts()
fig, ax = plt.subplots()
sns.barplot(x=gender_count.index, y=gender_count.values, palette="pastel", ax=ax)
ax.set_ylabel("Count")
st.pyplot(fig)

# Age Distribution
st.write("### 🎂 Age Distribution")
fig, ax = plt.subplots()
sns.histplot(demographics["Age"], kde=True, bins=15, color="blue", ax=ax)
ax.set_xlabel("Age")
st.pyplot(fig)

# Diagnosis Breakdown
st.write("### 🔬 Diagnosis Breakdown")
fig, ax = plt.subplots()
sns.countplot(data=demographics, x="Diagnosis", palette=["#ff4b4b", "#50C878"], ax=ax)
st.pyplot(fig)

# Footer
st.markdown('<p class="footer">Developed with ❤️ using Streamlit</p>', unsafe_allow_html=True)
