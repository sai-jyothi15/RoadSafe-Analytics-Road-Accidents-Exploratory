import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="RoadSafe Analytics Dashboard",
    page_icon="🚦",
    layout="wide"
)

sns.set_style("darkgrid")

# -------------------------------------------------
# LOAD DATA (Optimized)
# -------------------------------------------------
@st.cache_data
def load_data():
    cols = [
        'Start_Time', 'Severity', 'State', 'City', 'Weather_Condition',
        'Visibility(mi)', 'Distance(mi)', 'Temperature(F)',
        'Start_Lat', 'Start_Lng', 'Sunrise_Sunset', 'Traffic_Signal'
    ]
    df = pd.read_csv("cleaned_us_accidents.csv", usecols=cols)
    df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')
    df['Hour'] = df['Start_Time'].dt.hour
    df['Weekday'] = df['Start_Time'].dt.day_name()
    df['Month'] = df['Start_Time'].dt.month_name()

    return df

df = load_data()

# -------------------------------------------------
# SIDEBAR FILTERS
# -------------------------------------------------
st.sidebar.title("🔍 Filters")

if st.sidebar.button("🔄 Reset Filters"):
    st.rerun()

states = st.sidebar.multiselect(
    "State",
    options=sorted(df['State'].dropna().unique()),
    default=sorted(df['State'].dropna().unique())
)

severity = st.sidebar.multiselect(
    "Severity",
    sorted(df['Severity'].unique()),
    default=sorted(df['Severity'].unique())
)

weather = st.sidebar.multiselect(
    "Weather Condition",
    df['Weather_Condition'].dropna().unique(),
    default=df['Weather_Condition'].dropna().unique()[:3]
)

day_night = st.sidebar.multiselect(
    "Day / Night",
    options=["Day", "Night"],
    default=["Day", "Night"]
)


visibility = st.sidebar.slider(
    "Visibility (miles)",
    float(df['Visibility(mi)'].min()),
    float(df['Visibility(mi)'].max()),
    (0.0, 10.0)
)

date_range = st.sidebar.date_input(
    "Date Range",
    [df['Start_Time'].min(), df['Start_Time'].max()]
)

# -------------------------------------------------
# APPLY FILTERS
# -------------------------------------------------
filtered_df = df.copy()
filtered_df = filtered_df[filtered_df['State'].isin(states)]

filtered_df = filtered_df[
    (filtered_df['Severity'].isin(severity)) &
    (filtered_df['Weather_Condition'].isin(weather)) &
    (filtered_df['Visibility(mi)'].between(*visibility)) &
    (filtered_df['Start_Time'].dt.date >= date_range[0]) &
    (filtered_df['Start_Time'].dt.date <= date_range[1])
]

filtered_df = filtered_df[
    filtered_df['Sunrise_Sunset'].isin(day_night)
]


st.sidebar.metric("Filtered Records", len(filtered_df))

# -------------------------------------------------
# HEADER & KPIs
# -------------------------------------------------
st.title("🚦 RoadSafe Analytics – US Road Accident Dashboard")

st.markdown("""
**End-to-End Exploratory Data Analysis & Visualization Project**  
 RoadSafe Analytics is an Exploratory Data Analysis (EDA) project focused on understanding road accident patterns in the United States using a real-world dataset from Kaggle.

The project analyzes how time, weather, visibility, traffic signals, and location influence accident frequency and severity. The dataset contains millions of accident records across multiple states and cities, making it suitable for both temporal and geospatial analysis.

**The key objectives of this project are to**:

Explore and understand large-scale accident data

Identify peak accident times and high-risk conditions

Analyze relationships between accident severity and influencing factors

Detect accident-prone states and cities

Test common assumptions related to road accidents

This dashboard represents the final consolidated outcome of the internship project.
""")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Accidents", len(filtered_df))
k2.metric("Average Severity", round(filtered_df['Severity'].mean(), 2))
k3.metric("States Covered", filtered_df['State'].nunique())
k4.metric("Weather Types", filtered_df['Weather_Condition'].nunique())

st.markdown("---")

# -------------------------------------------------
# TABS (Week-wise)
# -------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📁 Data Overview",
    "📊 Univariate Analysis",
    "📈 Bivariate & Multivariate Analysis",
    "🔥 Correlation",
    "🗺️ Geospatial",
    "🧪 Hypothesis Testing",
    "ℹ️ About & Help"
])

# -------------------------------------------------
# TAB 1: DATA OVERVIEW (WEEK 1)
# -------------------------------------------------
with tab1:
    st.subheader("Dataset Structure & Quality")

    colA, colB = st.columns(2)

    with colA:
        st.markdown("### Column Data Types")
        dtype_counts = filtered_df.dtypes.value_counts()
        fig, ax = plt.subplots()
        dtype_counts.plot(kind='bar', ax=ax)
        st.pyplot(fig)

    with colB:
        st.markdown("### Missing Values (%) – Top Columns")
        missing_pct = (filtered_df.isnull().sum() / len(filtered_df)) * 100
        missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=False).head(10)
        fig, ax = plt.subplots()
        missing_pct.plot(kind='bar', ax=ax)
        st.pyplot(fig)

# -------------------------------------------------
# TAB 2: UNIVARIATE (WEEK 3)
# -------------------------------------------------
with tab2:
    st.subheader("Univariate Analysis")
    st.write("This section analyzes individual variables to understand accident patterns.")

    # 1️⃣ Distribution of Accident Severity
    st.markdown("### 1. Distribution of Accident Severity")
    plt.figure(figsize=(6,4))
    sns.countplot(x='Severity', data=df)
    plt.title("Distribution of Accident Severity")
    plt.xlabel("Severity Level")
    plt.ylabel("Number of Accidents")
    st.pyplot(plt.gcf())
    plt.clf()

    # 2️⃣ Accidents by Hour of Day
    st.markdown("### 2. Accidents by Hour of Day")
    plt.figure(figsize=(8,4))
    sns.histplot(df['Hour'], bins=24)
    plt.title("Accidents by Hour of Day")
    plt.xlabel("Hour")
    plt.ylabel("Accident Count")
    st.pyplot(plt.gcf())
    plt.clf()

    # 3️⃣ Accidents by Day of Week
    st.markdown("### 3. Accidents by Day of Week")
    order_days = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    plt.figure(figsize=(8,4))
    sns.countplot(x='Weekday', data=df, order=order_days)
    plt.xticks(rotation=45)
    plt.title("Accidents by Day of Week")
    st.pyplot(plt.gcf())
    plt.clf()

    # 4️⃣ Accidents by Month
    st.markdown("### 4. Accidents by Month")
    order_months = [
        'January','February','March','April','May','June',
        'July','August','September','October','November','December'
    ]
    plt.figure(figsize=(8,4))
    sns.countplot(x='Month', data=df, order=order_months)
    plt.xticks(rotation=45)
    plt.title("Accidents by Month")
    st.pyplot(plt.gcf())
    plt.clf()

    # 5️⃣ Top 10 Weather Conditions
    st.markdown("### 5. Top 10 Weather Conditions During Accidents")
    top_weather = df['Weather_Condition'].value_counts().head(10)
    plt.figure(figsize=(8,4))
    top_weather.plot(kind='bar')
    plt.title("Top 10 Weather Conditions During Accidents")
    plt.xlabel("Weather Condition")
    plt.ylabel("Count")
    st.pyplot(plt.gcf())
    plt.clf()

    # 6️⃣ Accidents Near Traffic Signals
    st.markdown("### 6. Accidents Near Traffic Signals")
    plt.figure(figsize=(5,5))
    df['Traffic_Signal'].value_counts().plot(kind='pie', autopct='%1.1f%%')
    plt.title("Accidents Near Traffic Signals")
    plt.ylabel("")
    st.pyplot(plt.gcf())
    plt.clf()

# -------------------------------------------------
# TAB 3: BIVARIATE & MULTIVARIATE (WEEK 4)
# -------------------------------------------------
with tab3:
    st.subheader("Bivariate Analysis")
    st.write(
        "Bivariate analysis helps understand the relationship between accident severity "
        "and other influencing factors such as visibility, weather, traffic signals, and weekdays."

    )

    # 1️⃣ Severity vs Visibility
    st.markdown("### 1. Accident Severity vs Visibility")
    plt.figure(figsize=(7,4))
    sns.boxplot(x='Severity', y='Visibility(mi)', data=df)
    plt.title("Accident Severity vs Visibility")
    plt.xlabel("Severity Level")
    plt.ylabel("Visibility (miles)")
    st.pyplot(plt.gcf())
    plt.clf()

    # 2️⃣ Severity vs Weather Condition
    st.markdown("### 2. Severity vs Weather Condition")
    top_weather = df['Weather_Condition'].value_counts().head(5).index
    df_weather = df[df['Weather_Condition'].isin(top_weather)]

    plt.figure(figsize=(8,4))
    sns.countplot(x='Weather_Condition', hue='Severity', data=df_weather)
    plt.xticks(rotation=30)
    plt.title("Severity vs Weather Condition")
    st.pyplot(plt.gcf())
    plt.clf()

    # 3️⃣ Severity vs Traffic Signal Presence
    st.markdown("### 3. Severity vs Traffic Signal Presence")
    plt.figure(figsize=(6,4))
    sns.countplot(x='Traffic_Signal', hue='Severity', data=df)
    plt.title("Severity vs Traffic Signal Presence")
    plt.xlabel("Traffic Signal Present")
    plt.ylabel("Accident Count")
    st.pyplot(plt.gcf())
    plt.clf()

    st.subheader("Multivariate Analysis")
    st.write("Multivariate analysis studies the relationship between two or more variables at the same time to understand how multiple factors together influence accident severity.")
        
    # 4️⃣ Correlation Heatmap
    st.markdown("### 4. Correlation Heatmap (Severity, Visibility, Hour)")
    numeric_cols = df[['Severity', 'Visibility(mi)', 'Hour']]

    plt.figure(figsize=(6,4))
    sns.heatmap(numeric_cols.corr(), annot=True, cmap='coolwarm')
    plt.title("Correlation Heatmap")
    st.pyplot(plt.gcf())
    plt.clf()

    # 5️⃣ Severity Distribution Across Weekdays
    st.markdown("### 5. Severity Distribution Across Weekdays")
    plt.figure(figsize=(8,4))
    sns.boxplot(
        x='Weekday',
        y='Severity',
        data=df,
        order=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    )
    plt.xticks(rotation=45)
    plt.title("Severity Distribution Across Weekdays")
    st.pyplot(plt.gcf())
    plt.clf()

    # 6️⃣ Pair Plot (Sample Data)
    st.markdown("### 6. Pair Plot (Sample Data)")
    sample_df = df[['Severity', 'Visibility(mi)', 'Hour']].sample(5000, random_state=42)
    pair_fig = sns.pairplot(sample_df)
    st.pyplot(pair_fig)


# -------------------------------------------------
# TAB 4: CORRELATION (WEEK 6)
# -------------------------------------------------
with tab4:
    st.subheader("Correlation Heatmap")

    corr_cols = ['Severity', 'Visibility(mi)', 'Distance(mi)', 'Temperature(F)']
    corr_df = filtered_df[corr_cols].dropna()

    fig, ax = plt.subplots()
    sns.heatmap(corr_df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# -------------------------------------------------
# TAB 5: GEOSPATIAL (WEEK 5)
# -------------------------------------------------
with tab5:
    st.subheader("Geospatial Analysis")
    st.write(
        "This section focuses on location-based accident patterns, "
        "including accident hotspots and regions with the highest accident frequency."
    )

    st.subheader("Accident Hotspots Map")
    map_df = filtered_df[['Start_Lat', 'Start_Lng']].dropna()
    st.map(map_df.rename(columns={'Start_Lat':'lat','Start_Lng':'lon'}))

    st.subheader("Top 10 Accident-Prone States")
    top_states = filtered_df['State'].value_counts().head(10)
    fig, ax = plt.subplots()
    top_states.plot(kind='bar', ax=ax)
    st.pyplot(fig)

    # 1️⃣ Accident Hotspots (Scatter Plot)
    st.markdown("### 1. Accident Hotspots (Sample Data)")
    plt.figure(figsize=(6,6))
    plt.scatter(
        df['Start_Lat'][0:5000],
        df['Start_Lng'][0:5000],
        s=2,
        alpha=0.3
    )
    plt.title("Accident Hotspots (Sample Data)")
    plt.xlabel("Latitude")
    plt.ylabel("Longitude")
    st.pyplot(plt.gcf())
    plt.clf()

    # 2️⃣ Top 5 Most Accident-Prone States
    st.markdown("### 2. Top 5 Accident-Prone States")
    top_states = df['State'].value_counts().head(5)

    plt.figure(figsize=(7,4))
    top_states.plot(kind='bar', color='orange')
    plt.title("Top 5 Accident-Prone States")
    plt.xlabel("State")
    plt.ylabel("Accident Count")
    st.pyplot(plt.gcf())
    plt.clf()

    # 3️⃣ Top 5 Most Accident-Prone Cities
    st.markdown("### 3. Top 5 Accident-Prone Cities")
    top_cities = df['City'].value_counts().head(5)

    plt.figure(figsize=(8,4))
    top_cities.plot(kind='bar', color='purple')
    plt.title("Top 5 Accident Cities")
    plt.xlabel("City")
    plt.ylabel("Accident Count")
    plt.xticks(rotation=45)
    st.pyplot(plt.gcf())
    plt.clf()


# -------------------------------------------------
# TAB 6: HYPOTHESIS TESTING (WEEK 6)
# -------------------------------------------------
with tab6:
    st.subheader("Hypothesis Testing")
    st.write(
        "This section tests common assumptions about road accidents "
        "using visual evidence from the dataset."
    )

    st.markdown("""
    **Hypothesis 1:** Accidents peak during rush hours  
    ✔ Confirmed via hourly distribution.

    **Hypothesis 2:** Poor visibility increases severity  
    ✔ Confirmed using boxplots.

    **Hypothesis 3:** Weather affects accident severity  
    ✔ Rain and fog show higher severity levels.
    """)

    # QUESTION 1
    st.markdown("### Q1. What time of day has the most accidents?")
    st.write(
        "This plot examines accident frequency across different hours of the day "
        "to identify peak accident periods."
    )

    plt.figure(figsize=(8,4))
    sns.countplot(x=df['Hour'], palette="viridis")
    plt.title("Accidents by Hour of Day")
    plt.xlabel("Hour")
    plt.ylabel("Accident Count")
    st.pyplot(plt.gcf())
    plt.clf()

    st.markdown(
        "**Observation:** Accidents peak during morning and evening hours, "
        "indicating higher risk during rush hours."
    )

    # QUESTION 2
    st.markdown("### Q2. Are accidents more severe during rain or fog?")
    st.write(
        "This plot compares accident severity under Rain, Fog, and Clear weather conditions."
    )

    weather_focus = df[df['Weather_Condition'].isin(['Rain', 'Fog', 'Clear'])]

    plt.figure(figsize=(7,4))
    sns.boxplot(x='Weather_Condition', y='Severity', data=weather_focus)
    plt.title("Severity during Rain, Fog & Clear Weather")
    st.pyplot(plt.gcf())
    plt.clf()

    st.markdown(
        "**Observation:** Accidents occurring during rain and fog "
        "show higher severity compared to clear weather."
    )

    # QUESTION 3
    st.markdown("### Q3. Is visibility correlated with accident severity?")
    st.write(
        "This heatmap shows the correlation between visibility and accident severity."
    )

    corr_df = df[['Severity', 'Visibility(mi)']].dropna().corr()

    plt.figure(figsize=(6,4))
    sns.heatmap(corr_df, annot=True, cmap="coolwarm")
    plt.title("Correlation between Visibility & Severity")
    st.pyplot(plt.gcf())
    plt.clf()

    st.markdown(
        "**Observation:** A negative correlation indicates that lower visibility "
        "is associated with higher accident severity."
    )


# -------------------------------------------------
# TAB 7: ABOUT & HELP (WEEK 1–6)
# -------------------------------------------------
with tab7:
    st.subheader("About the Project")
    st.write("""
    **Project Title:** RoadSafe Analytics – Exploratory Data Analysis of US Road Accidents

    **Project Type:** Data Analytics / Exploratory Data Analysis (EDA)

    **Objective:**  
    The objective of this project is to analyze road accident data to identify
    patterns, trends, and key factors influencing accident frequency and severity.

    **Dataset Information:**  
    - Dataset Name: US Accidents Dataset  
    - Source: Kaggle  
    - Data Type: Real-world, large-scale accident records  
    - Coverage: Multiple states and cities across the United States  

    **Key Parameters Analyzed:**  
    - Accident Severity  
    - Time (Hour, Day, Month)  
    - Weather Conditions  
    - Visibility  
    - Traffic Signals  
    - Location (State, City, Latitude & Longitude)  

    **Tools & Technologies Used:**  
    - Programming Language: Python  
    - Libraries: Pandas, NumPy, Matplotlib, Seaborn  
    - Visualization Platform: Streamlit  
    - Development Environment: Jupyter Notebook, VS Code  

    **Internship Scope:**  
    This dashboard represents the final consolidated output of the internship,
    integrating all analysis performed from Week 1 to Week 6.
    """)

    st.write("""
    **Outcome:**  
    The project demonstrates how data analytics and visualization can transform
    large datasets into meaningful insights that support road safety awareness
    and decision-making.
    """)
    
    st.subheader("Help & Usage Guide")

    st.write("""
    **How to Use This Dashboard:**

    - Use the **tabs at the top** to navigate through different types of analysis
      such as Univariate, Bivariate, Geospatial, and Hypothesis Testing.
    - Each tab corresponds to a specific stage of the data analysis process
      carried out during the internship.
    """)

    st.write("""
    **Understanding the Tabs:**
    - **Project Overview:** Provides background, objectives, and scope of the project.
    - **Univariate Analysis:** Shows patterns in individual variables such as time,
      severity, weather, and traffic signals.
    - **Bivariate Analysis:** Explores relationships between severity and other factors.
    - **Correlation Analysis:** Displays statistical relationships between numeric variables.
    - **Geospatial Analysis:** Highlights accident hotspots and high-risk regions.
    - **Hypothesis Testing:** Validates assumptions using visual evidence.
    - **Insights:** Summarizes key findings and conclusions.
    """)

    st.write("""
    **Important Notes:**
    - All plots shown in this dashboard are generated from the analysis performed
      during the internship.
    - No external or synthetic data has been added.
    - Some plots use sample data for performance reasons due to the large dataset size.
    """)

    st.write("""
    **Purpose of the Dashboard:**
    This dashboard is designed to present complex accident data in a clear,
    interactive, and easy-to-understand format for academic evaluation and learning.
    """)


# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.markdown("---")
st.caption("🚦 RoadSafe Analytics | Complete Dashboard |")   