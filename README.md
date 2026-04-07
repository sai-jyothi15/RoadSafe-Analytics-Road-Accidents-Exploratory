# RoadSafe-Analytics-Road-Accidents-Exploratory
A data analysis project exploring the US Road Accidents dataset. Includes data cleaning, preprocessing, and exploratory analysis to identify key patterns influencing road safety. Helps uncover trends related to location, weather, severity, and time to improve understanding of accident risks.
# US_Accidents_Data
https://drive.google.com/file/d/1H7a9gfsV7K_lJxMFX3ZgcXYfHVu9xbra/view?usp=drivesdk
🚦 RoadSafe Analytics
Intelligent Road Accident Analysis & Visualization Dashboard
📌 Project Overview
RoadSafe Analytics is an interactive data analytics dashboard designed to analyze large-scale US road accident data and uncover meaningful insights.
The project focuses on identifying:
High-risk accident zones
Peak accident time periods
Impact of weather and visibility
Severity patterns across regions
This dashboard transforms millions of accident records into clear, actionable insights using visualization techniques.
🎯 Objectives
Analyze accident trends based on time (hour, day, month)
Identify accident-prone states and cities
Study the impact of weather conditions and visibility
Understand severity distribution and influencing factors
Build a dynamic and interactive dashboard
📊 Features
🔍 Dynamic Filters (State, Severity, Weather, Date, Visibility, Day/Night)
📈 Univariate Analysis (Distribution, trends, counts)
📊 Bivariate & Multivariate Analysis
🔥 Correlation Heatmaps
🗺️ Geospatial Analysis (Hotspots & Maps)
🧪 Hypothesis Testing Section
⚡ Real-time Plot Updates Based on Filters
📉 Interactive Charts & Visual Insights
🧰 Technology Stack
Python – Core programming
Pandas & NumPy – Data preprocessing
Matplotlib & Seaborn – Data visualization
Plotly – Interactive charts
Streamlit – Dashboard development
Folium – Geospatial mapping
📂 Dataset Information
Dataset: US Accidents Dataset
Source: Kaggle
Type: Real-world accident data
Records: Millions of entries
Coverage: Multiple US states & cities
Key Attributes:
Date & Time
Severity
State & City
Weather Condition
Visibility
Traffic Signals
Latitude & Longitude
⚙️ Project Workflow
Data Collection
Loaded dataset from Kaggle
Data Cleaning
Handled missing values
Converted datetime fields
Feature engineering (Hour, Weekday, Month)
Exploratory Data Analysis (EDA)
Univariate Analysis
Bivariate Analysis
Multivariate Analysis
Correlation Analysis
Visualization
Bar Charts
Line Charts
Pie Charts
Boxplots
Heatmaps
Geospatial Maps
Dashboard Development
Built using Streamlit
Added dynamic filtering
Performance Optimization
Used caching (st.cache_data)
Sampled large datasets for maps
🚀 How to Run the Project
Step 1: Clone Repository
Bash
git clone https://github.com/your-username/roadsafe-analytics.git
cd roadsafe-analytics
Step 2: Install Dependencies
Bash
pip install -r requirements.txt
Step 3: Run the Dashboard
Bash
streamlit run app.py
📈 Key Insights
Accidents peak during morning and evening rush hours
Rain and fog increase accident severity
Low visibility is strongly linked to higher severity
Certain states and cities show high accident density
Night accidents tend to be more severe
🧪 Hypothesis Tested
✔ Accidents peak during rush hours
✔ Poor visibility increases severity
✔ Weather conditions affect accidents
✔ Night accidents show higher severity
⚠️ Limitations
No real-time data integration
Does not include driver behavior data
Limited predictive modeling
🔮 Future Enhancements
Real-time accident data integration
Machine learning-based severity prediction
AI-based risk alerts
Mobile-friendly dashboard
Integration with traffic sensor data
🎓 Learning Outcomes
Data Analysis using Python
Handling large real-world datasets
Data Visualization best practices
Dashboard development with Streamlit
Problem-solving with data
📌 Conclusion
This project demonstrates how data analytics and visualization can transform complex accident data into actionable insights, supporting road safety planning and decision-making.
Local URL: http://localhost:8501
Network URL: http://10.38.57.17:8501
