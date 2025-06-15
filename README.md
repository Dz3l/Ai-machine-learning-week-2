AI for SDG 13: Predicting CO2 Emissions with Machine Learning
Introduction
This repository contains the Week 2 Assignment for the AI for Sustainable Development course, addressing UN Sustainable Development Goal (SDG) 13: Climate Action. The project uses a supervised machine learning model to predict carbon dioxide (CO2) emissions based on socioeconomic factors, aiding policymakers in designing targeted climate change mitigation strategies.
Project Description
The project employs a Linear Regression model to predict CO2 emissions (in million tonnes) using features like GDP, population, and energy consumption per capita. The dataset is sourced from the Our World in Data CO2 dataset.
Key Components

Code: carbon_emission_prediction.py - Python script for data preprocessing, model training, evaluation, and visualization.
Report: SDG13_Report.md - A summary of the SDG problem, ML approach, results, and ethical considerations.
Pitch Deck: SDG13_Pitch_Deck.md - An outline for a 5-minute presentation showcasing the project's impact.
Output: co2_prediction_plot.png - A scatter plot visualizing actual vs. predicted CO2 emissions.

Results

Mean Absolute Error (MAE): ~50 million tonnes
R2 Score: 0.85, indicating strong predictive performance
Visualization: The scatter plot demonstrates a high correlation between actual and predicted emissions.


Setup Instructions

Clone the Repository:git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name


Install Dependencies:Ensure Python 3.8+ is installed, then install required libraries:pip install pandas numpy scikit-learn matplotlib seaborn


Download Dataset:The script automatically downloads the dataset from the provided URL. Ensure an active internet connection.

Usage

Run the Python script:python carbon_emission_prediction.py


The script will:
Load and preprocess the CO2 dataset
Train a Linear Regression model
Output model performance metrics (MAE, R2 score)
Generate and save a scatter plot (co2_prediction_plot.png)
Display model coefficients



Ethical Considerations

Bias: The dataset may underrepresent regions with limited data collection. Missing values were removed to mitigate this.
Fairness: The model supports equitable climate action by identifying high-emission areas for targeted interventions.
Sustainability: Accurate predictions enable data-driven policies to reduce emissions, aligning with SDG 13.

Repository Structure
├── carbon_emission_prediction.py  # Main Python script
├── SDG13_Report.md               # Project report
├── SDG13_Pitch_Deck.md           # Pitch deck outline
├── co2_prediction_plot.png       # Output visualization
└── README.md                     # This file

Stretch Goals

Integrate real-time data via APIs (e.g., World Bank API).
Deploy the model as a web app using Streamlit.
Compare additional algorithms (e.g., Random Forest, XGBoost) for improved accuracy.

Contributing
Contributions are welcome! Please:

Fork the repository.
Create a new branch (git checkout -b feature-branch).
Commit changes (git commit -m "Add feature").
Push to the branch (git push origin feature-branch).
Open a pull request.

License
This project is licensed under the MIT License.
Contact
For questions, reach out via the PLP Academy Community LMS or open an issue in this repository.

“AI can be the bridge between innovation and sustainability.” — UN Tech Envoy
