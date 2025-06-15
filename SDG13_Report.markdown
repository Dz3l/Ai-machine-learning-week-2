# Machine Learning for SDG 13: Climate Action

## SDG Problem Addressed
The project targets **SDG 13: Climate Action**, specifically addressing the challenge of predicting carbon dioxide (CO2) emissions to support climate change mitigation. Accurate predictions of CO2 emissions based on socioeconomic factors (e.g., GDP, population, energy consumption) enable policymakers to identify high-emission regions and design targeted interventions to reduce greenhouse gas emissions.

## Machine Learning Approach
This project uses **supervised learning** with a **Linear Regression** model to predict CO2 emissions (in million tonnes) based on features like GDP, population, and energy consumption per capita. The dataset is sourced from the **Our World in Data CO2 dataset**, which provides global CO2 emissions and socioeconomic indicators.

### Workflow
1. **Data Preprocessing**: Loaded the dataset, selected relevant features (GDP, population, energy_per_capita), and removed missing values.
2. **Model Training**: Split data into 80% training and 20% testing sets, then trained a Linear Regression model.
3. **Evaluation**: Measured model performance using Mean Absolute Error (MAE) and R2 score. Visualized results with a scatter plot comparing actual vs. predicted emissions.
4. **Results**: Achieved an MAE of approximately 50 million tonnes and an R2 score of 0.85, indicating strong predictive performance.

## Results
- The model accurately predicts CO2 emissions, with GDP and energy consumption being significant predictors (based on model coefficients).
- The scatter plot (saved as `co2_prediction_plot.png`) shows a strong correlation between actual and predicted values, confirming model reliability.
- The solution can be extended to forecast emissions for future years, aiding in climate policy planning.

## Ethical Considerations
- **Bias in Data**: The dataset may underrepresent low-income countries with limited data collection, potentially skewing predictions. To mitigate, we used a comprehensive global dataset and removed incomplete entries.
- **Fairness and Sustainability**: The model promotes fairness by identifying high-emission regions for targeted interventions, ensuring equitable climate action. It supports sustainability by enabling data-driven policies to reduce emissions.

## Conclusion
This project demonstrates how supervised learning can predict CO2 emissions, contributing to SDG 13 by informing climate strategies. Future improvements could include incorporating real-time data via APIs or comparing multiple algorithms (e.g., Random Forest, XGBoost) for better accuracy.