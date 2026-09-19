# SpaceX Falcon 9 First Stage Landing Prediction

## IBM Applied Data Science Capstone

This project is part of IBM's **Applied Data Science Capstone** course (Coursera). The goal is to predict whether the first stage of a **SpaceX Falcon 9** rocket will land successfully, using historical launch data, exploratory analysis, and machine learning models.

## Problem context

SpaceX advertises its launches at a cost of $62 million, much lower than other providers (which charge upward of $165 million), largely because it **reuses the rocket's first stage** instead of discarding it. If we can predict whether a landing will be successful, a competing company could estimate the real cost of a SpaceX launch and use that to bid against it for a launch contract.

**Main question:** Can we predict whether the first stage of a Falcon 9 will land successfully, and which variables best correlate with that success?

## Repository structure

| File | Description |
|---|---|
| [`1. jupyter-labs-spacex-data-collection-api.ipynb`](./1.%20jupyter-labs-spacex-data-collection-api.ipynb) | Data collection via the SpaceX API |
| [`2. jupyter-labs-webscraping.ipynb`](./2.%20jupyter-labs-webscraping.ipynb) | Data collection via web scraping (Wikipedia) |
| [`3. labs-jupyter-spacex-Data wrangling.ipynb`](./3.%20labs-jupyter-spacex-Data%20wrangling.ipynb) | Data cleaning and creation of the `Class` variable |
| [`4. jupyter-labs-eda-sql-coursera_sqllite.ipynb`](./4.%20jupyter-labs-eda-sql-coursera_sqllite.ipynb) | EDA using SQL queries |
| [`5. jupyter-labs-edadataviz.ipynb`](./5.%20jupyter-labs-edadataviz.ipynb) | EDA with visualization (matplotlib/seaborn) and one-hot encoding |
| [`6. lab_jupyter_launch_site_location.ipynb`](./6.%20lab_jupyter_launch_site_location.ipynb) | Interactive visual analytics with Folium |
| [`7. SpaceX_Machine Learning Prediction_Part_5.ipynb`](./7.%20SpaceX_Machine%20Learning%20Prediction_Part_5.ipynb) | Machine learning modeling and prediction |
| [`spacex_dash_app.py`](./spacex_dash_app.py) | Interactive Plotly Dash dashboard |
| `spacex_dash_app_screenshot.png` | Screenshot of the dashboard in action |
| `my_data1.db` | SQLite database used in the SQL analysis (notebook 4) |

## Module overview

The capstone is organized into the following modules:

### 1. Data Collection

- **Via API:** Requests to the SpaceX REST API (`api.spacexdata.com/v4`) to retrieve data on rockets, launch sites, payloads, and cores for each launch. This data is merged into a single pandas DataFrame.
- **Via Web Scraping:** Extraction of the Falcon 9 launch table from Wikipedia using `BeautifulSoup`, parsing the HTML table to obtain date, time, booster version, launch site, payload, orbit, customer, launch outcome, and booster landing outcome.

### 2. Data Wrangling

Cleaning and transforming the combined data:
- Analysis of missing values and column data types.
- Frequency counts by launch site and orbit type.
- Creation of the binary target variable **`Class`** (1 = successful landing, 0 = failed), derived from the `Outcome` column.
- Overall success rate obtained: **66.7%**.

### 3. Exploratory Data Analysis (EDA)

- **With SQL:** Queries against the `SPACEXTBL` table (distinct launch site names, total and average payload mass, date of the first successful drone ship landing, boosters with a successful landing within a given payload mass range, launch outcome counts, etc.), using the `my_data1.db` database.
- **With visualization:** Charts built with `matplotlib` and `seaborn` (`scatter`, `catplot`, `barplot`, `lineplot`) to explore relationships between variables such as payload mass, orbit, launch site, year, and success rate. **One-hot encoding** is applied to categorical variables (`Orbit`, `LaunchSite`, `LandingPad`, `Serial`) to prepare the data for modeling.

### 4. Interactive Visual Analytics with Folium

- **Task 1:** Marking all 4 launch sites (`CCAFS LC-40`, `CCAFS SLC-40`, `KSC LC-39A`, `VAFB SLC-4E`) on an interactive map.
- **Task 2:** Color-coded markers by launch outcome (green = success, red = failure), grouped with `MarkerCluster` to simplify the visualization.
- **Task 3:** Calculating distances between each launch site and nearby points of interest (coastline, railways, highways, cities) using the Haversine formula.

**Findings:** launch sites are very close to the coastline (to allow the first stage to safely fall into the ocean) and to transportation infrastructure (railways/highways, for rocket logistics), but kept well away from major cities for safety reasons.

### 5. Interactive Plotly Dash Dashboard

An interactive dashboard (`spacex_dash_app.py`, see screenshot `spacex_dash_app_screenshot.png`) that allows the user to:
- Filter by launch site using a dropdown menu.
- Visualize the proportion of successful/failed launches per site (pie chart).
- Filter by payload mass range using a slider.
- View the correlation between payload mass and launch success, colored by booster version (scatter plot).

### 6. Machine Learning Prediction

Several classification models are trained and compared to predict the `Class` variable:

| Step | Description |
|---|---|
| Preparation | `Y = data['Class'].to_numpy()`; standardizing `X` with `StandardScaler`; train/test split (80/20, `random_state=2`) → 18 test samples |
| Models evaluated | Logistic Regression, Support Vector Machine (SVM), Decision Tree, K-Nearest Neighbors (KNN) |
| Method | `GridSearchCV` with 10-fold cross-validation (`cv=10`) to tune each model's hyperparameters |
| Evaluation | Accuracy (`score`) on the test set and confusion matrix for each model |

**Result:** the `best_score_` from cross-validation is compared across the four models to determine which generalizes best; in this case, the **Decision Tree** achieved the best score (≈ 0.89), followed by KNN (≈ 0.85).

## General conclusions

- The Falcon 9 landing success rate has increased over time, reflecting SpaceX's growing technological maturity.
- Launch site, target orbit, and payload mass all influence the probability of a successful landing.
- Tree-based classification models (Decision Tree) achieved the best predictive performance on this dataset, though all evaluated models reached a reasonable accuracy (>80%).

## Technologies used

- **Python:** pandas, NumPy, requests, BeautifulSoup
- **Visualization:** matplotlib, seaborn, folium, Plotly Dash
- **Database:** SQL (SQLite, via `ipython-sql`)
- **Machine Learning:** scikit-learn (`LogisticRegression`, `SVC`, `DecisionTreeClassifier`, `KNeighborsClassifier`, `GridSearchCV`, `StandardScaler`, `train_test_split`)

## Author

Pablo Durán Hernández — Applied Data Science Capstone (IBM / Coursera)
