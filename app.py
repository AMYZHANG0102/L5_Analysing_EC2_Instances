# =========================================================
# IMPORT LIBRARIES
# =========================================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


# STREAMLIT PAGE SETUP

st.set_page_config(
    page_title="Amazon EC2 Cost Analysis",
    layout="wide"
)

st.title("Analyzing Amazon EC2 Instance Costs")

st.write("Amy Zhang")

st.write(
    "Analysis of Amazon EC2 instance pricing, cost distributions, "
    "instance families, and On-Demand cost prediction."
)


# =========================================================
# PART I
# =========================================================


st.header("Part I: EC2 Instance Cost and Performance Analysis")

file_path = "ec2dataset.csv"


# 1. LOAD THE DATASET
data = pd.read_csv(file_path)

st.subheader("1. Dataset Information")

col1, col2 = st.columns(2)

with col1:
    st.metric("Number of Rows", data.shape[0])
with col2:
    st.metric("Number of Columns", data.shape[1])

dataset_information = pd.DataFrame(
    {
        "Column": data.columns,
        "Data Type": data.dtypes.astype(str).values,
        "Non-Null Values": data.notnull().sum().values,
        "Missing Values": data.isnull().sum().values
    }
)

st.dataframe(
    dataset_information,
    use_container_width=True,
    hide_index=True
)


# 2. DATASET PREVIEW
st.subheader("2. Dataset Preview")

st.dataframe(
    data.head(),
    use_container_width=True
)


# 3. CLEAN THE DATA
st.subheader("3. Data Cleaning")

cost_columns = [
    "On Demand",
    "Linux Reserved cost",
    "Linux Spot Minimum cost",
    "Windows On Demand cost",
    "Windows Reserved cost"
]

# Remove '$' and 'hourly' and convert columns to numeric
for column in cost_columns:
    data[column] = pd.to_numeric(
        data[column].str.replace(
            "[$, hourly]",
            "",
            regex=True
        ),
        errors="coerce"
    )

# Check for missing values after conversion
missing_values = data[cost_columns].isnull().sum()
missing_values_table = missing_values.reset_index()

missing_values_table.columns = [
    "Cost Column",
    "Missing Values"
]

st.write(
    "Missing values after converting the cost columns to numeric:"
)

st.dataframe(
    missing_values_table,
    use_container_width=True,
    hide_index=True
)


# 4. SUMMARY ANALYSIS
st.subheader("4. Summary Analysis")

cost_summary = data[cost_columns].describe()

st.dataframe(
    cost_summary,
    use_container_width=True
)


# 5. VISUALIZE THE DATA
st.subheader("5. Cost Comparison")

sns.set(style="whitegrid")

fig1, ax1 = plt.subplots(figsize=(12, 6))

sns.boxplot(
    data=data[cost_columns],
    palette="Set2",
    ax=ax1
)

ax1.set_title(
    "Cost Comparison of Amazon EC2 Instances (Hourly)",
    fontsize=16
)

ax1.set_ylabel(
    "Cost (USD)",
    fontsize=12
)

ax1.tick_params(
    axis="x",
    rotation=45
)

plt.tight_layout()
st.pyplot(fig1)
plt.close(fig1)


# 6. IDENTIFY OUTLIERS
st.subheader("6. On-Demand Cost Outliers")

def detect_outliers(column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    return data[
        (data[column] < lower_bound)
        |
        (data[column] > upper_bound)
    ]

outliers_on_demand = detect_outliers(
    "On Demand"
)

st.metric(
    "Number of On-Demand Outliers",
    len(outliers_on_demand)
)

st.dataframe(
    outliers_on_demand,
    use_container_width=True
)


# 7. COST-EFFECTIVE INSTANCES
st.subheader("7. Top 10 Cost-Effective Instances")

cost_comparison = data[
    [
        "Name",
        "On Demand",
        "Linux Reserved cost"
    ]
].dropna().sort_values(
    "On Demand"
)

st.dataframe(
    cost_comparison.head(10),
    use_container_width=True,
    hide_index=True
)


# 8. FILTER AND COMPARE INSTANCE FAMILIES
st.subheader("8. T2 and T3 Instance Family Comparison")

def filter_instance_family(family):
    return data[
        data["Name"].str.startswith(family)
    ]

t2_instances = filter_instance_family(
    "T2"
)

t3_instances = filter_instance_family(
    "T3"
)

# T2 AND T3 SUMMARY STATISTICS
col1, col2 = st.columns(2)

with col1:
    st.markdown("### T2 Instance Costs Summary")
    t2_summary = t2_instances[
        cost_columns
    ].describe()

    st.dataframe(
        t2_summary,
        use_container_width=True
    )

with col2:
    st.markdown("### T3 Instance Costs Summary")
    t3_summary = t3_instances[
        cost_columns
    ].describe()

    st.dataframe(
        t3_summary,
        use_container_width=True
    )

# T2 AND T3 VISUALIZATIONS
st.markdown("### T2 and T3 Cost Distributions")

graph1, graph2 = st.columns(2)

# T2 GRAPH
with graph1:
    fig2, ax2 = plt.subplots(
        figsize=(10, 6)
    )

    sns.boxplot(
        data=t2_instances[cost_columns],
        palette="Blues",
        showmeans=True,
        ax=ax2
    )

    ax2.set_title(
        "Cost Distribution for T2 Instances"
    )

    ax2.set_ylabel(
        "Cost (USD)"
    )

    ax2.tick_params(
        axis="x",
        rotation=45
    )

    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)


# T3 GRAPH
with graph2:
    fig3, ax3 = plt.subplots(
        figsize=(10, 6)
    )

    sns.boxplot(
        data=t3_instances[cost_columns],
        palette="Greens",
        showmeans=True,
        ax=ax3
    )

    ax3.set_title(
        "Cost Distribution for T3 Instances"
    )

    ax3.set_ylabel(
        "Cost (USD)"
    )

    ax3.tick_params(
        axis="x",
        rotation=45
    )

    plt.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)

# T2 AND T3 LOWEST COST COMPARISON
st.markdown("### 10 Lowest-Cost T2 and T3 Instances")

comparison = pd.concat(
    [
        t2_instances[
            [
                "Name",
                "On Demand",
                "Linux Reserved cost"
            ]
        ],

        t3_instances[
            [
                "Name",
                "On Demand",
                "Linux Reserved cost"
            ]
        ]
    ]
)

comparison_sorted = comparison.dropna().sort_values(
    "On Demand"
)

st.dataframe(
    comparison_sorted.head(10),
    use_container_width=True,
    hide_index=True
)


# =========================================================
# PART II
# =========================================================

st.divider()

st.header(
    "Part II: Predicting EC2 Instance Costs Using Regression Analysis"
)


# LOAD DATASET
data = pd.read_csv(file_path)

# Clean cost columns
for column in cost_columns:
    data[column] = pd.to_numeric(
        data[column].str.replace(
            "[$, hourly]",
            "",
            regex=True
        ),
        errors="coerce"
    )


# 1. FEATURE ENGINEERING
st.subheader("1. Feature Engineering")

# Convert Instance Memory to numeric
data["Instance Memory"] = pd.to_numeric(
    data["Instance Memory"].str.replace(
        " GiB",
        ""
    )
)

# Extract the number of vCPUs
data["vCPUs"] = pd.to_numeric(
    data["vCPUs"].str.extract(
        r"(\d+)",
        expand=False
    )
)

st.write(
    "Converted Instance Memory and vCPU values:"
)

st.dataframe(
    data[
        [
            "Instance Memory",
            "vCPUs"
        ]
    ].head(),
    use_container_width=True,
    hide_index=True
)


# 2. HANDLE MISSING DATA
st.subheader("2. Handle Missing Data")

data_cleaned = data.dropna(
    subset=[
        "On Demand",
        "Instance Memory",
        "vCPUs"
    ]
)

# Only display missing values for the columns used by the regression model.
regression_columns = [
    "On Demand",
    "Instance Memory",
    "vCPUs"
]

missing_cleaned = (
    data_cleaned[regression_columns]
    .isnull()
    .sum()
    .reset_index()
)

missing_cleaned.columns = [
    "Column",
    "Missing Values"
]

st.dataframe(
    missing_cleaned,
    use_container_width=True,
    hide_index=True
)


# 3. SPLIT THE DATA
st.subheader("3. Training and Testing Data")

X = data_cleaned[
    [
        "Instance Memory",
        "vCPUs"
    ]
]

y = data_cleaned[
    "On Demand"
]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

train_col, test_col = st.columns(2)

with train_col:
    st.metric(
        "Training Samples",
        len(X_train)
    )

with test_col:
    st.metric(
        "Testing Samples",
        len(X_test)
    )


# 4. TRAIN LINEAR REGRESSION MODEL
st.subheader("4. Linear Regression Model")

model = LinearRegression()

model.fit(
    X_train,
    y_train
)

model_col1, model_col2, model_col3 = st.columns(3)

with model_col1:
    st.metric(
        "Intercept",
        f"{model.intercept_:.4f}"
    )

with model_col2:
    st.metric(
        "Memory Coefficient",
        f"{model.coef_[0]:.4f}"
    )

with model_col3:
    st.metric(
        "vCPU Coefficient",
        f"{model.coef_[1]:.4f}"
    )


# 5. EVALUATE THE MODEL
st.subheader("5. Model Evaluation")

y_pred = model.predict(
    X_test
)

mae = mean_absolute_error(
    y_test,
    y_pred
)

mse = mean_squared_error(
    y_test,
    y_pred
)

rmse = mse ** 0.5

metric1, metric2, metric3 = st.columns(3)

with metric1:
    st.metric(
        "Mean Absolute Error (MAE)",
        f"{mae:.4f}"
    )

with metric2:
    st.metric(
        "Mean Squared Error (MSE)",
        f"{mse:.4f}"
    )

with metric3:
    st.metric(
        "Root Mean Squared Error (RMSE)",
        f"{rmse:.4f}"
    )


# 6. VISUALIZE ACTUAL VS PREDICTED COSTS
st.subheader(
    "6. Actual vs Predicted On-Demand Costs"
)

fig4, ax4 = plt.subplots(
    figsize=(9, 6)
)

ax4.scatter(
    y_test,
    y_pred,
    alpha=0.7
)

ax4.plot(
    [
        min(y_test),
        max(y_test)
    ],

    [
        min(y_test),
        max(y_test)
    ],

    linestyle="--"
)

ax4.set_title(
    "Actual vs Predicted On-Demand Costs"
)

ax4.set_xlabel(
    "Actual On-Demand Cost"
)

ax4.set_ylabel(
    "Predicted On-Demand Cost"
)

plt.tight_layout()
st.pyplot(fig4)
plt.close(fig4)


# 7. MAKE A PREDICTION
st.subheader("7. Example Prediction")

# Example from the activity:
# 4 GiB memory and 2 vCPUs
new_instance = pd.DataFrame(
    [
        [4, 2]
    ],
    columns=[
        "Instance Memory",
        "vCPUs"
    ]
)

predicted_cost = model.predict(
    new_instance
)

st.metric(
    "Predicted On-Demand Cost",
    f"${predicted_cost[0]:.4f} per hour"
)