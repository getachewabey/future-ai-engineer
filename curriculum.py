
# AION Coach Curriculum Data

# Helper to generate common days
def integration_day(day_num, phase_topic):
    return {
        "day": day_num,
        "title": f"Integration Day ({phase_topic})",
        "phase": "Integration",
        "goal": "Combine this week's concepts into the AION Trainer app.",
        "concepts": ["Refactoring", "Integration Testing", "UI Polish"],
        "build": [
            "Review code from the last 5 days.",
            "Merge feature branches into main.",
            "Update the Streamlit interface to reflect new capabilities.",
            "Run full regression tests."
        ],
        "deliverables": ["Merged PR", "Updated App Deployment"],
        "prompts": ["How do I refactor this for better modularity?", "Write integration tests for these components."],
        "streamlit_integration": "Update Dashboard and Sidebar.",
        "quiz": [],
        "exit_criteria": "App runs without errors on main branch.",
        "time": "120+ minutes"
    }

def review_day(day_num, phase_topic):
    return {
        "day": day_num,
        "title": f"Review & catch-up ({phase_topic})",
        "phase": "Review",
        "goal": "Solidify knowledge, fix bugs, and prepare for next week.",
        "concepts": ["Code Review", "Documentation", "Optimization"],
        "build": [
            "Complete any unfinished tasks.",
            "Improve documentation (docstrings, README).",
            "Optimize simplest slow function.",
            "Read one external article on this week's topic."
        ],
        "deliverables": ["Updated README", "Blog post draft (optional)"],
        "prompts": ["Explain this concept to me like I'm 5.", "Review my project structure."],
        "streamlit_integration": "Add 'Weekly Reflection' to progress tracker.",
        "quiz": [{"question": "What was the hardest concept this week?", "type": "open"}],
        "exit_criteria": "All tests pass, ready for next week.",
        "time": "60 minutes"
    }

CURRICULUM = [
    # --- PHASE 1: FOUNDATIONS (Days 1-28) ---
    # Week 1: Environment & Python Basics
    {
        "day": 1,
        "title": "Environment & First Streamlit Skeleton",
        "phase": "Phase 1: Foundations",
        "goal": "Install toolchain (VS Code, Python, Git) and launch your first 'Hello World' Streamlit app.",
        "concepts": ["venv management", "pip/uv", "Streamlit basics", ".env Secrets"],
        "build": [
            "Install Python 3.11+ and git.",
            "Initialize project folder `future-ai-engineer`.",
            "Create venv: `python -m venv .venv` and activate it.",
            "Install: `pip install streamlit google-generativeai python-dotenv`.",
            "Create `app.py`: Display 'Hello World' and a sidebar input for 'Name'.",
            "Secure it: Create `.env` and add a dummy key `MY_SECRET=123`. Print it in app to verify (then remove)."
        ],
        "deliverables": ["Screenshot of running app (localhost:8501)", "Requirements.txt file"],
        "prompts": ["Act as code reviewer: check my venv setup.", "Explain why we use .env files."],
        "streamlit_integration": "Create the main 'Home' page. Input your name and have the app greet you.",
        "quiz": [{"question": "Which file should you ALWAYS add to .gitignore?", "options": [".venv", "app.py", "requirements.txt"], "answer": ".venv"}],
        "exit_criteria": "App runs locally without errors.",
        "time": "60 minutes"
    },
    {
        "day": 2,
        "title": "Git Workflow & Repo Structure",
        "phase": "Phase 1: Foundations",
        "goal": "Professionalize your setup. Treat code as a product.",
        "concepts": ["git init/commit", "Branching Strategy", ".gitignore patterns"],
        "build": [
            "Initialize git in your folder.",
            "Create `.gitignore` (exclude .venv, .env, __pycache__).",
            "Commit Day 1 code to `main`.",
            "Create a NEW branch `feature/day2-structure`.",
            "Restructure: Move `app.py` to root, create `src/` and `tests/` folders.",
            "Push to GitHub (optional but recommended)."
        ],
        "deliverables": ["GitHub Repo Link", "Clean `git status` output"],
        "prompts": ["Write a semantic commit message for adding .gitignore."],
        "streamlit_integration": "Add a 'Version' badge in the sidebar (e.g., v0.1.0).",
        "quiz": [{"question": "Why use feature branches?", "options": ["Isolation", "Speed", "Security"], "answer": "Isolation"}],
        "exit_criteria": "Repo checks out cleanly; app still runs after move.",
        "time": "60 minutes"
    },
    {
        "day": 3,
        "title": "Python Essentials I - Modular Functions",
        "phase": "Phase 1: Foundations",
        "goal": "Stop writing scripts; start writing modules.",
        "concepts": ["Type Hinting", "Docstrings (Google Style)", "Importing local modules"],
        "build": [
            "Create `utils.py` in the root.",
            "Write a function `calculate_growth(start: float, rate: float) -> float`.",
            "Add strict type hints and a docstring explaining args/returns.",
            "IMPORT this function into `app.py`.",
            "Create a 'Playground' page in Streamlit to test this function with sliders."
        ],
        "deliverables": ["utils.py file", "Screenshot of Playground page working"],
        "prompts": ["Generate a docstring for this function."],
        "streamlit_integration": "New Page: 'Playground'. Use sliders to input `start` and `rate`, display result.",
        "quiz": [{"question": "What does `-> float` denote?", "type": "open"}],
        "exit_criteria": "App successfully imports and uses utils.py.",
        "time": "90 minutes"
    },
    {
        "day": 4,
        "title": "Python Essentials II - Data Structures & Classes",
        "phase": "Phase 1: Foundations",
        "goal": "Manage state with Classes and Lists.",
        "concepts": ["Classes vs Dicts", "List Comprehensions", "Session State Basics"],
        "build": [
            "Create `todo.py` module.",
            "Implement a `TaskManager` class with methods `add_task(task)`, `get_pending_tasks()`.",
            "Integrate into `app.py`: Create a 'Task Tracker' page.",
            "Use `st.session_state` to keep the TaskManager instance alive across reruns."
        ],
        "deliverables": ["todo.py with Class", "Working Task List in App"],
        "prompts": ["Explain why we need session_state here."],
        "streamlit_integration": "Task page: Input box to add task, list below to show them.",
        "quiz": [],
        "exit_criteria": "Tasks persist when I click buttons.",
        "time": "90 minutes"
    },
    {
        "day": 5,
        "title": "Debugging & Logging",
        "phase": "Phase 1: Foundations",
        "goal": "Visibility into your application.",
        "concepts": ["Python logging module", "Streamlit error handling", "Try/Except blocks"],
        "build": [
            "Configure a standardized logger in `app.py`.",
            "Add `logger.info()` calls in your `utils.py` and `todo.py` methods.",
            "Add a deliberate bug (e.g., division by zero) in a new function.",
            "Wrap it in a `try/except` block and log the error."
        ],
        "deliverables": ["Console output showing logs", "App handling error gracefully"],
        "prompts": ["How to configure logging to file and console?"],
        "streamlit_integration": "Display 'System Status: Healthy' unless an error was caught.",
        "quiz": [],
        "exit_criteria": "Logs appear in terminal when using the app.",
        "time": "60 minutes"
    },
    integration_day(6, "Python Foundations"),
    review_day(7, "Python Foundations"),

    # Week 2: Data Handling (Numpy/Pandas) & Testing
    {
        "day": 8,
        "title": "Numpy Essentials",
        "phase": "Phase 1: Foundations",
        "goal": "Vectorized operations over loops.",
        "concepts": ["Arrays", "Broadcasting", "Vectorization"],
        "build": [
            "Create specific `data_analysis.py`.",
            "Generate random dataset using numpy.",
            "Perform matrix multiplication and stats."
        ],
        "deliverables": ["Numpy script", "Performance check"],
        "prompts": ["Explain broadcasting in Numpy."],
        "streamlit_integration": "Visualize numpy array data (st.dataframe).",
        "quiz": [],
        "exit_criteria": "Zero loops used for array math.",
        "time": "90 minutes"
    },
    {
        "day": 9,
        "title": "Pandas DataFrames",
        "phase": "Phase 1: Foundations",
        "goal": "Tabular data mastery.",
        "concepts": ["DataFrame", "Series", "read_csv", "groupby"],
        "build": [
            "Load a CSV (e.g., Titanic or Iris).",
            "Clean missing values.",
            "Group by and aggregate stats."
        ],
        "deliverables": ["Cleaned CSV", "Aggregation report"],
        "prompts": ["Pandas groupby explanation."],
        "streamlit_integration": "Interactive table with sorting/filtering.",
        "quiz": [],
        "exit_criteria": "Data loaded and displayed.",
        "time": "90 minutes"
    },
    {
        "day": 10,
        "title": "Data Visualization",
        "phase": "Phase 1: Foundations",
        "goal": "Tell stories with data.",
        "concepts": ["Matplotlib", "Plotly", "Streamlit charts"],
        "build": [
            "Visualize the data from Day 9.",
            "Create Bar, Line, and Scatter plots.",
            "Use Plotly for interactive charts."
        ],
        "deliverables": ["3 distinct charts"],
        "prompts": ["Matplotlib vs Plotly pros/cons."],
        "streamlit_integration": "Embed Plotly charts in app.",
        "quiz": [],
        "exit_criteria": "Charts render correctly.",
        "time": "90 minutes"
    },
    {
        "day": 11,
        "title": "Testing with Pytest",
        "phase": "Phase 1: Foundations",
        "goal": "Write tests before/during code.",
        "concepts": ["Fixtures", "Asserts", "Parametrization"],
        "build": [
            "Install `pytest`.",
            "Write tests for `utils.py` and `data_analysis.py`.",
            "Use a fixture for data loading."
        ],
        "deliverables": ["tests/ folder", "passing test suite"],
        "prompts": ["Generate pytest cases for this function."],
        "streamlit_integration": "Display 'Last Test Run' status (mocked or real).",
        "quiz": [],
        "exit_criteria": "All tests pass.",
        "time": "90 minutes"
    },
    {
        "day": 12,
        "title": "Code Quality (Linting)",
        "phase": "Phase 1: Foundations",
        "goal": "Enforce cleaner code automatically.",
        "concepts": ["Black (formatter)", "Ruff (linter)", "Pre-commit"],
        "build": [
            "Install `black` and `ruff`.",
            "Run them on the entire repo.",
            "Add a `.pre-commit-config.yaml` (optional/advanced)."
        ],
        "deliverables": ["Formatted code", "Zero lint errors"],
        "prompts": ["Why use a linter?"],
        "streamlit_integration": "Add a 'Code Audit' button that runs ruff on input text.",
        "quiz": [],
        "exit_criteria": "Repo clean.",
        "time": "60 minutes"
    },
    integration_day(13, "Data & Testing"),
    review_day(14, "Data & Testing"),

    # Week 3: Math & Algorithms for ML
    {
        "day": 15,
        "title": "Linear Algebra Basics",
        "phase": "Phase 1: Foundations",
        "goal": "Understand vectors and matrices intuitively.",
        "concepts": ["Dot product", "Matrix Multiplication", "Eigenvalues (basic)"],
        "build": [
            "Implement dot product from scratch (python list).",
            "Verify with Numpy.",
            "Visualize a vector transformation."
        ],
        "deliverables": ["Math notebook/script"],
        "prompts": ["Explain dot product geometrically."],
        "streamlit_integration": "Vector viz page.",
        "quiz": [],
        "exit_criteria": "Correct implementation.",
        "time": "90 minutes"
    },
    {
        "day": 16,
        "title": "Calculus: Gradients",
        "phase": "Phase 1: Foundations",
        "goal": "Understand how models learn.",
        "concepts": ["Derivative", "Partial Derivative", "Gradient Descent (concept)"],
        "build": [
            "Implement numerical differentiation (finite difference).",
            "Simulate gradient descent on `y = x^2`."
        ],
        "deliverables": ["Gradient descent simulation script"],
        "prompts": ["Explain gradient descent to a beginner."],
        "streamlit_integration": "Animate gradient descent steps.",
        "quiz": [],
        "exit_criteria": "Simulation converges to minimum.",
        "time": "90 minutes"
    },
    {
        "day": 17,
        "title": "Probability & Stats",
        "phase": "Phase 1: Foundations",
        "goal": "Understand distributions.",
        "concepts": ["Mean/Median/Mode", "Standard Deviation", "Normal Distribution"],
        "build": [
            "Generate distributions using Numpy.",
            "Plot histograms.",
            "Calculate z-scores."
        ],
        "deliverables": ["Stats visualization"],
        "prompts": ["What is a normal distribution?"],
        "streamlit_integration": "Distribution explorer widget.",
        "quiz": [],
        "exit_criteria": "Stats calculated correctly.",
        "time": "90 minutes"
    },
    {
        "day": 18,
        "title": "APIs & JSON",
        "phase": "Phase 1: Foundations",
        "goal": "Fetch data from the web (real world data).",
        "concepts": ["REST APIs", "JSON parsing", "Requests/Httpx"],
        "build": [
            "Fetch data from a public API (e.g., Weather, Crypto, Joke).",
            "Parse JSON response.",
            "Display data."
        ],
        "deliverables": ["Data fetcher script"],
        "prompts": ["How to handle API rate limits?"],
        "streamlit_integration": "Live data dashboard.",
        "quiz": [],
        "exit_criteria": "Live data visible in app.",
        "time": "60 minutes"
    },
    {
        "day": 19,
        "title": "SQL Basics (SQLite)",
        "phase": "Phase 1: Foundations",
        "goal": "Persistent data storage.",
        "concepts": ["SELECT", "INSERT", "JOIN", "SQLite"],
        "build": [
            "Create a SQLite DB for the AION App (store quiz scores).",
            "Write a script to insert/retrieve scores."
        ],
        "deliverables": ["database.db", "db_manager.py"],
        "prompts": ["Write a SQL query to find top scores."],
        "streamlit_integration": "Show quiz history from DB.",
        "quiz": [],
        "exit_criteria": "Data persists after restart.",
        "time": "90 minutes"
    },
    integration_day(20, "Math & Data"),
    review_day(21, "Math & Data"),

    # Week 4: Scikit-Learn First Steps
    {
        "day": 22,
        "title": "Intro to ML & Scikit-Learn",
        "phase": "Phase 1: Foundations",
        "goal": "Train your first model.",
        "concepts": ["Supervised Learning", "Fit/Predict API", "Train/Test Split"],
        "build": [
            "Load Iris dataset.",
            "Split data.",
            "Train a Logistic Regression model.",
            "Evaluate accuracy."
        ],
        "deliverables": ["ml_basics.py", "Accuracy score"],
        "prompts": ["Explain train/test split."],
        "streamlit_integration": "Button: 'Train Model' -> show accuracy.",
        "quiz": [],
        "exit_criteria": "Model accuracy > 80%.",
        "time": "90 minutes"
    },
    {
        "day": 23,
        "title": "Linear Regression (Prediction)",
        "phase": "Phase 1: Foundations",
        "goal": "Predict continuous values.",
        "concepts": ["MSE", "Coefficients", "Linear Relationship"],
        "build": [
            "Generate synthetic linear data.",
            "Train LinearRegression.",
            "Plot regression line over data."
        ],
        "deliverables": ["Regression Plot"],
        "prompts": ["Difference between classification and regression."],
        "streamlit_integration": "Interactive regression demo (tweak noise).",
        "quiz": [],
        "exit_criteria": "Line fits data.",
        "time": "90 minutes"
    },
    {
        "day": 24,
        "title": "Classification (KNN/Logistic)",
        "phase": "Phase 1: Foundations",
        "goal": "Classify distinct categories.",
        "concepts": ["Decision Boundaries", "K-Nearest Neighbors"],
        "build": [
            "Use a more complex dataset (Wine or Breast Cancer).",
            "Compare KNN vs Logistic Regression."
        ],
        "deliverables": ["Comparison Report"],
        "prompts": ["How does KNN work?"],
        "streamlit_integration": "Model selector dropdown.",
        "quiz": [],
        "exit_criteria": "Two models trained.",
        "time": "90 minutes"
    },
    {
        "day": 25,
        "title": "Model Evaluation Metrics",
        "phase": "Phase 1: Foundations",
        "goal": "Beyond accuracy.",
        "concepts": ["Confusion Matrix", "Precision/Recall", "F1 Score"],
        "build": [
            "Calculate Precision/Recall for Day 24 models.",
            "Plot Confusion Matrix."
        ],
        "deliverables": ["Evaluation plots"],
        "prompts": ["Explain Precision vs Recall."],
        "streamlit_integration": "Display confusion matrix.",
        "quiz": [],
        "exit_criteria": "Metrics displayed.",
        "time": "90 minutes"
    },
    {
        "day": 26,
        "title": "Saving & Loading Models",
        "phase": "Phase 1: Foundations",
        "goal": "Persistence for ML.",
        "concepts": ["Pickle", "Joblib", "Model Inference"],
        "build": [
            "Save the best trained model using `joblib`.",
            "Write a separate script to load and predict."
        ],
        "deliverables": ["model.joblib", "inference_script.py"],
        "prompts": ["Risks of using pickle."],
        "streamlit_integration": "Upload CSV -> Load Model -> Download Predictions.",
        "quiz": [],
        "exit_criteria": "Model loads and predicts correctly.",
        "time": "60 minutes"
    },
    integration_day(27, "Scikit-Learn Basics"),
    review_day(28, "Phase 1 Complete"),

    # --- PHASE 2: ML ENGINEERING CORE (Days 29-56) ---
    # Week 5: Feature Engineering
    {
        "day": 29,
        "title": "Data Preprocessing",
        "phase": "Phase 2: ML Core",
        "goal": "Garbage in, garbage out - fix the data.",
        "concepts": ["Scaling (Standard/MinMax)", "Imputation", "Encoding"],
        "build": [
            "Create a dirty dataset.",
            "Apply StandardScaler.",
            "One-hot encode categorical vars."
        ],
        "deliverables": ["preprocessing.py"],
        "prompts": ["Why scale data?"],
        "streamlit_integration": "Data cleaner widget.",
        "quiz": [],
        "exit_criteria": "Data ready for training.",
        "time": "90 minutes"
    },
    {
        "day": 30,
        "title": "Pipeline Objects",
        "phase": "Phase 2: ML Core",
        "goal": "Automate the workflow.",
        "concepts": ["sklearn.pipeline.Pipeline", "ColumnTransformer"],
        "build": [
            "Chain scaler and model into a Pipeline.",
            "Train the pipeline as a single object."
        ],
        "deliverables": ["pipeline_script.py"],
        "prompts": ["Benefits of ML pipelines."],
        "streamlit_integration": "Show pipeline steps viz.",
        "quiz": [],
        "exit_criteria": "Pipeline trains successfully.",
        "time": "90 minutes"
    },
    {
        "day": 31,
        "title": "Cross-Validation",
        "phase": "Phase 2: ML Core",
        "goal": "Robust performance estimation.",
        "concepts": ["K-Fold", "Stratified K-Fold"],
        "build": [
            "Implement K-Fold CV on previous pipeline.",
            "Report mean and std of scores."
        ],
        "deliverables": ["CV report"],
        "prompts": ["Why is CV better than simple split?"],
        "streamlit_integration": "Show CV results stability.",
        "quiz": [],
        "exit_criteria": "CV runs without error.",
        "time": "90 minutes"
    },
    {
        "day": 32,
        "title": "Hyperparameter Tuning",
        "phase": "Phase 2: ML Core",
        "goal": "Optimize model performance.",
        "concepts": ["GridSearchCV", "RandomizedSearchCV"],
        "build": [
            "Tune hyperparameters for the pipeline.",
            "Find best params."
        ],
        "deliverables": ["Optimized model"],
        "prompts": ["Grid vs Random Search."],
        "streamlit_integration": "Tuning progress bar.",
        "quiz": [],
        "exit_criteria": "Found better parameters.",
        "time": "90 minutes"
    },
    {
        "day": 33,
        "title": "Feature Selection",
        "phase": "Phase 2: ML Core",
        "goal": "Less is more.",
        "concepts": ["VarianceThreshold", "SelectKBest", "RFE"],
        "build": [
            "Apply feature selection to a high-dim dataset.",
            "Evaluate impact on speed/accuracy."
        ],
        "deliverables": ["Feature selection report"],
        "prompts": ["Curse of dimensionality."],
        "streamlit_integration": "Feature importance chart.",
        "quiz": [],
        "exit_criteria": "Reduced feature set.",
        "time": "90 minutes"
    },
    integration_day(34, "Feature Eng & Tuning"),
    review_day(35, "Feature Eng & Tuning"),

    # Week 6: Advanced Classical ML
    {
        "day": 36,
        "title": "Decision Trees & Forests",
        "phase": "Phase 2: ML Core",
        "goal": "Ensemble learning.",
        "concepts": ["RandomForest", "Entropy/Gini", "Overfitting"],
        "build": [
            "Train a Decision Tree (visualize it).",
            "Train a Random Forest."
        ],
        "deliverables": ["Tree visualization"],
        "prompts": ["How does Random Forest reduce variance?"],
        "streamlit_integration": "Interactive tree depth slider.",
        "quiz": [],
        "exit_criteria": "RF outperforms single tree.",
        "time": "90 minutes"
    },
    {
        "day": 37,
        "title": "Gradient Boosting (XGBoost/LightGBM)",
        "phase": "Phase 2: ML Core",
        "goal": "State of the art tabular ML.",
        "concepts": ["Boosting", "XGBoost basics"],
        "build": [
            "Install XGBoost.",
            "Train XGBClassifier.",
            "Compare with Random Forest."
        ],
        "deliverables": ["XGBoost model"],
        "prompts": ["Bagging vs Boosting."],
        "streamlit_integration": "Leaderboard of models.",
        "quiz": [],
        "exit_criteria": "XGBoost trained.",
        "time": "90 minutes"
    },
    {
        "day": 38,
        "title": "Unsupervised Learning: Clustering",
        "phase": "Phase 2: ML Core",
        "goal": "Find structure without labels.",
        "concepts": ["K-Means", "Elbow Method", "Silhouette Score"],
        "build": [
            "Cluster Iris dataset (ignore labels).",
            "Determine optimal K."
        ],
        "deliverables": ["Clustering plots"],
        "prompts": ["What is K-Means?"],
        "streamlit_integration": "Cluster visualizer (2D PCA).",
        "quiz": [],
        "exit_criteria": "Clusters form sensibly.",
        "time": "90 minutes"
    },
    {
        "day": 39,
        "title": "Dimensionality Reduction: PCA",
        "phase": "Phase 2: ML Core",
        "goal": "Visualize high-D data.",
        "concepts": ["PCA", "Variance Explained"],
        "build": [
            "Apply PCA to reduce specific dataset to 2D.",
            "Plot the result."
        ],
        "deliverables": ["PCA plot"],
        "prompts": ["Explain PCA."],
        "streamlit_integration": "3D PCA scatter plot (Plotly).",
        "quiz": [],
        "exit_criteria": "2D projection visible.",
        "time": "90 minutes"
    },
    {
        "day": 40,
        "title": "Anomaly Detection",
        "phase": "Phase 2: ML Core",
        "goal": "Find the outliers.",
        "concepts": ["Isolation Forest", "Outlier detection"],
        "build": [
            "Inject anomalies into dataset.",
            "Use Isolation Forest to detect them."
        ],
        "deliverables": ["Anomaly report"],
        "prompts": ["Use cases for anomaly detection."],
        "streamlit_integration": "Highlight outliers in table.",
        "quiz": [],
        "exit_criteria": "Outliers formatted red.",
        "time": "90 minutes"
    },
    integration_day(41, "Advanced ML"),
    review_day(42, "Advanced ML"),

    # Week 7: Classical ML Project (Month 2 Capstone)
    {
        "day": 43,
        "title": "Project Scoping & Data",
        "phase": "Phase 2: ML Core",
        "goal": "Start the Month 2 Portfolio Project.",
        "concepts": ["Problem Definition", "Data sourcing (Kaggle/API)"],
        "build": [
            "Select a dataset (Housing, Churn, or Credit).",
            "Define business metric (e.g., reduce churn by 5%).",
            "Initialize project structure in `projects/month2_ml`."
        ],
        "deliverables": ["Project Proposal", "Data Profile"],
        "prompts": ["Help me scope this ML project."],
        "streamlit_integration": "Project Tracker: Status 'InProgress'.",
        "quiz": [],
        "exit_criteria": "Scope defined.",
        "time": "120 minutes"
    },
    {
        "day": 44,
        "title": "Deep Dive EDA",
        "phase": "Phase 2: ML Core",
        "goal": "Understand the data fully.",
        "concepts": ["Correlation", "Distribution Analysis", "Pairplots"],
        "build": [
            "Generate comprehensive EDA notebook.",
            "Identify target leakage or bias."
        ],
        "deliverables": ["EDA Notebook"],
        "prompts": ["What to look for in EDA?"],
        "streamlit_integration": "Embed EDA plots in project page.",
        "quiz": [],
        "exit_criteria": "Hypotheses formed.",
        "time": "120 minutes"
    },
    {
        "day": 45,
        "title": "Training & Tuning",
        "phase": "Phase 2: ML Core",
        "goal": "Build the best model.",
        "concepts": ["Model Selection", "Hyperparameter Optimization"],
        "build": [
            "Train baseline model.",
            "Try 3 different algorithms.",
            "Tune the best one."
        ],
        "deliverables": ["Trained Model Artifacts"],
        "prompts": ["Strategy for model selection."],
        "streamlit_integration": "Live training log (optional).",
        "quiz": [],
        "exit_criteria": "Best model selected.",
        "time": "120 minutes"
    },
    {
        "day": 46,
        "title": "Evaluation & Reporting",
        "phase": "Phase 2: ML Core",
        "goal": "Translate metrics to business value.",
        "concepts": ["ROC/AUC", "Business Impact Analysis"],
        "build": [
            "Create final evaluation report.",
            "Translate 'Accuracy' to 'Revenue Saved' (hypothetical)."
        ],
        "deliverables": ["Final Report"],
        "prompts": ["Explain ROC AUC to business stakeholder."],
        "streamlit_integration": "Business Metrics Dashboard.",
        "quiz": [],
        "exit_criteria": "Report ready.",
        "time": "120 minutes"
    },
    {
        "day": 47,
        "title": "Deployment (Local)",
        "phase": "Phase 2: ML Core",
        "goal": "Serve the model.",
        "concepts": ["Streamlit frontend for Model", "Inference"],
        "build": [
            "Build a dedicated Streamlit app for this project.",
            "Allow user input -> predict."
        ],
        "deliverables": ["Working App"],
        "prompts": ["Best practices for ML UI."],
        "streamlit_integration": "Link Month 2 project in main app.",
        "quiz": [],
        "exit_criteria": "App functional.",
        "time": "120 minutes"
    },
    integration_day(48, "Month 2 Project"),
    review_day(49, "Month 2 Project"),

    # Week 8: MLOps Foundations
    {
        "day": 50,
        "title": "Reproducibility",
        "phase": "Phase 2: ML Core",
        "goal": "Ensure results are consistent.",
        "concepts": ["Random Seeds", "Environment.yml", "Docker basics"],
        "build": [
            "Fix all random seeds in `train.py`.",
            "Export exact environment.",
            "Write a `Reproducibility.md` guide."
        ],
        "deliverables": ["Reproducible repo"],
        "prompts": ["Why does my model change every run?"],
        "streamlit_integration": "Display env info in Admin.",
        "quiz": [],
        "exit_criteria": "Re-run gives exact same score.",
        "time": "90 minutes"
    },
    {
        "day": 51,
        "title": "Experiment Tracking",
        "phase": "Phase 2: ML Core",
        "goal": "Stop using spreadsheets for results.",
        "concepts": ["MLflow (intro)", "Weights & Biases (intro)"],
        "build": [
            "Sign up for free WandB or setup local MLflow.",
            "Log params and metrics from the Month 2 project."
        ],
        "deliverables": ["WandB/MLflow dashboard link"],
        "prompts": ["Metric logging best practices."],
        "streamlit_integration": "Link to experiment dashboard.",
        "quiz": [],
        "exit_criteria": "Experiments logged.",
        "time": "90 minutes"
    },
    {
        "day": 52,
        "title": "Model Registry Concept",
        "phase": "Phase 2: ML Core",
        "goal": "Version control for models.",
        "concepts": ["Versioning", "Staging/Prod tags"],
        "build": [
            "Simulate a registry: folder structure `models/v1`, `models/v2`.",
            "Write specific metadata for each version."
        ],
        "deliverables": ["Model versioning system"],
        "prompts": ["Why version models?"],
        "streamlit_integration": "Model selector picks specific version.",
        "quiz": [],
        "exit_criteria": "Multiple versions manageable.",
        "time": "60 minutes"
    },
    {
        "day": 53,
        "title": "Serving with FastAPI",
        "phase": "Phase 2: ML Core",
        "goal": "Decouple model from UI.",
        "concepts": ["REST API", "FastAPI", "Pydantic"],
        "build": [
            "Create `api.py` with FastAPI.",
            "Endpoint `/predict`.",
            "Call this API from Streamlit (instead of direct import)."
        ],
        "deliverables": ["api.py"],
        "prompts": ["FastAPI vs Flask for ML."],
        "streamlit_integration": "App now calls localhost:8000.",
        "quiz": [],
        "exit_criteria": "API valid response.",
        "time": "120 minutes"
    },
    {
        "day": 54,
        "title": "Dockerizing",
        "phase": "Phase 2: ML Core",
        "goal": "It works on my machine -> It works everywhere.",
        "concepts": ["Dockerfile", "Image", "Container"],
        "build": [
            "Write Dockerfile for the API.",
            "Build and run container.",
        ],
        "deliverables": ["Dockerfile"],
        "prompts": ["Explain Dockerfile commands."],
        "streamlit_integration": "N/A (Backend task).",
        "quiz": [],
        "exit_criteria": "Container runs API.",
        "time": "120 minutes"
    },
    integration_day(55, "MLOps Basics"),
    review_day(56, "Phase 2 Complete"),

    # --- PHASE 3: DEEP LEARNING & NLP (Days 57-98) ---
    # Week 9: PyTorch Basics
    {
        "day": 57,
        "title": "Tensors & Operations",
        "phase": "Phase 3: Deep Learning",
        "goal": "Numpy, but on GPU.",
        "concepts": ["Torch Tensors", "GPU/CUDA", "Reshaping"],
        "build": [
            "Install PyTorch.",
            "Do tensor math (matmul, reshape).",
            "Move tensors to/from GPU (if available) or check availability."
        ],
        "deliverables": ["torch_basics.py"],
        "prompts": ["Torch vs Numpy differences."],
        "streamlit_integration": "Show 'CUDA Available' status.",
        "quiz": [],
        "exit_criteria": "Torch installed.",
        "time": "90 minutes"
    },
    {
        "day": 58,
        "title": "Autograd",
        "phase": "Phase 3: Deep Learning",
        "goal": "Automatic differentiation magic.",
        "concepts": ["Computational Graph", ".backward()", "Gradients"],
        "build": [
            "Create a tensor with `requires_grad=True`.",
            "Compute a function.",
            "Backpropagate and inspect grads."
        ],
        "deliverables": ["autograd_demo.py"],
        "prompts": ["How does Autograd work?"],
        "streamlit_integration": "Viz gradients (optional).",
        "quiz": [],
        "exit_criteria": "Gradients computed.",
        "time": "90 minutes"
    },
    {
        "day": 59,
        "title": "The Training Loop",
        "phase": "Phase 3: Deep Learning",
        "goal": "The heartbeat of DL.",
        "concepts": ["Forward", "Loss", "Backward", "Step"],
        "build": [
            "Implement a manual training loop for Linear Regression in Torch.",
            "Visualize loss decreasing."
        ],
        "deliverables": ["manual_training.py"],
        "prompts": ["Write a standard training loop skeleton."],
        "streamlit_integration": "Live loss plot.",
        "quiz": [],
        "exit_criteria": "Loss converges.",
        "time": "90 minutes"
    },
    {
        "day": 60,
        "title": "Datasets & DataLoaders",
        "phase": "Phase 3: Deep Learning",
        "goal": "Efficient data batiching.",
        "concepts": ["Dataset Class", "DataLoader", "Batches"],
        "build": [
            "Create a custom Dataset class.",
            "Wrap it in a DataLoader.",
            "Iterate through batches."
        ],
        "deliverables": ["data_loader.py"],
        "prompts": ["Why use DataLoaders?"],
        "streamlit_integration": "Visualize a batch.",
        "quiz": [],
        "exit_criteria": "Batches yield correctly.",
        "time": "90 minutes"
    },
    {
        "day": 61,
        "title": "First Neural Net (MLP)",
        "phase": "Phase 3: Deep Learning",
        "goal": "Beyond linear lines.",
        "concepts": ["nn.Module", "Linear Layers", "ReLU"],
        "build": [
            "Define a simple MLP class.",
            "Train on MNIST (or simple digits dataset).",
        ],
        "deliverables": ["mlp_mnist.py"],
        "prompts": ["Explain ReLU."],
        "streamlit_integration": "Digit recognizer UI.",
        "quiz": [],
        "exit_criteria": "Training works.",
        "time": "120 minutes"
    },
    integration_day(62, "PyTorch Basics"),
    review_day(63, "PyTorch Basics"),

    # Week 10: NN Fundamentals
    {
        "day": 64,
        "title": "Activation & Loss Functions",
        "phase": "Phase 3: Deep Learning",
        "goal": "Designing the architecture.",
        "concepts": ["Sigmoid/Tanh/ReLU", "CrossEntropy", "MSE"],
        "build": [
            "Compare activations.",
            "Implement Softmax from scratch.",
        ],
        "deliverables": ["activation_plot.py"],
        "prompts": ["Effect of vanishing gradients."],
        "streamlit_integration": "Activation function explorer.",
        "quiz": [],
        "exit_criteria": "Softmax understood.",
        "time": "90 minutes"
    },
    {
        "day": 65,
        "title": "Optimizers (SGD, Adam)",
        "phase": "Phase 3: Deep Learning",
        "goal": "Training faster and better.",
        "concepts": ["Learning Rate", "Momentum", "Adam"],
        "build": [
            "Train same model with SGD vs Adam.",
            "Compare convergence speed."
        ],
        "deliverables": ["Optimizer comparison plot"],
        "prompts": ["Adam vs SGD."],
        "streamlit_integration": "Optimizer selector.",
        "quiz": [],
        "exit_criteria": "Adam wins (usually).",
        "time": "90 minutes"
    },
    {
        "day": 66,
        "title": "Regularization",
        "phase": "Phase 3: Deep Learning",
        "goal": "Prevent overfitting.",
        "concepts": ["Dropout", "Weight Decay (L2)", "BatchNorm"],
        "build": [
            "Add Dropout to Day 61 model.",
            "Observe test accuracy change."
        ],
        "deliverables": ["Regularized model"],
        "prompts": ["When to use Dropout?"],
        "streamlit_integration": "Toggle regularization in UI.",
        "quiz": [],
        "exit_criteria": "Model generalizes better.",
        "time": "90 minutes"
    },
    {
        "day": 67,
        "title": "Visualizing Training",
        "phase": "Phase 3: Deep Learning",
        "goal": "See what's happening.",
        "concepts": ["TensorBoard", "Matplotlib (dynamic)"],
        "build": [
            "Integrate TensorBoard (or simple plot updating).",
            "Monitor Loss and Accuracy."
        ],
        "deliverables": ["Training logs"],
        "prompts": ["How to read loss curves?"],
        "streamlit_integration": "Embed TensorBoard (if possible) or plots.",
        "quiz": [],
        "exit_criteria": "Real-time graphs.",
        "time": "90 minutes"
    },
    {
        "day": 68,
        "title": "Checkpoints",
        "phase": "Phase 3: Deep Learning",
        "goal": "Don't lose progress.",
        "concepts": ["torch.save", "torch.load", "State Dict"],
        "build": [
            "Implement 'Save Best Model' callback.",
            "Resume training from checkpoint."
        ],
        "deliverables": ["checkpoint.pth"],
        "prompts": ["What is state_dict?"],
        "streamlit_integration": "Load checkpoint button.",
        "quiz": [],
        "exit_criteria": "Training resumes correctly.",
        "time": "90 minutes"
    },
    integration_day(69, "NN Fundamentals"),
    review_day(70, "NN Fundamentals"),

    # Week 11: Computer Vision (CNNs)
    {
        "day": 71,
        "title": "Convolutions",
        "phase": "Phase 3: Deep Learning",
        "goal": "Seeing efficiently.",
        "concepts": ["Kernels/Filters", "Stride", "Padding"],
        "build": [
            "Apply edge detection filter manually to an image.",
            "Visualize the output feature map."
        ],
        "deliverables": ["Filter demo script"],
        "prompts": ["Explain convolution operation."],
        "streamlit_integration": "Image filter playground.",
        "quiz": [],
        "exit_criteria": "Edges detected.",
        "time": "90 minutes"
    },
    {
        "day": 72,
        "title": "CNN Architecture",
        "phase": "Phase 3: Deep Learning",
        "goal": "Stacking layers.",
        "concepts": ["Conv2d", "MaxPool2d", "Flatten"],
        "build": [
            "Build a small CNN for CIFAR-10.",
            "Train for 5 epochs."
        ],
        "deliverables": ["simple_cnn.py"],
        "prompts": ["Why pooling?"],
        "streamlit_integration": "Predict uploaded image.",
        "quiz": [],
        "exit_criteria": "Better than MLP results.",
        "time": "120 minutes"
    },
    {
        "day": 73,
        "title": "Transfer Learning",
        "phase": "Phase 3: Deep Learning",
        "goal": "Standing on shoulders of giants.",
        "concepts": ["ResNet", "Pretrained Weights", "Freezing Layers"],
        "build": [
            "Load pretrained ResNet18.",
            "Replace final layer for generic binary classification (e.g., ants vs bees).",
            "Fine-tune."
        ],
        "deliverables": ["transfer_learning.py"],
        "prompts": ["When to fine-tune vs feature extract?"],
        "streamlit_integration": "Smart object classifier.",
        "quiz": [],
        "exit_criteria": "High accuracy with few epochs.",
        "time": "120 minutes"
    },
    {
        "day": 74,
        "title": "Data Augmentation",
        "phase": "Phase 3: Deep Learning",
        "goal": "Free data.",
        "concepts": ["Transforms", "Rotation/Flip", "Normalization"],
        "build": [
            "Add transforms to the training pipeline.",
            "Visualize augmented images."
        ],
        "deliverables": ["Augmentation script"],
        "prompts": ["Common augmentations for images."],
        "streamlit_integration": "Show 'Augmented' version of upload.",
        "quiz": [],
        "exit_criteria": "More robust model.",
        "time": "90 minutes"
    },
    {
        "day": 75,
        "title": "Vision Project Mini-Capstone",
        "phase": "Phase 3: Deep Learning",
        "goal": "Ship a vision app.",
        "concepts": ["End-to-End Vision"],
        "build": [
            "Build a 'Is it a Hotdog?' app (or similar).",
            "Use Transfer Learning.",
        ],
        "deliverables": ["Vision App"],
        "prompts": ["MobileNet vs ResNet."],
        "streamlit_integration": "Dedicated Vision page.",
        "quiz": [],
        "exit_criteria": "Fun working demo.",
        "time": "120 minutes"
    },
    integration_day(76, "Computer Vision"),
    review_day(77, "Computer Vision"),

    # Week 12: NLP Basics (RNNs)
    {
        "day": 78,
        "title": "NLP Preprocessing",
        "phase": "Phase 3: Deep Learning",
        "goal": "Text to numbers.",
        "concepts": ["Tokenization", "Vocab Building", "Stopwords"],
        "build": [
            "Implement simple tokenizer (split).",
            "Build vocabulary index.",
            "Convert sentence to tensor."
        ],
        "deliverables": ["tokenizer.py"],
        "prompts": ["Challenges of text data."],
        "streamlit_integration": "Token viewer.",
        "quiz": [],
        "exit_criteria": "Text -> List[int].",
        "time": "90 minutes"
    },
    {
        "day": 79,
        "title": "Embeddings",
        "phase": "Phase 3: Deep Learning",
        "goal": "Meaning in vectors.",
        "concepts": ["nn.Embedding", "Word2Vec idea", "Semantic Similarity"],
        "build": [
            "Train a small custom embedding.",
            "Visualize similarity (cosine)."
        ],
        "deliverables": ["embeddings.py"],
        "prompts": ["One-hot vs Embeddings."],
        "streamlit_integration": "Word arithmetic (King - Man + Woman).",
        "quiz": [],
        "exit_criteria": "Similar words close.",
        "time": "90 minutes"
    },
    {
        "day": 80,
        "title": "RNNs & LSTMs",
        "phase": "Phase 3: Deep Learning",
        "goal": "Sequential memory.",
        "concepts": ["Hidden State", "Vanishing Gradient", "LSTM Cell"],
        "build": [
            "Use nn.LSTM.",
            "Feed a sequence and check output shapes."
        ],
        "deliverables": ["lstm_basics.py"],
        "prompts": ["Why LSTM over RNN?"],
        "streamlit_integration": "N/A",
        "quiz": [],
        "exit_criteria": "Shapes understood.",
        "time": "90 minutes"
    },
    {
        "day": 81,
        "title": "Sequence Classification",
        "phase": "Phase 3: Deep Learning",
        "goal": "Classify text.",
        "concepts": ["Many-to-One", "Sentiment Analysis"],
        "build": [
            "Train LSTM on IMDB reviews (or similar).",
            "Classify pos/neg."
        ],
        "deliverables": ["sentiment_model.pth"],
        "prompts": ["Handling variable length sequences (padding)."],
        "streamlit_integration": "Sentiment analyzer text box.",
        "quiz": [],
        "exit_criteria": "Acc > 80%.",
        "time": "120 minutes"
    },
    {
        "day": 82,
        "title": "Text Generation (Char-RNN)",
        "phase": "Phase 3: Deep Learning",
        "goal": "Generate Shakespeare.",
        "concepts": ["Many-to-Many", "Sampling"],
        "build": [
            "Train a char-level RNN to generate text.",
            "Sample from the model."
        ],
        "deliverables": ["text_gen.py"],
        "prompts": ["Temperature in sampling."],
        "streamlit_integration": "Text generator.",
        "quiz": [],
        "exit_criteria": "Generates readable-ish text.",
        "time": "120 minutes"
    },
    integration_day(83, "RNNs/NLP"),
    review_day(84, "RNNs/NLP"),

    # Week 13: Transformers (Deep Dive)
    {
        "day": 85,
        "title": "Attention Mechanism",
        "phase": "Phase 3: Deep Learning",
        "goal": "All you need is attention.",
        "concepts": ["Self-Attention", "Q/K/V Matrices", "Scalability"],
        "build": [
            "Implement Scaled Dot-Product Attention from scratch.",
            "Visualise attention weights for a sentence."
        ],
        "deliverables": ["attention_mechanism.py"],
        "prompts": ["Explain Q, K, V analogy."],
        "streamlit_integration": "Attention heatmap viz.",
        "quiz": [],
        "exit_criteria": "Attention matrix shape correct.",
        "time": "90 minutes"
    },
    {
        "day": 86,
        "title": "The Transformer Arch",
        "phase": "Phase 3: Deep Learning",
        "goal": "Encoder-Decoder structure.",
        "concepts": ["Multi-Head Attention", "Feed Forward", "Normalisation"],
        "build": [
            "Review 'Attention is All You Need' paper.",
            "Build a single Transformer Block class."
        ],
        "deliverables": ["transformer_block.py"],
        "prompts": ["LayerNorm vs BatchNorm in NLP."],
        "streamlit_integration": "N/A",
        "quiz": [],
        "exit_criteria": "Block runs.",
        "time": "90 minutes"
    },
    {
        "day": 87,
        "title": "BERT (Encoder)",
        "phase": "Phase 3: Deep Learning",
        "goal": "Understanding Bi-directional context.",
        "concepts": ["Masked LM", "Next Sentence Prediction", "Fine-tuning"],
        "build": [
            "Use `transformers` to load BERT.",
            "Extract embeddings for sentences.",
            "Compare [CLS] token embedding."
        ],
        "deliverables": ["bert_explorer.py"],
        "prompts": ["Why is BERT better than LSTM?"],
        "streamlit_integration": "Sentence compatibility checker.",
        "quiz": [],
        "exit_criteria": "BERT runs.",
        "time": "90 minutes"
    },
    {
        "day": 88,
        "title": "GPT (Decoder)",
        "phase": "Phase 3: Deep Learning",
        "goal": "Generative Pre-training.",
        "concepts": ["Causal Masking", "Autoregressive generation"],
        "build": [
            "Load GPT-2 small.",
            "Generate text with different prompts."
        ],
        "deliverables": ["gpt_generation.py"],
        "prompts": ["Difference between BERT and GPT."],
        "streamlit_integration": "GPT-2 Writer assistant.",
        "quiz": [],
        "exit_criteria": "Text generated.",
        "time": "90 minutes"
    },
    {
        "day": 89,
        "title": "Hugging Face Transformers",
        "phase": "Phase 3: Deep Learning",
        "goal": "The standard library of NLP.",
        "concepts": ["Pipeline API", "AutoTokenizer", "AutoModel"],
        "build": [
            "Build a sentiment classifier using `pipeline` (1 line).",
            "Fine-tune a DistilBERT model on a small dataset."
        ],
        "deliverables": ["hf_finetune.py"],
        "prompts": ["What is the HF Hub?"],
        "streamlit_integration": "HF Model downloader/runner.",
        "quiz": [],
        "exit_criteria": "Model fine-tuned.",
        "time": "120 minutes"
    },
    integration_day(90, "Transformers"),
    review_day(91, "Transformers"),

    # Week 14: Month 3 Project (NLP Capstone)
    {
        "day": 92,
        "title": "Project Scoping (NLP)",
        "phase": "Phase 3: Deep Learning",
        "goal": "Solve a language problem.",
        "concepts": ["Text Classification", "NER", "Summarization"],
        "build": [
            "Select task (e.g. Summarize Legal Docs).",
            "Find dataset on HF Hub.",
            "Init project."
        ],
        "deliverables": ["Proposal"],
        "prompts": ["Business value of summarization."],
        "streamlit_integration": "Project setup.",
        "quiz": [],
        "exit_criteria": "Dataset loaded.",
        "time": "120 minutes"
    },
    {
        "day": 93,
        "title": "Data Prep & Tokenization",
        "phase": "Phase 3: Deep Learning",
        "goal": "Mapping text to model inputs.",
        "concepts": ["Subword tokenization", "Padding/Truncation"],
        "build": [
            "Explore tokenizers.",
            "Prepare dataset for Trainer."
        ],
        "deliverables": ["Tokenized Dataset"],
        "prompts": ["BPE vs WordPiece."],
        "streamlit_integration": "Show tokens.",
        "quiz": [],
        "exit_criteria": "Data ready.",
        "time": "120 minutes"
    },
    {
        "day": 94,
        "title": "Training with HF Trainer",
        "phase": "Phase 3: Deep Learning",
        "goal": "Efficient fine-tuning.",
        "concepts": ["Trainer Arguments", "Learning Rate Scheduler"],
        "build": [
            "Setup `Trainer`.",
            "Run training (use Colab/Kaggle GPU if local weak).",
            "Save model."
        ],
        "deliverables": ["Finetuned Model"],
        "prompts": ["What is warmup steps?"],
        "streamlit_integration": "Training progress link.",
        "quiz": [],
        "exit_criteria": "Loss decreases.",
        "time": "120 minutes"
    },
    {
        "day": 95,
        "title": "NLP Evaluation",
        "phase": "Phase 3: Deep Learning",
        "goal": "Is it good?",
        "concepts": ["BLEU", "ROUGE", "Perplexity"],
        "build": [
            "Evaluate model on test set.",
            "Compute relevant metrics.",
        ],
        "deliverables": ["Eval Report"],
        "prompts": ["Limitations of BLEU."],
        "streamlit_integration": "Metrics display.",
        "quiz": [],
        "exit_criteria": "Scores computed.",
        "time": "120 minutes"
    },
    {
        "day": 96,
        "title": "Interactive NLP Demo",
        "phase": "Phase 3: Deep Learning",
        "goal": "Showcase.",
        "concepts": ["Streamlit Text Input", "Latency"],
        "build": [
            "Build user-friendly UI.",
            "Highlight entity extractions or show summary.",
        ],
        "deliverables": ["App UI"],
        "prompts": ["Optimizing inference latency."],
        "streamlit_integration": "Main App Integration.",
        "quiz": [],
        "exit_criteria": "Demo works.",
        "time": "120 minutes"
    },
    integration_day(97, "NLP Capstone"),
    review_day(98, "Phase 3 Complete"),

    # --- PHASE 4: LLM ENGINEERING (Days 99-140) ---
    # Week 15: Prompt Engineering & APIs
    {
        "day": 99,
        "title": "LLM API 101",
        "phase": "Phase 4: LLM Engineering",
        "goal": "Programmable Intelligence.",
        "concepts": ["OpenAI/Gemini API", "Roles (System/User/Assistant)"],
        "build": [
            "Get API keys.",
            "Send first request via Python.",
            "Build a simple CLI chat loop."
        ],
        "deliverables": ["basic_chat.py"],
        "prompts": ["System prompt best practices."],
        "streamlit_integration": "Chat interface (re-using Gemini integration).",
        "quiz": [],
        "exit_criteria": "Chat works.",
        "time": "60 minutes"
    },
    {
        "day": 100,
        "title": "Prompting Strategies",
        "phase": "Phase 4: LLM Engineering",
        "goal": "Steering the model.",
        "concepts": ["Zero-shot", "Few-shot", "Chain of Thought"],
        "build": [
            "Implement a function that dynamically constructs few-shot prompts.",
            "Compare zero-shot vs few-shot performance on a math problem."
        ],
        "deliverables": ["prompt_lib.py"],
        "prompts": ["Write a CoT prompt."],
        "streamlit_integration": "Prompt playground.",
        "quiz": [],
        "exit_criteria": "Few-shot wins.",
        "time": "90 minutes"
    },
    {
        "day": 101,
        "title": "Structured Outputs",
        "phase": "Phase 4: LLM Engineering",
        "goal": "Getting JSON back.",
        "concepts": ["JSON Mode", "Function/Tool Definition", "Pydantic"],
        "build": [
            "Force model to return JSON schema.",
            "Validate with Pydantic."
        ],
        "deliverables": ["extractor.py"],
        "prompts": ["How to ensure valid JSON?"],
        "streamlit_integration": "Form auto-filler demo.",
        "quiz": [],
        "exit_criteria": "Valid JSON parsed.",
        "time": "90 minutes"
    },
    {
        "day": 102,
        "title": "Chatbot with Memory",
        "phase": "Phase 4: LLM Engineering",
        "goal": "Context management.",
        "concepts": ["Context Window", "History Pruning", "Summary Memory"],
        "build": [
            "Implement a chat class that stores history.",
            "Implement a pruning strategy (last N messages).",
        ],
        "deliverables": ["memory_chat.py"],
        "prompts": ["Context window limits."],
        "streamlit_integration": "Chat with long memory.",
        "quiz": [],
        "exit_criteria": "Remembers name from start.",
        "time": "90 minutes"
    },
    {
        "day": 103,
        "title": "Prompt Evaluation (Evals)",
        "phase": "Phase 4: LLM Engineering",
        "goal": "Data-driven prompting.",
        "concepts": ["LLM-as-a-Judge", "Unit Testing Prompts"],
        "build": [
            "Create a dataset of questions + golden answers.",
            "Use an LLM to grade your chatbot's answers against golden answers."
        ],
        "deliverables": ["eval_script.py"],
        "prompts": ["Design a rubric for LLM grading."],
        "streamlit_integration": "Eval dashboard.",
        "quiz": [],
        "exit_criteria": "Eval scores generated.",
        "time": "90 minutes"
    },
    integration_day(104, "Prompt Engineering"),
    review_day(105, "Prompt Engineering"),

    # Week 16: RAG Foundations
    {
        "day": 106,
        "title": "Vector Embeddings II",
        "phase": "Phase 4: LLM Engineering",
        "goal": "Search infrastructure.",
        "concepts": ["Dense Retrieval", "Cosine Similarity"],
        "build": [
            "Embed a text corpus (e.g., Wikipedia articles).",
            "Perform search query by dot product."
        ],
        "deliverables": ["naive_search.py"],
        "prompts": ["Dense vs Sparse vectors."],
        "streamlit_integration": "Semantic search bar.",
        "quiz": [],
        "exit_criteria": "Query finds relevant doc.",
        "time": "90 minutes"
    },
    {
        "day": 107,
        "title": "Vector Databases (FAISS)",
        "phase": "Phase 4: LLM Engineering",
        "goal": "Scale search.",
        "concepts": ["FAISS/Chroma", "Indexing", "ANN"],
        "build": [
            "Set up FAISS index locally.",
            "Add embeddings.",
            "Search.",
        ],
        "deliverables": ["vector_db.py"],
        "prompts": ["What is HNSW?"],
        "streamlit_integration": "Indexed corpus search.",
        "quiz": [],
        "exit_criteria": "Fast retrieval.",
        "time": "90 minutes"
    },
    {
        "day": 108,
        "title": "Naive RAG Pipeline",
        "phase": "Phase 4: LLM Engineering",
        "goal": "Chat with data.",
        "concepts": ["Retrieve-Augment-Generate"],
        "build": [
            "Combine Retriever + Prompt + LLM.",
            "Build 'Chat with this PDF' script (parsing text first)."
        ],
        "deliverables": ["rag_basic.py"],
        "prompts": ["The RAG triad."],
        "streamlit_integration": "PDF Chatbot UI.",
        "quiz": [],
        "exit_criteria": "Answers based on PDF.",
        "time": "120 minutes"
    },
    {
        "day": 109,
        "title": "Chunking Strategies",
        "phase": "Phase 4: LLM Engineering",
        "goal": "Better context.",
        "concepts": ["Fixed-size", "RecursiveCharacter", "Semantic Chunking"],
        "build": [
            "Compare search quality with different chunk sizes.",
            "Implement sliding window chunking."
        ],
        "deliverables": ["chunker.py"],
        "prompts": ["Optimal chunk size."],
        "streamlit_integration": "Chunk visualizer.",
        "quiz": [],
        "exit_criteria": "Better chunks = better answers.",
        "time": "90 minutes"
    },
    {
        "day": 110,
        "title": "Retrieval Evaluation",
        "phase": "Phase 4: LLM Engineering",
        "goal": "Measuring recall.",
        "concepts": ["Hit Rate", "MRR (Mean Reciprocal Rank)", "Ragas (library)"],
        "build": [
            "Create synthetic Questions/Context pairs from docs.",
            "Measure Hit Rate of retriever."
        ],
        "deliverables": ["retriever_eval.py"],
        "prompts": ["How to evaluate retrieval?"],
        "streamlit_integration": "Metric score display.",
        "quiz": [],
        "exit_criteria": "Hit rate calculated.",
        "time": "90 minutes"
    },
    integration_day(111, "RAG Basics"),
    review_day(112, "RAG Basics"),

    # Week 17: RAG Advanced
    {
        "day": 113,
        "title": "Hybrid Search",
        "phase": "Phase 4: LLM Engineering",
        "goal": "Best of both worlds.",
        "concepts": ["BM25 (Keyword)", "Reciprocal Rank Fusion (RRF)"],
        "build": [
            "Implement BM25 (using library).",
            "Combine with Vector search using RRF."
        ],
        "deliverables": ["hybrid_search.py"],
        "prompts": ["When does semantic search fail?"],
        "streamlit_integration": "Hybrid search toggle.",
        "quiz": [],
        "exit_criteria": "Keyword matches prioritized when needed.",
        "time": "90 minutes"
    },
    {
        "day": 114,
        "title": "Re-ranking",
        "phase": "Phase 4: LLM Engineering",
        "goal": "Precision filter.",
        "concepts": ["Cross-Encoder", "Cohere/BGE Reranker"],
        "build": [
            "Retrieve top 25 docs.",
            "Use a Cross-Encoder to re-rank top 5."
        ],
        "deliverables": ["rerank.py"],
        "prompts": ["Bi-encoder vs Cross-encoder."],
        "streamlit_integration": "Re-ranker demo.",
        "quiz": [],
        "exit_criteria": "Top result improvement.",
        "time": "90 minutes"
    },
    {
        "day": 115,
        "title": "Query Transformations",
        "phase": "Phase 4: LLM Engineering",
        "goal": "Fixing user queries.",
        "concepts": ["HyDE (Hypothetical Document Embeddings)", "Query Expansion"],
        "build": [
            "Use LLM to generate a hypothetical answer -> embed that.",
            "Or generate 3 sub-queries."
        ],
        "deliverables": ["query_transform.py"],
        "prompts": ["Explain HyDE."],
        "streamlit_integration": "Show transformed queries.",
        "quiz": [],
        "exit_criteria": "Better retrieval.",
        "time": "90 minutes"
    },
    {
        "day": 116,
        "title": "RAG with Tools",
        "phase": "Phase 4: LLM Engineering",
        "goal": "Actionable RAG.",
        "concepts": ["Function Calling", "Tool Use"],
        "build": [
            "Add a 'Calculator' tool.",
            "Let LLM decide to Search or Calculate."
        ],
        "deliverables": ["agentic_rag.py"],
        "prompts": ["Tool selection logic."],
        "streamlit_integration": "Calculator tool demo.",
        "quiz": [],
        "exit_criteria": "LLM calls tool.",
        "time": "90 minutes"
    },
    {
        "day": 117,
        "title": "Router Pattern",
        "phase": "Phase 4: LLM Engineering",
        "goal": "Routing queries.",
        "concepts": ["Semantic Routing", "Classification"],
        "build": [
            "Train/Prompt a classifier to route query to 'SQL DB' vs 'Vector DB'.",
        ],
        "deliverables": ["router.py"],
        "prompts": ["Designing a router."],
        "streamlit_integration": "Route visualizer.",
        "quiz": [],
        "exit_criteria": "Correct routing.",
        "time": "90 minutes"
    },
    integration_day(118, "Advanced RAG"),
    review_day(119, "Advanced RAG"),

    # Week 18: Agents
    {
        "day": 120,
        "title": "Function Calling Deep Dive",
        "phase": "Phase 4: LLM Engineering",
        "goal": "Connecting to the world.",
        "concepts": ["OpenAI Tool Spec", "Pydantic Args"],
        "build": [
            "Define complex tools with multi-argument inputs.",
            "Execute function calls safely."
        ],
        "deliverables": ["tools_engine.py"],
        "prompts": ["Security risks of tool use."],
        "streamlit_integration": "Tool execution logs.",
        "quiz": [],
        "exit_criteria": "Complex tool called.",
        "time": "90 minutes"
    },
    {
        "day": 121,
        "title": "ReAct Pattern",
        "phase": "Phase 4: LLM Engineering",
        "goal": "Reason + Act.",
        "concepts": ["Thought-Action-Observation loop"],
        "build": [
            "Implement a manual ReAct loop: Prompt -> LLM -> Parse -> Act -> Prompt...",
        ],
        "deliverables": ["react_agent.py"],
        "prompts": ["Why ReAct loops?"],
        "streamlit_integration": "Agent thought trace.",
        "quiz": [],
        "exit_criteria": "Agent solves multi-step task.",
        "time": "120 minutes"
    },
    {
        "day": 122,
        "title": "LangChain / LlamaIndex",
        "phase": "Phase 4: LLM Engineering",
        "goal": "Don't reinvent the wheel (yet).",
        "concepts": ["Chains", "Agents", "Memory"],
        "build": [
            "Re-implement previous RAG/Agent using a framework.",
            "Compare complexity vs manual."
        ],
        "deliverables": ["framework_compare.py"],
        "prompts": ["Pros/Cons of LangChain."],
        "streamlit_integration": "Framework agent.",
        "quiz": [],
        "exit_criteria": "Working agent.",
        "time": "90 minutes"
    },
    {
        "day": 123,
        "title": "Custom Agent Loop",
        "phase": "Phase 4: LLM Engineering",
        "goal": "Full control.",
        "concepts": ["State Management", "Error Handling"],
        "build": [
            "Design a robust state machine for an agent.",
            "Handle API failures."
        ],
        "deliverables": ["robust_agent.py"],
        "prompts": ["State machines in agents."],
        "streamlit_integration": "State viewer.",
        "quiz": [],
        "exit_criteria": "Handles errors gracefully.",
        "time": "90 minutes"
    },
    {
        "day": 124,
        "title": "Multi-Agent Systems Concept",
        "phase": "Phase 4: LLM Engineering",
        "goal": "Teamwork.",
        "concepts": ["Manager/Worker", "Handoffs"],
        "build": [
            "Create two agents: 'Researcher' and 'Writer'.",
            "Manually pass output of one to another."
        ],
        "deliverables": ["multi_agent.py"],
        "prompts": ["Benefits of multi-agent."],
        "streamlit_integration": "Multi-agent chat.",
        "quiz": [],
        "exit_criteria": "Collaboration works.",
        "time": "90 minutes"
    },
    integration_day(125, "Agents"),
    review_day(126, "Agents"),

    # Week 19: LLM Finetuning
    {
        "day": 127,
        "title": "Finetuning Concepts",
        "phase": "Phase 4: LLM Engineering",
        "goal": "Specializing models.",
        "concepts": ["SFT (Supervised Fine-Tuning)", "RLHF (Concept)", "Full vs PEFT"],
        "build": [
            "Read LoRA paper abstract.",
            "Set up 'Unsloth' or 'PEFT' environment (Colab usually).",
        ],
        "deliverables": ["finetune_env_check.ipynb"],
        "prompts": ["When to finetune vs RAG?"],
        "streamlit_integration": "Show GPU memory reqs.",
        "quiz": [],
        "exit_criteria": "Env ready.",
        "time": "90 minutes"
    },
    {
        "day": 128,
        "title": "LoRA & QLoRA",
        "phase": "Phase 4: LLM Engineering",
        "goal": "Efficient tuning.",
        "concepts": ["Low Rank Adapters", "Quantization (4-bit)"],
        "build": [
            "Load a base model in 4-bit.",
            "Attach LoRA adapters.",
            "Print trainable parameters count."
        ],
        "deliverables": ["lora_setup.py"],
        "prompts": ["Explain Rank (r) in LoRA."],
        "streamlit_integration": "LoRA config generator.",
        "quiz": [],
        "exit_criteria": "Model loads.",
        "time": "90 minutes"
    },
    {
        "day": 129,
        "title": "Dataset Preparation",
        "phase": "Phase 4: LLM Engineering",
        "goal": "It's all about the data.",
        "concepts": ["Chat Templates", "Alpaca format", "ShareGPT format"],
        "build": [
            "Convert a raw text dataset into Chat Template format.",
            "Visualize the actual tokens fed to model."
        ],
        "deliverables": ["dataset_formatter.py"],
        "prompts": ["Why use specific chat templates?"],
        "streamlit_integration": "Dataset previewer.",
        "quiz": [],
        "exit_criteria": "Data matches template.",
        "time": "90 minutes"
    },
    {
        "day": 130,
        "title": "Running the Trainer",
        "phase": "Phase 4: LLM Engineering",
        "goal": "Press play.",
        "concepts": ["SFTTrainer (TRL library)", "Packing"],
        "build": [
            "Run SFT on a small dataset (e.g. quote generation).",
            "Monitor loss."
        ],
        "deliverables": ["finetune_run.ipynb"],
        "prompts": ["What is packing?"],
        "streamlit_integration": "Training status link.",
        "quiz": [],
        "exit_criteria": "Training finishes.",
        "time": "120 minutes"
    },
    {
        "day": 131,
        "title": "Inference & Merging",
        "phase": "Phase 4: LLM Engineering",
        "goal": "Using the tailored model.",
        "concepts": ["Merging Adapters", "GGUF export (concept)"],
        "build": [
            "Load base model + adapter.",
            "Run inference.",
            "Merge adapter into base."
        ],
        "deliverables": ["merged_model.pth"],
        "prompts": ["Pros/cons of merging."],
        "streamlit_integration": "Custom model chat.",
        "quiz": [],
        "exit_criteria": "Model replies in new style.",
        "time": "90 minutes"
    },
    integration_day(132, "Finetuning"),
    review_day(133, "Finetuning"),

    # Week 20: Month 5 Project (RAG + Agent)
    {
        "day": 134,
        "title": "Project Scoping (Capstone Prep)",
        "phase": "Phase 4: LLM Engineering",
        "goal": "A complex vertical app.",
        "concepts": ["Vertical AI", "UX for GenAI"],
        "build": [
            "Scope a 'Legal Assistant' or 'Medical Coder' app.",
            "Define RAG sources and Tools needed."
        ],
        "deliverables": ["Spec Doc"],
        "prompts": ["UX patterns for AI."],
        "streamlit_integration": "New project created.",
        "quiz": [],
        "exit_criteria": "Scope locked.",
        "time": "120 minutes"
    },
    {
        "day": 135,
        "title": "Architecture & Tools",
        "phase": "Phase 4: LLM Engineering",
        "goal": "System Design.",
        "concepts": ["Database Schema", "API Design"],
        "build": [
            "Set up Vector DB collection.",
            "Write tool definitions (e.g., 'Search Case Law')."
        ],
        "deliverables": ["Architecture Diagram"],
        "prompts": ["Microservices vs Monolith for AI?"],
        "streamlit_integration": "Diagram upload.",
        "quiz": [],
        "exit_criteria": "DB ready.",
        "time": "120 minutes"
    },
    {
        "day": 136,
        "title": "Core RAG Implementation",
        "phase": "Phase 4: LLM Engineering",
        "goal": "The brain.",
        "concepts": ["Advanced Retrieval"],
        "build": [
            "Implement Hybrid Search for the domain data.",
            "Test retrieval quality."
        ],
        "deliverables": ["rag_core.py"],
        "prompts": ["Improving recall on legal text."],
        "streamlit_integration": "Search debug UI.",
        "quiz": [],
        "exit_criteria": "Good context retrieved.",
        "time": "120 minutes"
    },
    {
        "day": 137,
        "title": "Agentic Layer",
        "phase": "Phase 4: LLM Engineering",
        "goal": "The hands.",
        "concepts": ["Planning"],
        "build": [
            "Implement the agent loop to use tools.",
            "Handle 'requires clarification' state."
        ],
        "deliverables": ["agent_core.py"],
        "prompts": ["Handling ambiguous queries."],
        "streamlit_integration": "Agent chat.",
        "quiz": [],
        "exit_criteria": "Agent asks clarifying questions.",
        "time": "120 minutes"
    },
    {
        "day": 138,
        "title": "Eval Harness",
        "phase": "Phase 4: LLM Engineering",
        "goal": "Trust but verify.",
        "concepts": ["Ragas", "G-Eval"],
        "build": [
            "Build an automated eval suite (20 questions).",
            "Run against the agent."
        ],
        "deliverables": ["eval_results.csv"],
        "prompts": ["Designing gold sets."],
        "streamlit_integration": "Eval report details.",
        "quiz": [],
        "exit_criteria": "Baseline score established.",
        "time": "120 minutes"
    },
    integration_day(139, "Month 5 Project"),
    review_day(140, "Phase 4 Complete"),

    # --- PHASE 5: PRODUCTION & CAREER (Days 141-168) ---
    # Week 21: Production Engineering
    {
        "day": 141,
        "title": "Serving with vLLM",
        "phase": "Phase 5: Production",
        "goal": "High throughput.",
        "concepts": ["PagedAttention", "Continous Batching"],
        "build": [
            "Set up a vLLM server (Docker).",
            "Benchmark tok/sec vs standard HuggingFace."
        ],
        "deliverables": ["benchmark_report.md"],
        "prompts": ["What is PagedAttention?"],
        "streamlit_integration": "N/A (Backend).",
        "quiz": [],
        "exit_criteria": "vLLM running.",
        "time": "90 minutes"
    },
    {
        "day": 142,
        "title": "Latency & Caching",
        "phase": "Phase 5: Production",
        "goal": "Faster.",
        "concepts": ["KV Cache", "Semantic Caching (Redis)"],
        "build": [
            "Implement a semantic cache (if query similar, return cached).",
            "Measure latency reduction."
        ],
        "deliverables": ["caching.py"],
        "prompts": ["Cache invalidation strategies."],
        "streamlit_integration": "Cache hit counter.",
        "quiz": [],
        "exit_criteria": "Latency drop on repeat queries.",
        "time": "90 minutes"
    },
    {
        "day": 143,
        "title": "Guardrails",
        "phase": "Phase 5: Production",
        "goal": "Safe.",
        "concepts": ["NeMo Guardrails", "PII Redaction", "Jailbreak detection"],
        "build": [
            "Implement input filter for PII.",
            "Implement output filter for toxicity."
        ],
        "deliverables": ["guardrails.py"],
        "prompts": ["How to prevent prompt injection?"],
        "streamlit_integration": "Safety status indicator.",
        "quiz": [],
        "exit_criteria": "Toxic prompt blocked.",
        "time": "90 minutes"
    },
    {
        "day": 144,
        "title": "CI/CD for AI",
        "phase": "Phase 5: Production",
        "goal": "Automated pipeline.",
        "concepts": ["GitHub Actions", "Model Regression setup"],
        "build": [
            "Write a GitHub Action that runs the Eval harness on push.",
        ],
        "deliverables": ["ci_cd.yaml"],
        "prompts": ["Diffs for models?"],
        "streamlit_integration": "Show last build status.",
        "quiz": [],
        "exit_criteria": "Action passes.",
        "time": "90 minutes"
    },
    {
        "day": 145,
        "title": "Monitoring",
        "phase": "Phase 5: Production",
        "goal": "Observability.",
        "concepts": ["LangSmith", "Traces", "Drift"],
        "build": [
            "Instrument the Month 5 project with LangSmith/Arize.",
            "View a trace of a complex run."
        ],
        "deliverables": ["Monitoring Dashboard"],
        "prompts": ["What is data drift?"],
        "streamlit_integration": "Stats embedded.",
        "quiz": [],
        "exit_criteria": "Traces visible.",
        "time": "90 minutes"
    },
    integration_day(146, "Production"),
    review_day(147, "Production"),

    # Week 22: Final Capstone Part 1
    {
        "day": 148,
        "title": "Capstone Proposal",
        "phase": "Phase 5: Production",
        "goal": "The Magnum Opus.",
        "concepts": ["Product Requirements Document"],
        "build": [
            "Write full PRD for Final App.",
            "Must include: RAG, Agents, Eval, Monitoring.",
        ],
        "deliverables": ["PRD.md"],
        "prompts": ["Critique my PRD."],
        "streamlit_integration": "Capstone Tracker.",
        "quiz": [],
        "exit_criteria": "Proposal approved (by self/coach).",
        "time": "120 minutes"
    },
    {
        "day": 149,
        "title": "Data & Eval First",
        "phase": "Phase 5: Production",
        "goal": "TDD for AI.",
        "concepts": ["Golden Dataset"],
        "build": [
            "Create the evaluation dataset BEFORE building.",
            "Define success metrics."
        ],
        "deliverables": ["gold_data.json"],
        "prompts": ["Eval-Driven Development."],
        "streamlit_integration": "Eval baseline.",
        "quiz": [],
        "exit_criteria": "Eval ready.",
        "time": "120 minutes"
    },
    {
        "day": 150,
        "title": "MVP: Core Model",
        "phase": "Phase 5: Production",
        "goal": "Proof of Concept.",
        "concepts": ["Prototyping"],
        "build": [
            "Build the core Logic/Chain.",
            "Verify against subset of Eval."
        ],
        "deliverables": ["core.py"],
        "prompts": ["Rapid prototyping."],
        "streamlit_integration": "Core demo.",
        "quiz": [],
        "exit_criteria": "Core logic works.",
        "time": "120 minutes"
    },
    {
        "day": 151,
        "title": "MVP: Backend",
        "phase": "Phase 5: Production",
        "goal": "Scalable API.",
        "concepts": ["FastAPI", "Async"],
        "build": [
            "Wrap core in robust FastAPI.",
            "Add DB persistence."
        ],
        "deliverables": ["backend/"],
        "prompts": ["Async python benefits."],
        "streamlit_integration": "Connect to API.",
        "quiz": [],
        "exit_criteria": "API responsive.",
        "time": "120 minutes"
    },
    {
        "day": 152,
        "title": "MVP: Frontend",
        "phase": "Phase 5: Production",
        "goal": "User delight.",
        "concepts": ["Streamlit Advanced", "React (optional)"],
        "build": [
            "Build the public-facing UI.",
            "Focus on latency feedback (spinners, streaming)."
        ],
        "deliverables": ["frontend/"],
        "prompts": ["UI for streaming text."],
        "streamlit_integration": "N/A (Is the app).",
        "quiz": [],
        "exit_criteria": "UI usable.",
        "time": "120 minutes"
    },
    integration_day(153, "Capstone MVP"),
    review_day(154, "Capstone MVP"),

    # Week 23: Final Capstone Part 2
    {
        "day": 155,
        "title": "Testing & Safety",
        "phase": "Phase 5: Production",
        "goal": "Bulletproof.",
        "concepts": ["Red Teaming", "Unit Tests"],
        "build": [
            "Try to break the app (Red Team).",
            "Write regression tests."
        ],
        "deliverables": ["test_report.md"],
        "prompts": ["How to red team LLMs?"],
        "streamlit_integration": "Safety score.",
        "quiz": [],
        "exit_criteria": "Tests pass.",
        "time": "120 minutes"
    },
    {
        "day": 156,
        "title": "Docker & Cloud",
        "phase": "Phase 5: Production",
        "goal": "Live on internet.",
        "concepts": ["Dockerfile", "Cloud Run/Render"],
        "build": [
            "Dockerize everything (docker-compose).",
            "Deploy to cloud provider."
        ],
        "deliverables": ["Live URL"],
        "prompts": ["Docker optimization."],
        "streamlit_integration": "Deployment status.",
        "quiz": [],
        "exit_criteria": "Publicly accessible.",
        "time": "120 minutes"
    },
    {
        "day": 157,
        "title": "Load Testing",
        "phase": "Phase 5: Production",
        "goal": "Scale test.",
        "concepts": ["Locust", "Concurrency"],
        "build": [
            "Use Locust to simulate 50 users.",
            "Identify bottlenecks."
        ],
        "deliverables": ["load_test.py"],
        "prompts": ["Scaling strategies."],
        "streamlit_integration": "Load stats.",
        "quiz": [],
        "exit_criteria": "Bottlenecks found.",
        "time": "120 minutes"
    },
    {
        "day": 158,
        "title": "Docs & Video",
        "phase": "Phase 5: Production",
        "goal": "Marketing.",
        "concepts": ["Documentation", "Demo Reel"],
        "build": [
            "Write comprehensive README.",
            "Record 2 min demo video."
        ],
        "deliverables": ["README.md", "video.mp4"],
        "prompts": ["What makes a good Readme?"],
        "streamlit_integration": "Embed video.",
        "quiz": [],
        "exit_criteria": "Video done.",
        "time": "120 minutes"
    },
    {
        "day": 159,
        "title": "Polish & Refactor",
        "phase": "Phase 5: Production",
        "goal": "Professional sheen.",
        "concepts": ["Code cleanup"],
        "build": [
            "Final linting pass.",
            "UI tweaks (colors, spacing)."
        ],
        "deliverables": ["Polished Codebase"],
        "prompts": ["Refactoring strategies."],
        "streamlit_integration": "Final look.",
        "quiz": [],
        "exit_criteria": "Looks great.",
        "time": "120 minutes"
    },
    integration_day(160, "Capstone Final"),
    review_day(161, "Capstone Final"),

    # Week 24: Career Prep
    {
        "day": 162,
        "title": "Portfolio Website",
        "phase": "Phase 5: Career",
        "goal": "Showcase.",
        "concepts": ["Personal Branding"],
        "build": [
            "Build a personal portfolio site (or Notion page).",
            "Link all 4 major projects."
        ],
        "deliverables": ["Portfolio URL"],
        "prompts": ["Portfolio checklist."],
        "streamlit_integration": "Portfolio section final.",
        "quiz": [],
        "exit_criteria": "Online portfolio.",
        "time": "90 minutes"
    },
    {
        "day": 163,
        "title": "Resume Review",
        "phase": "Phase 5: Career",
        "goal": "Get the interview.",
        "concepts": ["ATS Optimization", "Action Verbs"],
        "build": [
            "Update resume with 'AI Engineer' title.",
            "Add project metrics (e.g., 'Reduced latency by 20%')."
        ],
        "deliverables": ["Resume.pdf"],
        "prompts": ["Critique my resume bullet points."],
        "streamlit_integration": "Resume uploader (for Gemini critique).",
        "quiz": [],
        "exit_criteria": "Resume updated.",
        "time": "90 minutes"
    },
    {
        "day": 164,
        "title": "System Design Interview",
        "phase": "Phase 5: Career",
        "goal": "Architecture interview.",
        "concepts": ["Scalability", "Trade-offs"],
        "build": [
            "Practice 'Design a RAG system for Twitter'.",
            "Draw architecture diagrams."
        ],
        "deliverables": ["Design Doc"],
        "prompts": ["System design interview tips."],
        "streamlit_integration": "Interview sim.",
        "quiz": [],
        "exit_criteria": "Practice done.",
        "time": "90 minutes"
    },
    {
        "day": 165,
        "title": "Coding Interview",
        "phase": "Phase 5: Career",
        "goal": "Leetcode style.",
        "concepts": ["Python Dsa"],
        "build": [
            "Solve 3 medium Leetcode problems related to arrays/graphs.",
        ],
        "deliverables": ["Solutions"],
        "prompts": ["Explain this solution."],
        "streamlit_integration": "Coding challenge timer.",
        "quiz": [],
        "exit_criteria": "Problems solved.",
        "time": "90 minutes"
    },
    {
        "day": 166,
        "title": "Outreach Strategy",
        "phase": "Phase 5: Career",
        "goal": "Finding opportunities.",
        "concepts": ["Networking", "Cold DMs"],
        "build": [
            "Identify 20 companies.",
            "Draft cold outreach messages."
        ],
        "deliverables": ["Target List"],
        "prompts": ["Cold email templates."],
        "streamlit_integration": "Company tracker.",
        "quiz": [],
        "exit_criteria": "Messages drafted.",
        "time": "90 minutes"
    },
    {
        "day": 167,
        "title": "Final Submission",
        "phase": "Phase 5: Career",
        "goal": "Wrapping up.",
        "concepts": ["Reflection"],
        "build": [
            "Package everything.",
            "Write a 'Learnings' blog post."
        ],
        "deliverables": ["Final Submission"],
        "prompts": ["Reflection questions."],
        "streamlit_integration": "Celebrate button.",
        "quiz": [],
        "exit_criteria": "Submitted.",
        "time": "90 minutes"
    },
    {
        "day": 168,
        "title": "Graduation",
        "phase": "Phase 5: Career",
        "goal": "You are an AI Engineer.",
        "concepts": ["Lifelong Learning"],
        "build": [
            "Plan next learning path.",
            "Apply to 5 jobs."
        ],
        "deliverables": ["Job Applications"],
        "prompts": ["Future of AI Engineering."],
        "streamlit_integration": "Certificate generation.",
        "quiz": [],
        "exit_criteria": "Applied.",
        "time": "90 minutes"
    },
]
