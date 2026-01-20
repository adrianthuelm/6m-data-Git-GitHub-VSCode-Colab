# Collaborative Development Workflow Bootcamp
## Git, GitHub, VS Code & Google Colab for Data Science

## DETAILED LEARNING PLAN

### Key Principles

1. **Start with the WHY** – Ground every tool in real data science workflows
2. **Learning by Doing** – 70% hands-on exercises, 30% explanation
3. **Minimal Cognitive Load** – Introduce one concept at a time, build complexity gradually
4. **Immediate Application** – Every demo followed by a guided hands-on exercise
5. **Practical Mistakes** – Include deliberate error scenarios for recovery practice
6. **Real-world Relevance** – Frame everything around collaborative data science projects they'll do in the bootcamp

### Session Flow Architecture

```
Total Time: 180 minutes
├─ Opening Context (5 min)
├─ Section 1: Git Fundamentals & Local Workflow (60 min)
├─ Section 2: GitHub Collaboration & Pull Requests (55 min)
├─ Section 3: VS Code + Colab Integration & Deployment (55 min)
└─ Closing & Next Steps (5 min)
```

---

## SECTION BREAKDOWN

### SECTION 1: GIT FUNDAMENTALS & LOCAL WORKFLOW
**Duration:** 60 minutes  
**Focus:** Master Git locally, understand version control philosophy, practice essential commands

#### Learning Outcomes
- Understand what Git is and why it's essential for data science teams
- Create and initialize a Git repository
- Make meaningful commits with clear commit messages
- View project history and navigate changes
- Practice recovering from common mistakes

#### Content Flow

**A. OPENING CONTEXT (5 min) – Why Git Matters**

*Narrative Frame:*  
"Imagine you're working on a machine learning model with 5 teammates. Sarah modifies the feature engineering pipeline. You add new validation code. Ahmed refactors the model. How do you all sync changes without overwriting each other's work? How do you know who changed what and when? That's where Git comes in."

**B. DEMO 1: Git Workflow Visualization (8 min)**

*Instructor Shows:*
- 3-panel visualization: Working Directory → Staging Area → Repository
- Analogy: "Like drafting a paper → peer review → final publication"
- Show a typical sequence: make changes → `git add` → `git commit` → view history
- Use a GitHub or GitLab project as visual reference (live browser demo)

*Key Metaphors:*
- Repository = "Time machine for your project"
- Commit = "Snapshot of your work with a message"
- Staging area = "Holding area before finalizing changes"

**C. HANDS-ON EXERCISE 1: Git Setup & First Commit (22 min)**

*Environment Setup (3 min):*
1. Download & install Git (Mac/Windows installer links pre-provided)
2. Configure Git with name and email:
   ```bash
   git config --global user.name "Your Name"
   git config --global user.email "your.email@bootcamp.com"
   ```

*Guided Walkthrough (19 min):*

**Step 1: Initialize Repository (5 min)**
- Create a project folder: `mkdir my_first_ds_project`
- Navigate in: `cd my_first_ds_project`
- Initialize Git: `git init`
- **Explain:** Git created a hidden `.git` folder that tracks everything
- Check status: `git status` (show the clean slate)

**Step 2: Create & Track a File (7 min)**
- Create a Python file with sample code (provided template):
  ```python
  # iris_analysis.py
  import pandas as pd
  
  # Load iris dataset
  df = pd.read_csv("iris.csv")
  print(f"Dataset shape: {df.shape}")
  ```
- **Instructor narrates:** "Now we have code. Let's tell Git to track it."
- Stage the file: `git add iris_analysis.py`
- Show staged changes: `git status` (file appears in "Changes to be committed")
- **Explain:** "We've staged it. Now let's officially save this snapshot."

**Step 3: Make First Commit (5 min)**
- Commit with message: `git commit -m "Add initial iris analysis script"`
- **Emphasize:** "Use clear, action-oriented messages. Not 'stuff' but 'Add initial iris analysis script'"
- Show commit in history: `git log` (display commit hash, author, date, message)
- **Highlight:** "Each commit is uniquely identified and timestamped"

**Step 4: Practice Making Changes (2 min)**
- Edit the file (add a comment or one line of code)
- Check status: `git status` (file shows as modified, not staged)
- Stage and commit: `git add` → `git commit -m "Add data summary statistics"`
- View history: `git log --oneline` (compact view showing commits)

*Checkpoint:* Each learner shows their `git log` output to confirm 2+ commits

**D. DEMO 2: Undoing Mistakes Gracefully (8 min)**

*Scenario-Based Teaching:*
"You realize you committed code with a bug. Don't panic—Git can help."

- **Scenario A: Undo unstaged changes** (file was modified but not staged)
  ```bash
  git checkout -- filename.py
  ```
  *Narrative:* "Throws away your changes, resets to the last commit"

- **Scenario B: Unstage a file** (file staged but not committed)
  ```bash
  git reset HEAD filename.py
  ```
  *Narrative:* "Removes from staging but keeps your changes"

- **Scenario C: View what changed** (curiosity about last commit)
  ```bash
  git diff
  git diff --staged
  ```

**E. HANDS-ON EXERCISE 2: Mistake Recovery Practice (10 min)**

*Guided Scenario:*

1. **Make a "mistake" on purpose:**
   - Modify the iris analysis file and add a deliberately bad line
   - Stage it: `git add`
   - Commit: `git commit -m "Add bad code"`
   - **Point out:** "Oops, we committed a mistake!"

2. **Recover using Git:**
   - Show the mistake in `git log`
   - Use `git show <commit-hash>` to see the problematic commit
   - **Explain:** "Don't panic. Let's revert this commit."
   - Revert: `git revert <commit-hash>`
   - Check log: `git log` (shows original bad commit + new revert commit)
   - **Narrative:** "Reverting is safer than erasing history. Everyone can see what happened."

3. **Each learner practices:**
   - Add a test file, commit it, then remove it using `git reset --soft HEAD~1`
   - Confirm the file is back in staging area

*Checkpoint:* Demo navigating through 3+ commits using `git checkout <commit-hash>` to view project state at different points in time

---

### SECTION 2: GITHUB COLLABORATION & PULL REQUESTS
**Duration:** 55 minutes  
**Focus:** Bridge local Git work to remote collaboration, master pull requests, resolve conflicts

#### Learning Outcomes
- Create a GitHub account and connect local Git to GitHub
- Push local commits to a remote repository
- Create and review pull requests
- Merge code safely with basic conflict resolution
- Understand branching as a collaboration strategy

#### Content Flow

**A. OPENING CONTEXT (4 min) – From Local to Team**

*Narrative Frame:*  
"Git works on your computer. GitHub is where your team meets. When you push your code to GitHub, your teammates can see it, review it, and help you improve it before merging to the main project."

*Visual Metaphor:*  
"Git = your personal notebook | GitHub = team workspace on the wall"

**B. DEMO 1: GitHub Workflow Setup (10 min)**

*Instructor Demo:*

1. **Create a Repository on GitHub (3 min)**
   - Log in to github.com
   - Click "New Repository"
   - Name: `bootcamp_ds_project`
   - Initialize with README
   - **Explain:** "This is your project's home on the internet"

2. **Connect Local Git to GitHub (4 min)**
   - Copy repository URL
   - Show terminal command: `git remote add origin https://github.com/user/bootcamp_ds_project.git`
   - **Explain:** "Now your local Git knows where to send code"
   - Verify: `git remote -v`

3. **Push Code to GitHub (3 min)**
   - Push local commits: `git push origin main`
   - **Emphasize:** "First time push establishes connection"
   - Show code appearing on GitHub website in real-time
   - **Narrative:** "Your teammates can now see all your commits and changes"

**C. HANDS-ON EXERCISE 1: Personal Repository Setup (15 min)**

*Guided Walkthrough (Each learner completes):*

1. **Create GitHub Account (if needed) – 3 min**
   - Go to github.com/signup
   - Use email from bootcamp
   - Verify email
   - **Instructor tip:** "Use a professional username you'll use throughout the bootcamp"

2. **Create Your First Repository – 5 min**
   - "New Repository" → `my_ds_project`
   - ✓ Initialize with README
   - Copy the provided git commands

3. **Push Local Work to GitHub – 7 min**
   - In terminal (in your local project folder):
     ```bash
     git remote add origin https://github.com/YOUR_USERNAME/my_ds_project.git
     git branch -M main
     git push -u origin main
     ```
   - **Explain each line:**
     - `remote add` = connect to GitHub
     - `branch -M main` = rename branch (modern GitHub standard)
     - `push -u origin main` = upload and track origin/main
   - Refresh GitHub page → see your commits appear
   - **Checkpoint:** Each learner shows their GitHub page with code visible

*Common Issue Handling:*
- "Permission denied" → Check SSH keys or use HTTPS token
- Instructor circulates with solution steps printed

**D. DEMO 2: Branching Strategy & Pull Requests (12 min)**

*Scenario:*  
"You're on a 4-person team. Sarah is working on feature A. You're working on feature B. You both need to change the same data loading function. Without branching, you'd destroy each other's work. Branches let you work independently."

*Instructor Shows:*

1. **Why Branches Matter (3 min)**
   - Diagram on screen: `main` branch (production) vs `feature` branches (experimental)
   - **Metaphor:** "Main is the published version. Branches are drafts."

2. **Create a Feature Branch (4 min)**
   ```bash
   git checkout -b add_data_validation
   ```
   - **Explain:** "We're creating a new branch and switching to it"
   - Make a change: add new Python function with data validation
   - Commit: `git commit -m "Add input validation function"`

3. **Push Branch to GitHub (2 min)**
   ```bash
   git push origin add_data_validation
   ```
   - Show branch appearing on GitHub website

4. **Create Pull Request on GitHub (3 min)**
   - Navigate to repository on GitHub
   - GitHub shows: "Your branch add_data_validation has recent pushes"
   - Click "Compare & pull request"
   - **Explain:** "PR = formal request to merge my branch into main"
   - Fill in PR title and description
   - Submit PR
   - **Narrative:** "Now your team reviews your changes"

**E. HANDS-ON EXERCISE 2: Create & Merge Your First Pull Request (18 min)**

*Scaffolded Steps (Learners follow instructor):*

1. **Create Feature Branch (4 min)**
   - On your machine:
     ```bash
     git checkout -b feature/add_statistics
     ```
   - Verify branch switch: `git branch` (shows * next to current)

2. **Make a Meaningful Change (5 min)**
   - Edit iris_analysis.py:
     ```python
     def calculate_statistics(df):
         """Calculate summary statistics for iris data"""
         return df.describe()
     
     # Add to main code
     stats = calculate_statistics(df)
     print(stats)
     ```
   - Add/commit: `git add iris_analysis.py` → `git commit -m "Add statistics function"`

3. **Push to GitHub (3 min)**
   ```bash
   git push origin feature/add_statistics
   ```
   - GitHub notification appears

4. **Create & Review PR (4 min)**
   - Go to GitHub, click "Compare & pull request"
   - Add title: "Add statistics function for data analysis"
   - Add description: "This function calculates summary statistics using pandas.describe()"
   - **Point out:** "In real projects, teammates would review and comment here"
   - Merge PR: Click "Merge pull request" (for this exercise, you approve your own)
   - **Explain:** "Main branch now includes your changes"

5. **Sync Local Main Branch (2 min)**
   ```bash
   git checkout main
   git pull origin main
   ```
   - **Verify:** Your new function is now in local code
   - **Narrative:** "Local and remote are back in sync"

*Checkpoint:* Each learner shows GitHub with merged PR and updated main branch code

**F. BONUS DEMO: Conflict Resolution (8 min)** *(if time permits)*

*Scenario:*  
"Two people edit the same line of code on different branches. When merging, Git flags this as a conflict."

*Instructor Demonstrates:*
1. Create 2 branches from main
2. Edit the same line differently on each
3. Merge first branch (succeeds)
4. Merge second branch (conflict!)
5. Open conflicted file, show conflict markers:
   ```python
   <<<<<<< HEAD
   # Sarah's version
   ======= 
   # Your version
   >>>>>>> feature/branch_name
   ```
6. **Explain:** "Manual resolution needed. Choose which version to keep, or combine both."
7. Resolve, add, commit, push

*Learner Practice (if time):* Each learner practices editing a shared file on two branches and resolving conflict

---

### SECTION 3: VS CODE + GOOGLE COLAB INTEGRATION & DEPLOYMENT
**Duration:** 55 minutes  
**Focus:** Complete data science workflow: write code locally, experiment in cloud, sync with team

#### Learning Outcomes
- Set up VS Code environment for Python/data science development
- Create Jupyter notebooks in VS Code
- Connect to Google Colab for cloud-based experimentation
- Load data and run models in Colab
- Save results and push to GitHub for team access

#### Content Flow

**A. OPENING CONTEXT (3 min) – Bringing It Together**

*Narrative Frame:*  
"Now you have local Git, GitHub collaboration, and branching. Let's add the tools for actual data science: VS Code for coding, Google Colab for GPU-powered experiments, and Git for sharing results."

*Workflow Diagram:*  
```
VS Code (write code locally)
    ↓
Google Colab (run on cloud GPU)
    ↓
Push results to GitHub
    ↓
Team downloads & builds on your work
```

**B. DEMO 1: VS Code Setup & Python Environment (10 min)**

*Instructor Shows Step-by-Step:*

1. **Install VS Code (2 min)**
   - Download from code.visualstudio.com
   - Install Python extension
   - **Explain:** "VS Code is a lightweight but powerful code editor. Python extension adds coding assistance."

2. **Open Project Folder in VS Code (2 min)**
   - File → Open Folder
   - Select your Git project folder (`my_ds_project`)
   - **Show:** VS Code displays Git integration in sidebar
     - Branch name visible
     - Source Control panel shows staged/unstaged changes

3. **Create Python Environment (3 min)**
   - Open terminal in VS Code: Ctrl+`
   - Create virtual environment:
     ```bash
     python3 -m venv ds_env
     source ds_env/bin/activate  # Mac
     # or
     ds_env\Scripts\activate  # Windows
     ```
   - **Explain:** "Virtual environment isolates your project's packages"

4. **Install Data Science Libraries (3 min)**
   - Install essentials:
     ```bash
     pip install pandas numpy scikit-learn jupyter
     ```
   - **Narrative:** "These are your data science toolkit"

**C. HANDS-ON EXERCISE 1: VS Code Notebook Creation (12 min)**

*Guided Walkthrough:*

1. **Create a Jupyter Notebook in VS Code (4 min)**
   - New file: `analysis.ipynb`
   - VS Code recognizes `.ipynb` extension
   - Select Python kernel
   - **Show:** Notebook interface with cells

2. **Write & Run Data Code (8 min)**

   *Cell 1: Imports & Data Loading*
   ```python
   import pandas as pd
   import numpy as np
   from sklearn.datasets import load_iris
   
   # Load iris dataset
   iris = load_iris()
   df = pd.DataFrame(iris.data, columns=iris.feature_names)
   df['target'] = iris.target
   print(f"Dataset shape: {df.shape}")
   df.head()
   ```
   - Click "Run Cell"
   - **Explain:** "Notebook mixes code and output. Great for exploration."

   *Cell 2: Exploratory Analysis*
   ```python
   # Summary statistics
   print("Summary Statistics:")
   print(df.describe())
   
   # Check for missing values
   print("\nMissing values:")
   print(df.isnull().sum())
   ```

   *Cell 3: Simple Model*
   ```python
   from sklearn.model_selection import train_test_split
   from sklearn.ensemble import RandomForestClassifier
   
   X = df.iloc[:, :-1]
   y = df['target']
   
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   
   model = RandomForestClassifier(n_estimators=10)
   model.fit(X_train, y_train)
   
   accuracy = model.score(X_test, y_test)
   print(f"Model Accuracy: {accuracy:.2%}")
   ```
   - Run each cell sequentially

3. **Save & Commit Notebook (2 min)**
   - Save: Ctrl+S
   - Git integration shows file as modified
   - Stage & commit:
     ```bash
     git add analysis.ipynb
     git commit -m "Add iris dataset analysis notebook"
     ```

*Checkpoint:* Each learner shows VS Code with running notebook and Git commit

**D. DEMO 2: Bridge to Google Colab (10 min)**

*Instructor Demonstrates:*

1. **Why Google Colab? (2 min)**
   - Free access to GPUs/TPUs
   - Pre-installed data science libraries
   - Easy sharing & collaboration
   - No setup required
   - **Metaphor:** "Colab = renting a powerful computer in the cloud"

2. **Upload Notebook to GitHub & Open in Colab (4 min)**
   - Push local notebook to GitHub:
     ```bash
     git push origin main
     ```
   - Go to GitHub, navigate to `analysis.ipynb`
   - **Show:** GitHub renders notebook nicely
   - Copy raw notebook URL
   - Go to colab.research.google.com
   - Menu: File → Open notebook → Upload → paste GitHub URL
   - **Explain:** "Colab loads your notebook from GitHub"

3. **Run in Colab (2 min)**
   - Click "Run all cells"
   - **Show:** Same results, but running on Google's servers
   - **Narrative:** "Same code, cloud-powered execution"

4. **Modify & Save (2 min)**
   - Add a new cell in Colab:
     ```python
     import matplotlib.pyplot as plt
     
     # Visualization
     pd.DataFrame(model.feature_importances_, 
                  index=iris.feature_names, 
                  columns=['Importance']).plot(kind='barh')
     plt.title('Feature Importance')
     plt.show()
     ```
   - Run cell
   - Download notebook: File → Download → `.ipynb`
   - **Explain:** "Colab experiments can be saved and versioned back in Git"

**E. HANDS-ON EXERCISE 2: Complete Workflow (20 min)**

*Real-World Scenario:*  
"You're tasked with building an iris classifier. You develop locally, refine in Colab, then push results for the team."

*Step-by-Step (Learners follow):*

1. **Local Development in VS Code (5 min)**
   - Create new file: `iris_classifier.py`
   - Write reusable code:
     ```python
     import pandas as pd
     from sklearn.datasets import load_iris
     from sklearn.model_selection import train_test_split
     from sklearn.ensemble import RandomForestClassifier
     from sklearn.metrics import accuracy_score, confusion_matrix
     
     class IrisClassifier:
         def __init__(self, n_estimators=10):
             self.model = RandomForestClassifier(n_estimators=n_estimators)
             self.X_train = None
             self.X_test = None
             self.y_train = None
             self.y_test = None
         
         def load_data(self):
             iris = load_iris()
             df = pd.DataFrame(iris.data, columns=iris.feature_names)
             df['target'] = iris.target
             return df
         
         def train(self, df, test_size=0.2):
             X = df.iloc[:, :-1]
             y = df['target']
             self.X_train, self.X_test, self.y_train, self.y_test = \
                 train_test_split(X, y, test_size=test_size, random_state=42)
             self.model.fit(self.X_train, self.y_train)
         
         def evaluate(self):
             y_pred = self.model.predict(self.X_test)
             accuracy = accuracy_score(self.y_test, y_pred)
             cm = confusion_matrix(self.y_test, y_pred)
             return {'accuracy': accuracy, 'confusion_matrix': cm}
     ```
   - **Explain:** "Modular code is easier to test, share, and improve"
   - Commit:
     ```bash
     git add iris_classifier.py
     git commit -m "Add modular iris classifier class"
     ```

2. **Push & Create Feature Branch (4 min)**
   - Push to GitHub:
     ```bash
     git push origin main
     ```
   - Create experimental branch:
     ```bash
     git checkout -b experiment/hyperparameter_tuning
     ```
   - Modify classifier to add hyperparameter optimization (add GridSearchCV)
   - Commit:
     ```bash
     git commit -m "Add hyperparameter tuning experiment"
     git push origin experiment/hyperparameter_tuning
     ```

3. **Test in Google Colab (6 min)**
   - Go to GitHub, navigate to branch `experiment/hyperparameter_tuning`
   - Create new Colab notebook
   - Clone repo and import class:
     ```python
     !git clone https://github.com/YOUR_USERNAME/my_ds_project.git
     import sys
     sys.path.append('/content/my_ds_project')
     from iris_classifier import IrisClassifier
     
     # Test classifier
     clf = IrisClassifier()
     df = clf.load_data()
     clf.train(df)
     results = clf.evaluate()
     print(f"Accuracy: {results['accuracy']:.2%}")
     ```
   - Run and verify
   - **Narrative:** "You've tested your experiment in Colab without affecting main branch"

4. **Create Pull Request & Merge (5 min)**
   - Go to GitHub
   - Create PR: Compare `experiment/hyperparameter_tuning` to `main`
   - Add description: "Improved model with hyperparameter tuning using GridSearchCV"
   - Merge PR
   - Pull locally:
     ```bash
     git checkout main
     git pull origin main
     ```
   - **Checkpoint:** Inspect local code—hyperparameter tuning code is now integrated

**F. FINAL DEMO: Team Collaboration Scenario (5 min)**

*Instructor walks through a real collaboration example:*

1. Two teammates create branches for different features
2. Both push to GitHub
3. GitHub shows pending PRs
4. Teammate 1's PR is merged first
5. Teammate 2 pulls latest main, resolves conflicts if needed, re-pushes
6. Teammate 2's PR is merged
7. Both developers pull updated main

**Message:** "This is how teams stay synchronized and avoid conflict."

---

## CLOSING (5 min)
- **Recap:** You've mastered Git locally, GitHub collaboration, and cloud experimentation
- **Next Steps:** Use this workflow for all bootcamp projects
- **FAQ Session:** Address any remaining questions
- **Handout:** Cheat sheet with common commands

---

