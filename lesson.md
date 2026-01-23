# Git/GitHub/VS Code/Colab Bootcamp Lesson
## Designed for Adult Learners (3 Hours | 48 Participants)

---

## LEARNING OBJECTIVES (3 Clear, SMART-Based)

### LO1: Git Essentials & Mental Model
**By the end of Section 1, learners will:**
- Understand Git as a version control system (not file backup)
- Execute core Git commands: init, add, commit, log, status
- Explain the three Git states (working directory → staging → committed) using a real project example
- Create their first local repository and make 3 commits independently

**Assessment:** Can they explain why they'd use `git add` before `git commit`?

---

### LO2: GitHub as Collaboration Hub & VS Code Integration
**By the end of Section 2, learners will:**
- Understand GitHub's role as cloud storage + collaboration platform (not just Git storage)
- Push/pull repositories between local machine and GitHub
- Create a branch, make changes, and submit a pull request (PR) workflow
- Use VS Code's Git integration to perform commits, pushes, and branch switching visually

**Assessment:** Can they push their local repo to GitHub and explain what happened in each step?

---

### LO3: Practical Collaboration & Alternative Setups
**By the end of Section 3, learners will:**
- Execute a team collaboration workflow: clone → branch → PR → merge
- Resolve simple merge conflicts with a partner
- Understand when/why to use Colab as an alternative to local setup
- Set up a Jupyter notebook in Colab linked to their GitHub repo

**Assessment:** Can they clone a team partner's repo, create a feature branch, and submit a PR?

---

## DETAILED LEARNING PLAN (3-Hour Breakdown)

### **SECTION 1: Git Essentials (50 minutes)**
**Theme:** "Git is a time machine for your code"

#### 1.1 Context Setting (5 min)
- **Hook:** Show 2-minute video or live demo of project disaster (file named "final_v3_REAL_final_fix.py") → GitHub restore
- **Why now:** Data science is collaborative; Git prevents chaos
- **Adult learning principle:** Start with relevance to their bootcamp projects

#### 1.2 Conceptual Foundation (15 min) - DEMO
- Git ≠ GitHub (Git is local tool, GitHub is cloud platform)
- Three states explanation with visuals:
  - Working Directory (your laptop files)
  - Staging Area (files ready to commit)
  - Repository (committed history)
- Analogy: "Like saving drafts → selecting which drafts to publish → publishing"
- Show sample `git status` output explained line-by-line

#### 1.3 Hands-On Workshop 1: First Repository (30 min)
**Learning by Doing:**
1. **Setup (5 min):**
   - Create ~/bootcamp_practice folder
   - `git init` and show hidden .git folder (explain what it is)
   - Configure: `git config user.name` & `git config user.email`

2. **Create & Commit Cycle (15 min):**
   - Create simple data_analysis.py (5 lines of Python)
   - `git add data_analysis.py`
   - `git commit -m "Initial data loading function"`
   - `git log` to see history
   - **Instructor checkpoint:** Walk room, verify each learner has 1 commit

3. **Make Changes & Understand States (10 min):**
   - Modify data_analysis.py (add a comment)
   - `git status` (show "modified" state)
   - `git add` and `git commit` again
   - Repeat 1 more time (total: 3 commits)
   - `git log --oneline` to see commit history visually
   - **Learner checkpoint:** "Show me your git log with 3 commits"

**Troubleshooting Station:** Have 2 TAs ready for:
- "fatal: not a git repository" (forgot `git init`)
- `git config` issues (first-time Git users)

---

### **SECTION 2: GitHub & VS Code Integration (70 minutes)**
**Theme:** "From local to cloud; from terminal to GUI"

#### 2.1 GitHub Setup & Mental Model (10 min) - DEMO
- **Live walkthrough:** Create repo on GitHub UI (https://github.com/new)
- Explain: remote repository = cloud backup + team collaboration hub
- Show SSH key setup (or HTTPS for simplicity—decide based on your group's comfort)
- Mental model: "Local repo (yours) ↔️ Remote repo (shared)"

#### 2.2 Push/Pull Workflow (10 min) - DEMO
- Connect local to remote: `git remote add origin https://...`
- `git branch -M main` (rename default branch for clarity)
- `git push -u origin main` (push first time with tracking)
- Show repo on GitHub—"your code is now in the cloud"
- Demo: edit file on GitHub UI → `git pull` locally (show file changed)

#### 2.3 Hands-On Workshop 2: Push to GitHub & Use VS Code (40 min)

**Part A: Connect & Push to GitHub (15 min)**
1. Each learner creates a NEW public GitHub repo (empty, no README)
2. Copy SSH/HTTPS URL
3. In terminal:
   ```
   git remote add origin [their_url]
   git branch -M main
   git push -u origin main
   ```
4. Verify on GitHub.com—"Refresh and see your code online"
5. **Checkpoint:** Screenshot of their repo on GitHub posted in Slack

**Part B: Introduce VS Code Git Integration (10 min)**
- Open their project folder in VS Code
- Show Source Control panel (left sidebar icon with 3 circles)
- Explain: VS Code GUI replaces terminal commands
- Demo equivalent workflow in VS Code:
  - Edit file
  - Stage (click + button)
  - Commit (fill message box, click checkmark)
  - Sync/Push (click "Sync Changes")

**Part C: Practice with VS Code (15 min)**
1. Add a new file (README.md with project description)
2. In VS Code Source Control:
   - Stage the file
   - Write commit message
   - Commit
   - Sync/Push
3. Verify on GitHub—file appears online
4. Repeat: edit README locally in VS Code → commit → push
5. **Checkpoint:** "Your README now has 2 commits visible on GitHub"

**Troubleshooting Station:**
- SSH key issues (have HTTPS fallback ready)
- "Permission denied" (verify GitHub credentials)

#### 2.4 Branch Basics (10 min) - DEMO
- Why branches? "Experiment without breaking main"
- Create feature branch: `git checkout -b feature/add-model`
- In VS Code: click branch name → "Create Branch"
- Show: you can commit to feature branch without touching main
- **Key insight:** "Branches isolate work; makes collaboration safe"

---

### **SECTION 3: Team Collaboration, Merge Conflicts & Colab Alternative (40 minutes)**
**Theme:** "Real-world data science workflows"

#### 3.1 Pull Request Workflow (10 min) - DEMO
- Why PRs? "Code review before merging"
- Demo: create branch → make changes → push → open PR on GitHub
- Show PR interface: description, commit history, "Merge" button
- Explain: "PR = proposal to change code; teammate reviews"
- Real scenario: "Your partner adds a new preprocessing function; you review via PR"

#### 3.2 Hands-On Workshop 3: Team Collaboration & Merge Conflict (25 min)

**Part A: Pair Programming Setup (5 min)**
- Learners pair up (24 pairs)
- Each pair creates a shared GitHub repo (one person creates, invites partner as collaborator)
- Both clone repo to their laptop

**Part B: Create Feature Branches & PRs (12 min)**
1. **Partner A:** Create branch `feature/data-cleaning`
   - Add function to repo (simple preprocessing)
   - Commit & push
   - Open PR on GitHub with description
2. **Partner B:** 
   - Meanwhile, create branch `feature/eda-plots`
   - Add function to repo (simple visualization)
   - Commit & push
   - Open PR
3. **Both:** Review each other's PR → approve → merge to main

**Part C: Intentional Merge Conflict (8 min)**
1. **Setup conflict:** Both partners edit same line in same file
   - Partner A: `main_analysis.py` line 5 → `threshold = 0.5`
   - Partner B: `main_analysis.py` line 5 → `threshold = 0.75`
2. Both push to different branches → create PRs
3. Merge Partner A's PR first (no conflict)
4. Try to merge Partner B's PR → **CONFLICT!**
5. **Live resolution (instructor-led, 5 min):**
   - Show VS Code conflict markers
   - Explain: choose which version (0.5 or 0.75)
   - Resolve manually
   - Commit conflict resolution
   - Merge successfully
6. **Key learning:** "Conflicts are normal; don't panic—Git helps you manage them"

**Checkpoint:** "Both your PRs merged; repo now has all code"

#### 3.3 Colab Alternative Setup (5 min) - DEMO + OPTIONAL HANDS-ON

**When to use Colab:**
- Python environment issues on local machine
- Collaborative real-time editing
- GPU access (free)
- No installation needed

**Live Demo:**
1. Create Jupyter notebook in Google Colab
2. Link to GitHub:
   - Clone repo: `!git clone https://[repo_url]`
   - Work in Colab
   - Push changes back: setup Git auth in Colab (or download notebook)
3. Show: "Your Colab notebook can commit to GitHub"

**Optional:** Learners with local Python issues try Colab setup (TA support)
