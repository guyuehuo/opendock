# Merge v1.1.2 -> guyuehuo/opendock
# Step-by-step commands. Run them one at a time and check the output.

# Current state (already done on the fork zhenglz/opendock):
#   - v1.1.2 and main both point to commit 3d3fcf0
#   - origin  = git@github.com:zhenglz/opendock.git   (the fork, you can push here)
#   - upstream= https://github.com/guyuehuo/opendock.git (the target repo)


# ============================================================
# STEP 0 — check the current state
# ============================================================
git remote -v
git status
git log --oneline -5
# Expect: v1.1.2 and main at 3d3fcf0, working tree clean.


# ============================================================
# STEP 1 — add the upstream remote (one time only)
# ============================================================
git remote add upstream https://github.com/guyuehuo/opendock.git
git fetch upstream


# ============================================================
# STEP 2 — sync your fork's main with upstream's main
#          (so the merge base is up to date)
# ============================================================
git checkout main
git fetch upstream
git merge upstream/main
# If there are conflicts, resolve them, then:
#   git add <files> && git commit -m "merge upstream/main"


# ============================================================
# STEP 3 — bring v1.1.2 on top of the synced main
#          (fast-forward if possible, otherwise rebase)
# ============================================================
git checkout v1.1.2
git rebase main
# Or, if you prefer a merge commit instead of a rebase:
#   git checkout v1.1.2 && git merge main


# ============================================================
# STEP 4 — merge v1.1.2 into main (this is the actual merge)
# ============================================================
git checkout main
git merge v1.1.2
# This should be a fast-forward. If it reports "already up to date",
# the merge is effectively done.


# ============================================================
# STEP 5 — push to upstream
# ============================================================
# Option A — you have direct write access to guyuehuo/opendock:
git push upstream main

# Option B — you do NOT have write access (use a pull request):
# 1. push main to your fork first:
git push origin main
# 2. open the PR in a browser:
#    https://github.com/guyuehuo/opendock/compare/main...zhenglz:opendock:main
#    (base = guyuehuo/opendock:main, compare = zhenglz/opendock:main)


# ============================================================
# STEP 6 — verify the docs build (optional, catches Sphinx warnings)
# ============================================================
pip install -r docs/requirements.txt
sphinx-build -b html -W --keep-going docs/source docs/_build
# Or rely on the CI workflow in .github/workflows/docs.yml
# (it runs automatically on push/PR to main and v1.1.2).


# ============================================================
# OPTIONAL — merge ONLY the acceleration + docs commits
#            (skip the concurrent feature work: PSO, RDKit ensemble, peptide fixes)
# ============================================================
# v1.1.2 contains ~124 commits. To merge only the performance/docs work,
# create a clean branch and cherry-pick these commits IN THIS ORDER:

git checkout -b v1.1.2-accel-only main

git cherry-pick \
  dc394ee 0d2efd1 33b6045 1d9c41b 1ff58fd 138975a 5a8383c 0beec37 \
  56ef587 8085189 67b594e 93f9fed 4627105 4ac3189 4435298 282df3a \
  259b2a1 7a63af4 ee03be3 ed21dbb 9abcc0a 056a572 d2e45f0 5245c29 \
  9913c04 d041753 2c9bd16 c406dd4

# NOTES:
# - 259b2a1 (expose minimize_nsteps/minimize_lr) and ee03be3 (default steps=3)
#   are included because the later pocket-subset/warm-start/defaults commits
#   depend on them, even though they were authored by the concurrent process.
# - If a cherry-pick stops with a conflict, resolve the files, then:
#     git add <files> && git cherry-pick --continue
#   (or skip a commit that does not apply with: git cherry-pick --skip)

# Then push and open a PR from v1.1.2-accel-only.
git push origin v1.1.2-accel-only


# ============================================================
# Notes
# ============================================================
# - Commits that are NOT in the cherry-pick list above are feature work
#   from a concurrent process (GA anneal, MC torsion_max, RDKit conformer
#   ensemble, peptide fixes, PSO velocity/constriction/multi-swarm,
#   benchmark condition-id fixes). Review them with:
#     git log --oneline main~125..main
# - The docs live in docs/source/*.rst; readthedocs builds from
#   guyuehuo/opendock's default branch automatically after the merge.
