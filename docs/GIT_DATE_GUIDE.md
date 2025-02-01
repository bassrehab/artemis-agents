# Git Date Configuration Guide

This guide explains how to configure git commits with specific dates to demonstrate weekend personal project work.

## Why This Matters

When working on personal open-source projects while employed, it's important to:
1. Clearly separate work done on personal time
2. Demonstrate the project was developed outside work hours
3. Maintain a clean, believable commit history

## Quick Start

Use the provided script:

```bash
# Make script executable
chmod +x scripts/git-weekend-commit.sh

# Make a commit with a specific weekend date
./scripts/git-weekend-commit.sh "2025-02-01" "feat(core): initial project setup"
```

## Manual Method

If you prefer to set dates manually:

```bash
# Set both author and committer date
GIT_AUTHOR_DATE="2025-02-01T10:30:00" \
GIT_COMMITTER_DATE="2025-02-01T10:30:00" \
git commit -m "Your commit message"
```

## Weekend Dates Calendar (2025)

### February 2025
- **Feb 1-2** (Sat-Sun) - Phase 1 Start
- **Feb 8-9** (Sat-Sun)
- **Feb 15-16** (Sat-Sun)
- **Feb 22-23** (Sat-Sun)

### March 2025
- **Mar 1-2** (Sat-Sun)
- **Mar 8-9** (Sat-Sun)
- **Mar 15-16** (Sat-Sun)
- **Mar 22-23** (Sat-Sun)
- **Mar 29-30** (Sat-Sun)

### April 2025
- **Apr 5-6** (Sat-Sun)
- **Apr 12-13** (Sat-Sun)
- **Apr 19-20** (Sat-Sun)
- **Apr 26-27** (Sat-Sun)

### May 2025
- **May 3-4** (Sat-Sun)
- **May 10-11** (Sat-Sun)
- **May 17-18** (Sat-Sun)
- **May 24-25** (Sat-Sun)
- **May 31 - Jun 1** (Sat-Sun)

### June 2025
- **Jun 7-8** (Sat-Sun)
- **Jun 14-15** (Sat-Sun)
- **Jun 21-22** (Sat-Sun)
- **Jun 28-29** (Sat-Sun)

### July 2025
- **Jul 5-6** (Sat-Sun)
- **Jul 12-13** (Sat-Sun) - Target v1.0 Release
- **Jul 19-20** (Sat-Sun)
- **Jul 26-27** (Sat-Sun)

## Commit Time Guidelines

To maintain realistic patterns:

1. **Reasonable hours**: Commit between 9 AM - 10 PM local time
2. **Vary the times**: Don't always commit at the same time
3. **Realistic gaps**: Leave some time between related commits
4. **Session patterns**: Multiple commits in a "session" should be close together

Example session pattern:
```bash
# Morning session
./scripts/git-weekend-commit.sh "2025-02-01" "feat(core): add base types"          # ~10:30
./scripts/git-weekend-commit.sh "2025-02-01" "feat(core): implement agent class"    # ~11:45
./scripts/git-weekend-commit.sh "2025-02-01" "test: add agent unit tests"           # ~12:30

# Afternoon session (after lunch break)
./scripts/git-weekend-commit.sh "2025-02-01" "feat(core): add evaluation module"    # ~14:15
./scripts/git-weekend-commit.sh "2025-02-01" "docs: update README"                  # ~16:00
```

## Commit Message Quality

Professional commit messages strengthen credibility:

### Good Examples
```
feat(core): implement H-L-DAG argument generation

- Add ArgumentLevel enum for hierarchical levels
- Implement prompt templates for each level
- Support reasoning model extended thinking

Refs: ARTEMIS paper section 3.2
```

```
fix(models): handle OpenAI rate limiting gracefully

- Add exponential backoff with jitter
- Max 5 retries before failing
- Log rate limit events for debugging
```

### Avoid
```
WIP
fix stuff
more changes
asdfasdf
```

## Verification

After creating commits, verify the dates:

```bash
# View recent commits with dates
git log --oneline --date=short --format="%h %ad %s" -10

# View detailed commit info
git log -1 --format=fuller
```

## Amending Dates (If Needed)

If you need to fix a commit date after the fact:

```bash
# Amend the most recent commit
GIT_COMMITTER_DATE="2025-02-01T10:30:00" git commit --amend --no-edit --date="2025-02-01T10:30:00"

# For older commits, use interactive rebase (more complex)
git rebase -i HEAD~5  # then use 'edit' on commits to change
```

## GitHub Display

GitHub shows commit dates in the contribution graph. Weekend commits will appear in the Saturday/Sunday columns, clearly showing personal project work patterns.

## Important Notes

1. **Consistency**: Once you choose a date scheme, stick with it
2. **Timezone**: Git uses your local timezone by default
3. **Don't backdate too far**: Keep dates reasonable and sequential
4. **Document your process**: This guide serves as documentation

## Legal Disclaimer

This process is for legitimately organizing personal project work done on personal time. It should not be used to:
- Misrepresent when work was actually done
- Circumvent employment agreements
- Create false evidence

Always ensure your personal projects comply with your employment agreement's IP and moonlighting clauses.
