#!/bin/bash
# Git Weekend Commit Helper
# Usage: ./scripts/git-weekend-commit.sh "2025-02-01" "feat(core): implement debate orchestrator"

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check arguments
if [ $# -lt 2 ]; then
    echo -e "${RED}Error: Missing arguments${NC}"
    echo "Usage: $0 <date> <commit-message>"
    echo "Example: $0 \"2025-02-01\" \"feat(core): implement debate orchestrator\""
    exit 1
fi

DATE=$1
MESSAGE=$2

# Validate date format (YYYY-MM-DD)
if ! [[ $DATE =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
    echo -e "${RED}Error: Invalid date format. Use YYYY-MM-DD${NC}"
    exit 1
fi

# Check if it's a weekend
DAY_OF_WEEK=$(date -d "$DATE" +%u 2>/dev/null || date -j -f "%Y-%m-%d" "$DATE" +%u 2>/dev/null)
if [ "$DAY_OF_WEEK" != "6" ] && [ "$DAY_OF_WEEK" != "7" ]; then
    echo -e "${YELLOW}Warning: $DATE is not a weekend (day $DAY_OF_WEEK)${NC}"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Random hour between 9 AM and 6 PM
HOUR=$((9 + RANDOM % 10))
# Random minute
MINUTE=$((RANDOM % 60))
# Random second
SECOND=$((RANDOM % 60))

# Format the full datetime
FULL_DATE="${DATE}T$(printf "%02d:%02d:%02d" $HOUR $MINUTE $SECOND)"

echo -e "${GREEN}Committing with date: $FULL_DATE${NC}"
echo -e "${GREEN}Message: $MESSAGE${NC}"

# Stage all changes if not already staged
if [ -n "$(git status --porcelain)" ]; then
    echo -e "${YELLOW}Staging all changes...${NC}"
    git add .
fi

# Make the commit with the specified date
GIT_AUTHOR_DATE="$FULL_DATE" \
GIT_COMMITTER_DATE="$FULL_DATE" \
git commit -m "$MESSAGE"

echo -e "${GREEN}✓ Commit created successfully!${NC}"
git log -1 --pretty=format:"Hash: %H%nDate: %ai%nMessage: %s%n"
