#!/bin/bash
# Script to remove large PDF files from git history

echo "Removing data folder from git history..."
git filter-branch --force --index-filter \
  'git rm -r --cached --ignore-unmatch data/' \
  --prune-empty --tag-name-filter cat -- --all

echo "Cleaning up refs..."
rm -rf .git/refs/original/
git reflog expire --expire=now --all
git gc --prune=now --aggressive

echo "Done! Now force push to origin:"
echo "git push origin --force --all"
echo ""
echo "Then manually trigger the GitHub Action to sync to Hugging Face"
