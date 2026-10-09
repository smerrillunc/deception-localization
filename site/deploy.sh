#!/usr/bin/env bash
# Publish site/ to the gh-pages branch of a remote you name explicitly.
#
# GitHub Pages can only serve a branch's root or its /docs folder, never an
# arbitrary subfolder, so site/ is pushed to the root of a separate gh-pages
# branch. `git subtree push` refuses to fast-forward once gh-pages has diverged,
# which it will as soon as anyone edits it, so this splits and force-pushes
# instead - gh-pages is generated output and is safe to overwrite.
#
# The remote is a required argument on purpose, so nothing is published to a
# default remote by accident.
#
# Usage:
#   ./deploy.sh <remote> [branch]
#
#   git remote add pages git@github.com:<you>/<repo>.git
#   ./deploy.sh pages
set -euo pipefail

PREFIX="site"

if [[ $# -lt 1 ]]; then
  cat >&2 <<'EOF'
usage: ./deploy.sh <remote> [branch]

Name the remote explicitly - there is no default. To publish somewhere new:

  git remote add pages git@github.com:<you>/<repo>.git
  ./site/deploy.sh pages

Configured remotes:
EOF
  git remote -v | sed 's/^/  /' >&2
  exit 2
fi

REMOTE="$1"
BRANCH="${2:-gh-pages}"

cd "$(git rev-parse --show-toplevel)"

if ! git remote get-url "$REMOTE" >/dev/null 2>&1; then
  echo "error: no remote named '$REMOTE'. Configured remotes:" >&2
  git remote -v | sed 's/^/  /' >&2
  exit 2
fi

URL="$(git remote get-url "$REMOTE")"

if [[ -n "$(git status --porcelain -- "$PREFIX")" ]]; then
  echo "error: $PREFIX/ has uncommitted changes. Commit them first:" >&2
  echo "  git add $PREFIX && git commit -m 'Update site'" >&2
  exit 1
fi

echo "Publishing $PREFIX/ -> $REMOTE ($URL), branch $BRANCH"
SHA="$(git subtree split --prefix "$PREFIX" HEAD)"
git push --force "$REMOTE" "$SHA:refs/heads/$BRANCH"

# strip any .git suffix first - sed has no lazy quantifier to do it inline
CLEAN_URL="${URL%.git}"
PAGES_URL="$(printf '%s' "$CLEAN_URL" | sed -E 's#.*[:/]([^/]+)/([^/]+)$#https://\1.github.io/\2/#')"
cat <<EOF

Pushed. If this is the first deploy, enable it once:
  Settings -> Pages -> Source: "Deploy from a branch" -> $BRANCH / (root)

The site will be at roughly:
  $PAGES_URL

The first build takes a minute or two.
EOF
