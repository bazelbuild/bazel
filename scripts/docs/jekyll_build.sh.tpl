#!/bin/sh
HOST="${HOST-localhost}"
PORT="${PORT-12345}"
RUNFILES=$(cd ${JAVA_RUNFILES-$0.runfiles}/%{workspace_name} && pwd -P)
SOURCE_DIR="$RUNFILES/%{source_dir}"
prod_dir="$RUNFILES/%{prod_dir}"
bucket="%{bucket}"

function serve() {
  TDIR=$(mktemp -d)
  RDIR=$(mktemp -d)
  trap "rm -fr $RDIR $TDIR" EXIT
  (cd $RDIR && \
    jekyll serve --host "$HOST" --port "$PORT" -s "$SOURCE_DIR" -d "$TDIR")
}

function push() {
  # Get gsutil
  gs="${GSUTIL:-$(which gsutil 2>/dev/null || : )}"
  if [ ! -x "${gs}" ]; then
    echo "Please set GSUTIL to the path the gsutil binary." >&2
    echo "gsutil (https://cloud.google.com/storage/docs/gsutil/) is the" >&2
    echo "command-line interface to google cloud." >&2
    exit 1
  fi

  # Rsync:
  #   -r: recursive
  #   -c: compute checksum even though the input is from the filesystem
  #   -d: remove deleted files
  cd "${prod_dir}"
  "${gs}" -m rsync -r -c -d . "gs://${bucket}"
  "${gs}" web set -m index.html -e 404.html "gs://${bucket}"
  "${gs}" -m acl ch -R -u AllUsers:R "gs://${bucket}"
}

case "${1-}" in
  --push)
    push
    ;;
  --serve|"")
    serve
    ;;
  *)
    echo "Usage: $0 [--push|--serve]" >&2
    exit 1
    ;;
esac
