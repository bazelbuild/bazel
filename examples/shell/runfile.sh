#!/bin/bash

SOME_FILE="examples/runfile.txt"

root="$(runfiles_current_repository)"
if [ -z "$root" ]; then
  root="_main"
fi
real_path_to_some_file="$(rlocation "${root}/${SOME_FILE}")"

echo "The content of the runfile is:"
cat "${real_path_to_some_file}"
