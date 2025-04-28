# Merges an arbitrary number of MODULE.bazel.lock files.
#
# Input: an array of MODULE.bazel.lock JSON objects (as produced by `jq -s`).
# Output: a single MODULE.bazel.lock JSON object.
#
# This script assumes that all files are valid JSON and have a numeric
# "lockFileVersion" field. It will not fail on any such files, but only
# preserves information for files with a version of 10 or higher.
#
# The first file is considered to be the base when deciding which values to
# keep in case of conflicts.

# Like unique, but preserves the order of the first occurrence of each element.
def stable_unique:
  reduce .[] as $item ([]; if index($item) == null then . + [$item] else . end);

# Given an array of objects, shallowly merges the result of applying f to each
# object into a single object, with a few special properties:
# 1. Values are uniquified before merging and then merged with last-wins
#    semantics. Assuming that the first value is the base, this ensures that
#    later occurrences of the base value do not override other values. For
#    example, when this is called with B A1 A2 and A1 contains changes to a
#    field but A2 does not (compared to B), the changes in A1 will be preserved.
# 2. Object keys on the top level are sorted lexicographically after merging,
#    but are additionally split on ":". This ensures that module extension IDs,
#    which start with labels, sort as strings in the same way as they due as
#    structured objects in Bazel (that is, //python/extensions:python.bzl
#    sorts before //python/extensions/private:internal_deps.bzl).
def shallow_merge(f):
  map(f) | stable_unique | add | to_entries | sort_by(.key | split(":")) | from_entries;

(
    # Ignore all MODULE.bazel.lock files that do not have the maximum
    # lockFileVersion.
    (map(.lockFileVersion) | max) as $maxVersion
    | map(select(.lockFileVersion == $maxVersion))
    | {
        lockFileVersion: $maxVersion,
        registryFileHashes: shallow_merge(.registryFileHashes),
        selectedYankedVersions: shallow_merge(.selectedYankedVersions),
        # Group extension results by extension ID across all lockfiles with
        # shallowly merged factors map, then shallowly merge the results.
        moduleExtensions:  (map(.moduleExtensions | to_entries)
                           | flatten
                           | group_by(.key)
                           | shallow_merge({(.[0].key): shallow_merge(.value)})),
        facts: .[0].facts,
    }
    # Filter out null values for missing top-level keys such as facts.
    | with_entries(select(.value != null))
)? //
    # We get here if the lockfiles with the highest lockFileVersion could not be
    # processed, for example because all lockfiles have lockFileVersion < 10.
    # In this case Bazel 7.2.0+ would ignore all lockfiles, so we might as well
    # return the first lockfile for the proper "mismatched version" error
    # message.
    .[0]
