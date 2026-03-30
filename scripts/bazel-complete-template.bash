# -*- sh -*- (Bash only)
#
# Copyright 2015 The Bazel Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# The template is expanded at build time using tables of commands/options
# derived from the bazel executable built in the same client; the expansion is
# written to bazel-complete.bash.
#
# Don't use this script directly. Generate the final script with
# bazel build //scripts:bash_completion instead.

# This script expects a header to be prepended to it that defines the following
# nullary functions:
#
# _bazel_completion_use_query - Has a successful exit code if
# BAZEL_COMPLETION_USE_QUERY is "true".
#
# _bazel_completion_allow_tests_for_run - Has a successful exit code if
# BAZEL_COMPLETION_ALLOW_TESTS_FOR_RUN is "true".

# The package path used by the completion routines.  Unfortunately
# this isn't necessarily the same as the actual package path used by
# Bazel, but that's ok.  (It's impossible for us to reliably know what
# the relevant package-path, so this is just a good guess.  Users can
# override it if they want.)
: ${BAZEL_COMPLETION_PACKAGE_PATH:=%workspace%}

# Some commands might interfer with the important one, so don't complete them
: ${BAZEL_IGNORED_COMMAND_REGEX:="__none__"}

# bazel & ibazel commands
: ${BAZEL:=bazel}
: ${IBAZEL:=ibazel}

# Pattern to match for looking for a target
#  BAZEL_BUILD_MATCH_PATTERN__* give the pattern for label-*
#  when looking in the build file.
#  BAZEL_QUERY_MATCH_PATTERN__* give the pattern for label-*
#  when using 'bazel query'.
# _RUNTEST is a special case for _bazel_completion_allow_tests_for_run.
: ${BAZEL_BUILD_MATCH_PATTERN__test:='(.*_test|test_suite)'}
: ${BAZEL_QUERY_MATCH_PATTERN__test:='(test|test_suite)'}
: ${BAZEL_BUILD_MATCH_PATTERN__bin:='.*_binary'}
: ${BAZEL_QUERY_MATCH_PATTERN__bin:='(binary)'}
: ${BAZEL_BUILD_MATCH_PATTERN_RUNTEST__bin:='(.*_(binary|test)|test_suite)'}
: ${BAZEL_QUERY_MATCH_PATTERN_RUNTEST__bin:='(binary|test)'}
: ${BAZEL_BUILD_MATCH_PATTERN__:='.*'}
: ${BAZEL_QUERY_MATCH_PATTERN__:=''}

# Usage: _bazel__get_rule_match_pattern <command>
# Determine what kind of rules to match, based on command.
_bazel__get_rule_match_pattern() {
  local var_name pattern
  if _bazel_completion_use_query; then
    var_name="BAZEL_QUERY_MATCH_PATTERN"
  else
    var_name="BAZEL_BUILD_MATCH_PATTERN"
  fi
  if [[ "$1" =~ ^label-?([a-z]*)$ ]]; then
    pattern=${BASH_REMATCH[1]:-}
    if _bazel_completion_allow_tests_for_run; then
      eval "echo \"\${${var_name}_RUNTEST__${pattern}:-\$${var_name}__${pattern}}\""
    else
      eval "echo \"\$${var_name}__${pattern}\""
    fi
  fi
}

# Compute workspace directory. Search for the innermost
# enclosing directory with a boundary file (see
# src/main/cpp/workspace_layout.cc).
_bazel__get_workspace_path() {
  local workspace=$PWD
  while true; do
    if [ -f "${workspace}/WORKSPACE" ] || \
       [ -f "${workspace}/WORKSPACE.bazel" ] || \
       [ -f "${workspace}/MODULE.bazel" ] || \
       [ -f "${workspace}/REPO.bazel" ]; then
      break
    elif [ -z "$workspace" ] || [ "$workspace" = "/" ]; then
      workspace=$PWD
      break;
    fi
    workspace=${workspace%/*}
  done
  echo $workspace
}

# Find the current piece of the line to complete, but only do word breaks at
# certain characters. In particular, ignore these: "':=@
# This method also takes into account the current cursor position.
#
# Works with both bash 3 and 4! Bash 3 and 4 perform different word breaks when
# computing the COMP_WORDS array. We need this here because Bazel options are of
# the form --a=b, and labels of the form //some/label:target.
_bazel__get_cword() {
  local cur=${COMP_LINE:0:$COMP_POINT}
  # This expression finds the last word break character, as defined in the
  # COMP_WORDBREAKS variable, but without '@', '=' or ':', which is not
  # preceded by a slash. Quote characters are also excluded.
  local wordbreaks="$COMP_WORDBREAKS"
  wordbreaks="${wordbreaks//\'/}"
  wordbreaks="${wordbreaks//\"/}"
  wordbreaks="${wordbreaks//:/}"
  wordbreaks="${wordbreaks//=/}"
  wordbreaks="${wordbreaks//@/}"
  local word_start=$(expr "$cur" : '.*[^\]['"${wordbreaks}"']')
  echo "${cur:$word_start}"
}


# Usage: _bazel__package_path <workspace> <displacement>
#
# Prints a list of package-path root directories, displaced using the
# current displacement from the workspace.  All elements have a
# trailing slash.
_bazel__package_path() {
  local workspace=$1 displacement=$2 root
  IFS=:
  for root in ${BAZEL_COMPLETION_PACKAGE_PATH//\%workspace\%/$workspace}; do
    unset IFS
    echo "$root/$displacement"
  done
}

# Usage: _bazel__options_for <command>
#
# Prints the set of options for a given Bazel command, e.g. "build".
_bazel__options_for() {
  local options
  if [[ "${BAZEL_COMMAND_LIST}" =~ ^(.* )?$1( .*)?$ ]]; then
      # assumes option names only use ASCII characters
      local option_name=$(echo $1 | tr a-z A-Z | tr "-" "_")
      eval "echo \${BAZEL_COMMAND_${option_name}_FLAGS}" | tr " " "\n"
  fi
}
# Usage: _bazel__expansion_for <command>
#
# Prints the completion pattern for a given Bazel command, e.g. "build".
_bazel__expansion_for() {
  local options
  if [[ "${BAZEL_COMMAND_LIST}" =~ ^(.* )?$1( .*)?$ ]]; then
      # assumes option names only use ASCII characters
      local option_name=$(echo $1 | tr a-z A-Z | tr "-" "_")
      eval "echo \${BAZEL_COMMAND_${option_name}_ARGUMENT}"
  fi
}

# Usage: _bazel__matching_targets <kind> <prefix>
#
# Prints target names of kind <kind> and starting with <prefix> in the BUILD
# file given as standard input.  <kind> is a basic regex (BRE) used to match the
# bazel rule kind and <prefix> is the prefix of the target name.
_bazel__matching_targets() {
  local kind_pattern="$1"
  local target_prefix="$2"
  # The following commands do respectively:
  #   Remove BUILD file comments
  #   Replace \n by spaces to have the BUILD file in a single line
  #   Extract all rule types and target names
  #   Grep the kind pattern and the target prefix
  #   Returns the target name
  sed 's/#.*$//' \
      | tr "\n" " " \
      | sed 's/\([a-zA-Z0-9_]*\) *(\([^)]* \)\{0,1\}name *= *['\''"]\([a-zA-Z0-9_/.+=,@~-]*\)['\''"][^)]*)/\
type:\1 name:\3\
/g' \
      | "grep" -E "^type:$kind_pattern name:$target_prefix" \
      | cut -d ':' -f 3
}


# Usage: _bazel__is_true <string>
#
# Returns true or false based on the input string. The following are
# valid true values (the rest are false): "1", "true".
_bazel__is_true() {
  local str="$1"
  [[ "$str" == "1" || "$str" == "true" ]]
}

# Usage: _bazel__expand_rules_in_package <workspace> <displacement>
#                                        <current> <label-type>
#
# Expands rules in specified packages, exploring all roots of
# $BAZEL_COMPLETION_PACKAGE_PATH, not just $(pwd).  Only rules
# appropriate to the command are printed.  Sets $COMPREPLY array to
# result.
#
# If _bazel_completion_use_query has a successful exit code, 'bazel query' is
# used instead, with the actual Bazel package path;
# $BAZEL_COMPLETION_PACKAGE_PATH is ignored in this case, since the actual Bazel
# value is likely to be more accurate.
_bazel__expand_rules_in_package() {
  local workspace=$1 displacement=$2 current=$3 label_type=$4
  local package_name=$(echo "$current" | cut -f1 -d:)
  local rule_prefix=$(echo "$current" | cut -f2 -d:)
  local root buildfile rule_pattern r result

  result=
  pattern=$(_bazel__get_rule_match_pattern "$label_type")
  if _bazel_completion_use_query; then
    package_name=$(echo "$package_name" | tr -d "'\"") # remove quotes
    result=$(${BAZEL} --output_base=/tmp/${BAZEL}-completion-$USER query \
                   --keep_going --noshow_progress --output=label \
      "kind('$pattern rule', '$package_name:*')" 2>/dev/null |
      cut -f2 -d: | "grep" "^$rule_prefix")
  else
    for root in $(_bazel__package_path "$workspace" "$displacement"); do
      buildfile="$root/$package_name/BUILD.bazel"
      if [ ! -f "$buildfile" ]; then
        buildfile="$root/$package_name/BUILD"
      fi
      if [ -f "$buildfile" ]; then
        result=$(_bazel__matching_targets \
                   "$pattern" "$rule_prefix" <"$buildfile")
        break
      fi
    done
  fi

  index=$(echo $result | wc -w)
  if [ -n "$result" ]; then
      echo "$result" | tr " " "\n" | sed 's|$| |'
  fi
  # Include ":all" wildcard if there was no unique match.  (The zero
  # case is tricky: we need to include "all" in that case since
  # otherwise we won't expand "a" to "all" in the absence of rules
  # starting with "a".)
  if [ $index -ne 1 ] && expr all : "\\($rule_prefix\\)" >/dev/null; then
    echo "all "
  fi
}

# Usage: _bazel__expand_package_name <workspace> <displacement> <current-word>
#                                    <label-type>
#
# Expands directories, but explores all roots of
# BAZEL_COMPLETION_PACKAGE_PATH, not just $(pwd).  When a directory is
# a bazel package, the completion offers "pkg:" so you can expand
# inside the package.
# Sets $COMPREPLY array to result.
_bazel__expand_package_name() {
  local workspace=$1 displacement=$2 current=$3 type=${4:-} root dir index
  for root in $(_bazel__package_path "$workspace" "$displacement"); do
    found=0
    for dir in $(compgen -d $root$current); do
      [ -L "$dir" ] && continue  # skip symlinks (e.g. bazel-bin)
      [[ "$dir" =~ ^(.*/)?\.[^/]*$ ]] && continue  # skip dotted dir (e.g. .git)
      found=1
      echo "${dir#$root}/"
      if [ -f $dir/BUILD.bazel -o -f $dir/BUILD ]; then
        if [ "${type}" = "label-package" ]; then
          echo "${dir#$root} "
        else
          echo "${dir#$root}:"
        fi
      fi
    done
    # The loop over the compgen -d output above does not include the top-level
    # package.
    if [ -f $root$current/BUILD.bazel -o -f $root$current/BUILD ]; then
      found=1
      if [ "${type}" != "label-package" ]; then
        echo "${current}:"
      fi
    fi
    [ $found -gt 0 ] && break  # Stop searching package path upon first match.
  done
}

# Usage: _bazel__filter_repo_mapping <filter> <field>
#
# Returns all entries of the main repo's repository mapping whose apparent repo
# name, followed by a double quote, matches the given filter. To return the
# matching apparent names, set field to 2. To return the matching canonical
# names, set field to 4.
# Note: Instead of returning an empty canonical name for the main repository,
# this function returns the string "_main" so that this case can be
# distinguished from that of no match.
_bazel__filter_repo_mapping() {
  local filter=$1 field=$2
  # 1. dump_repo_mapping '' returns a single line consisting of a minified JSON
  #    object.
  # 2. Transform JSON to have lines of the form "apparent_name":"canonical_name".
  # 3. Filter by apparent repo name.
  # 4. Replace an empty canonical name with "_main".
  # 5. Cut out either the apparent or canonical name.
  ${BAZEL} mod dump_repo_mapping '' --noshow_progress 2>/dev/null |
    tr '{},' '\n' |
    "grep" "^\"${filter}" |
    sed 's|:""$|:"_main"|' |
    cut -d'"' -f${field}
}

# Usage: _bazel__expand_repo_name <current>
#
# Returns completions for apparent repository names. Each line is of the form
# @apparent_name or @apparent_name//, where apparent_name starts with current.
_bazel__expand_repo_name() {
  local current=$1
  # If current exactly matches a repo name, also provide the @current//
  # completion so that users can tab through to package completion, but also
  # complete just the shorthand for "@repo_name//:repo_name".
  _bazel__filter_repo_mapping "${current#@}" 2 |
    sed 's|^|@|' |
    sed "s|^${current}\$|${current} ${current}//|"
}

# Usage: _bazel__repo_root <workspace> <repo>
#
# Returns the absolute path to the root of the repository identified by the
# repository part <repo> of a label. <repo> can be either of the form
# "@apparent_name" or "@@canonical_name" and may also refer to the main
# repository.
_bazel__repo_root() {
  local workspace=$1 repo=$2
  local canonical_repo
  if [[ "$repo" == @@ ]]; then
    # Match the sentinel value for the main repository used by
    # _bazel__filter_repo_mapping.
    canonical_repo=_main
  elif [[ "$repo" =~ ^@@ ]]; then
    # Canonical repo names should not go through repo mapping.
    canonical_repo=${repo#@@}
  else
    canonical_repo=$(_bazel__filter_repo_mapping "${repo#@}\"" 4)
  fi
  if [ -z "$canonical_repo" ]; then
    return
  fi
  if [ "$canonical_repo" == "_main" ]; then
    echo "$workspace"
    return
  fi
  local output_base="$(${BAZEL} info output_base --noshow_progress 2>/dev/null)"
  if [ -z "$output_base" ]; then
    return
  fi
  local repo_root="$output_base/external/$canonical_repo"
  echo "$repo_root"
}

# Usage: _bazel__expand_package_name <workspace> <current> <label-type>
#
# Expands packages under the potentially external repository pointed to by
# <current>, which is expected to start with "@repo//".
_bazel__expand_external_package_name() {
  local workspace=$1 current=$2 label_syntax=$3
  local repo=$(echo "$current" | cut -f1 -d/)
  local package=$(echo "$current" | cut -f3- -d/)
  local repo_root=$(_bazel__repo_root "$workspace" "$repo")
  if [ -z "$repo_root" ]; then
    return
  fi
  _bazel__expand_package_name "$repo_root" "" "$package" "$label_syntax" |
    sed "s|^|${repo}//|"
}

# Usage: _bazel__expand_rules_in_external_package <workspace> <current>
#                                                 <label-type>
#
# Expands rule names in the potentially external package pointed to by
# <current>, which is expected to start with "@repo//some/pkg:".
_bazel__expand_rules_in_external_package() {
  local workspace=$1 current=$2 label_syntax=$3
  local repo=$(echo "$current" | cut -f1 -d/)
  local package=$(echo "$current" | cut -f3- -d/ | cut -f1 -d:)
  local name=$(echo "$current" | cut -f2 -d:)
  local repo_root=$(_bazel__repo_root "$workspace" "$repo")
  if [ -z "$repo_root" ]; then
    return
  fi
  _bazel__expand_rules_in_package "$repo_root" "" "//$package:$name" "$label_syntax"
}

# Usage: _bazel__expand_target_pattern <workspace> <displacement>
#                                      <word> <label-syntax>
#
# Expands "word" to match target patterns, using the current workspace
# and displacement from it.  "command" is used to filter rules.
# Sets $COMPREPLY array to result.
_bazel__expand_target_pattern() {
  local workspace=$1 displacement=$2 current=$3 label_syntax=$4
  case "$current" in
    @*//*:*) # Expand rule names within external repository.
      _bazel__expand_rules_in_external_package "$workspace" "$current" "$label_syntax"
      ;;
    @*/*) # Expand package names within external repository.
      # Append a second slash after the repo name before performing completion
      # if there is no second slash already.
      if [[ "$current" =~ ^@[^/]*/$ ]]; then
        current="$current/"
      fi
      _bazel__expand_external_package_name "$workspace" "$current" "$label_syntax"
      ;;
    @*) # Expand external repository names.
      # Do not expand canonical repository names: Users are not expected to
      # compose them manually and completing them based on the contents of the
      # external directory has a high risk of returning stale results.
      if [[ "$current" =~ ^@@ ]]; then
        return
      fi
      _bazel__expand_repo_name "$current"
      ;;
    //*:*) # Expand rule names within package, no displacement.
      if [ "${label_syntax}" = "label-package" ]; then
        compgen -S " " -W "BUILD" "$(echo current | cut -f ':' -d2)"
      else
        _bazel__expand_rules_in_package "$workspace" "" "$current" "$label_syntax"
      fi
      ;;
    *:*) # Expand rule names within package, displaced.
      if [ "${label_syntax}" = "label-package" ]; then
        compgen -S " " -W "BUILD" "$(echo current | cut -f ':' -d2)"
      else
        _bazel__expand_rules_in_package \
          "$workspace" "$displacement" "$current" "$label_syntax"
      fi
      ;;
    //*) # Expand filenames using package-path, no displacement
      _bazel__expand_package_name "$workspace" "" "$current" "$label_syntax"
      ;;
    *) # Expand filenames using package-path, displaced.
      if [ -n "$current" ]; then
        _bazel__expand_package_name "$workspace" "$displacement" "$current" "$label_syntax"
      fi
      ;;
  esac
}

_bazel__get_command() {
  for word in "${COMP_WORDS[@]:1:COMP_CWORD-1}"; do
    if echo "$BAZEL_COMMAND_LIST" | "grep" -wsq -e "$word"; then
      echo $word
      break
    fi
  done
}

# Returns the displacement to the workspace given in $1
_bazel__get_displacement() {
  if [[ "$PWD" =~ ^$1/.*$ ]]; then
    echo ${PWD##$1/}/
  fi
}


# Usage: _bazel__complete_pattern <workspace> <displacement> <current>
#                                 <type>
#
# Expand a word according to a type. The currently supported types are:
#  - {a,b,c}: an enum that can take value a, b or c
#  - label: a label of any kind
#  - label-bin: a label to a runnable rule (basically to a _binary rule)
#  - label-test: a label to a test rule
#  - info-key: an info key as listed by `bazel help info-keys`
#  - command: the name of a command
#  - path: a file path
#  - combinaison of previous type using | as separator
_bazel__complete_pattern() {
  local workspace=$1 displacement=$2 current=$3 types=$4
  for type in $(echo $types | tr "|" "\n"); do
    case "$type" in
      label*)
        _bazel__expand_target_pattern "$workspace" "$displacement" \
            "$current" "$type"
        ;;
      info-key)
    compgen -S " " -W "${BAZEL_INFO_KEYS}" -- "$current"
        ;;
      "command")
        local commands=$(echo "${BAZEL_COMMAND_LIST}" \
          | tr " " "\n" | "grep" -v "^${BAZEL_IGNORED_COMMAND_REGEX}$")
    compgen -S " " -W "${commands}" -- "$current"
        ;;
      path)
        for file in $(compgen -f -- "$current"); do
          if [[ -d "$file" ]]; then
            echo "$file/"
          else
            echo "$file "
          fi
        done
        ;;
      *)
        compgen -S " " -W "$type" -- "$current"
        ;;
    esac
  done
}

# Usage: _bazel__expand_options <workspace> <displacement> <current-word>
#                               <options>
#
# Expands options, making sure that if current-word contains an equals sign,
# it is handled appropriately.
_bazel__expand_options() {
  local workspace="$1" displacement="$2" cur="$3" options="$4"
  if [[ $cur =~ = ]]; then
    # also expands special labels
    current=$(echo "$cur" | cut -f2 -d=)
    _bazel__complete_pattern "$workspace" "$displacement" "$current" \
    "$(compgen -W "$options" -- "$cur" | cut -f2 -d=)" \
        | sort -u
  else
    compgen -W "$(echo "$options" | sed 's|=.*$|=|')" -- "$cur" \
    | sed 's|\([^=]\)$|\1 |'
  fi
}

# Usage: _bazel__abspath <file>
#
#
# Returns the absolute path to a file
_bazel__abspath() {
    echo "$(cd "$(dirname "$1")"; pwd)/$(basename "$1")"
 }

# Usage: _bazel__rc_imports <workspace> <rc-file>
#
#
# Returns the list of other RC imported (or try-imported) by a given RC file
# Only return files we can actually find, and only return absolute paths
_bazel__rc_imports() {
  local workspace="$1" rc_dir rc_file="$2" import_files
  rc_dir=$(dirname $rc_file)
  import_files=$(cat $rc_file \
      | sed 's/#.*//' \
      | sed -E "/^(try-){0,1}import/!d" \
      | sed -E "s/^(try-){0,1}import ([^ ]*).*$/\2/" \
      | sort -u)

  local confirmed_import_files=()
  for import in $import_files; do
    # rc imports can use %workspace% to refer to the workspace, and we need to substitute that here
    import=${import//\%workspace\%/$workspace}
    if [[ "${import:0:1}" != "/" ]] ; then
      import="$rc_dir/$import"
    fi
    import=$(_bazel__abspath $import)
    # Don't bother dealing with it further if we can't find it
    if [ -f "$import" ] ; then
      confirmed_import_files+=($import)
    fi
  done
  echo "${confirmed_import_files[@]}"
}

# Usage: _bazel__rc_expand_imports <workspace> <processed-rc-files ...> __new__ <new-rc-files ...>
#
#
# Function that receives a workspace and two lists. The first list is a list of RC files that have
# already been processed, and the second list contains new RC files that need processing. Each new file will be
# processed for "{try-}import" lines to discover more RC files that need parsing.
# Any lines we find in "{try-}import" will be checked against known files (and not processed again if known).
_bazel__rc_expand_imports() {
  local workspace="$1" rc_file new_found="no" processed_files=() to_process_files=() discovered_files=()
  # We've consumed workspace
  shift
  # Now grab everything else
  local all_files=($@)
  for rc_file in ${all_files[@]} ; do
    if [ "$rc_file" == "__new__" ] ; then
      new_found="yes"
      continue
    elif [ "$new_found" == "no" ] ; then
      processed_files+=($rc_file)
    else
      to_process_files+=($rc_file)
    fi
  done

  # For all the non-processed files, get the list of imports out of each of those files
  for rc_file in "${to_process_files[@]}"; do
    local potential_new_files+=($(_bazel__rc_imports "$workspace" "$rc_file"))
    processed_files+=($rc_file)
    for potential_new_file in ${potential_new_files[@]} ; do
      if [[ ! " ${processed_files[@]} " =~ " ${potential_new_file} " ]] ; then
        discovered_files+=($potential_new_file)
      fi
    done
  done

  # Finally, return two lists (separated by __new__) of the files that have been fully processed, and
  # the files that need processing.
  echo "${processed_files[@]}" "__new__" "${discovered_files[@]}"
}

# Usage: _bazel__rc_files <workspace>
#
#
# Returns the list of RC files to examine, given the current command-line args.
_bazel__rc_files() {
  local workspace="$1" new_files=() processed_files=()
  # Handle the workspace RC unless --noworkspace_rc says not to.
  if [[ ! "${COMP_LINE}" =~ "--noworkspace_rc" ]]; then
    local workspacerc="$workspace/.bazelrc"
    if [ -f "$workspacerc" ] ; then
      new_files+=($(_bazel__abspath $workspacerc))
    fi
  fi
  # Handle the $HOME RC unless --nohome_rc says not to.
  if [[ ! "${COMP_LINE}" =~ "--nohome_rc" ]]; then
    local homerc="$HOME/.bazelrc"
    if [ -f "$homerc" ] ; then
      new_files+=($(_bazel__abspath $homerc))
    fi
  fi
  # Handle the system level RC unless --nosystem_rc says not to.
  if [[ ! "${COMP_LINE}" =~ "--nosystem_rc" ]]; then
    local systemrc="/etc/bazel.bazelrc"
    if [ -f "$systemrc" ] ; then
      new_files+=($(_bazel__abspath $systemrc))
    fi
  fi
  # Finally, if the user specified any on the command-line, then grab those
  # keeping in mind that there may be several.
  if [[ "${COMP_LINE}" =~ "--bazelrc=" ]]; then
    # There's got to be a better way to do this, but... it gets the job done,
    # even if there are multiple --bazelrc on the command line. The sed command
    # SHOULD be simpler, but capturing several instances of the same pattern
    # from the same line is tricky because of the greedy nature of .*
    # Instead we transform it to multiple lines, and then back.
    local cmdlnrcs=$(echo ${COMP_LINE} | sed -E "s/--bazelrc=/\n--bazelrc=/g" | sed -E "/--bazelrc/!d;s/^--bazelrc=([^ ]*).*$/\1/g" | tr "\n" " ")
    for rc_file in $cmdlnrcs; do
      if [ -f "$rc_file" ] ; then
        new_files+=($(_bazel__abspath $rc_file))
      fi
    done
  fi

  # Each time we call _bazel__rc_expand_imports, it may find new files which then need to be expanded as well,
  # so we loop until we've processed all new files.
  while (( ${#new_files[@]} > 0 )) ; do
    local all_files=($(_bazel__rc_expand_imports "$workspace" "${processed_files[@]}" "__new__" "${new_files[@]}"))
    local new_found="no"
    new_files=()
    processed_files=()
    for file in ${all_files[@]} ; do
      if [ "$file" == "__new__" ] ; then
        new_found="yes"
        continue
      elif [ "$new_found" == "no" ] ; then
        processed_files+=($file)
      else
        new_files+=($file)
      fi
    done
  done

  echo "${processed_files[@]}"
}

# Usage: _bazel__all_configs <workspace> <command>
#
#
# Gets contents of all RC files and searches them for config names
# that could be used for expansion.
_bazel__all_configs() {
  local workspace="$1" command="$2" rc_files

  # Start out getting a list of all RC files that we can look for configs in
  # This respects the various command line options documented at
  # https://bazel.build/docs/bazelrc
  rc_files=$(_bazel__rc_files "$workspace")

  # Commands can inherit configs from other commands, so build up command_match, which is
  # a match list of the various commands that we can match against, given the command
  # specified by the user
  local build_inherit=("aquery" "clean" "coverage" "cquery" "info" "mobile-install" "print_action" "run" "test")
  local test_inherit=("coverage")
  local command_match="$command|common|always"
  if [[ "${build_inherit[@]}" =~ "$command" ]]; then
    command_match="$command_match|build"
  fi
  if [[ "${test_inherit[@]}" =~ "$command" ]]; then
    command_match="$command_match|test"
  fi

  # The following commands do respectively:
  #   Gets the contents of all relevant/allowed RC files
  #   Remove file comments
  #   Filter only the configs relevant to the current command
  #   Extract the config names
  #   Filters out redundant names and returns the results
  cat $rc_files \
      | sed 's/#.*//' \
      | sed -E "/^($command_match):/!d" \
      | sed -E "s/^($command_match):([^ ]*).*$/\2/" \
      | sort -u
}

# Usage: _bazel__expand_config <workspace> <command> <current-word>
#
#
# Expands configs, checking through the allowed rc files and parsing for configs
# relevant to the current command
_bazel__expand_config() {
  local workspace="$1" command="$2" cur="$3" rc_files all_configs
  all_configs=$(_bazel__all_configs "$workspace" "$command")
  compgen -S " " -W "$all_configs" -- "$cur"
}

_bazel__is_after_doubledash() {
  for word in "${COMP_WORDS[@]:1:COMP_CWORD-1}"; do
    if [[ "$word" == "--" ]]; then
      return 0
    fi
  done
  return 1
}

_bazel__complete_stdout() {
  local cur=$(_bazel__get_cword) word command displacement workspace

  # Determine command: "" (startup-options) or one of $BAZEL_COMMAND_LIST.
  command="$(_bazel__get_command)"

  workspace="$(_bazel__get_workspace_path)"
  displacement="$(_bazel__get_displacement ${workspace})"

  if _bazel__is_after_doubledash && [[ "$command" == "run" ]]; then
    _bazel__complete_pattern "$workspace" "$displacement" "${cur#*=}" "path"
  else
    case "$command" in
      "") # Expand startup-options or commands
        local commands=$(echo "${BAZEL_COMMAND_LIST}" \
          | tr " " "\n" | "grep" -v "^${BAZEL_IGNORED_COMMAND_REGEX}$")
        _bazel__expand_options  "$workspace" "$displacement" "$cur" \
            "${commands}\
            ${BAZEL_STARTUP_OPTIONS}"
        ;;

      *)
        case "$cur" in
          --config=*) # Expand options:
            _bazel__expand_config  "$workspace" "$command" "${cur#"--config="}"
            ;;
          -*) # Expand options:
            _bazel__expand_options  "$workspace" "$displacement" "$cur" \
                "$(_bazel__options_for $command)"
            ;;
          *)  # Expand target pattern
        expansion_pattern="$(_bazel__expansion_for $command)"
            NON_QUOTE_REGEX="^[\"']"
            if [[ $command = query && $cur =~ $NON_QUOTE_REGEX ]]; then
              : # Ideally we would expand query expressions---it's not
                # that hard, conceptually---but readline is just too
                # damn complex when it comes to quotation.  Instead,
                # for query, we just expand target patterns, unless
                # the first char is a quote.
            elif [ -n "$expansion_pattern" ]; then
              _bazel__complete_pattern \
          "$workspace" "$displacement" "$cur" "$expansion_pattern"
            fi
            ;;
        esac
        ;;
    esac
  fi
}

_bazel__to_compreply() {
  local replies="$1"
  COMPREPLY=()
  # Trick to preserve whitespaces
  while IFS="" read -r reply; do
    COMPREPLY+=("${reply}")
  done < <(echo "${replies}")
  # Null may be set despite there being no completions
  if [ ${#COMPREPLY[@]} -eq 1 ] && [ -z ${COMPREPLY[0]} ]; then
    COMPREPLY=()
  fi
}

_bazel__complete() {
  _bazel__to_compreply "$(_bazel__complete_stdout)"
}

# Some users have aliases such as bt="bazel test" or bb="bazel build", this
# completion function allows them to have auto-completion for these aliases.
_bazel__complete_target_stdout() {
  local cur=$(_bazel__get_cword) word command displacement workspace

  # Determine command: "" (startup-options) or one of $BAZEL_COMMAND_LIST.
  command="$1"

  workspace="$(_bazel__get_workspace_path)"
  displacement="$(_bazel__get_displacement ${workspace})"

  _bazel__to_compreply "$(_bazel__expand_target_pattern "$workspace" "$displacement" \
      "$cur" "$(_bazel__expansion_for $command)")"
}

# default completion for bazel
complete -F _bazel__complete -o nospace "${BAZEL}"
complete -F _bazel__complete -o nospace "${IBAZEL}"
