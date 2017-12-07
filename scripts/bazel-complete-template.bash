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
#
#
# Bash completion of Bazel commands.
#
# The template is expanded at build time using tables of commands/options
# derived from the bazel executable built in the same client; the expansion is
# written to bazel-complete.bash.
#
# Provides command-completion for:
# - bazel prefix options (e.g. --host_jvm_args)
# - blaze command-set (e.g. build, test)
# - blaze command-specific options (e.g. --copts)
# - values for enum-valued options
# - package-names, exploring all package-path roots.
# - targets within packages.

# The package path used by the completion routines.  Unfortunately
# this isn't necessarily the same as the actual package path used by
# Bazel, but that's ok.  (It's impossible for us to reliably know what
# the relevant package-path, so this is just a good guess.  Users can
# override it if they want.)
#
# Don't use it directly. Generate the final script with
# bazel build //scripts:bash_completion instead.
#
: ${BAZEL_COMPLETION_PACKAGE_PATH:=%workspace%}


# If true, Bazel query is used for autocompletion.  This is more
# accurate than the heuristic grep, especially for strangely-formatted
# BUILD files.  But it can be slower, especially if the Bazel server
# is busy, and more brittle, if the BUILD file contains serious
# errors.   This is an experimental feature.
: ${BAZEL_COMPLETION_USE_QUERY:=false}


# If true, Bazel run allows autocompletion for test targets. This is convenient
# for users who run a lot of tests/benchmarks locally with blaze run.
: ${BAZEL_COMPLETION_ALLOW_TESTS_FOR_RUN:=false}

# Some commands might interfer with the important one, so don't complete them
: ${BAZEL_IGNORED_COMMAND_REGEX:="__none__"}

# bazel & ibazel commands
: ${BAZEL:=bazel}
: ${IBAZEL:=ibazel}

# Pattern to match for looking for a target
#  BAZEL_BUILD_MATCH_PATTERN__* give the pattern for label-*
#  when looking in the the build file.
#  BAZEL_QUERY_MATCH_PATTERN__* give the pattern for label-*
#  when using 'bazel query'.
# _RUNTEST are special case when BAZEL_COMPLETION_ALLOW_TESTS_FOR_RUN
# is on.
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
  if _bazel__is_true "$BAZEL_COMPLETION_USE_QUERY"; then
    var_name="BAZEL_QUERY_MATCH_PATTERN"
  else
    var_name="BAZEL_BUILD_MATCH_PATTERN"
  fi
  if [[ "$1" =~ ^label-?([a-z]*)$ ]]; then
    pattern=${BASH_REMATCH[1]:-}
    if _bazel__is_true "$BAZEL_COMPLETION_ALLOW_TESTS_FOR_RUN"; then
      eval "echo \"\${${var_name}_RUNTEST__${pattern}:-\$${var_name}__${pattern}}\""
    else
      eval "echo \"\$${var_name}__${pattern}\""
    fi
  fi
}

# Compute workspace directory. Search for the innermost
# enclosing directory with a WORKSPACE file.
_bazel__get_workspace_path() {
  local workspace=$PWD
  while true; do
    if [ -f "${workspace}/WORKSPACE" ]; then
      break
    elif [ -z "$workspace" -o "$workspace" = "/" ]; then
      workspace=$PWD
      break;
    fi
    workspace=${workspace%/*}
  done
  echo $workspace
}


# Find the current piece of the line to complete, but only do word breaks at
# certain characters. In particular, ignore these: "':=
# This method also takes into account the current cursor position.
#
# Works with both bash 3 and 4! Bash 3 and 4 perform different word breaks when
# computing the COMP_WORDS array. We need this here because Bazel options are of
# the form --a=b, and labels of the form //some/label:target.
_bazel__get_cword() {
  local cur=${COMP_LINE:0:$COMP_POINT}
  # This expression finds the last word break character, as defined in the
  # COMP_WORDBREAKS variable, but without '=' or ':', which is not preceeded by
  # a slash. Quote characters are also excluded.
  local wordbreaks="$COMP_WORDBREAKS"
  wordbreaks="${wordbreaks//\'/}"
  wordbreaks="${wordbreaks//\"/}"
  wordbreaks="${wordbreaks//:/}"
  wordbreaks="${wordbreaks//=/}"
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
      local option_name=$(echo $1 | perl -ne 'print uc' | tr "-" "_")
      eval "echo \${BAZEL_COMMAND_${option_name}_FLAGS}" | tr " " "\n"
  fi
}
# Usage: _bazel__expansion_for <command>
#
# Prints the completion pattern for a given Bazel command, e.g. "build".
_bazel__expansion_for() {
  local options
  if [[ "${BAZEL_COMMAND_LIST}" =~ ^(.* )?$1( .*)?$ ]]; then
      local option_name=$(echo $1 | perl -ne 'print uc' | tr "-" "_")
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
# If $BAZEL_COMPLETION_USE_QUERY is true, 'bazel query' is used
# instead, with the actual Bazel package path;
# $BAZEL_COMPLETION_PACKAGE_PATH is ignored in this case, since the
# actual Bazel value is likely to be more accurate.
_bazel__expand_rules_in_package() {
  local workspace=$1 displacement=$2 current=$3 label_type=$4
  local package_name=$(echo "$current" | cut -f1 -d:)
  local rule_prefix=$(echo "$current" | cut -f2 -d:)
  local root buildfile rule_pattern r result

  result=
  pattern=$(_bazel__get_rule_match_pattern "$label_type")
  if _bazel__is_true "$BAZEL_COMPLETION_USE_QUERY"; then
    package_name=$(echo "$package_name" | tr -d "'\"") # remove quotes
    result=$(${BAZEL} --output_base=/tmp/${BAZEL}-completion-$USER query \
                   --keep_going --noshow_progress \
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
    [ $found -gt 0 ] && break  # Stop searching package path upon first match.
  done
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
        compgen -f -- "$current"
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


_bazel__complete_stdout() {
  local cur=$(_bazel__get_cword) word command displacement workspace

  # Determine command: "" (startup-options) or one of $BAZEL_COMMAND_LIST.
  command="$(_bazel__get_command)"

  workspace="$(_bazel__get_workspace_path)"
  displacement="$(_bazel__get_displacement ${workspace})"

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
}

_bazel__to_compreply() {
  local replies="$1"
  COMPREPLY=()
  # Trick to preserve whitespaces
  while IFS="" read -r reply; do
    COMPREPLY+=("${reply}")
  done < <(echo "${replies}")
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
