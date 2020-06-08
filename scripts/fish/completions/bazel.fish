# Copyright 2020 The Bazel Authors. All rights reserved.
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

# fish completion for bazel

set __bazel_command "bazel"
set __bazel_completion_text ($__bazel_command help completion 2>/dev/null)
set __bazel_help_text ($__bazel_command help 2>/dev/null)
set __bazel_startup_options_text ($__bazel_command help startup_options 2>/dev/null)

function __bazel_get_completion_variable \
    -d 'Print contents of a completion helper variable from `bazel help completion`'
    set -l var $argv[1]
    set -l regex (string join '' "$var=\"\\([^\"]*\\)\"")
    echo $__bazel_completion_text | grep -o $regex | sed "s/$regex/\\1/" | string trim
end

set __bazel_subcommands (__bazel_get_completion_variable BAZEL_COMMAND_LIST | string split ' ')

function __bazel_seen_subcommand \
    -d 'Check whether the current command line contains a bazel subcommand'
    set -l subcommand $argv[1]
    if test -n "$subcommand"
        __fish_seen_subcommand_from $subcommand
    else
        __fish_seen_subcommand_from $__bazel_subcommands
    end
end

function __bazel_get_options \
    -d 'Parse bazel help text for options and print each option and its description'
    set -l help_text_lines $argv
    set -l regex '^[[:space:]]*--\(\[no\]\)\?\([_[:alnum:]]\+\)[[:space:]]\+(\(.*\))$'
    printf '%s\n' $help_text_lines | grep $regex | sed "s/$regex/\\2 \\3/"
end

function __bazel_complete_option \
    -d 'Set up completion for a bazel option with a given condition'
    set -l condition $argv[1]
    set -l option $argv[2]
    set -l desc $argv[3..-1]

    set -l complete_opts -c $__bazel_command -n $condition
    if string match -qr boolean $desc
        complete $complete_opts -l "$option"
        complete $complete_opts -l "no$option"
    else if test -n "$desc"
        complete $complete_opts -rl "$option" -d "$desc"
    else
        complete $complete_opts -rl "$option"
    end
end

function __bazel_complete_startup_options \
    -d 'Set up completion for all bazel startup options'
    for line in (__bazel_get_options (printf '%s\n' $__bazel_startup_options_text))
        __bazel_complete_option "not __bazel_seen_subcommand" (string split ' ' $line)
    end
end

function __bazel_describe_subcommand \
    -d 'Print description text for a bazel subcommand'
    set -l subcommand $argv[1]
    set -l regex (string join '' '^[[:space:]]*' $subcommand '[[:space:]]\+\(.*\)$')
    printf '%s\n' $__bazel_help_text | grep -m1 $regex | sed "s/$regex/\\1/"
end

function __bazel_get_subcommand_arg_type \
    -d 'Print the expected argument type of a bazel subcommand'
    set -l subcommand $argv[1]
    set -l formatted_subcommand (string upper $subcommand | tr '-' '_')
    set -l var (string join '' 'BAZEL_COMMAND_' $formatted_subcommand '_ARGUMENT')
    __bazel_get_completion_variable $var
end

function __bazel_get_subcommand_args \
    -d 'Print an argument string for subcommand completion'
    set -l subcommand $argv[1]
    set -l arg_type (__bazel_get_subcommand_arg_type $subcommand)

    switch $arg_type
        case "label"
            echo "($__bazel_command query -k '//...' 2>/dev/null)"
        case "label-bin"
            echo "($__bazel_command query -k 'kind(\".*_binary\", //...)' 2>/dev/null)"
        case "label-test"
            echo "($__bazel_command query -k 'tests(//...)' 2>/dev/null)"
        case "command*"
            printf '%s\n' $__bazel_subcommands
            echo $arg_type | sed 's/command|{\(.*\)}/\1/' | string split ','
        case "info-key"
            __bazel_get_completion_variable BAZEL_INFO_KEYS | string split ' '
    end
end

function __bazel_complete_subcommand \
    -d 'Set up completion for a given bazel subcommand'
    set -l subcommand $argv[1]

    set -l desc (__bazel_describe_subcommand $subcommand)
    if test -n "$desc"
        complete -c $__bazel_command -n "not __bazel_seen_subcommand" -xa $subcommand -d $desc
    else
        complete -c $__bazel_command -n "not __bazel_seen_subcommand" -xa $subcommand
    end

    set -l opts (__bazel_get_options (bazel help $subcommand 2>/dev/null))
    if test -n "$opts"
        for line in $opts
            __bazel_complete_option "__bazel_seen_subcommand $subcommand" (string split ' ' $line)
        end
    end

    set -l args (__bazel_get_subcommand_args $subcommand)
    if test -n "$args"
        complete -c $__bazel_command -n "__bazel_seen_subcommand $subcommand" -fa "$args"
    else
        complete -c $__bazel_command -n "__bazel_seen_subcommand $subcommand"
    end
end

function __bazel_complete_subcommands \
    -d 'Set up completion for all bazel subcommands'
    for subcommand in $__bazel_subcommands
        __bazel_complete_subcommand $subcommand
    end
end

__bazel_complete_startup_options
__bazel_complete_subcommands
