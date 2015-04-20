#!/bin/bash
# Copyright 2015 Google Inc. All rights reserved.
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
# Generates eclipse files for Bazel

set -eu

progname=$0
function usage() {
    echo "Usage: $progname command args..." >&2
    echo "Possible commands are:" >&2
    echo "    classpath java_paths lib_paths jre output_path" >&2
    echo "    factorypath project_name plugin_paths" >&2
    echo "    project project_name" >&2
    echo "    apt_settings output_path" >&2
    exit 1
}

function read_entry() {
    if [[ -e "${1// /_}" ]]; then
        cat "$1"
    else
        echo "$1"
    fi
}

function generate_classpath() {
    if [[ "$#" != 4 ]]; then
        usage
    fi

    java_paths="$(read_entry "$1")"
    lib_paths="$(read_entry "$2")"
    jre="$3"
    output_path="$4"

    cat <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<classpath>
EOF

    for path in $java_paths; do
        echo "    <classpathentry kind=\"src\" path=\"$path\"/>"
    done

    for path_pair in $lib_paths; do
        path_arr=(${path_pair//:/ })
        jar=${path_arr[0]}
        source_path=${path_arr[1]-}
        if [ -n "${source_path}" ]; then
            echo "    <classpathentry kind=\"lib\" path=\"${jar}\" sourcepath=\"${source_path}\"/>"
        else
            echo "    <classpathentry kind=\"lib\" path=\"${jar}\"/>"
        fi
    done

    # Write the end of the .classpath file
    cat <<EOF
    <classpathentry kind="output" path="${output_path}"/>
    <classpathentry kind="con" path="org.eclipse.jdt.launching.JRE_CONTAINER/org.eclipse.jdt.internal.debug.ui.launcher.StandardVMType/${jre}">
      <accessrules>
        <accessrule kind="accessible" pattern="**"/>
      </accessrules>
    </classpathentry>
</classpath>
EOF
}

function generate_factorypath() {
    if [ "$#" != 2 ]; then
        usage
    fi
    project_name="$1"
    plugin_paths="$(read_entry "$2")"

    cat <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<factorypath>
EOF

    for path in $plugin_paths; do
        echo "    <factorypathentry kind=\"WKSPJAR\" id=\"/${project_name}/${path}\" enabled=\"true\" runInBatchMode=\"false\" />"
    done

    # Write the end of the .factorypath file
    cat <<EOF
</factorypath>
EOF
}

function generate_project() {
    if [ "$#" != 1 ]; then
        usage
    fi
    project_name=$1
    cat <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<projectDescription>
    <name>${project_name}</name>
    <comment></comment>
    <projects>
    </projects>
    <buildSpec>
        <buildCommand>
            <name>org.eclipse.jdt.core.javabuilder</name>
            <arguments>
            </arguments>
        </buildCommand>
    </buildSpec>
    <natures>
        <nature>org.eclipse.jdt.core.javanature</nature>
    </natures>
</projectDescription>
EOF
}

function generate_apt_settings() {
    if [ "$#" != 1 ]; then
        usage
    fi
    output_path=$1
    cat <<EOF
eclipse.preferences.version=1
org.eclipse.jdt.apt.aptEnabled=true
org.eclipse.jdt.apt.genSrcDir=${output_path}
org.eclipse.jdt.apt.reconcileEnabled=true
EOF
}

command=$1
shift
case "${command}" in
    classpath)
        generate_classpath "$@"
        ;;
    factorypath)
        generate_factorypath "$@"
        ;;
    project)
        generate_project "$@"
        ;;
    apt_settings)
        generate_apt_settings "$@"
        ;;
    *)
        usage
        ;;
esac
