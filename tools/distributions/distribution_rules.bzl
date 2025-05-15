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

load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_java//java:defs.bzl", _java_import = "java_import")

def distrib_java_import(name, visibility = None, enable_distributions = [], **kwargs):
    """A macro for java_import rule to support distributions build (eg. Debian)"""
    checked_in_name = name + "_checked_in"

    _java_import(name = checked_in_name, visibility = visibility, **kwargs)

    conditions = {
        "//conditions:default": ":" + checked_in_name,
    }

    if "debian" in enable_distributions:
        conditions["//src/conditions:debian_build"] = "@debian_java_deps//:" + name

    if "applicable_licenses" in kwargs:
        licenses = kwargs["applicable_licenses"]
    else:
        licenses = None

    native.alias(
        name = name,
        actual = select(conditions),
        visibility = visibility,
        applicable_licenses = licenses,
    )

def distrib_cc_library(name, visibility = None, enable_distributions = [], **kwargs):
    """A macro for cc_library rule to support distributions build (eg. Debian)"""
    checked_in_name = name + "_checked_in"
    cc_library(name = checked_in_name, visibility = visibility, **kwargs)

    conditions = {
        "//conditions:default": ":" + checked_in_name,
    }

    if "debian" in enable_distributions:
        conditions["//src/conditions:debian_build"] = "@debian_cc_deps//:" + name

    native.alias(name = name, actual = select(conditions), visibility = visibility)

def distrib_cc_binary(name, visibility = None, enable_distributions = [], **kwargs):
    """A macro for cc_binary rule to support distributions build (eg. Debian)"""
    checked_in_name = name + "_checked_in"
    cc_binary(name = checked_in_name, visibility = visibility, **kwargs)

    conditions = {
        "//conditions:default": ":" + checked_in_name,
    }

    if "debian" in enable_distributions:
        conditions["//src/conditions:debian_build"] = "@debian_bin_deps//:" + name

    native.alias(name = name, actual = select(conditions), visibility = visibility)

def distrib_jar_filegroup(name, visibility = None, enable_distributions = [], **kwargs):
    """A macro for filegroup rule to support distributions build (eg. Debian)"""
    checked_in_name = name + "_checked_in"

    native.filegroup(name = checked_in_name, visibility = visibility, **kwargs)

    conditions = {
        "//conditions:default": ":" + checked_in_name,
    }

    if "debian" in enable_distributions:
        conditions["//src/conditions:debian_build"] = "@debian_java_deps//:" + name

    native.alias(name = name, actual = select(conditions), visibility = visibility)
