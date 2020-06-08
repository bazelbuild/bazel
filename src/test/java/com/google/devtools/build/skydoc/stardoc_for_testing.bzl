# Copyright 2019 The Bazel Authors. All rights reserved.
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

"""A wrapper on the stardoc rule for convenience of testing."""

load("@io_bazel_skydoc//stardoc:html_tables_stardoc.bzl", "html_tables_stardoc")
load("@io_bazel_skydoc//stardoc:stardoc.bzl", _stardoc = "stardoc")

def stardoc(format = "html_tables", **kwargs):
    """A wrapper on the stardoc rule for convenience of testing.

    Args:
        format: The output format of stardoc.
            Valid values: "custom", "html_tables", "markdown_tables", or "proto".
            "html_tables" by default.
        **kwargs: Attributes to pass through to the stardoc rule."""
    if format == "markdown_tables" or format == "proto" or format == "custom":
        if format == "markdown_tables" or format == "custom":
            # Stardoc's format "markdown" is technically "markdown with html tables",
            # and the user can specify custom templates adhoc if the format is "markdown".
            format_val = "markdown"
        else:
            format_val = "proto"
        _stardoc(
            format = format_val,
            **kwargs
        )
    elif format == "html_tables":
        html_tables_stardoc(**kwargs)
    else:
        fail("parameter 'format' must be one of " +
             "['custom', 'html_tables', 'markdown_tables', 'proto'], " +
             "but was " + format)
