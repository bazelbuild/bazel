# Copyright 2022 The Bazel Authors. All rights reserved.
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
"""Attributes for Python rules."""

load(":common/cc/cc_info.bzl", _CcInfo = "CcInfo")
load(":common/python/common.bzl", "union_attrs")
load(":common/python/providers.bzl", "PyInfo")
load(
    ":common/python/semantics.bzl",
    "DEPS_ATTR_ALLOW_RULES",
    "PLATFORMS_LOCATION",
    "SRCS_ATTR_ALLOW_FILES",
)

PackageSpecificationInfo = _builtins.toplevel.PackageSpecificationInfo

_STAMP_VALUES = [-1, 0, 1]

def create_stamp_attr(**kwargs):
    return {"stamp": attr.int(values = _STAMP_VALUES, **kwargs)}

default_srcs_attr_doc = """
The list of source (<code>.py</code>) files that are processed to create the target.
This includes all your checked-in code and any generated source files.
"""

def create_srcs_attr(*, mandatory, doc = default_srcs_attr_doc):
    return {
        "srcs": attr.label_list(
            # Google builds change the set of allowed files.
            allow_files = SRCS_ATTR_ALLOW_FILES,
            doc = doc,
            mandatory = mandatory,
            # Necessary for --compile_one_dependency to work.
            flags = ["DIRECT_COMPILE_TIME_INPUT"],
        ),
    }

SRCS_VERSION_ALL_VALUES = ["PY2", "PY2ONLY", "PY2AND3", "PY3", "PY3ONLY"]
SRCS_VERSION_NON_CONVERSION_VALUES = ["PY2AND3", "PY2ONLY", "PY3ONLY"]

DEFAULT_SRCS_VERSION_ATTR_DOC = """
This attribute declares the target's <code>srcs</code> to be compatible with either Python
2, Python 3, or both. To actually set the Python runtime version, use the
<a href="${link py_binary.python_version}"><code>python_version</code></a> attribute of an
executable Python rule (<code>py_binary</code> or <code>py_test</code>).

<p>Allowed values are: <code>"PY2AND3"</code>, <code>"PY2"</code>, and <code>"PY3"</code>.
The values <code>"PY2ONLY"</code> and <code>"PY3ONLY"</code> are also allowed for historic
reasons, but they are essentially the same as <code>"PY2"</code> and <code>"PY3"</code>
and should be avoided.

<p>Note that only the executable rules (<code>py_binary</code> and <code>py_library
</code>) actually verify the current Python version against the value of this attribute.
(This is a feature; since <code>py_library</code> does not change the current Python
version, if it did the validation, it'd be impossible to build both <code>PY2ONLY</code>
and <code>PY3ONLY</code> libraries in the same invocation.) Furthermore, if there is a
version mismatch, the error is only reported in the execution phase. In particular, the
error will not appear in a <code>bazel build --nobuild</code> invocation.)

<p>To get diagnostic information about which dependencies introduce version requirements,
you can run the <code>find_requirements</code> aspect on your target:
<pre>
bazel build &lt;your target&gt; \\
    --aspects=@rules_python//python:defs.bzl%find_requirements \\
    --output_groups=pyversioninfo
</pre>
This will build a file with the suffix <code>-pyversioninfo.txt</code> giving information
about why your target requires one Python version or another. Note that it works even if
the given target failed to build due to a version conflict.
"""

def create_srcs_version_attr(values, srcs_version_attr_doc = DEFAULT_SRCS_VERSION_ATTR_DOC):
    return {
        "srcs_version": attr.string(
            default = "PY2AND3",
            values = values,
            doc = srcs_version_attr_doc,
        ),
    }

def copy_common_binary_kwargs(kwargs):
    return {
        key: kwargs[key]
        for key in BINARY_ATTR_NAMES
        if key in kwargs
    }

def copy_common_test_kwargs(kwargs):
    return {
        key: kwargs[key]
        for key in TEST_ATTR_NAMES
        if key in kwargs
    }

NATIVE_RULES_ALLOWLIST_ATTRS = {
    "_native_rules_allowlist": attr.label(
        default = configuration_field(
            fragment = "py",
            name = "native_rules_allowlist",
        ),
        providers = [PackageSpecificationInfo],
    ),
}

def create_common_attrs(*, data_attr_doc = None):
    # Attributes common to all rules.
    return union_attrs(
        # NOTE: The "flags" attribute is deprecated, but there isn't an alternative
        # way to specify that constraints should be ignored.
        {"data": attr.label_list(
            allow_files = True,
            flags = ["SKIP_CONSTRAINTS_OVERRIDE"],
            doc = data_attr_doc,
        )},
        NATIVE_RULES_ALLOWLIST_ATTRS,
        # buildifier: disable=attr-licenses
        {
            # TODO(b/148103851): This attribute is deprecated and slated for
            # removal.
            # NOTE: The license attribute is missing in some Java integration tests,
            # so fallback to a regular string_list for that case.
            # buildifier: disable=attr-license
            "licenses": attr.license() if hasattr(attr, "license") else attr.string_list(),
            # NOTE: This attribute is deprecated and slated for removal.
            "distribs": attr.string_list(),
        },
        allow_none = True,
    )

COMMON_ATTRS = create_common_attrs()

DEFAULT_DEPS_ATTR_DOC = """
The list of other libraries to be linked in to the binary target. See general comments about
<code>deps</code> at <a href="${link common-definitions#typical-attributes}">Typical attributes
defined by most build rules</a>. These are generally
<a href="${link py_library}"><code>py_library</code></a> rules.
"""

def _create_py_srcs_attrs(deps_attr_doc):
    return union_attrs(
        {
            # Required attribute, but details vary by rule.
            # Use create_srcs_attr to create one.
            "srcs": None,
            "deps": attr.label_list(
                doc = deps_attr_doc,
                providers = [[PyInfo], [_CcInfo]],
                # TODO(b/228692666): Google-specific; remove these allowances once
                # the depot is cleaned up.
                allow_rules = DEPS_ATTR_ALLOW_RULES,
            ),
            # NOTE: In Google, this attribute is deprecated, and can only
            # effectively be PY3 or PY3ONLY. Externally, with Bazel, this attribute
            # has a separate story.
            # Required attribute, but the details vary by rule.
            # Use create_srcs_version_attr to create one.
            "srcs_version": None,
        },
        allow_none = True,
    )

# Attributes common to rules accepting Python sources and deps.
PY_SRCS_ATTRS = _create_py_srcs_attrs(DEFAULT_DEPS_ATTR_DOC)

def create_py_srcs_attrs(deps_attr_doc):
    return PY_SRCS_ATTRS if deps_attr_doc == DEFAULT_DEPS_ATTR_DOC else _create_py_srcs_attrs(deps_attr_doc)

# Attributes specific to Python executable-equivalent rules. Such rules may not
# accept Python sources (e.g. some packaged-version of a py_test/py_binary), but
# still accept Python source-agnostic settings.
AGNOSTIC_EXECUTABLE_ATTRS = union_attrs(
    {
        # NOTE: The "flags" attribute is deprecated, but there isn't an alternative
        # way to specify that constraints should be ignored.
        "data": attr.label_list(
            allow_files = True,
            flags = ["SKIP_CONSTRAINTS_OVERRIDE"],
        ),
        "env": attr.string_dict(
            doc = """\
Dictionary of strings; optional; values are subject to `$(location)` and "Make
variable" substitution.

Specifies additional environment variables to set when the target is executed by
`test` or `run`.
""",
        ),
        # The value is required, but varies by rule and/or rule type. Use
        # create_stamp_attr to create one.
        "stamp": None,
    },
    allow_none = True,
)

# Attributes specific to Python test-equivalent executable rules. Such rules may
# not accept Python sources (e.g. some packaged-version of a py_test/py_binary),
# but still accept Python source-agnostic settings.
AGNOSTIC_TEST_ATTRS = union_attrs(
    AGNOSTIC_EXECUTABLE_ATTRS,
    # Tests have stamping disabled by default.
    create_stamp_attr(default = 0),
    {
        "env_inherit": attr.string_list(
            doc = """\
List of strings; optional

Specifies additional environment variables to inherit from the external
environment when the test is executed by bazel test.
""",
        ),
        # TODO(b/176993122): Remove when Bazel automatically knows to run on darwin.
        "_apple_constraints": attr.label_list(
            default = [
                PLATFORMS_LOCATION + "/os:ios",
                PLATFORMS_LOCATION + "/os:macos",
                PLATFORMS_LOCATION + "/os:tvos",
                PLATFORMS_LOCATION + "/os:visionos",
                PLATFORMS_LOCATION + "/os:watchos",
            ],
        ),
    },
)

# Attributes specific to Python binary-equivalent executable rules. Such rules may
# not accept Python sources (e.g. some packaged-version of a py_test/py_binary),
# but still accept Python source-agnostic settings.
AGNOSTIC_BINARY_ATTRS = union_attrs(
    AGNOSTIC_EXECUTABLE_ATTRS,
    create_stamp_attr(default = -1),
)

# Attribute names common to all Python rules
COMMON_ATTR_NAMES = [
    "compatible_with",
    "deprecation",
    "distribs",  # NOTE: Currently common to all rules, but slated for removal
    "exec_compatible_with",
    "exec_properties",
    "features",
    "restricted_to",
    "tags",
    "target_compatible_with",
    # NOTE: The testonly attribute requires careful handling: None/unset means
    # to use the `package(default_testonly`) value, which isn't observable
    # during the loading phase.
    "testonly",
    "toolchains",
    "visibility",
] + COMMON_ATTRS.keys()

# Attribute names common to all test=True rules
TEST_ATTR_NAMES = COMMON_ATTR_NAMES + [
    "args",
    "size",
    "timeout",
    "flaky",
    "shard_count",
    "local",
] + AGNOSTIC_TEST_ATTRS.keys()

# Attribute names common to all executable=True rules
BINARY_ATTR_NAMES = COMMON_ATTR_NAMES + [
    "args",
    "output_licenses",  # NOTE: Common to all rules, but slated for removal
] + AGNOSTIC_BINARY_ATTRS.keys()
