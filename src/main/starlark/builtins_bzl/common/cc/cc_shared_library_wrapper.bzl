# Copyright 2024 The Bazel Authors. All rights reserved.
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
load(":common/cc/cc_shared_library.bzl", _cc_shared_library = "cc_shared_library")

"""Implementation of cc_shared_library's macro wrapper"""

def cc_shared_library(**kwargs):
    return _cc_shared_library(**_canonicalize_exports_filter(kwargs))

def _canonicalize_exports_filter(kwargs):
    """Converts labels in exports_filter into canonical form relative to the current repository.

    This conversion can only be done in a macro as it requires access to the repository mapping of
    the repository containing the cc_shared_library target. This mapping is automatically
    applied to label attributes, but exports_filter is a list of strings attribute.
    """
    if "exports_filter" not in kwargs:
        return kwargs

    raw_exports_filter = kwargs["exports_filter"]
    if type(raw_exports_filter) != type([]):
        # TODO: Also canonicalize labels in selects once macros can operate on them.
        # https://github.com/bazelbuild/bazel/issues/14157
        return kwargs

    canonical_exports_filter = [
        str(_builtins.native.package_relative_label(s))
        for s in raw_exports_filter
    ]
    return kwargs | {"exports_filter": canonical_exports_filter}
