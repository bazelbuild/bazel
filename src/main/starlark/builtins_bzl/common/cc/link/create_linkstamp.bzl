# Copyright 2021 The Bazel Authors. All rights reserved.
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
"""Starlark implementation of create_linkstamp."""

load(":common/cc/cc_helper_internal.bzl", "wrap_with_check_private_api")

# A linkstamp that also knows about its declared includes.
#
# This object is required because linkstamp files may include other headers which will have to
# be provided during compilation.
_LinkstampInfo = provider(
    doc = "Information about a C++ linkstamp.",
    fields = {
        "file": "The linkstamp source file, a C++ source file to be compiled and linked.",
        "hdrs": "The headers needed to compile the linkstamp artifact.",
    },
)

def create_linkstamp(linkstamp, headers):
    """Creates a linkstamp.

    Args:
      linkstamp: the linkstamp source file.
      headers: a CcCompilationContext from which to get the declared_include_srcs.

    Returns:
      A LinkstampInfo provider.
    """

    return _LinkstampInfo(
        file = wrap_with_check_private_api(linkstamp),
        hdrs = wrap_with_check_private_api(headers),
    )
