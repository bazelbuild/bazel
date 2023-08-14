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

"""Thin sh_test macro to run Android integration shell tests against versioned
and HEAD android_tools.tar.

This is required to ensure coverage for developing unbundled Android tools,
like the ResourceProcessorBusyBox and Desugar.

For more information, see //tools/android/runtime_deps.
"""

# Check that an SDK exists, with a clearer build error when it doesn't.
CHECK_FOR_ANDROID_SDK = select(
    {
        "//external:has_androidsdk": [],
    },
    no_match_error = "This test requires an android SDK, and one isn't present. Make sure to uncomment the android rules in the WORKSPACE.",
)

def android_sh_test(create_test_with_released_tools = True, **kwargs):
    """Creates versions of the test with and without platforms and head android tools.

    Args:
        create_test_with_released_tools: Whether to create a version of the test with the released
        android tools, for when the code under test relies on not-yet-released code.
        **kwargs: Args to sh_test
    """
    name = kwargs.pop("name")
    data = kwargs.pop("data")
    if not data:
        data = []
    data = data + CHECK_FOR_ANDROID_SDK

    if create_test_with_released_tools:
        # Test with released android_tools version.
        native.sh_test(
            name = name,
            args = ["--without_platforms"],
            data = data,
            **kwargs
        )

        # Test with platform-based toolchain resolution.
        native.sh_test(
            name = name + "_with_platforms",
            data = data,
            args = ["--with_platforms"],
            **kwargs
        )

    # Test with android_tools version that's built at the same revision
    # as the test itself.
    native.sh_test(
        name = name + "_with_head_android_tools",
        args = ["--without_platforms"],
        data = data + [
            "//tools/android/runtime_deps:android_tools.tar",
        ],
        **kwargs
    )
