# Copyright 2018 The Bazel Authors. All rights reserved.
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

"""Defines a rule that can assert the number of output files in another rule.

=== Use ===

Use this rule to assert the size of a filegroup or any other rule and catch
sudden, unexpected changes in the size.

The `margin` attribute allows specifying a tolerance value (percentage), to
allow for organic, expected growth or shrinkage of the target rule.

=== Example ===

the "resources_size_test" test fails if the number of files in
"resources" changes from 123 by more than 3 percent:

    filegroup(
        name = "resources",
        srcs = glob(["**"]) + [
            "//foo/bar:resources"
            "//baz:resources",
        ],
    )

    assert_size(
        name = "watch_resources_size",
        src = ":resources",

        # Expect 123 files in ":resources", with an error margin of 3% to allow
        # for slight changes.
        expect = 123,
        margin = 3,
    )

    sh_test(
        name = "resources_size_test",
        srcs = ["dummy.sh"],  # does nothing
        data = [":watch_resources_size"],
    )

"""

def _impl(ctx):
  if ctx.attr.expect < 0:
    fail("ERROR: assert_size.expect must be positive")

  if ctx.attr.margin < 0 or ctx.attr.margin > 100:
    # Do not allow more than 100% change in size.
    fail("ERROR: assert_size.margin must be in range [0..100]")

  if ctx.attr.expect == 0 and ctx.attr.margin != 0:
    # Allow no margin when expecting 0 files, to avoid division by zero.
    fail("ERROR: assert_size.margin must be 0 when assert_size.expect is 0") 

  amount = len(ctx.attr.src[DefaultInfo].files)

  if ctx.attr.margin > 0:
    if amount >= ctx.attr.expect:
      diff = amount - ctx.attr.expect
    else:
      diff = ctx.attr.expect - amount

    if ((diff * 100) / ctx.attr.expect) > ctx.attr.margin:
      fail(("ERROR: assert_size: expected %d file(s) within %d%% error margin, "
            + "got %d file(s) (%d%% difference)") % (
                ctx.attr.expect, ctx.attr.margin, amount,
                (diff * 100) / ctx.attr.expect))
  elif amount != ctx.attr.expect:
    fail(
        "ERROR: assert_size: expected exactly %d file(s), got %d file(s)" % (
            ctx.attr.expect, amount))

  ctx.actions.do_nothing(mnemonic = "AssertSizeAction")
  

assert_size = rule(
    implementation = _impl,
    attrs = {
        # The target whose number of output files this rule asserts. The number
        # of output files is the size of the target's DefaultInfo.files field.
        "src": attr.label(allow_files = True),
        # A non-negative integer, the expected number of files that the target
        # in `src` outputs. If 0, then `margin` must also be 0.
        "expect": attr.int(mandatory = True),
        # A percentage value, in the range of [0..100]. Allows for tolerance in
        # the difference between expected and actual number of files in `src`.
        # If 0, then the target in `src` must output exactly `expect` many
        # files.
        "margin": attr.int(mandatory = True),
    })
