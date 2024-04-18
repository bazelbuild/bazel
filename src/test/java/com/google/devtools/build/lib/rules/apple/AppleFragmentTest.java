// Copyright 2022 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.lib.rules.apple;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.Provider;
import com.google.devtools.build.lib.packages.StarlarkProvider;
import com.google.devtools.build.lib.packages.StructImpl;
import com.google.devtools.build.lib.testutil.TestConstants;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for the Starlark interface of Apple fragment. */
@RunWith(JUnit4.class)
public class AppleFragmentTest extends BuildViewTestCase {

  @Before
  public void setup() throws Exception {
    scratch.file(
        "rules.bzl",
        """
        MyInfo = provider()

        def _my_binary_impl(ctx):
            out = ctx.actions.declare_file(ctx.label.name)
            ctx.actions.write(out, "")
            return [
                DefaultInfo(executable = out),
                MyInfo(
                    exec_cpu = ctx.fragments.apple.single_arch_cpu,
                ),
            ]

        my_binary = rule(
            fragments = ["apple"],
            implementation = _my_binary_impl,
        )

        def _my_rule_impl(ctx):
            return ctx.attr._tool[MyInfo]

        my_rule = rule(
            _my_rule_impl,
            attrs = {
                "_tool": attr.label(
                    cfg = "exec",
                    executable = True,
                    default = ("//:bin"),
                ),
            },
        )
        """);
    scratch.file(
        "BUILD",
        "load(':rules.bzl', 'my_binary', 'my_rule')",
        "my_binary(name = 'bin')",
        "my_rule(name = 'a')",
        "platform(",
        "    name = 'macos_arm64',",
        "    constraint_values = [",
        "        '" + TestConstants.CONSTRAINTS_PACKAGE_ROOT + "cpu:aarch64',",
        "        '" + TestConstants.CONSTRAINTS_PACKAGE_ROOT + "os:osx',",
        "    ],",
        ")");
    scratch.file(
        "/workspace/platform_mappings",
        "platforms:",
        "  //:macos_arm64",
        "    --cpu=darwin_arm64",
        "flags:",
        "  --cpu=darwin_arm64",
        "  --apple_platform_type=macos",
        "    //:macos_arm64");
    invalidatePackages(false);
  }

  @Test
  public void appleFragmentSingleArchCpuOnExtraExecPlatform() throws Exception {
    // Test that ctx.fragments.apple.single_arch_cpu returns the execution
    // platform's cpu in a tool's rule context.
    useConfiguration("--extra_execution_platforms=//:macos_arm64");
    ConfiguredTarget configuredTarget = getConfiguredTarget("//:a");
    Provider.Key key = new StarlarkProvider.Key(Label.parseCanonical("//:rules.bzl"), "MyInfo");
    StructImpl myInfo = (StructImpl) configuredTarget.get(key);
    assertThat((String) myInfo.getValue("exec_cpu")).isEqualTo("arm64");
  }
}
