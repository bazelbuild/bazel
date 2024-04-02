// Copyright 2017 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.analysis.platform;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableMap;
import com.google.common.testing.EqualsTester;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import net.starlark.java.syntax.Location;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests of {@link ToolchainInfo}. */
@RunWith(JUnit4.class)
public class ToolchainInfoTest extends BuildViewTestCase {

  @Test
  public void toolchainInfoConstructor() throws Exception {
    scratch.file(
        "test/toolchain/my_toolchain.bzl",
        """
        def _impl(ctx):
            toolchain = platform_common.ToolchainInfo(
                extra_label = ctx.attr.extra_label,
                extra_str = ctx.attr.extra_str,
            )
            return [toolchain]

        my_toolchain = rule(
            implementation = _impl,
            attrs = {
                "extra_label": attr.label(),
                "extra_str": attr.string(),
            },
        )
        """);
    scratch.file(
        "test/toolchain/BUILD",
        """
        load("//test/toolchain:my_toolchain.bzl", "my_toolchain")

        toolchain_type(name = "my_toolchain_type")

        filegroup(name = "dep")

        my_toolchain(
            name = "toolchain",
            extra_label = ":dep",
            extra_str = "foo",
        )
        """);

    ConfiguredTarget toolchain = getConfiguredTarget("//test/toolchain:toolchain");
    assertThat(toolchain).isNotNull();

    ToolchainInfo provider = PlatformProviderUtils.toolchain(toolchain);
    assertThat(provider).isNotNull();

    ConfiguredTarget extraLabel = (ConfiguredTarget) provider.getValue("extra_label");
    assertThat(extraLabel).isNotNull();
    assertThat(extraLabel.getLabel())
        .isEqualTo(Label.parseCanonicalUnchecked("//test/toolchain:dep"));
    assertThat(provider.getValue("extra_str")).isEqualTo("foo");
  }

  @Test
  public void toolchainInfo_equalsTester() throws Exception {
    new EqualsTester()
        .addEqualityGroup(
            // Base case.
            new ToolchainInfo(
                ImmutableMap.<String, Object>of("foo", "val1", "bar", "val2"), Location.BUILTIN),
            new ToolchainInfo(
                ImmutableMap.<String, Object>of("foo", "val1", "bar", "val2"), Location.BUILTIN))
        .addEqualityGroup(
            // Different data.
            new ToolchainInfo(
                ImmutableMap.<String, Object>of("foo", "val1", "bar", "val3"), Location.BUILTIN))
        .testEquals();
  }
}
