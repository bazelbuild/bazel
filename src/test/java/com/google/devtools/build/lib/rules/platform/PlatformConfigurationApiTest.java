// Copyright 2018 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.rules.platform;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.StarlarkProvider;
import com.google.devtools.build.lib.packages.StructImpl;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests Starlark API for Platform configuration fragments. */
@RunWith(JUnit4.class)
public class PlatformConfigurationApiTest extends BuildViewTestCase {

  @Test
  public void testHostPlatform() throws Exception {
    scratch.file("platforms/BUILD", "platform(name = 'test_platform')");

    scratch.file(
        "verify/verify.bzl",
        """
        result = provider()

        def _impl(ctx):
            platformConfig = ctx.fragments.platform
            host_platform = platformConfig.host_platform
            return [result(
                host_platform = host_platform,
            )]

        verify = rule(
            implementation = _impl,
            fragments = ["platform"],
        )
        """);
    scratch.file(
        "verify/BUILD",
        """
        load(":verify.bzl", "verify")

        verify(name = "verify")
        """);

    useConfiguration("--host_platform=//platforms:test_platform");

    ConfiguredTarget myRuleTarget = getConfiguredTarget("//verify:verify");
    StructImpl info =
        (StructImpl)
            myRuleTarget.get(
                new StarlarkProvider.Key(Label.parseCanonical("//verify:verify.bzl"), "result"));

    Label hostPlatform = (Label) info.getValue("host_platform");
    assertThat(hostPlatform).isEqualTo(Label.parseCanonicalUnchecked("//platforms:test_platform"));
  }

  @Test
  public void testTargetPlatform_single() throws Exception {
    scratch.file("platforms/BUILD", "platform(name = 'test_platform')");

    scratch.file(
        "verify/verify.bzl",
        """
        result = provider()

        def _impl(ctx):
            platformConfig = ctx.fragments.platform
            target_platform = platformConfig.platform
            return [result(
                target_platform = target_platform,
            )]

        verify = rule(
            implementation = _impl,
            fragments = ["platform"],
        )
        """);
    scratch.file(
        "verify/BUILD",
        """
        load(":verify.bzl", "verify")

        verify(name = "verify")
        """);

    useConfiguration("--platforms=//platforms:test_platform");

    ConfiguredTarget myRuleTarget = getConfiguredTarget("//verify:verify");
    StructImpl info =
        (StructImpl)
            myRuleTarget.get(
                new StarlarkProvider.Key(Label.parseCanonical("//verify:verify.bzl"), "result"));

    Label targetPlatform = (Label) info.getValue("target_platform");
    assertThat(targetPlatform)
        .isEqualTo(Label.parseCanonicalUnchecked("//platforms:test_platform"));
  }
}
