// Copyright 2026 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.proto;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.skyframe.BzlLoadValue.keyForBuild;

import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.StarlarkProvider;
import com.google.devtools.build.lib.packages.StructImpl;
import com.google.devtools.build.lib.packages.util.MockProtoSupport;
import java.io.IOException;
import java.util.List;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Test for proto_common module. */
@RunWith(JUnit4.class)
public class ProtoCommonTest extends BuildViewTestCase {

  @Before
  public void setUp() throws IOException {
    MockProtoSupport.setup(mockToolsConfig);
  }

  @Test
  public void protocOpts_accessibleInStarlark() throws Exception {
    useConfiguration("--protocopt=--cpp_opt=foo");
    scratch.file(
        "foo/custom_rule.bzl",
        """
        result = provider()

        def _impl(ctx):
            return [result(protoc_opts = ctx.fragments.proto.experimental_protoc_opts)]

        proto_custom_library = rule(
            implementation = _impl,
            fragments = ["proto"],
        )
        """);
    scratch.file(
        "foo/BUILD",
        """
        load(":custom_rule.bzl", "proto_custom_library")

        proto_custom_library(name = "custom")
        """);

    ConfiguredTarget configuredTarget = getConfiguredTarget("//foo:custom");

    StructImpl info =
        (StructImpl)
            configuredTarget.get(
                new StarlarkProvider.Key(
                    keyForBuild(Label.parseCanonical("//foo:custom_rule.bzl")), "result"));
    assertThat(info.getValue("protoc_opts", List.class)).containsExactly("--cpp_opt=foo");
  }

  @Test
  public void strictProtoDeps_isPrivateAPI() throws Exception {
    scratch.file(
        "foo/custom_rule.bzl",
        """
        def _impl(ctx):
            ctx.fragments.proto.strict_proto_deps()
            return []

        proto_custom_library = rule(
            implementation = _impl,
            fragments = ["proto"],
        )
        """);
    scratch.file(
        "foo/BUILD",
        """
        load(":custom_rule.bzl", "proto_custom_library")

        proto_custom_library(name = "custom")
        """);
    reporter.removeHandler(failFastHandler);

    getConfiguredTarget("//foo:custom");

    assertContainsEvent("file '//foo:custom_rule.bzl' cannot use private API");
  }

  @Test
  public void experimentalProtoDescriptorsetsIncludeSourceInfo_isPrivateAPI() throws Exception {
    scratch.file(
        "foo/custom_rule.bzl",
        """
        def _impl(ctx):
            ctx.fragments.proto.experimental_proto_descriptorsets_include_source_info()
            return []

        proto_custom_library = rule(
            implementation = _impl,
            fragments = ["proto"],
        )
        """);
    scratch.file(
        "foo/BUILD",
        """
        load(":custom_rule.bzl", "proto_custom_library")

        proto_custom_library(name = "custom")
        """);
    reporter.removeHandler(failFastHandler);

    getConfiguredTarget("//foo:custom");

    assertContainsEvent("file '//foo:custom_rule.bzl' cannot use private API");
  }
}
