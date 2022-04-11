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
package com.google.devtools.build.lib.view.go;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.packages.util.MockProtoSupport;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests Starlark API for Go rules. */
@RunWith(JUnit4.class)
public class GoStarlarkApiTest extends BuildViewTestCase {

  /**
   * Tests that go_proto_library's aspect exposes a Starlark provider "aspect_proto_go_api_info".
   */
  @Test
  public void testGoProtoLibraryAspectProviders() throws Exception {
    MockProtoSupport.setup(mockToolsConfig);
    scratch.file(
        "x/aspect.bzl",
        "def _foo_aspect_impl(target,ctx):",
        "  proto_found = hasattr(target, 'aspect_proto_go_api_info')",
        "  if hasattr(ctx.rule.attr, 'deps'):",
        "    for dep in ctx.rule.attr.deps:",
        "      proto_found = proto_found or dep.proto_found",
        "  return struct(proto_found = proto_found)",
        "foo_aspect = aspect(",
        "    _foo_aspect_impl,",
        "    attr_aspects = ['deps'],",
        "    required_aspect_providers=['aspect_proto_go_api_info'])",
        "def _rule_impl(ctx):",
        "  return struct(result = ctx.attr.dep.proto_found)",
        "foo_rule = rule(_rule_impl, attrs = {'dep' : attr.label(aspects = [foo_aspect])})");
    scratch.file(
        "x/BUILD",
        "load(':aspect.bzl', 'foo_rule')",
        "go_proto_library(",
        "    name = 'foo_go_proto',",
        "    deps = ['foo_proto'],",
        ")",
        "proto_library(",
        "    name = 'foo_proto',",
        "    srcs = ['foo.proto'],",
        ")",
        "foo_rule(",
        "    name = 'foo_rule',",
        "    dep = 'foo_go_proto',",
        ")");
    ConfiguredTarget target = getConfiguredTarget("//x:foo_rule");
    Boolean result = (Boolean) target.get("result");

    // 'result' is true iff "aspect_proto_go_api_info" was found on the proto_library +
    // go_proto_library aspect.
    assertThat(result).isTrue();
  }
}
