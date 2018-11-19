// Copyright 2016 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class BazelProtoSourceProviderStarlarkTest extends BuildViewTestCase /*SkylarkTestCase*/ {

  @Before
  public void setUp() throws Exception {
    useConfiguration("--proto_compiler=//proto:compiler"); // TODO check do we need that.
    scratch.file("proto/BUILD", "licenses(['notice'])", "exports_files(['compiler'])");
  }

  @Test
  public void testProtoSourceRootExportedInStarlark() throws Exception {

    scratch.file(
        "foo/myTsetRule.bzl",
        "",
        "def _my_test_rule_impl(ctx):",
        "    return struct(",
        "        fetched_proto_source_root = ctx.attr.protodep.proto.proto_source_root",
        "    )",
        "",
        "my_test_rule = rule(",
        "    implementation = _my_test_rule_impl,",
        "    attrs = {'protodep': attr.label()},",
        ")");

    scratch.file(
        "foo/BUILD",
        "",
        "load(':myTsetRule.bzl', 'my_test_rule')",
        "my_test_rule(",
        "  name = 'myRule',",
        "  protodep = ':myProto',",
        ")",
        "proto_library(",
        "  name = 'myProto',",
        "  srcs = ['myProto.proto'],",
        "  proto_source_root = 'foo',",
        ")");

    ConfiguredTarget ct = getConfiguredTarget("//foo:myRule");
    String protoSourceRoot = (String) ct.get("fetched_proto_source_root");

    assertThat(protoSourceRoot).isEqualTo("foo");
  }
}
