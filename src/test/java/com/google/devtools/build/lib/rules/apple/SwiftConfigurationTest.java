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

package com.google.devtools.build.lib.rules.apple;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.Provider;
import com.google.devtools.build.lib.packages.StarlarkProvider;
import com.google.devtools.build.lib.packages.StructImpl;
import java.util.List;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for Starlark interface to SwiftConfiguration. */
@RunWith(JUnit4.class)
public class SwiftConfigurationTest extends BuildViewTestCase {
  @Before
  public void setupMyInfo() throws Exception {
    scratch.file("myinfo/myinfo.bzl", "MyInfo = provider()");

    scratch.file("myinfo/BUILD");
  }

  private StructImpl getMyInfoFromTarget(ConfiguredTarget configuredTarget) throws Exception {
    Provider.Key key =
        new StarlarkProvider.Key(
            Label.parseAbsolute("//myinfo:myinfo.bzl", ImmutableMap.of()), "MyInfo");
    return (StructImpl) configuredTarget.get(key);
  }

  @Test
  public void testStarlarkApi() throws Exception {
    scratch.file("examples/rule/BUILD");
    scratch.file(
        "examples/rule/apple_rules.bzl",
        "load('//myinfo:myinfo.bzl', 'MyInfo')",
        "def swift_binary_impl(ctx):",
        "   copts = ctx.fragments.swift.copts()",
        "   return MyInfo(",
        "      copts=copts,",
        "   )",
        "swift_binary = rule(",
        "   implementation = swift_binary_impl,",
        "   fragments = ['swift']",
        ")");

    scratch.file("examples/swift_starlark/a.m");
    scratch.file(
        "examples/swift_starlark/BUILD",
        "package(default_visibility = ['//visibility:public'])",
        "load('//examples/rule:apple_rules.bzl', 'swift_binary')",
        "swift_binary(",
        "   name='my_target',",
        ")");

    useConfiguration("--swiftcopt=foo", "--swiftcopt=bar");
    ConfiguredTarget starlarkTarget = getConfiguredTarget("//examples/swift_starlark:my_target");

    @SuppressWarnings("unchecked")
    List<String> copts = (List<String>) getMyInfoFromTarget(starlarkTarget).getValue("copts");

    assertThat(copts).containsAtLeast("foo", "bar");
  }

  @Test
  public void testHostSwiftcopt() throws Exception {
    scratch.file("examples/rule/BUILD");
    scratch.file(
        "examples/rule/apple_rules.bzl",
        "load('//myinfo:myinfo.bzl', 'MyInfo')",
        "def swift_binary_impl(ctx):",
        "   copts = ctx.fragments.swift.copts()",
        "   return MyInfo(",
        "      copts=copts,",
        "   )",
        "swift_binary = rule(",
        "   implementation = swift_binary_impl,",
        "   fragments = ['swift']",
        ")");

    scratch.file(
        "examples/swift_starlark/BUILD",
        "load('//examples/rule:apple_rules.bzl', 'swift_binary')",
        "swift_binary(",
        "   name='my_target',",
        ")");

    useConfiguration("--swiftcopt=foo", "--host_swiftcopt=bar", "--host_swiftcopt=baz");
    ConfiguredTarget target =
        getConfiguredTarget("//examples/swift_starlark:my_target", getHostConfiguration());

    @SuppressWarnings("unchecked")
    List<String> copts = (List<String>) getMyInfoFromTarget(target).getValue("copts");

    assertThat(copts).doesNotContain("foo");
    assertThat(copts).containsAtLeast("bar", "baz");
  }
}
