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

package com.google.devtools.build.lib.rules.objc;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.CommandAction;
import com.google.devtools.build.lib.rules.objc.ObjcCommandLineOptions.ObjcCrosstoolMode;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Legacy test: These tests test --experimental_objc_crosstool=off. See README.
 */
@RunWith(JUnit4.class)
public class LegacyAppleStaticLibraryTest extends AppleStaticLibraryTest {
  @Override
  protected ObjcCrosstoolMode getObjcCrosstoolMode() {
    return ObjcCrosstoolMode.OFF;
  }

  @Test
  public void testAvoidDepsObjects() throws Exception {
    scratch.file("package/BUILD",
        "apple_static_library(",
        "    name = 'test',",
        "    deps = [':objcLib'],",
        "    avoid_deps = [':avoidLib'],",
        "    platform_type = 'ios',",
        ")",
        "objc_library(name = 'objcLib', srcs = [ 'b.m' ], deps = [':avoidLib', ':baseLib'])",
        "objc_library(name = 'baseLib', srcs = [ 'base.m' ])",
        "objc_library(name = 'avoidLib', srcs = [ 'c.m' ])");

    CommandAction action = linkLibAction("//package:test");
    assertThat(Artifact.toRootRelativePaths(action.getInputs())).containsExactly(
        "package/libobjcLib.a", "package/libbaseLib.a", MOCK_LIBTOOL_PATH);
  }

  @Test
  // Tests that if there is a cc_library in avoid_deps, all of its dependencies are
  // transitively avoided, even if it is not present in deps.
  public void testAvoidDepsObjects_avoidViaCcLibrary() throws Exception {
    scratch.file("package/BUILD",
        "apple_static_library(",
        "    name = 'test',",
        "    deps = [':objcLib'],",
        "    avoid_deps = [':avoidCclib'],",
        "    platform_type = 'ios',",
        ")",
        "cc_library(name = 'avoidCclib', srcs = ['cclib.c'], deps = [':avoidLib'])",
        "objc_library(name = 'objcLib', srcs = [ 'b.m' ], deps = [':avoidLib'])",
        "objc_library(name = 'avoidLib', srcs = [ 'c.m' ])");

    useConfiguration("--experimental_disable_go", "--experimental_disable_jvm");
    CommandAction action = linkLibAction("//package:test");
    assertThat(Artifact.toRootRelativePaths(action.getInputs())).containsExactly(
        "package/libobjcLib.a", MOCK_LIBTOOL_PATH);
  }

  @Test
  // Tests that if there is a cc_library in avoid_deps, and it is present in deps, it will
  // be avoided, as well as its transitive dependencies.
  public void testAvoidDepsObjects_avoidCcLibrary() throws Exception {
    scratch.file("package/BUILD",
        "apple_static_library(",
        "    name = 'test',",
        "    deps = [':objcLib', ':avoidCclib'],",
        "    avoid_deps = [':avoidCclib'],",
        "    platform_type = 'ios',",
        ")",
        "cc_library(name = 'avoidCclib', srcs = ['cclib.c'], deps = [':avoidLib'])",
        "objc_library(name = 'objcLib', srcs = [ 'b.m' ])",
        "objc_library(name = 'avoidLib', srcs = [ 'c.m' ])");

    useConfiguration("--experimental_disable_go", "--experimental_disable_jvm");
    CommandAction action = linkLibAction("//package:test");
    assertThat(Artifact.toRootRelativePaths(action.getInputs())).containsExactly(
        "package/libobjcLib.a", MOCK_LIBTOOL_PATH);
  }

}
