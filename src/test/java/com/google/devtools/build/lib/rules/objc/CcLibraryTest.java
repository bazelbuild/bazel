// Copyright 2021 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.actions.CommandAction;
import com.google.devtools.build.lib.analysis.util.ScratchAttributeWriter;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for cc_library with Apple-specific logic. */
@RunWith(JUnit4.class)
public class CcLibraryTest extends ObjcRuleTestCase {
  @Override
  protected ScratchAttributeWriter createLibraryTargetWriter(String labelString) {
    return ScratchAttributeWriter.fromLabelString(this, "cc_library", labelString);
  }

  @Test
  public void testCompilationActionsWithEmbeddedBitcode() throws Exception {
    useConfiguration("--apple_platform_type=ios", "--cpu=ios_arm64", "--apple_bitcode=embedded");
    createLibraryTargetWriter("//cc:lib")
        .setAndCreateFiles("srcs", "a.cc", "b.cc", "private.h")
        .setAndCreateFiles("hdrs", "c.h")
        .write();

    CommandAction compileActionA = compileAction("//cc:lib", "a.o");

    assertThat(compileActionA.getArguments()).contains("-fembed-bitcode");
  }

  @Test
  public void testCompilationActionsWithEmbeddedBitcodeMarkers() throws Exception {
    useConfiguration(
        "--apple_platform_type=ios", "--cpu=ios_arm64", "--apple_bitcode=embedded_markers");

    createLibraryTargetWriter("//cc:lib")
        .setAndCreateFiles("srcs", "a.cc", "b.cc", "private.h")
        .setAndCreateFiles("hdrs", "c.h")
        .write();

    CommandAction compileActionA = compileAction("//cc:lib", "a.o");

    assertThat(compileActionA.getArguments()).contains("-fembed-bitcode-marker");
  }

  @Test
  public void testCompilationActionsWithNoBitcode() throws Exception {
    useConfiguration("--apple_platform_type=ios", "--cpu=ios_arm64", "--apple_bitcode=none");

    createLibraryTargetWriter("//cc:lib")
        .setAndCreateFiles("srcs", "a.cc", "b.cc", "private.h")
        .setAndCreateFiles("hdrs", "c.h")
        .write();

    CommandAction compileActionA = compileAction("//cc:lib", "a.o");

    assertThat(compileActionA.getArguments()).doesNotContain("-fembed-bitcode");
    assertThat(compileActionA.getArguments()).doesNotContain("-fembed-bitcode-marker");
  }

  /** Tests that bitcode is disabled for simulator builds even if enabled by flag. */
  @Test
  public void testCompilationActionsWithBitcode_simulator() throws Exception {
    useConfiguration("--apple_platform_type=ios", "--cpu=ios_x86_64", "--apple_bitcode=embedded");

    createLibraryTargetWriter("//cc:lib")
        .setAndCreateFiles("srcs", "a.cc", "b.cc", "private.h")
        .setAndCreateFiles("hdrs", "c.h")
        .write();

    CommandAction compileActionA = compileAction("//cc:lib", "a.o");

    assertThat(compileActionA.getArguments()).doesNotContain("-fembed-bitcode");
    assertThat(compileActionA.getArguments()).doesNotContain("-fembed-bitcode-marker");
  }

  @Test
  public void testCompilationActionsWithEmbeddedBitcodeForMultiplePlatformsWithMatch()
      throws Exception {
    useConfiguration(
        "--apple_platform_type=ios",
        "--cpu=ios_arm64",
        "--apple_bitcode=ios=embedded",
        "--apple_bitcode=watchos=embedded");
    createLibraryTargetWriter("//cc:lib")
        .setAndCreateFiles("srcs", "a.cc", "b.cc", "private.h")
        .setAndCreateFiles("hdrs", "c.h")
        .write();

    CommandAction compileActionA = compileAction("//cc:lib", "a.o");

    assertThat(compileActionA.getArguments()).contains("-fembed-bitcode");
  }

  @Test
  public void testCompilationActionsWithEmbeddedBitcodeForMultiplePlatformsWithoutMatch()
      throws Exception {
    useConfiguration(
        "--apple_platform_type=ios",
        "--cpu=ios_arm64",
        "--apple_bitcode=tvos=embedded",
        "--apple_bitcode=watchos=embedded");
    createLibraryTargetWriter("//cc:lib")
        .setAndCreateFiles("srcs", "a.cc", "b.cc", "private.h")
        .setAndCreateFiles("hdrs", "c.h")
        .write();

    CommandAction compileActionA = compileAction("//cc:lib", "a.o");

    assertThat(compileActionA.getArguments()).doesNotContain("-fembed-bitcode");
    assertThat(compileActionA.getArguments()).doesNotContain("-fembed-bitcode-marker");
  }

  @Test
  public void testLaterBitcodeOptionsOverrideEarlierOptionsForSamePlatform() throws Exception {
    useConfiguration(
        "--apple_platform_type=ios",
        "--cpu=ios_arm64",
        "--apple_bitcode=ios=embedded",
        "--apple_bitcode=ios=embedded_markers");
    createLibraryTargetWriter("//cc:lib")
        .setAndCreateFiles("srcs", "a.cc", "b.cc", "private.h")
        .setAndCreateFiles("hdrs", "c.h")
        .write();

    CommandAction compileActionA = compileAction("//cc:lib", "a.o");

    assertThat(compileActionA.getArguments()).doesNotContain("-fembed-bitcode");
    assertThat(compileActionA.getArguments()).contains("-fembed-bitcode-marker");
  }

  @Test
  public void testLaterBitcodeOptionWithoutPlatformOverridesEarlierOptionWithPlatform()
      throws Exception {
    useConfiguration(
        "--apple_platform_type=ios",
        "--cpu=ios_arm64",
        "--apple_bitcode=ios=embedded",
        "--apple_bitcode=embedded_markers");
    createLibraryTargetWriter("//cc:lib")
        .setAndCreateFiles("srcs", "a.cc", "b.cc", "private.h")
        .setAndCreateFiles("hdrs", "c.h")
        .write();

    CommandAction compileActionA = compileAction("//cc:lib", "a.o");

    assertThat(compileActionA.getArguments()).doesNotContain("-fembed-bitcode");
    assertThat(compileActionA.getArguments()).contains("-fembed-bitcode-marker");
  }

  @Test
  public void testLaterPlatformBitcodeOptionWithPlatformOverridesEarlierOptionWithoutPlatform()
      throws Exception {
    useConfiguration(
        "--apple_platform_type=ios",
        "--cpu=ios_arm64",
        "--apple_bitcode=embedded",
        "--apple_bitcode=ios=embedded_markers");
    createLibraryTargetWriter("//cc:lib")
        .setAndCreateFiles("srcs", "a.cc", "b.cc", "private.h")
        .setAndCreateFiles("hdrs", "c.h")
        .write();

    CommandAction compileActionA = compileAction("//cc:lib", "a.o");

    assertThat(compileActionA.getArguments()).doesNotContain("-fembed-bitcode");
    assertThat(compileActionA.getArguments()).contains("-fembed-bitcode-marker");
  }

  @Test
  public void testGenerateDsymFlagPropagatesToCcLibraryFeature() throws Exception {
    useConfiguration("--apple_generate_dsym");
    createLibraryTargetWriter("//cc/lib").setList("srcs", "a.cc").write();
    CommandAction compileAction = compileAction("//cc/lib", "a.o");
    assertThat(compileAction.getArguments()).contains("-DDUMMY_GENERATE_DSYM_FILE");
  }
}
