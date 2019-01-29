// Copyright 2015 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.rules.cpp;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.analysis.util.DefaultsPackageUtil;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.testutil.TestConstants;
import com.google.devtools.common.options.OptionsParsingException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests {@link CppOptions}.
 */
@RunWith(JUnit4.class)
public class CppOptionsTest {

  @Test
  public void testGetDefaultsPackage() throws Exception {
    String content = DefaultsPackageUtil.getDefaultsPackageForOptions(CppOptions.class);
    Label toolchainLabel =
        Label.parseAbsoluteUnchecked(TestConstants.TOOLS_REPOSITORY + "//tools/cpp:toolchain");
    assertThat(content)
        .contains(
            String.format(
                // Indentation matched literally.
                "filegroup(name = 'crosstool',\n          srcs = ['%s'])",
                toolchainLabel.getDefaultCanonicalForm()));
  }

  @Test
  public void testGetDefaultsPackageHostCrosstoolTop() throws OptionsParsingException {
    String content = DefaultsPackageUtil.getDefaultsPackageForOptions(
        CppOptions.class, "--host_crosstool_top=//some/package:crosstool");
    assertThat(content).contains("//some/package:crosstool");
  }

  @Test
  public void testGetDefaultsPackageGrteTop() throws OptionsParsingException {
    String content = DefaultsPackageUtil.getDefaultsPackageForOptions(
        CppOptions.class, "--grte_top=//some/grte:other");
    assertThat(content).contains("//some/grte:everything");
  }
}
