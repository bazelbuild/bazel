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
package com.google.devtools.build.lib.rules.android;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.packages.util.MockCcSupport;
import com.google.devtools.build.lib.rules.android.AndroidRuleClasses.MultidexMode;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests common for Android rules.
 */
@RunWith(JUnit4.class)
public class AndroidCommonTest extends BuildViewTestCase {

  @Before
  public void setupCcToolchain() throws Exception {
    getAnalysisMock()
        .ccSupport()
        .setupCrosstool(
            mockToolsConfig,
            /* appendToCurrentToolchain=*/ false,
            MockCcSupport.emptyToolchainForCpu("armeabi-v7a"));
  }

  @Before
  public final void createFile() throws Exception {
    scratch.file("java/srcs/a.properties", "foo");
  }

  // regression test for #3169099
  @Test
  public void testLibrarySrcs() throws Exception {
    scratch.file("java/srcs/BUILD",
        "android_library(name = 'valid', srcs = ['a.java', 'b.srcjar', ':gvalid', ':gmix'])",
        "android_library(name = 'invalid', srcs = ['a.properties', ':ginvalid'])",
        "android_library(name = 'mix', srcs = ['a.java', 'a.properties'])",
        "genrule(name = 'gvalid', srcs = ['a.java'], outs = ['b.java'], cmd = '')",
        "genrule(name = 'ginvalid', srcs = ['a.java'], outs = ['b.properties'], cmd = '')",
        "genrule(name = 'gmix', srcs = ['a.java'], outs = ['c.java', 'c.properties'], cmd = '')"
        );
    assertSrcsValidityForRuleType("android_library", ".java or .srcjar");
  }

  // regression test for #3169099
  @Test
  public void testBinarySrcs() throws Exception {
    scratch.file("java/srcs/BUILD",
        "android_binary(name = 'empty', manifest = 'AndroidManifest.xml', srcs = [])",
        "android_binary(name = 'valid', manifest = 'AndroidManifest.xml', "
            + "srcs = ['a.java', 'b.srcjar', ':gvalid', ':gmix'])",
        "android_binary(name = 'invalid', manifest = 'AndroidManifest.xml', "
            + "srcs = ['a.properties', ':ginvalid'])",
        "android_binary(name = 'mix', manifest = 'AndroidManifest.xml', "
            + "srcs = ['a.java', 'a.properties'])",
        "genrule(name = 'gvalid', srcs = ['a.java'], outs = ['b.java'], cmd = '')",
        "genrule(name = 'ginvalid', srcs = ['a.java'], outs = ['b.properties'], cmd = '')",
        "genrule(name = 'gmix', srcs = ['a.java'], outs = ['c.java', 'c.properties'], cmd = '')"
        );
    assertSrcsValidityForRuleType("android_binary", ".java or .srcjar");
  }

  private void assertSrcsValidityForRuleType(String rule, String expectedTypes) throws Exception {
    reporter.removeHandler(failFastHandler);
    String descriptionSingle = rule + " srcs file (expected " + expectedTypes + ")";
    String descriptionPlural = rule + " srcs files (expected " + expectedTypes + ")";
    String descriptionPluralFile = "(expected " + expectedTypes + ")";
    assertSrcsValidity(rule, "//java/srcs:valid", false, "need at least one " + descriptionSingle,
        "'//java/srcs:gvalid' is misplaced " + descriptionPlural,
        "'//java/srcs:gmix' does not produce any " + descriptionPlural);
    assertSrcsValidity(rule, "//java/srcs:invalid", true,
        "source file '//java/srcs:a.properties' is misplaced here " + descriptionPluralFile,
        "'//java/srcs:ginvalid' does not produce any " + descriptionPlural);
    assertSrcsValidity(rule, "//java/srcs:mix", true,
        "'//java/srcs:a.properties' does not produce any " + descriptionPlural);
  }

  /**
   * Tests expected values of
   * {@link com.google.devtools.build.lib.rules.android.AndroidRuleClasses.MultidexMode}.
   */
  @Test
  public void testMultidexModeEnum() throws Exception {
    assertThat(MultidexMode.getValidValues()).containsExactly("native", "legacy", "manual_main_dex",
        "off");
    assertThat(MultidexMode.fromValue("native")).isSameAs(MultidexMode.NATIVE);
    assertThat(MultidexMode.NATIVE.getAttributeValue()).isEqualTo("native");
    assertThat(MultidexMode.fromValue("legacy")).isSameAs(MultidexMode.LEGACY);
    assertThat(MultidexMode.LEGACY.getAttributeValue()).isEqualTo("legacy");
    assertThat(MultidexMode.fromValue("manual_main_dex")).isSameAs(MultidexMode.MANUAL_MAIN_DEX);
    assertThat(MultidexMode.MANUAL_MAIN_DEX.getAttributeValue()).isEqualTo("manual_main_dex");
    assertThat(MultidexMode.fromValue("off")).isSameAs(MultidexMode.OFF);
    assertThat(MultidexMode.OFF.getAttributeValue()).isEqualTo("off");
  }

  /**
   * Tests that each multidex mode produces the expected output dex classes file name.
   */
  @Test
  public void testOutputDexforMultidexModes() throws Exception {
    assertThat(MultidexMode.OFF.getOutputDexFilename()).isEqualTo("classes.dex");
    assertThat(MultidexMode.LEGACY.getOutputDexFilename()).isEqualTo("classes.dex.zip");
    assertThat(MultidexMode.NATIVE.getOutputDexFilename()).isEqualTo("classes.dex.zip");
  }
}
