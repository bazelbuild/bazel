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

import com.google.devtools.build.lib.rules.android.AndroidRuleClasses.MultidexMode;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests the multidex code of {@link com.google.devtools.build.lib.rules.android.AndroidBinary}. */
@RunWith(JUnit4.class)
public class AndroidBinaryMultidexTest extends AndroidMultidexBaseTest {

  @Before
  public void setup() throws Exception {
    getAnalysisMock().ccSupport().setupCcToolchainConfigForCpu(mockToolsConfig, "armeabi-v7a");
    useConfiguration(
        "--experimental_use_dex_splitter_for_incremental_dexing=false",
        "--experimental_incremental_dexing_after_proguard_by_default=false",
        "--experimental_incremental_dexing_after_proguard=1");
  }

  /**
   * Tests that when multidex = "off", a classes.dex file is generated directly from the input jar.
   */
  @Test
  public void testNonMultidexBuildStructure() throws Exception {
    scratch.file(
        "java/foo/BUILD",
        """
        android_binary(
            name = "nomultidex",
            srcs = ["a.java"],
            manifest = "AndroidManifest.xml",
            resource_files = glob(
                ["res/**"],
                allow_empty = True,
            ),
        )
        """);
    internalTestNonMultidexBuildStructure("//java/foo:nomultidex");
  }

  /** Tests that the default multidex setting is the same as when multidex = "off". */
  @Test
  public void testDefaultBuildStructure() throws Exception {
    scratch.file(
        "java/foo/BUILD",
        """
        android_binary(
            name = "default",
            srcs = ["a.java"],
            manifest = "AndroidManifest.xml",
            resource_files = glob(
                ["res/**"],
                allow_empty = True,
            ),
        )
        """);
    internalTestNonMultidexBuildStructure("//java/foo:default");
  }

  @Test
  public void testManualMainDexMode() throws Exception {
    scratch.file("java/foo/main_dex_list.txt", "android/A.class");
    scratch.file(
        "java/foo/BUILD",
        """
        android_binary(
            name = "manual_main_dex",
            srcs = ["a.java"],
            main_dex_list = "main_dex_list.txt",
            manifest = "AndroidManifest.xml",
            multidex = "manual_main_dex",
            resource_files = glob(
                ["res/**"],
                allow_empty = True,
            ),
        )
        """);
    internalTestMultidexBuildStructure("//java/foo:manual_main_dex", MultidexMode.MANUAL_MAIN_DEX);
  }

  /**
   * Tests that when multidex = "legacy", a classes.dex.zip file is generated from an intermediate
   * file with multidex mode specified and a "--main-dex-list" dx flag filled out with appropriate
   * input, This file is then filtered through a zip action to remove non-.dex files to produce the
   * final output.
   */
  @Test
  public void testLegacyMultidexBuildStructure() throws Exception {
    scratch.file(
        "java/foo/BUILD",
        """
        android_binary(
            name = "legacy",
            srcs = ["a.java"],
            manifest = "AndroidManifest.xml",
            multidex = "legacy",
            resource_files = glob(
                ["res/**"],
                allow_empty = True,
            ),
        )
        """);
    internalTestMultidexBuildStructure("//java/foo:legacy", MultidexMode.LEGACY);
  }

  /**
   * Tests that when multidex = "native", a classes.dex.zip file is generated from an intermediate
   * file with multidex mode specified. Unlike in "legacy" mode, no actions are required to set and
   * fill the "--main-dex-list" dx flag. The intermediate file is then filtered through a zip action
   * to remove non-.dex files to produce the final output.
   */
  @Test
  public void testNativeMultidexBuildStructure() throws Exception {
    scratch.file(
        "java/foo/BUILD",
        """
        android_binary(
            name = "native",
            srcs = ["a.java"],
            manifest = "AndroidManifest.xml",
            multidex = "native",
            resource_files = glob(
                ["res/**"],
                allow_empty = True,
            ),
        )
        """);
    internalTestMultidexBuildStructure("//java/foo:native", MultidexMode.NATIVE);
  }
}
