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

package com.google.devtools.build.lib.bazel.rules.java;


import com.google.common.collect.Maps;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.LicensingTests;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.License;
import java.util.Map;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Integration tests for {@link License}.
 */
@RunWith(JUnit4.class)
public class BazelLicensingTests extends LicensingTests {
  @Test
  public void testJavaPluginAllowsOutputLicenseDeclaration() throws Exception {
    scratch.file("ise/BUILD",
        "licenses(['restricted'])",
        "java_library(name = 'dependency',",
        "             srcs = ['dependency.java'])",
        "java_plugin(name = 'plugin',",
        "            deps = [':dependency'],",
        "            srcs = ['plugin.java'],",
        "            output_licenses = ['unencumbered'])");

    scratch.file("gsa/BUILD",
        "licenses(['unencumbered'])",
        "java_library(name = 'library',",
        "             srcs = ['library.java'],",
        "             plugins = ['//ise:plugin'])");

    ConfiguredTarget library = getConfiguredTarget("//gsa:library");
    Map<Label, License> actual = Maps.filterKeys(getTransitiveLicenses(library),
        CC_OR_JAVA_OR_SH_LABEL_FILTER);
    Map<Label, License> expected = licenses(
        "//ise:plugin", "unencumbered",
        "//gsa:library", "unencumbered"
    );

    assertSameMapEntries(expected, actual);
  }
}
