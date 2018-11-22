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

package com.google.devtools.build.lib.packages;

import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.collect.ImmutableSet;
import com.google.common.io.Files;
import com.google.devtools.build.lib.bazel.Bazel;
import com.google.devtools.build.lib.bazel.rules.BazelRuleClassProvider;
import com.google.devtools.build.lib.packages.util.DocumentationTestUtil;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.windows.util.WindowsTestUtil;
import com.google.devtools.build.runfiles.Runfiles;
import java.io.File;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Test for Bazel documentation.
 */
@RunWith(JUnit4.class)
public class BazelDocumentationTest {
  /**
   * Checks that the user-manual is in sync with the {@link
   * com.google.devtools.build.lib.analysis.config.BuildConfiguration}.
   */
  @Test
  public void testBazelUserManual() throws Exception {
    Runfiles runfiles = Runfiles.create();
    String documentationFilePath = runfiles.rlocation("io_bazel/site/docs/user-manual.html");
    final File documentationFile = new File(documentationFilePath);
    DocumentationTestUtil.validateUserManual(
        Bazel.BAZEL_MODULES,
        BazelRuleClassProvider.create(),
        Files.asCharSource(documentationFile, UTF_8).read(),
        ImmutableSet.of());
  }
}
