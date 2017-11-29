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
package com.google.devtools.build.lib.rules.java;

import static com.google.common.truth.Truth.assertWithMessage;

import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.syntax.Type;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Unit tests for java_test rile.
 */
@RunWith(JUnit4.class)
public class JavaTestTest extends BuildViewTestCase {

  @Test
  public void testMainClassWithoutUseTestRunner() throws Exception {

    ConfiguredTarget ct = scratchConfiguredTarget(
            "a",
            "UXMessageTest",
            "java_test("
                + "    name = 'UXMessageTest',"
                + "    srcs = ['UXMessageTest.java'],"
                + "    main_class = 'com.google.devtools.build.lib.view.uxmessage.UXMessage'"
                + ")");

    assertThatUseTestrunnerIsFalse(ct);
  }

  @Test
  public void testWithUseTestrunnerDefinedAsFalse() throws Exception {

    ConfiguredTarget ct = scratchConfiguredTarget(
            "a",
            "UXMessageTest",
            "java_test("
                + "    name = 'UXMessageTest',"
                + "    srcs = ['UXMessageTest.java'],"
                + "    main_class = 'com.google.devtools.build.lib.view.uxmessage.UXMessage',"
                + "    use_testrunner = 0"
                + ")");

    assertThatUseTestrunnerIsFalse(ct);
  }

  private void assertThatUseTestrunnerIsFalse(ConfiguredTarget configuredTarget) throws Exception {

    Boolean useTestrunner =
        getRuleContext(configuredTarget).attributes().get("use_testrunner", Type.BOOLEAN);

    assertWithMessage("attribute 'use_testrunner' value should be false")
        .that(useTestrunner)
        .isFalse();
  }

  @Test
  public void testWithUseTestrunnerDefinedAsTrue() throws Exception {

    checkError("java",
        "UXMessageTest",
        "cannot use use_testrunner with main_class specified.",
        "java_test("
            + "    name = 'UXMessageTest',"
            + "    srcs = ['UXMessageTest.java'],"
            + "    main_class = 'com.google.devtools.build.lib.view.uxmessage.UXMessage',"
            + "    use_testrunner = 1,"
            + ")");
  }
}
