// Copyright 2018 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.packages.util.ResourceLoader;
import com.google.devtools.build.lib.syntax.Mutability;
import com.google.devtools.build.lib.syntax.StarlarkList;
import com.google.devtools.build.lib.syntax.util.EvaluationTestCase;
import com.google.devtools.build.lib.testutil.TestConstants;
import java.io.IOException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for cc autoconfiguration. */
@RunWith(JUnit4.class)
public class SkylarkCcToolchainConfigureTest extends EvaluationTestCase {

  @Test
  public void testSplitEscaped() throws Exception {
    Mutability mu = null;
    newTest()
        .testExpression("split_escaped('a:b:c', ':')", StarlarkList.of(mu, "a", "b", "c"))
        .testExpression("split_escaped('a%:b', ':')", StarlarkList.of(mu, "a:b"))
        .testExpression("split_escaped('a%%b', ':')", StarlarkList.of(mu, "a%b"))
        .testExpression("split_escaped('a:::b', ':')", StarlarkList.of(mu, "a", "", "", "b"))
        .testExpression("split_escaped('a:b%:c', ':')", StarlarkList.of(mu, "a", "b:c"))
        .testExpression("split_escaped('a%%:b:c', ':')", StarlarkList.of(mu, "a%", "b", "c"))
        .testExpression("split_escaped(':a', ':')", StarlarkList.of(mu, "", "a"))
        .testExpression("split_escaped('a:', ':')", StarlarkList.of(mu, "a", ""))
        .testExpression("split_escaped('::a::', ':')", StarlarkList.of(mu, "", "", "a", "", ""))
        .testExpression("split_escaped('%%%:a%%%%:b', ':')", StarlarkList.of(mu, "%:a%%", "b"))
        .testExpression("split_escaped('', ':')", StarlarkList.of(mu))
        .testExpression("split_escaped('%', ':')", StarlarkList.of(mu, "%"))
        .testExpression("split_escaped('%%', ':')", StarlarkList.of(mu, "%"))
        .testExpression("split_escaped('%:', ':')", StarlarkList.of(mu, ":"))
        .testExpression("split_escaped(':', ':')", StarlarkList.of(mu, "", ""))
        .testExpression("split_escaped('a%%b', ':')", StarlarkList.of(mu, "a%b"))
        .testExpression("split_escaped('a%:', ':')", StarlarkList.of(mu, "a:"));
  }

  private Scenario newTest(String... skylarkOptions) throws IOException {
    return new Scenario(skylarkOptions)
        // A mock implementation of Label to be able to parse lib_cc_configure under default
        // Skylark environment (lib_cc_configure is meant to be used from the repository
        // environment).
        .setUp("def Label(arg):\n  return 42")
        .setUp(
            ResourceLoader.readFromResources(
                TestConstants.RULES_CC_REPOSITORY_EXECROOT
                    + "cc/private/toolchain/lib_cc_configure.bzl"));
  }
}
