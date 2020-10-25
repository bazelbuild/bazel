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
import com.google.devtools.build.lib.starlark.util.BazelEvaluationTestCase;
import com.google.devtools.build.lib.testutil.TestConstants;
import java.io.IOException;
import net.starlark.java.eval.Mutability;
import net.starlark.java.eval.StarlarkList;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for cc autoconfiguration. */
@RunWith(JUnit4.class)
public class StarlarkCcToolchainConfigureTest {

  private final BazelEvaluationTestCase ev = new BazelEvaluationTestCase();

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

  private BazelEvaluationTestCase.Scenario newTest(String... starlarkOptions) throws IOException {
    return ev.new Scenario(starlarkOptions)
        .setUp(
            ResourceLoader.readFromResources(
                TestConstants.RULES_CC_REPOSITORY_EXECROOT
                    + "cc/private/toolchain/lib_cc_configure.bzl"));
  }
}
