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

import static org.junit.Assert.assertEquals;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Lists;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCaseForJunit4;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.packages.Package.Builder;
import com.google.devtools.build.lib.syntax.Argument;
import com.google.devtools.build.lib.syntax.FuncallExpression;
import com.google.devtools.build.lib.syntax.Identifier;
import com.google.devtools.build.lib.testutil.TestRuleClassProvider;
import com.google.devtools.build.lib.vfs.Path;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.util.Map;

/**
 * Test for building external packages.
 */
@RunWith(JUnit4.class)
public class ExternalPackageTest extends BuildViewTestCaseForJunit4 {

  private Path workspacePath;

  @Before
  public final void setWorkspacePath() throws Exception {
    workspacePath = getOutputPath().getRelative("WORKSPACE");
  }

  @Test
  public void testWorkspaceName() {
    Builder builder = Package.newExternalPackageBuilder(workspacePath, "TESTING");
    builder.setWorkspaceName("foo");
    assertEquals("foo", builder.build().getWorkspaceName());
  }

  @Test
  public void testMultipleRulesWithSameName() throws Exception {
    Builder builder = Package.newExternalPackageBuilder(workspacePath, "TESTING");

    // The WORKSPACE file allows rules to be overridden.
    Location buildFile = Location.fromFile(getOutputPath().getRelative("WORKSPACE"));

    // Add first rule.
    RuleClass ruleClass =
        TestRuleClassProvider.getRuleClassProvider().getRuleClassMap().get("local_repository");
    RuleClass bindRuleClass =
        TestRuleClassProvider.getRuleClassProvider().getRuleClassMap().get("bind");

    Map<String, Object> kwargs = ImmutableMap.of("name", (Object) "my-rule");
    FuncallExpression ast =
        new FuncallExpression(
            new Identifier(ruleClass.getName()), Lists.<Argument.Passed>newArrayList());
    ast.setLocation(buildFile);
    builder
        .externalPackageData()
        .createAndAddRepositoryRule(builder, ruleClass, bindRuleClass, kwargs, ast);

    // Add another rule with the same name.
    ruleClass =
        TestRuleClassProvider.getRuleClassProvider().getRuleClassMap().get("new_local_repository");
    ast =
        new FuncallExpression(
            new Identifier(ruleClass.getName()), Lists.<Argument.Passed>newArrayList());
    ast.setLocation(buildFile);
    builder
        .externalPackageData()
        .createAndAddRepositoryRule(builder, ruleClass, bindRuleClass, kwargs, ast);
    Package pkg = builder.build();

    // Make sure the second rule "wins."
    assertEquals("new_local_repository rule", pkg.getTarget("my-rule").getTargetKind());
  }
}
