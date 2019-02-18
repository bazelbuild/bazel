// Copyright 2014 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.skylark.util;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.fail;

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Lists;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.skylark.BazelStarlarkContext;
import com.google.devtools.build.lib.analysis.skylark.SkylarkModules;
import com.google.devtools.build.lib.analysis.skylark.SkylarkRuleContext;
import com.google.devtools.build.lib.analysis.skylark.SymbolGenerator;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.PackageFactory;
import com.google.devtools.build.lib.packages.PackageFactory.PackageContext;
import com.google.devtools.build.lib.rules.platform.PlatformCommon;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.Environment.GlobalFrame;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.Runtime;
import com.google.devtools.build.lib.syntax.SkylarkUtils;
import com.google.devtools.build.lib.syntax.SkylarkUtils.Phase;
import com.google.devtools.build.lib.syntax.StarlarkSemantics;
import com.google.devtools.build.lib.syntax.util.EvaluationTestCase;
import com.google.devtools.build.lib.testutil.TestConstants;
import org.junit.Before;

/**
 * A class to contain the common functionality for Skylark tests.
 */
public abstract class SkylarkTestCase extends BuildViewTestCase {

  // We don't have multiple inheritance, so we fake it.
  protected EvaluationTestCase ev;

  @Before
  public final void setUpEvaluator() throws Exception {
    ev = createEvaluationTestCase(StarlarkSemantics.DEFAULT_SEMANTICS);
    ev.initialize();
  }

  private static final Environment.GlobalFrame getSkylarkGlobals() {
    ImmutableMap.Builder<String, Object> envBuilder = ImmutableMap.builder();

    SkylarkModules.addSkylarkGlobalsToBuilder(envBuilder);
    Runtime.setupSkylarkLibrary(envBuilder, new PlatformCommon());

    return GlobalFrame.createForBuiltins(envBuilder.build());
  }

  protected EvaluationTestCase createEvaluationTestCase(StarlarkSemantics semantics) {
    return new EvaluationTestCase() {
      @Override
      public Environment newEnvironment() throws Exception {
        BazelStarlarkContext context =
            new BazelStarlarkContext(
                TestConstants.TOOLS_REPOSITORY,
                /*repoMapping=*/ ImmutableMap.of(),
                new SymbolGenerator<>(new Object()));
        Environment env =
            Environment.builder(mutability)
                .setSemantics(semantics)
                .setEventHandler(getEventHandler())
                .setGlobals(
                    getSkylarkGlobals()
                        .withLabel(
                            Label.parseAbsoluteUnchecked("//test:label", /*defaultToMain=*/ false)))
                .setStarlarkContext(context)
                .build()
                .setupDynamic(
                    PackageFactory.PKG_CONTEXT,
                    // This dummy pkgContext works because no Skylark unit test attempts to actually
                    // create rules. Creating actual rules is tested in SkylarkIntegrationTest.
                    new PackageContext(null, null, getEventHandler(), null));
        SkylarkUtils.setPhase(env, Phase.LOADING);
        return env;
      }
    };
  }

  protected Object eval(String... input) throws Exception {
    return ev.eval(input);
  }

  protected void update(String name, Object value) throws Exception {
    ev.update(name, value);
  }

  protected Object lookup(String name) throws Exception {
    return ev.lookup(name);
  }

  protected void checkEvalError(String msg, String... input) throws Exception {
    ev.checkEvalError(msg, input);
  }

  protected void checkEvalErrorContains(String msg, String... input) throws Exception {
    ev.checkEvalErrorContains(msg, input);
  }

  protected SkylarkRuleContext dummyRuleContext() throws Exception {
    return createRuleContext("//foo:foo");
  }

  protected SkylarkRuleContext createRuleContext(String label) throws Exception {
    return new SkylarkRuleContext(getRuleContextForSkylark(getConfiguredTarget(label)), null,
        getSkylarkSemantics());
  }

  protected Object evalRuleContextCode(String... lines) throws Exception {
    return evalRuleContextCode(dummyRuleContext(), lines);
  }

  /**
   * RuleContext can't be null, SkylarkBuiltInFunctions checks it.
   * However not all built in functions use it, if usesRuleContext == false
   * the wrapping function won't need a ruleContext parameter.
   */
  protected Object evalRuleContextCode(SkylarkRuleContext ruleContext, String... code)
      throws Exception {
    setUpEvaluator();
    if (ruleContext != null) {
      update("ruleContext", ruleContext);
    }
    return eval(code);
  }

  protected void assertArtifactFilenames(Iterable<Artifact> artifacts, String... expected) {
    ImmutableList.Builder<String> artifactFilenames = ImmutableList.builder();
    for (Artifact artifact : artifacts) {
      artifactFilenames.add(artifact.getFilename());
    }
    assertThat(artifactFilenames.build()).containsAllIn(Lists.newArrayList(expected));
  }

  protected Object evalRuleClassCode(String... lines) throws Exception {
    setUpEvaluator();
    return eval("def impl(ctx): return None\n" + Joiner.on("\n").join(lines));
  }

  protected Object evalRuleClassCode(StarlarkSemantics semantics, String... lines)
      throws Exception {
    ev = createEvaluationTestCase(semantics);
    ev.initialize();
    return eval("def impl(ctx): return None\n" + Joiner.on("\n").join(lines));
  }

  protected void checkError(SkylarkRuleContext ruleContext, String errorMsg, String... lines)
      throws Exception {
    try {
      evalRuleContextCode(ruleContext, lines);
      fail();
    } catch (EvalException e) {
      assertThat(e).hasMessage(errorMsg);
    }
  }

  protected void checkErrorStartsWith(
      SkylarkRuleContext ruleContext, String errorMsg, String... lines) throws Exception {
    try {
      evalRuleContextCode(ruleContext, lines);
      fail();
    } catch (EvalException e) {
      assertThat(e).hasMessageThat().startsWith(errorMsg);
    }
  }

  protected void checkErrorContains(String errorMsg, String... lines) throws Exception {
    ev.setFailFast(false);
    try {
      eval(lines);
      fail("checkErrorContains(String, String...): There was no error");
    } catch (EvalException e) {
      assertThat(e).hasMessageThat().contains(errorMsg);
    }
  }

  protected void checkErrorContains(
      SkylarkRuleContext ruleContext, String errorMsg, String... lines) throws Exception {
    try {
      evalRuleContextCode(ruleContext, lines);
      fail("checkErrorContains(SkylarkRuleContext, String, String...): There was no error");
    } catch (EvalException e) {
      assertThat(e).hasMessageThat().contains(errorMsg);
    }
  }
}
