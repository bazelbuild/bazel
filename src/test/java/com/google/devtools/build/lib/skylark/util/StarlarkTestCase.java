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


import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.skylark.StarlarkModules;
import com.google.devtools.build.lib.analysis.skylark.StarlarkRuleContext;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.packages.BazelModuleContext;
import com.google.devtools.build.lib.packages.BazelStarlarkContext;
import com.google.devtools.build.lib.packages.SymbolGenerator;
import com.google.devtools.build.lib.rules.platform.PlatformCommon;
import com.google.devtools.build.lib.syntax.Module;
import com.google.devtools.build.lib.syntax.Mutability;
import com.google.devtools.build.lib.syntax.Starlark;
import com.google.devtools.build.lib.syntax.StarlarkThread;
import com.google.devtools.build.lib.syntax.util.EvaluationTestCase;
import com.google.devtools.build.lib.testutil.TestConstants;
import org.junit.Before;

/**
 * A class to contain the common functionality for analysis-phase Starlark tests. The less stuff
 * here, the better.
 */
public abstract class StarlarkTestCase extends BuildViewTestCase {

  protected EvaluationTestCase ev;

  // Subclasses must call this method after change StarlarkSemantics (among other things).
  @Before
  public final void reset() throws Exception {
    ev = createEvaluationTestCase();
  }

  private EvaluationTestCase createEvaluationTestCase() throws Exception {
    // Set up globals.
    ImmutableMap.Builder<String, Object> env = ImmutableMap.builder();
    StarlarkModules.addStarlarkGlobalsToBuilder(env);
    Starlark.addModule(env, new PlatformCommon());
    Module globals = Module.createForBuiltins(env.build());

    EvaluationTestCase ev =
        new EvaluationTestCase() {
          @Override
          public StarlarkThread newStarlarkThread() {
            Mutability mu = Mutability.create("test");
            StarlarkThread thread =
                StarlarkThread.builder(mu)
                    .setSemantics(getStarlarkSemantics())
                    .setGlobals(
                        globals.withClientData(
                            BazelModuleContext.create(
                                Label.parseAbsoluteUnchecked(
                                    "//test:label", /*defaultToMain=*/ false),
                                // dummy value for tests
                                /*bzlTransitiveDigest=*/ new byte[0])))
                    .build();
            thread.setPrintHandler(Event.makeDebugPrintHandler(getEventHandler()));

            // This StarlarkThread has no PackageContext, so attempts to create a rule will fail.
            // Rule creation is tested by StarlarkIntegrationTest.

            new BazelStarlarkContext(
                    BazelStarlarkContext.Phase.LOADING,
                    TestConstants.TOOLS_REPOSITORY,
                    /*fragmentNameToClass=*/ null,
                    /*repoMapping=*/ ImmutableMap.of(),
                    new SymbolGenerator<>(new Object()),
                    /*analysisRuleLabel=*/ null)
                .storeInThread(thread);

            return thread;
          }
        };
    ev.initialize();
    return ev;
  }

  protected final Object eval(String... input) throws Exception {
    return ev.eval(input);
  }

  protected final void exec(String... lines) throws Exception {
    ev.exec(lines);
  }

  protected final void update(String name, Object value) throws Exception {
    ev.update(name, value);
  }

  protected final Object lookup(String name) throws Exception {
    return ev.lookup(name);
  }

  protected final void checkEvalErrorContains(String msg, String... input) throws Exception {
    ev.checkEvalErrorContains(msg, input);
  }

  protected final StarlarkRuleContext createRuleContext(String label) throws Exception {
    return new StarlarkRuleContext(
        getRuleContextForStarlark(getConfiguredTarget(label)), null, getStarlarkSemantics());
  }

  // Disable BuildViewTestCase's overload to avoid unintended calls.
  @Override
  protected final Event checkError(String x, String y) throws Exception {
    throw new IllegalStateException("wrong method");
  }

}
