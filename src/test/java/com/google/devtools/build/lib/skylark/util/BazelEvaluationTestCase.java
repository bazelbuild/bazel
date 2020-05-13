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
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.packages.BazelModuleContext;
import com.google.devtools.build.lib.packages.BazelStarlarkContext;
import com.google.devtools.build.lib.packages.SymbolGenerator;
import com.google.devtools.build.lib.rules.platform.PlatformCommon;
import com.google.devtools.build.lib.syntax.Module;
import com.google.devtools.build.lib.syntax.Mutability;
import com.google.devtools.build.lib.syntax.Starlark;
import com.google.devtools.build.lib.syntax.StarlarkSemantics;
import com.google.devtools.build.lib.syntax.StarlarkThread;
import com.google.devtools.build.lib.syntax.util.EvaluationTestCase;
import com.google.devtools.build.lib.testutil.TestConstants;

/**
 * BazelEvaluationTestCase is a subclass of EvaluationTestCase that defines various Bazel built-ins
 * in the environment.
 */
// TODO(adonovan): this is (and has always been) a mess, and an abuse of inheritance.
// Once the production code API has disentangled Thread and Module, make this rational.
public final class BazelEvaluationTestCase extends EvaluationTestCase {

  // Caution: called by the base class constructor.
  @Override
  protected StarlarkThread newStarlarkThread(StarlarkSemantics semantics) {
    ImmutableMap.Builder<String, Object> env = ImmutableMap.builder();
    StarlarkModules.addStarlarkGlobalsToBuilder(env);
    Starlark.addModule(env, new PlatformCommon());

    StarlarkThread thread =
        StarlarkThread.builder(Mutability.create("test"))
            .setSemantics(semantics)
            .setGlobals(
                Module.createForBuiltins(env.build())
                    .withClientData(
                        BazelModuleContext.create(
                            Label.parseAbsoluteUnchecked("//test:label", /*defaultToMain=*/ false),
                            /*bzlTransitiveDigest=*/ new byte[0]))) // dummy value for tests
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
            /*analysisRuleLabel=*/ null) // dummy value for tests
        .storeInThread(thread);

    return thread;
  }
}
