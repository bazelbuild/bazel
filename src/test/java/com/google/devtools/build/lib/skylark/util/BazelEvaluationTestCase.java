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
import com.google.devtools.build.lib.analysis.starlark.StarlarkModules;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.BazelModuleContext;
import com.google.devtools.build.lib.packages.BazelStarlarkContext;
import com.google.devtools.build.lib.packages.SymbolGenerator;
import com.google.devtools.build.lib.rules.platform.PlatformCommon;
import com.google.devtools.build.lib.syntax.Starlark;
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

  @Override
  protected Object newModuleHook(ImmutableMap.Builder<String, Object> predeclared) {
    StarlarkModules.addStarlarkGlobalsToBuilder(predeclared);
    Starlark.addModule(predeclared, new PlatformCommon());

    // Return the module's client data. (This one uses dummy values for tests.)
    return BazelModuleContext.create(
        Label.parseAbsoluteUnchecked("//test:label", /*defaultToMain=*/ false),
        "test/label.bzl",
        /*loads=*/ ImmutableMap.of(),
        /*bzlTransitiveDigest=*/ new byte[0]);
  }

  @Override
  protected void newThreadHook(StarlarkThread thread) {
    // This StarlarkThread has no PackageContext, so attempts to create a rule will fail.
    // Rule creation is tested by StarlarkIntegrationTest.

    // This is a poor approximation to the thread that Blaze would create
    // for testing rule implementation functions. It has phase LOADING, for example.
    // TODO(adonovan): stop creating threads in tests. This is the responsibility of the
    // production code. Tests should provide only files and commands.
    new BazelStarlarkContext(
            BazelStarlarkContext.Phase.LOADING,
            TestConstants.TOOLS_REPOSITORY,
            /*fragmentNameToClass=*/ null,
            /*repoMapping=*/ ImmutableMap.of(),
            new SymbolGenerator<>(new Object()),
            /*analysisRuleLabel=*/ null) // dummy value for tests
        .storeInThread(thread);
  }
}
