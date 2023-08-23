// Copyright 2016 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.rules.test;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.analysis.RunEnvironmentInfo;
import com.google.devtools.build.lib.analysis.starlark.StarlarkRuleClassFunctions;
import com.google.devtools.build.lib.analysis.starlark.StarlarkRuleClassFunctions.StarlarkRuleFunction;
import com.google.devtools.build.lib.analysis.test.ExecutionInfo;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.events.EventKind;
import com.google.devtools.build.lib.events.StoredEventHandler;
import com.google.devtools.build.lib.packages.LabelConverter;
import com.google.devtools.build.lib.packages.PackageFactory.PackageContext;
import com.google.devtools.build.lib.starlarkbuildapi.test.TestingModuleApi;
import java.util.regex.Pattern;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Sequence;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkFunction;
import net.starlark.java.eval.StarlarkList;
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.eval.Tuple;

/** A class that exposes testing infrastructure to Starlark. */
public class StarlarkTestingModule implements TestingModuleApi {
  private static final Pattern RULE_NAME_PATTERN = Pattern.compile("[A-Za-z_][A-Za-z0-9_]*");

  @Override
  public ExecutionInfo.ExecutionInfoProvider executionInfo() {
    return ExecutionInfo.PROVIDER;
  }

  @Override
  public RunEnvironmentInfo testEnvironment(
      Dict<?, ?> environment /* <String, String> */,
      Sequence<?> inheritedEnvironment /* <String> */)
      throws EvalException {
    return new RunEnvironmentInfo(
        Dict.cast(environment, String.class, String.class, "environment"),
        StarlarkList.immutableCopyOf(
            Sequence.cast(inheritedEnvironment, String.class, "inherited_environment")),
        /* shouldErrorOnNonExecutableRule= */ false);
  }

  @Override
  public void analysisTest(
      String name,
      StarlarkFunction implementation,
      Object attrs,
      Sequence<?> fragments,
      Sequence<?> toolchains,
      Object attrValuesApi,
      StarlarkThread thread)
      throws EvalException, InterruptedException {
    PackageContext pkgContext = thread.getThreadLocal(PackageContext.class);
    RuleDefinitionEnvironment ruleDefinitionEnvironment =
        thread.getThreadLocal(RuleDefinitionEnvironment.class);
    // TODO(b/236456122): Refactor this check into a standard helper / error message
    if (pkgContext == null || ruleDefinitionEnvironment == null) {
      throw Starlark.errorf("analysis_test can only be called in a BUILD thread");
    }

    if (!RULE_NAME_PATTERN.matcher(name).matches()) {
      throw Starlark.errorf("'name' is limited to Starlark identifiers, got %s", name);
    }
    Dict<String, Object> attrValues =
        Dict.cast(attrValuesApi, String.class, Object.class, "attr_values");
    if (attrValues.containsKey("name")) {
      throw Starlark.errorf("'name' cannot be set or overridden in 'attr_values'");
    }

    // Get the callstack, sans the last entry, which is the builtin 'analysis_test' callable itself.
    ImmutableList<StarlarkThread.CallStackEntry> callStack = thread.getCallStack();
    callStack = callStack.subList(0, callStack.size() - 1);

    LabelConverter labelConverter = LabelConverter.forBzlEvaluatingThread(thread);

    // TODO(b/291752414): The digest of the rule class is incorrect. It is usually based on the
    // transitive digest of the .bzl being initialized. But since we're in a BUILD-evaluating
    // thread, we should use the transitive digest of the BUILD file. This is not currently
    // computed, but could be added to PackageFunction. In the meantime, we use a dummy empty digest
    // for all analysis_test-generated rule classes.
    Label dummyBzlFile = Label.createUnvalidated(PackageIdentifier.EMPTY_PACKAGE_ID, "dummy_label");
    byte[] dummyDigest = new byte[0];

    StarlarkRuleFunction starlarkRuleFunction =
        StarlarkRuleClassFunctions.createRule(
            // Contextual parameters.
            ruleDefinitionEnvironment,
            thread.getCallerLocation(),
            callStack,
            dummyBzlFile,
            dummyDigest,
            labelConverter,
            thread.getSemantics(),
            // rule() parameters.
            implementation,
            /* test= */ true,
            attrs,
            /* implicitOutputs= */ Starlark.NONE,
            /* executable= */ false,
            /* outputToGenfiles= */ false,
            /* fragments= */ fragments,
            /* starlarkTestable= */ false,
            /* toolchains= */ toolchains,
            /* doc= */ Starlark.NONE,
            /* providesArg= */ StarlarkList.empty(),
            /* execCompatibleWith= */ StarlarkList.empty(),
            /* analysisTest= */ Boolean.TRUE,
            /* buildSetting= */ Starlark.NONE,
            /* cfg= */ Starlark.NONE,
            /* execGroups= */ Starlark.NONE);

    // Export the rule.
    //
    // Because exporting can raise multiple errors, we need to accumulate them here into a single
    // EvalException. This is a code smell because any non-ERROR events will be lost, and any
    // location information in the events will be overwritten by the location of this rule's
    // definition.
    //
    // However, this is currently fine because StarlarkRuleFunction#export only creates events that
    // are ERRORs and that have the rule definition as their location.
    //
    // TODO(brandjon): Instead of accumulating events here, consider registering the rule in the
    // BazelStarlarkContext (or the appropriate subclass), and exporting such rules after module
    // evaluation in BzlLoadFunction#execAndExport.
    StoredEventHandler handler = new StoredEventHandler();
    starlarkRuleFunction.export(
        handler, pkgContext.getLabel(), name + "_test"); // export in BUILD thread
    if (handler.hasErrors()) {
      StringBuilder errors =
          handler.getEvents().stream()
              .filter(e -> e.getKind() == EventKind.ERROR)
              .reduce(
                  new StringBuilder(),
                  (sb, ev) -> sb.append("\n").append(ev.getMessage()),
                  StringBuilder::append);
      throw Starlark.errorf("Errors in exporting %s: %s", name, errors.toString());
    }

    // Instantiate the target
    Dict.Builder<String, Object> args = Dict.builder();
    args.put("name", name);
    args.putAll(attrValues);
    starlarkRuleFunction.call(thread, Tuple.of(), args.buildImmutable());
  }
}
