// Copyright 2014 Google Inc. All rights reserved.
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
package com.google.devtools.build.lib.rules;

import static com.google.devtools.build.lib.syntax.SkylarkFunction.cast;

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.packages.TargetUtils;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.syntax.ClassObject.SkylarkClassObject;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.Function;
import com.google.devtools.build.lib.syntax.SkylarkEnvironment;
import com.google.devtools.build.lib.syntax.SkylarkNestedSet;
import com.google.devtools.build.lib.view.ConfiguredTarget;
import com.google.devtools.build.lib.view.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.view.RuleContext;
import com.google.devtools.build.lib.view.Runfiles;
import com.google.devtools.build.lib.view.RunfilesProvider;
import com.google.devtools.build.lib.view.RunfilesSupport;

/**
 * A helper class to build Rule Configured Targets via runtime loaded rule implementations
 * defined using the Skylark Build Extension Language. This is experimental code.
 */
public final class SkylarkRuleConfiguredTargetBuilder {

  /**
   * Create a Rule Configured Target from the ruleContext and the ruleImplementation.
   */
  public static ConfiguredTarget buildRule(RuleContext ruleContext,
      Function ruleImplementation) {
    String expectError = ruleContext.attributes().get("expect_failure", Type.STRING);
    try {
      SkylarkRuleContext skylarkRuleContext = new SkylarkRuleContext(ruleContext);
      SkylarkEnvironment env =
          ruleContext.getRule().getRuleClassObject().getRuleDefinitionEnvironment().cloneEnv();
      // Collect the symbols to disable statically and pass at the next call, so we don't need to
      // clone the RuleDefinitionEnvironment.
      env.disableOnlyLoadingPhaseObjects();
      Object target = ruleImplementation.call(ImmutableList.<Object>of(skylarkRuleContext),
          ImmutableMap.<String, Object>of(), null, env);

      if (ruleContext.hasErrors()) {
        return null;
      } else if (!(target instanceof SkylarkClassObject) && target != Environment.NONE) {
        ruleContext.ruleError("Rule implementation doesn't return a struct");
        return null;
      } else if (!expectError.isEmpty()) {
        ruleContext.ruleError("Expected error not found: " + expectError);
        return null;
      }
      ConfiguredTarget configuredTarget = createTarget(ruleContext, target);
      checkOrphanArtifacts(ruleContext);
      return configuredTarget;

    } catch (InterruptedException e) {
      ruleContext.ruleError(e.getMessage());
      return null;
    } catch (EvalException e) {
      // If the error was expected, return an empty target.
      if (!expectError.isEmpty() && e.getMessage().matches(expectError)) {
        return new com.google.devtools.build.lib.view.RuleConfiguredTargetBuilder(ruleContext)
            .add(RunfilesProvider.class, RunfilesProvider.EMPTY)
            .build();
      }
      ruleContext.ruleError("\n" + e.print());
      return null;
    }
  }

  private static void checkOrphanArtifacts(RuleContext ruleContext) throws EvalException {
    ImmutableSet<Artifact> orphanArtifacts =
        ruleContext.getAnalysisEnvironment().getOrphanArtifacts();
    if (!orphanArtifacts.isEmpty()) {
      throw new EvalException(null, "The following files have no generating action:\n"
          + Joiner.on("\n").join(Iterables.transform(orphanArtifacts,
          new com.google.common.base.Function<Artifact, String>() {
            @Override
            public String apply(Artifact artifact) {
              return artifact.getRootRelativePathString();
            }
          })));
    }
  }

  private static ConfiguredTarget createTarget(RuleContext ruleContext, Object target)
      throws EvalException {
    RuleConfiguredTargetBuilder builder = new RuleConfiguredTargetBuilder(ruleContext);
    // Every target needs runfiles provider by default.
    builder.add(RunfilesProvider.class, RunfilesProvider.EMPTY);
    builder.setFilesToBuild(
        NestedSetBuilder.<Artifact>wrap(Order.STABLE_ORDER, ruleContext.getOutputArtifacts()));
    Location loc = null;
    if (target instanceof SkylarkClassObject) {
      SkylarkClassObject struct = (SkylarkClassObject) target;
      loc = struct.getCreationLoc();
      addStructFields(ruleContext, builder, struct);
    }
    try {
      return builder.build();
    } catch (IllegalArgumentException e) {
      throw new EvalException(loc, e.getMessage());
    }
  }

  private static void addStructFields(RuleContext ruleContext, RuleConfiguredTargetBuilder builder,
      SkylarkClassObject struct) throws EvalException {
    Location loc = struct.getCreationLoc();
    Runfiles defaultRunfiles = Runfiles.EMPTY;
    Artifact executable = ruleContext.getRule().getRuleClassObject().outputsDefaultExecutable()
        // This doesn't actually create a new Artifact just returns the one
        // created in SkylarkruleContext.
        ? ruleContext.createOutputArtifact() : null;
    for (String key : struct.getKeys()) {
      if (key.equals("files_to_build")) {
        builder.setFilesToBuild(cast(struct.getValue("files_to_build"),
                SkylarkNestedSet.class, "files_to_build", loc).getSet(Artifact.class));
      } else if (key.equals("runfiles")) {
        RunfilesProvider runfilesProvider =
            cast(struct.getValue("runfiles"), RunfilesProvider.class, "runfiles", loc);
        builder.add(RunfilesProvider.class, runfilesProvider);
        defaultRunfiles = runfilesProvider.getDefaultRunfiles();
      } else if (key.equals("executable")) {
        // We need this because of genrule.bzl. This overrides the default executable.
        executable = cast(struct.getValue("executable"), Artifact.class, "executable", loc);
      } else {
        builder.addSkylarkTransitiveInfo(key, struct.getValue(key));
      }
    }
    // This works because we only allowed to call a rule *_test iff it's a test type rule.
    boolean testRule = TargetUtils.isTestRuleName(ruleContext.getRule().getRuleClass());
    if (testRule && defaultRunfiles.isEmpty()) {
      throw new EvalException(loc, "Test rules have to define runfiles");
    }
    if (executable != null || testRule) {
      RunfilesSupport runfilesSupport = defaultRunfiles.isEmpty()
          ? null : RunfilesSupport.withExecutable(ruleContext, defaultRunfiles, executable);
      builder.setRunfilesSupport(runfilesSupport, executable);
    }
  }
}
