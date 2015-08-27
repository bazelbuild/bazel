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

import com.google.common.base.Function;
import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.RunfilesSupport;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.TargetUtils;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.syntax.BaseFunction;
import com.google.devtools.build.lib.syntax.ClassObject;
import com.google.devtools.build.lib.syntax.ClassObject.SkylarkClassObject;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.EvalExceptionWithStackTrace;
import com.google.devtools.build.lib.syntax.EvalUtils;
import com.google.devtools.build.lib.syntax.SkylarkEnvironment;
import com.google.devtools.build.lib.syntax.SkylarkNestedSet;
import com.google.devtools.build.lib.syntax.SkylarkType;

/**
 * A helper class to build Rule Configured Targets via runtime loaded rule implementations
 * defined using the Skylark Build Extension Language.
 */
public final class SkylarkRuleConfiguredTargetBuilder {

  /**
   * Create a Rule Configured Target from the ruleContext and the ruleImplementation.
   */
  public static ConfiguredTarget buildRule(RuleContext ruleContext,
      BaseFunction ruleImplementation) {
    String expectFailure = ruleContext.attributes().get("expect_failure", Type.STRING);
    try {
      SkylarkRuleContext skylarkRuleContext = new SkylarkRuleContext(ruleContext);
      SkylarkEnvironment env = ruleContext.getRule().getRuleClassObject()
          .getRuleDefinitionEnvironment().cloneEnv(
              ruleContext.getAnalysisEnvironment().getEventHandler());
      Object target = ruleImplementation.call(ImmutableList.<Object>of(skylarkRuleContext),
          ImmutableMap.<String, Object>of(), null, env);

      if (ruleContext.hasErrors()) {
        return null;
      } else if (!(target instanceof SkylarkClassObject) && target != Environment.NONE) {
        ruleContext.ruleError("Rule implementation doesn't return a struct");
        return null;
      } else if (!expectFailure.isEmpty()) {
        ruleContext.ruleError("Expected failure not found: " + expectFailure);
        return null;
      }
      ConfiguredTarget configuredTarget = createTarget(ruleContext, target);
      checkOrphanArtifacts(ruleContext);
      return configuredTarget;
    } catch (InterruptedException e) {
      ruleContext.ruleError(e.getMessage());
      return null;
    } catch (EvalException e) {
      addRuleToStackTrace(e, ruleContext.getRule());
      // If the error was expected, return an empty target.
      if (!expectFailure.isEmpty() && getMessageWithoutStackTrace(e).matches(expectFailure)) {
        return new com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder(ruleContext)
            .add(RunfilesProvider.class, RunfilesProvider.EMPTY)
            .build();
      }
      ruleContext.ruleError("\n" + e.print());
      return null;
    }
  }

  /**
   * Adds an entry for the given rule to the stack trace of the exception (if there is one).
   */
  private static void addRuleToStackTrace(EvalException ex, Rule rule) {
    if (ex instanceof EvalExceptionWithStackTrace) {
      ((EvalExceptionWithStackTrace) ex).registerRule(rule);
    }
  }

  /**
   * Returns the message of the given exception after removing the stack trace, if present.
   */
  private static String getMessageWithoutStackTrace(EvalException ex) {
    if (ex instanceof EvalExceptionWithStackTrace) {
      return ((EvalExceptionWithStackTrace) ex).getOriginalMessage();
    }
    return ex.getMessage();
  }

  private static void checkOrphanArtifacts(RuleContext ruleContext) throws EvalException {
    ImmutableSet<Artifact> orphanArtifacts =
        ruleContext.getAnalysisEnvironment().getOrphanArtifacts();
    if (!orphanArtifacts.isEmpty()) {
      throw new EvalException(null, "The following files have no generating action:\n"
          + Joiner.on("\n").join(Iterables.transform(orphanArtifacts,
        new Function<Artifact, String>() {
          @Override
          public String apply(Artifact artifact) {
            return artifact.getRootRelativePathString();
          }})));
    }
  }

  // TODO(bazel-team): this whole defaulting - overriding executable, runfiles and files_to_build
  // is getting out of hand. Clean this whole mess up.
  private static ConfiguredTarget createTarget(RuleContext ruleContext, Object target)
      throws EvalException {
    Artifact executable = getExecutable(ruleContext, target);
    RuleConfiguredTargetBuilder builder = new RuleConfiguredTargetBuilder(ruleContext);
    // Set the default files to build.
    NestedSetBuilder<Artifact> filesToBuild = NestedSetBuilder.<Artifact>stableOrder()
        .addAll(ruleContext.getOutputArtifacts());
    if (executable != null) {
      filesToBuild.add(executable);
    }
    builder.setFilesToBuild(filesToBuild.build());
    return addStructFields(ruleContext, builder, target, executable);
  }

  private static Artifact getExecutable(RuleContext ruleContext, Object target)
      throws EvalException {
    Artifact executable = ruleContext.getRule().getRuleClassObject().outputsDefaultExecutable()
        // This doesn't actually create a new Artifact just returns the one
        // created in SkylarkruleContext.
        ? ruleContext.createOutputArtifact() : null;
    if (target instanceof SkylarkClassObject) {
      SkylarkClassObject struct = (SkylarkClassObject) target;
      if (struct.getValue("executable") != null) {
        // We need this because of genrule.bzl. This overrides the default executable.
        executable = cast("executable", struct, Artifact.class, struct.getCreationLoc());
      }
    }
    return executable;
  }

  private static ConfiguredTarget addStructFields(RuleContext ruleContext,
      RuleConfiguredTargetBuilder builder, Object target, Artifact executable)
      throws EvalException {
    Location loc = null;
    Runfiles statelessRunfiles = null;
    Runfiles dataRunfiles = null;
    Runfiles defaultRunfiles = null;
    if (target instanceof SkylarkClassObject) {
      SkylarkClassObject struct = (SkylarkClassObject) target;
      loc = struct.getCreationLoc();
      for (String key : struct.getKeys()) {
        if (key.equals("files")) {
          // If we specify files_to_build we don't have the executable in it by default.
          builder.setFilesToBuild(cast("files", struct, SkylarkNestedSet.class, Artifact.class, loc)
              .getSet(Artifact.class));
        } else if (key.equals("runfiles")) {
          statelessRunfiles = cast("runfiles", struct, Runfiles.class, loc);
        } else if (key.equals("data_runfiles")) {
          dataRunfiles = cast("data_runfiles", struct, Runfiles.class, loc);
        } else if (key.equals("default_runfiles")) {
          defaultRunfiles = cast("default_runfiles", struct, Runfiles.class, loc);
        } else if (!key.equals("executable")) {
          // We handled executable already.
          builder.addSkylarkTransitiveInfo(key, struct.getValue(key), loc);
        }
      }
    }

    if ((statelessRunfiles != null) && (dataRunfiles != null || defaultRunfiles != null)) {
      throw new EvalException(loc, "Cannot specify the provider 'runfiles' "
          + "together with 'data_runfiles' or 'default_runfiles'");
    }

    if (statelessRunfiles == null && dataRunfiles == null && defaultRunfiles == null) {
      // No runfiles specified, set default
      statelessRunfiles = Runfiles.EMPTY;
    }

    RunfilesProvider runfilesProvider = statelessRunfiles != null
        ? RunfilesProvider.simple(merge(statelessRunfiles, executable, ruleContext))
        : RunfilesProvider.withData(
            // The executable doesn't get into the default runfiles if we have runfiles states.
            // This is to keep skylark genrule consistent with the original genrule.
            defaultRunfiles != null ? defaultRunfiles : Runfiles.EMPTY,
            dataRunfiles != null ? dataRunfiles : Runfiles.EMPTY);
    builder.addProvider(RunfilesProvider.class, runfilesProvider);

    Runfiles computedDefaultRunfiles = runfilesProvider.getDefaultRunfiles();
    // This works because we only allowed to call a rule *_test iff it's a test type rule.
    boolean testRule = TargetUtils.isTestRuleName(ruleContext.getRule().getRuleClass());
    if (testRule && computedDefaultRunfiles.isEmpty()) {
      throw new EvalException(loc, "Test rules have to define runfiles");
    }
    if (executable != null || testRule) {
      RunfilesSupport runfilesSupport = computedDefaultRunfiles.isEmpty()
          ? null : RunfilesSupport.withExecutable(ruleContext, computedDefaultRunfiles, executable);
      builder.setRunfilesSupport(runfilesSupport, executable);
    }
    try {
      return builder.build();
    } catch (IllegalArgumentException e) {
      throw new EvalException(loc, e.getMessage());
    }
  }

  private static <T> T cast(String paramName, ClassObject struct, Class<T> expectedGenericType,
      Class<?> expectedArgumentType, Location loc) throws EvalException {
    Object value = struct.getValue(paramName);
    return SkylarkType.cast(value, expectedGenericType, expectedArgumentType, loc,
        "expected %s for '%s' but got %s instead: %s",
        SkylarkType.of(expectedGenericType, expectedArgumentType),
        paramName, EvalUtils.getDataTypeName(value, true), value);
  }

  private static <T> T cast(String paramName, ClassObject struct, Class<T> expectedType,
      Location loc) throws EvalException {
    Object value = struct.getValue(paramName);
    return SkylarkType.cast(value, expectedType, loc,
        "expected %s for '%s' but got %s instead: %s",
        SkylarkType.of(expectedType),
        paramName, EvalUtils.getDataTypeName(value, false), value);
  }

  private static Runfiles merge(Runfiles runfiles, Artifact executable, RuleContext ruleContext) {
    if (executable == null) {
      return runfiles;
    }
    return new Runfiles.Builder(ruleContext.getWorkspaceName()).addArtifact(executable)
        .merge(runfiles).build();
  }
}
