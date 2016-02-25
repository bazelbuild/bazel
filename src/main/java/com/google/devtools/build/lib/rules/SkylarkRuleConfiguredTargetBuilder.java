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
package com.google.devtools.build.lib.rules;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.RunfilesSupport;
import com.google.devtools.build.lib.analysis.SkylarkProviderValidationUtil;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.TargetUtils;
import com.google.devtools.build.lib.rules.SkylarkRuleContext.Kind;
import com.google.devtools.build.lib.rules.test.InstrumentedFilesCollector;
import com.google.devtools.build.lib.rules.test.InstrumentedFilesCollector.InstrumentationSpec;
import com.google.devtools.build.lib.rules.test.InstrumentedFilesProvider;
import com.google.devtools.build.lib.skylarkinterface.SkylarkValue;
import com.google.devtools.build.lib.syntax.BaseFunction;
import com.google.devtools.build.lib.syntax.ClassObject;
import com.google.devtools.build.lib.syntax.ClassObject.SkylarkClassObject;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.EvalExceptionWithStackTrace;
import com.google.devtools.build.lib.syntax.EvalUtils;
import com.google.devtools.build.lib.syntax.Mutability;
import com.google.devtools.build.lib.syntax.Runtime;
import com.google.devtools.build.lib.syntax.SkylarkList;
import com.google.devtools.build.lib.syntax.SkylarkNestedSet;
import com.google.devtools.build.lib.syntax.SkylarkType;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.util.FileType;
import com.google.devtools.build.lib.util.FileTypeSet;

import java.util.Collections;
import java.util.List;
import java.util.Map;

/**
 * A helper class to build Rule Configured Targets via runtime loaded rule implementations
 * defined using the Skylark Build Extension Language.
 */
public final class SkylarkRuleConfiguredTargetBuilder {

  /**
   * Create a Rule Configured Target from the ruleContext and the ruleImplementation.
   */
  public static ConfiguredTarget buildRule(RuleContext ruleContext, BaseFunction ruleImplementation)
      throws InterruptedException {
    String expectFailure = ruleContext.attributes().get("expect_failure", Type.STRING);
    try (Mutability mutability = Mutability.create("configured target")) {
      SkylarkRuleContext skylarkRuleContext = new SkylarkRuleContext(ruleContext, Kind.RULE);
      Environment env = Environment.builder(mutability)
          .setSkylark()
          .setGlobals(
              ruleContext.getRule().getRuleClassObject().getRuleDefinitionEnvironment().getGlobals())
          .setEventHandler(ruleContext.getAnalysisEnvironment().getEventHandler())
          .build(); // NB: loading phase functions are not available: this is analysis already,
                    // so we do *not* setLoadingPhase().
      Object target = ruleImplementation.call(
          ImmutableList.<Object>of(skylarkRuleContext),
          ImmutableMap.<String, Object>of(),
          /*ast=*/null,
          env);

      if (ruleContext.hasErrors()) {
        return null;
      } else if (!(target instanceof SkylarkClassObject) && target != Runtime.NONE) {
        ruleContext.ruleError("Rule implementation doesn't return a struct");
        return null;
      } else if (!expectFailure.isEmpty()) {
        ruleContext.ruleError("Expected failure not found: " + expectFailure);
        return null;
      }
      ConfiguredTarget configuredTarget = createTarget(ruleContext, target);
      SkylarkProviderValidationUtil.checkOrphanArtifacts(ruleContext);
      return configuredTarget;
    } catch (EvalException e) {
      addRuleToStackTrace(e, ruleContext.getRule(), ruleImplementation);
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
   * Adds the given rule to the stack trace of the exception (if there is one).
   */
  private static void addRuleToStackTrace(EvalException ex, Rule rule, BaseFunction ruleImpl) {
    if (ex instanceof EvalExceptionWithStackTrace) {
      ((EvalExceptionWithStackTrace) ex)
          .registerPhantomFuncall(
              String.format("%s(name = '%s')", rule.getRuleClass(), rule.getName()),
              rule.getLocation(),
              ruleImpl);
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

  private static void addOutputGroups(Object value, Location loc,
      RuleConfiguredTargetBuilder builder)
      throws EvalException {
    Map<String, SkylarkValue> outputGroups =
        SkylarkType.castMap(value, String.class, SkylarkValue.class, "output_groups");

    for (String outputGroup : outputGroups.keySet()) {
      SkylarkValue objects = outputGroups.get(outputGroup);
      NestedSet<Artifact> artifacts;

      String typeErrorMessage =
          "Output group '%s' is of unexpected type. "
              + "Should be list or set of Files, but got '%s' instead.";

      if (objects instanceof SkylarkList) {
        NestedSetBuilder<Artifact> nestedSetBuilder = NestedSetBuilder.stableOrder();
        for (Object o : (SkylarkList) objects) {
          if (o instanceof Artifact) {
            nestedSetBuilder.add((Artifact) o);
          } else {
            throw new EvalException(
                loc,
                String.format(
                    typeErrorMessage,
                    outputGroup,
                    "list with an element of " + EvalUtils.getDataTypeNameFromClass(o.getClass())));
          }
        }
        artifacts = nestedSetBuilder.build();
      } else {
        artifacts =
            SkylarkType.cast(
                    objects,
                    SkylarkNestedSet.class,
                    Artifact.class,
                    loc,
                    typeErrorMessage,
                    outputGroup,
                    EvalUtils.getDataTypeName(objects, true))
                .getSet(Artifact.class);
      }
      builder.addOutputGroup(outputGroup, artifacts);
    }
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
        } else if (key.equals("output_groups")) {
          addOutputGroups(struct.getValue(key), loc, builder);
        } else if (key.equals("instrumented_files")) {
          SkylarkClassObject insStruct =
              cast("instrumented_files", struct, SkylarkClassObject.class, loc);
          Location insLoc = insStruct.getCreationLoc();
          FileTypeSet fileTypeSet = FileTypeSet.ANY_FILE;
          if (insStruct.getKeys().contains("extensions")) {
            List<String> exts = cast(
                "extensions", insStruct, SkylarkList.class, String.class, insLoc);
            if (exts.isEmpty()) {
              fileTypeSet = FileTypeSet.NO_FILE;
            } else {
              FileType[] fileTypes = new FileType[exts.size()];
              for (int i = 0; i < fileTypes.length; i++) {
                fileTypes[i] = FileType.of(exts.get(i));
              }
              fileTypeSet = FileTypeSet.of(fileTypes);
            }
          }
          List<String> dependencyAttributes = Collections.emptyList();
          if (insStruct.getKeys().contains("dependency_attributes")) {
            dependencyAttributes =
                cast("dependency_attributes", insStruct, SkylarkList.class, String.class, insLoc);
          }
          List<String> sourceAttributes = Collections.emptyList();
          if (insStruct.getKeys().contains("source_attributes")) {
            sourceAttributes =
                cast("source_attributes", insStruct, SkylarkList.class, String.class, insLoc);
          }
          InstrumentationSpec instrumentationSpec =
              new InstrumentationSpec(fileTypeSet)
                  .withSourceAttributes(sourceAttributes.toArray(new String[0]))
                  .withDependencyAttributes(dependencyAttributes.toArray(new String[0]));
          InstrumentedFilesProvider instrumentedFilesProvider =
              InstrumentedFilesCollector.collect(
                  ruleContext,
                  instrumentationSpec,
                  InstrumentedFilesCollector.NO_METADATA_COLLECTOR,
                  Collections.<Artifact>emptySet());
          builder.addProvider(InstrumentedFilesProvider.class, instrumentedFilesProvider);
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
