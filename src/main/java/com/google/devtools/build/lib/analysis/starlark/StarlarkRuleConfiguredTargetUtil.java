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
package com.google.devtools.build.lib.analysis.starlark;

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.actions.ActionConflictException;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.ActionsProvider;
import com.google.devtools.build.lib.analysis.CachingAnalysisEnvironment;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.DefaultInfo;
import com.google.devtools.build.lib.analysis.RequiredConfigFragmentsProvider;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.RunEnvironmentInfo;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.RunfilesSupport;
import com.google.devtools.build.lib.analysis.StarlarkProviderValidationUtil;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.packages.AdvertisedProviderSet;
import com.google.devtools.build.lib.packages.Info;
import com.google.devtools.build.lib.packages.Provider;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.StarlarkInfo;
import com.google.devtools.build.lib.packages.StarlarkProviderIdentifier;
import com.google.devtools.build.lib.packages.StructProvider;
import com.google.devtools.build.lib.packages.TargetUtils;
import com.google.devtools.build.lib.packages.Type;
import com.google.errorprone.annotations.FormatMethod;
import java.util.LinkedHashMap;
import java.util.Map;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Sequence;
import net.starlark.java.eval.Starlark;
import net.starlark.java.syntax.Location;

/**
 * A helper class to build Rule Configured Targets via runtime loaded rule implementations defined
 * using the Starlark Build Extension Language.
 */
public final class StarlarkRuleConfiguredTargetUtil {

  private StarlarkRuleConfiguredTargetUtil() {}

  /**
   * Evaluates the rule's implementation function and returns what it returns (raw providers).
   *
   * <p>If there were errors during the evaluation or the type of the returned object is obviously
   * wrong, it sets ruleErrors on the ruleContext and returns null.
   *
   * <p>Unchecked exception {@code UncheckedEvalException}s and {@code MissingDepException} may be
   * thrown.
   *
   * @param ruleContext the rule context
   * @param ruleClass the rule class for which to evaluate the implementation function gets. This
   *     serves extended rules, where for parent's implementation function needs to be evaluated.
   */
  // TODO(blaze-team): Legacy providers are preventing to change the return type to Sequence<Info>.
  @Nullable
  public static Object evalRule(RuleContext ruleContext, RuleClass ruleClass)
      throws InterruptedException {
    // TODO(blaze-team): expect_failure attribute is special for all rule classes, but it should
    // be special only for analysis tests
    String expectFailure = ruleContext.attributes().get("expect_failure", Type.STRING);
    Object providersRaw;

    try {
      // call rule.implementation(ctx)
      providersRaw =
          Starlark.positionalOnlyCall(
              ruleContext.getStarlarkThread(),
              ruleClass.getConfiguredTargetFunction(),
              ruleContext.getStarlarkRuleContext());

    } catch (Starlark.UncheckedEvalException ex) {
      // MissingDepException is expected to transit through Starlark execution.
      throw ex.getCause() instanceof CachingAnalysisEnvironment.MissingDepException
          ? (CachingAnalysisEnvironment.MissingDepException) ex.getCause()
          : ex;

    } catch (EvalException ex) {
      // An error occurred during the rule.implementation call

      // If the error was expected by an analysis test, return None, to produce an empty target.
      if (!expectFailure.isEmpty() && ex.getMessage().matches(expectFailure)) {
        return Starlark.NONE;
      }

      // Emit a single event that spans multiple lines:
      //     ERROR p/BUILD:1:1: in foo_library rule //p:p:
      //     Traceback:
      //        File foo.bzl, line 1, in foo_library_impl:
      //        ...
      ruleContext.ruleError("\n" + ex.getMessageWithStack());
      return null;
    }

    // Errors already reported?
    if (ruleContext.hasErrors()) {
      return null;
    }

    // Wrong result type?
    if (!(providersRaw instanceof Info
        || providersRaw == Starlark.NONE
        || providersRaw instanceof Iterable)) {
      ruleContext.ruleError(
          String.format(
              "Rule should return a struct or a list, but got %s", Starlark.type(providersRaw)));
      return null;
    }

    // Did the Starlark implementation function fail to fail as expected?
    if (!expectFailure.isEmpty()) {
      ruleContext.ruleError("Expected failure not found: " + expectFailure);
      return null;
    }

    return providersRaw;
  }

  private static void checkDeclaredProviders(
      ConfiguredTarget configuredTarget, AdvertisedProviderSet advertisedProviders)
      throws EvalException {
    for (StarlarkProviderIdentifier providerId : advertisedProviders.getStarlarkProviders()) {
      if (configuredTarget.get(providerId) == null) {
        throw Starlark.errorf(
            "rule advertised the '%s' provider, but this provider was not among those returned",
            providerId);
      }
    }
  }

  /**
   * Creates a Rule Configured Target from the raw providers returned by the rule's implementation
   * function.
   *
   * <p>If there are problems with the raw providers, it sets ruleErrors on the ruleContext and
   * returns null.
   */
  @Nullable
  public static ConfiguredTarget createTarget(
      RuleContext context,
      Object rawProviders,
      AdvertisedProviderSet advertisedProviders,
      boolean isDefaultExecutableCreated,
      @Nullable RequiredConfigFragmentsProvider requiredConfigFragmentsProvider)
      throws InterruptedException, ActionConflictException {
    RuleConfiguredTargetBuilder builder = new RuleConfiguredTargetBuilder(context);

    // Location of rule.implementation function.
    Location implLoc =
        context.getRule().getRuleClassObject().getConfiguredTargetFunction().getLocation();

    // TODO(adonovan): clean up addProviders' error handling,
    // reporting provider validity errors through ruleError
    // where possible. This allows for multiple events, with independent
    // locations, even for the same root cause.
    // The required change is fiddly due to frequent and nested use of
    // Structure.getField, Sequence.cast, and similar operators.
    try {
      addProviders(context, builder, rawProviders, implLoc, isDefaultExecutableCreated);
    } catch (EvalException ex) {
      // Emit a single event that spans two lines (see infoError).
      // The message typically starts with another location, e.g. of provider creation.
      //     ERROR p/BUILD:1:1: in foo_library rule //p:p:
      //     ...message...
      context.ruleError("\n" + ex.getMessage());
      return null;
    }

    // This provider is kept out of `addProviders` method, because it's not generated by the
    // Starlark rule and because `addProviders` will be simplified by the legacy providers removal
    // RequiredConfigFragmentsProvider may be removed with removal of Android feature flags.
    if (requiredConfigFragmentsProvider != null) {
      builder.addProvider(requiredConfigFragmentsProvider);
    }

    ConfiguredTarget ct;
    try {
      // This also throws InterruptedException from a convoluted dependency:
      // TestActionBuilder -> TestTargetExecutionSettings -> CommandLine -> Starlark.
      ct = builder.build(); // may be null
    } catch (IllegalArgumentException ex) {
      // TODO(adonovan): eliminate this abuse of unchecked exceptions.
      // Emit a single event that spans two lines (see infoError).
      // The message typically starts with another location, e.g. of provider creation.
      //     ERROR p/BUILD:1:1: in foo_library rule //p:p:
      //     ...message...
      context.ruleError("\n" + implLoc + ": " + ex.getMessage());
      return null;
    }

    if (ct != null) {
      // If there was error creating the ConfiguredTarget, no further validation is needed.
      // Null will be returned and the errors thus reported.
      try {
        // Check all artifacts have actions. Despite signature, must be done after build().
        StarlarkProviderValidationUtil.validateArtifacts(context);
        // Check all advertised providers were created.
        checkDeclaredProviders(ct, advertisedProviders);
      } catch (EvalException ex) {
        context.ruleError("\n" + implLoc + ": " + ex.getMessage());
        return null;
      }
    }

    return ct;
  }

  public static NestedSet<Artifact> convertToOutputGroupValue(String outputGroup, Object objects)
      throws EvalException {
    // regrettable preemptive allocation of error message
    String what = "output group '" + outputGroup + "'";
    return objects instanceof Sequence
        ? NestedSetBuilder.<Artifact>stableOrder()
            .addAll(Sequence.cast(objects, Artifact.class, what))
            .build()
        : Depset.cast(objects, Artifact.class, what);
  }

  private static void addProviders(
      RuleContext context,
      RuleConfiguredTargetBuilder builder,
      Object rawProviders,
      Location implLoc,
      boolean isDefaultExecutableCreated)
      throws EvalException, InterruptedException {
    Map<Provider.Key, Info> declaredProviders = new LinkedHashMap<>();
    if (rawProviders instanceof Info info) {
      if (getProviderKey(info).equals(StructProvider.STRUCT.getKey())) {
        throw infoError(
            info, "Returning a struct from a rule implementation function is deprecated.");
      }

      // A single declared provider (not in a list)
      if (info instanceof StarlarkInfo starlarkInfo) {
        info = starlarkInfo.unsafeOptimizeMemoryLayout();
      }
      Provider.Key providerKey = getProviderKey(info);
      // Single declared provider
      declaredProviders.put(providerKey, info);
    } else if (rawProviders instanceof Sequence) {
      // Sequence of declared providers
      for (Info provider :
          Sequence.cast(rawProviders, Info.class, "result of rule implementation function")) {
        if (provider instanceof StarlarkInfo) {
          // Provider instances are optimised recursively, without optimising elements of the list.
          // Tradeoff is that some object may be duplicated if they are reachable by more than one
          // path, but we don't expect that much in practice.
          provider = ((StarlarkInfo) provider).unsafeOptimizeMemoryLayout();
        }
        Provider.Key providerKey = getProviderKey(provider);
        if (declaredProviders.put(providerKey, provider) != null) {
          context.ruleError("Multiple conflicting returned providers with key " + providerKey);
        }
      }
    } else if (rawProviders != Starlark.NONE) {
      throw Starlark.errorf(
          "Expected a list of providers, but got %s", Starlark.type(rawProviders));
    }

    boolean defaultProviderProvidedExplicitly = false;

    for (Info declaredProvider : declaredProviders.values()) {
      if (declaredProvider instanceof DefaultInfo defaultInfo) {
        parseDefaultProviderFields(defaultInfo, context, builder, isDefaultExecutableCreated);
        defaultProviderProvidedExplicitly = true;
      } else if (getProviderKey(declaredProvider).equals(RunEnvironmentInfo.PROVIDER.getKey())
          && !(context.getRule().getRuleClassObject().isExecutableStarlark()
              || context.isTestTarget())) {
        String message =
            "Returning RunEnvironmentInfo from a non-executable, non-test target has no effect";
        RunEnvironmentInfo runEnvironmentInfo = (RunEnvironmentInfo) declaredProvider;
        if (runEnvironmentInfo.shouldErrorOnNonExecutableRule()) {
          context.ruleError(message);
        } else {
          context.ruleWarning(message);
          builder.addStarlarkDeclaredProvider(declaredProvider);
        }
      } else {
        builder.addStarlarkDeclaredProvider(declaredProvider);
      }
    }

    if (!defaultProviderProvidedExplicitly) {
      // TODO(b/308767456): Avoid creating an empty DefaultInfo, just to pass location for throwing
      // exceptions.
      parseDefaultProviderFields(
          DefaultInfo.createEmpty(implLoc), context, builder, isDefaultExecutableCreated);
    }
  }

  // Returns an EvalException whose message has the info's creation location as a prefix.
  // The exception is intended to be reported as ruleErrors by createTarget.
  @FormatMethod
  private static EvalException infoError(Info info, String format, Object... args) {
    return Starlark.errorf("%s: %s", info.getCreationLocation(), String.format(format, args));
  }

  /**
   * Returns the provider key from an info (provider instance).
   *
   * @throws EvalException if the provider for this info object has not been exported, which can
   *     occur if the provider was declared in a non-global scope (for example a rule implementation
   *     function)
   */
  private static Provider.Key getProviderKey(Info info) throws EvalException {
    Provider provider = info.getProvider();
    if (!provider.isExported()) {
      // TODO(adonovan): report separate error events at distinct locations:
      //  "cannot return non-exported provider" (at location of instantiation), and
      //  "provider definition not at top level" (at location of definition).
      throw infoError(
          info,
          "The rule implementation function returned an instance of an unnamed provider. "
              + "A provider becomes named by being assigned to a global variable in a .bzl file. "
              + "(Provider defined at %s.)",
          provider.getLocation());
    }
    return provider.getKey();
  }

  /** Parses fields of a default provider. */
  private static void parseDefaultProviderFields(
      DefaultInfo defaultInfo,
      RuleContext context,
      RuleConfiguredTargetBuilder builder,
      boolean isDefaultExecutableCreated)
      throws EvalException, InterruptedException {
    Depset files = defaultInfo.getFiles();
    Runfiles statelessRunfiles = defaultInfo.getStatelessRunfiles();
    Runfiles dataRunfiles = defaultInfo.getDataRunfiles();
    Runfiles defaultRunfiles = defaultInfo.getDefaultRunfiles();
    Artifact executable = defaultInfo.getExecutable();

    if (executable != null && !executable.getArtifactOwner().equals(context.getOwner())) {
      throw infoError(
          defaultInfo,
          "'executable' provided by an executable rule '%s' should be created "
              + "by the same rule.",
          context.getRule().getRuleClass());
    }

    boolean isExecutable = context.getRule().getRuleClassObject().isExecutableStarlark();
    if (executable != null && isExecutable && isDefaultExecutableCreated) {
      Artifact defaultExecutable = context.createOutputArtifact();
      if (!executable.equals(defaultExecutable)) {
        throw infoError(
            defaultInfo,
            "The rule '%s' both accesses 'ctx.outputs.executable' and provides "
                + "a different executable '%s'. Do not use 'ctx.output.executable'.",
            context.getRule().getRuleClass(),
            executable.getRootRelativePathString());
      }
    }

    if (context.getRule().isAnalysisTest()) {
      // The Starlark Build API should already throw exception if the rule implementation attempts
      // to register any actions. This is just a check of this invariant.
      Preconditions.checkState(
          context.getAnalysisEnvironment().getRegisteredActions().isEmpty(),
          "%s",
          context.getLabel());

      executable = context.createOutputArtifactScript();
    }

    if (executable == null && isExecutable) {
      if (isDefaultExecutableCreated) {
        // This doesn't actually create a new Artifact just returns the one
        // created in StarlarkRuleContext.
        executable = context.createOutputArtifact();
      } else {
        throw infoError(
            defaultInfo,
            "The rule '%s' is executable. "
                + "It needs to create an executable File and pass it as the 'executable' "
                + "parameter to the DefaultInfo it returns.",
            context.getRule().getRuleClass());
      }
    }

    addSimpleProviders(
        builder, context, executable, files, statelessRunfiles, dataRunfiles, defaultRunfiles);
  }

  private static void addSimpleProviders(
      RuleConfiguredTargetBuilder builder,
      RuleContext ruleContext,
      Artifact executable,
      @Nullable Depset files,
      Runfiles statelessRunfiles,
      Runfiles dataRunfiles,
      Runfiles defaultRunfiles)
      throws EvalException, InterruptedException {

    // TODO(bazel-team) if both 'files' and 'executable' are provided, 'files' overrides
    // 'executable'
    NestedSetBuilder<Artifact> filesToBuild =
        NestedSetBuilder.<Artifact>stableOrder().addAll(ruleContext.getOutputArtifacts());
    if (executable != null) {
      filesToBuild.add(executable);
    }
    builder.setFilesToBuild(filesToBuild.build());

    if (files != null) {
      // If we specify files_to_build we don't have the executable in it by default.
      builder.setFilesToBuild(Depset.cast(files, Artifact.class, "files"));
    }

    if (statelessRunfiles == null && dataRunfiles == null && defaultRunfiles == null) {
      // No runfiles specified, set default
      statelessRunfiles = Runfiles.EMPTY;
    }

    // This works because we only allowed to call a rule *_test iff it's a test type rule.
    boolean testRule = TargetUtils.isTestRuleName(ruleContext.getRule().getRuleClass());
    boolean isExecutableOrTest = executable != null || testRule;
    RunfilesProvider runfilesProvider;
    if (statelessRunfiles != null) {
      runfilesProvider =
          RunfilesProvider.simple(mergeFiles(statelessRunfiles, executable, ruleContext));
    } else {
      var mergedDefaultRunfiles = defaultRunfiles != null ? defaultRunfiles : Runfiles.EMPTY;
      if (isExecutableOrTest) {
        // The executable is only merged in if needed when using stateful runfiles to preserve
        // long-standing behavior.
        mergedDefaultRunfiles = mergeFiles(mergedDefaultRunfiles, executable, ruleContext);
      }
      runfilesProvider =
          RunfilesProvider.withData(
              mergedDefaultRunfiles, dataRunfiles != null ? dataRunfiles : Runfiles.EMPTY);
    }
    builder.addProvider(RunfilesProvider.class, runfilesProvider);

    Runfiles computedDefaultRunfiles = runfilesProvider.getDefaultRunfiles();
    if (testRule && computedDefaultRunfiles.isEmpty()) {
      throw Starlark.errorf("Test rules have to define runfiles");
    }
    if (isExecutableOrTest) {
      RunfilesSupport runfilesSupport = null;
      if (!computedDefaultRunfiles.isEmpty()) {
        Preconditions.checkNotNull(executable, "executable must not be null");
        runfilesSupport =
            RunfilesSupport.withExecutable(ruleContext, computedDefaultRunfiles, executable);
      }
      builder.setRunfilesSupport(runfilesSupport, executable);
    }

    if (ruleContext.getRule().getRuleClassObject().isStarlarkTestable()) {
      Info actions =
          ActionsProvider.create(ruleContext.getAnalysisEnvironment().getRegisteredActions());
      builder.addStarlarkDeclaredProvider(actions);
    }
  }

  private static Runfiles mergeFiles(
      Runfiles runfiles, Artifact executable, RuleContext ruleContext) {
    if (executable == null) {
      return runfiles;
    }
    return new Runfiles.Builder(ruleContext.getWorkspaceName())
        .addArtifact(executable)
        .merge(runfiles)
        .build();
  }
}
