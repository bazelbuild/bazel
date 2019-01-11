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
package com.google.devtools.build.lib.analysis.skylark;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.analysis.ActionsProvider;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.DefaultInfo;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.RunfilesSupport;
import com.google.devtools.build.lib.analysis.SkylarkProviderValidationUtil;
import com.google.devtools.build.lib.analysis.Whitelist;
import com.google.devtools.build.lib.analysis.test.InstrumentedFilesCollector;
import com.google.devtools.build.lib.analysis.test.InstrumentedFilesCollector.InstrumentationSpec;
import com.google.devtools.build.lib.analysis.test.InstrumentedFilesInfo;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.packages.AdvertisedProviderSet;
import com.google.devtools.build.lib.packages.FunctionSplitTransitionWhitelist;
import com.google.devtools.build.lib.packages.InfoInterface;
import com.google.devtools.build.lib.packages.NativeProvider;
import com.google.devtools.build.lib.packages.Provider;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.packages.SkylarkProviderIdentifier;
import com.google.devtools.build.lib.packages.StructImpl;
import com.google.devtools.build.lib.packages.StructProvider;
import com.google.devtools.build.lib.packages.TargetUtils;
import com.google.devtools.build.lib.skylarkinterface.SkylarkValue;
import com.google.devtools.build.lib.syntax.BaseFunction;
import com.google.devtools.build.lib.syntax.ClassObject;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.EvalExceptionWithStackTrace;
import com.google.devtools.build.lib.syntax.EvalUtils;
import com.google.devtools.build.lib.syntax.Mutability;
import com.google.devtools.build.lib.syntax.Runtime;
import com.google.devtools.build.lib.syntax.SkylarkList;
import com.google.devtools.build.lib.syntax.SkylarkNestedSet;
import com.google.devtools.build.lib.syntax.SkylarkSemantics;
import com.google.devtools.build.lib.syntax.SkylarkType;
import com.google.devtools.build.lib.syntax.Type;
import com.google.devtools.build.lib.util.FileType;
import com.google.devtools.build.lib.util.FileTypeSet;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * A helper class to build Rule Configured Targets via runtime loaded rule implementations defined
 * using the Skylark Build Extension Language.
 */
public final class SkylarkRuleConfiguredTargetUtil {

  private SkylarkRuleConfiguredTargetUtil() {}

  private static final ImmutableSet<String> DEFAULT_PROVIDER_FIELDS =
      ImmutableSet.of("files", "runfiles", "data_runfiles", "default_runfiles", "executable");

  /**
   * Create a Rule Configured Target from the ruleContext and the ruleImplementation. Returns null
   * if there were errors during target creation.
   */
  @Nullable
  public static ConfiguredTarget buildRule(
      RuleContext ruleContext,
      AdvertisedProviderSet advertisedProviders,
      BaseFunction ruleImplementation,
      Location location,
      SkylarkSemantics skylarkSemantics,
      String toolsRepository)
      throws InterruptedException, RuleErrorException, ActionConflictException {
    String expectFailure = ruleContext.attributes().get("expect_failure", Type.STRING);
    SkylarkRuleContext skylarkRuleContext = null;
    try (Mutability mutability = Mutability.create("configured target")) {
      skylarkRuleContext = new SkylarkRuleContext(ruleContext, null, skylarkSemantics);
      Environment env =
          Environment.builder(mutability)
              .setCallerLabel(ruleContext.getLabel())
              .setSemantics(skylarkSemantics)
              .setEventHandler(ruleContext.getAnalysisEnvironment().getEventHandler())
              .setStarlarkContext(
                  new BazelStarlarkContext(
                      toolsRepository, ruleContext.getTarget().getPackage().getRepositoryMapping()))
              .build(); // NB: loading phase functions are not available: this is analysis already,
      // so we do *not* setLoadingPhase().

      RuleClass ruleClass = ruleContext.getRule().getRuleClassObject();
      if (ruleClass.getRuleClassType().equals(RuleClass.Builder.RuleClassType.WORKSPACE)) {
        ruleContext.ruleError(
            "Found reference to a workspace rule in a context where a build"
                + " rule was expected; probably a reference to a target in that external"
                + " repository, properly specified as @reponame//path/to/package:target,"
                + " should have been specified by the requesting rule.");
        return null;
      }
      if (ruleClass.hasFunctionTransitionWhitelist()
          && !Whitelist.isAvailable(ruleContext, FunctionSplitTransitionWhitelist.WHITELIST_NAME)) {
          ruleContext.ruleError("Non-whitelisted use of function-base split transition");
      }

      Object target =
          ruleImplementation.call(
              /*args=*/ ImmutableList.of(skylarkRuleContext),
              /*kwargs*/ ImmutableMap.of(),
              /*ast=*/ null,
              env);

      if (ruleContext.hasErrors()) {
        return null;
      } else if (!(target instanceof InfoInterface)
          && target != Runtime.NONE
          && !(target instanceof Iterable)) {
        ruleContext.ruleError(
            String.format(
                "Rule should return a struct or a list, but got %s", SkylarkType.typeOf(target)));
        return null;
      } else if (!expectFailure.isEmpty()) {
        ruleContext.ruleError("Expected failure not found: " + expectFailure);
        return null;
      }
      ConfiguredTarget configuredTarget = createTarget(skylarkRuleContext, target);
      if (configuredTarget != null) {
        // If there was error creating the ConfiguredTarget, no further validation is needed.
        // Null will be returned and the errors thus reported.
        SkylarkProviderValidationUtil.validateArtifacts(ruleContext);
        checkDeclaredProviders(configuredTarget, advertisedProviders, location);
      }
      return configuredTarget;
    } catch (EvalException e) {
      addRuleToStackTrace(e, ruleContext.getRule(), ruleImplementation);
      // If the error was expected, return an empty target.
      if (!expectFailure.isEmpty() && getMessageWithoutStackTrace(e).matches(expectFailure)) {
        return new RuleConfiguredTargetBuilder(ruleContext)
            .add(RunfilesProvider.class, RunfilesProvider.EMPTY)
            .build();
      }
      ruleContext.ruleError("\n" + e.print());
      return null;
    } finally {
      if (skylarkRuleContext != null) {
        skylarkRuleContext.nullify();
      }
    }
  }

  private static void checkDeclaredProviders(
      ConfiguredTarget configuredTarget, AdvertisedProviderSet advertisedProviders, Location loc)
      throws EvalException {
    for (SkylarkProviderIdentifier providerId : advertisedProviders.getSkylarkProviders()) {
      if (configuredTarget.get(providerId) == null) {
        throw new EvalException(
            loc,
            String.format(
                "rule advertised the '%s' provider, but this provider was not among those returned",
                providerId.toString()));
      }
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

  @Nullable
  private static ConfiguredTarget createTarget(SkylarkRuleContext context, Object target)
      throws EvalException, RuleErrorException, ActionConflictException {
    RuleConfiguredTargetBuilder builder = new RuleConfiguredTargetBuilder(
        context.getRuleContext());
    // Set the default files to build.

    Location loc =
        context.getRuleContext()
            .getRule()
            .getRuleClassObject()
            .getConfiguredTargetFunction()
            .getLocation();
    addProviders(context, builder, target, loc);

    try {
      return builder.build();
    } catch (IllegalArgumentException e) {
      throw new EvalException(loc, e.getMessage());
    }
  }

  private static void addOutputGroups(Object value, Location loc,
      RuleConfiguredTargetBuilder builder)
      throws EvalException {
    Map<String, SkylarkValue> outputGroups =
        SkylarkType.castMap(value, String.class, SkylarkValue.class, "output_groups");

    for (String outputGroup : outputGroups.keySet()) {
      SkylarkValue objects = outputGroups.get(outputGroup);
      NestedSet<Artifact> artifacts = convertToOutputGroupValue(loc, outputGroup, objects);
      builder.addOutputGroup(outputGroup, artifacts);
    }
  }

  private static void addInstrumentedFiles(
      StructImpl insStruct, RuleContext ruleContext, RuleConfiguredTargetBuilder builder)
      throws EvalException {
    Location insLoc = insStruct.getCreationLoc();
    FileTypeSet fileTypeSet = FileTypeSet.ANY_FILE;
    if (insStruct.getFieldNames().contains("extensions")) {
      @SuppressWarnings("unchecked")
      List<String> exts = cast("extensions", insStruct, SkylarkList.class, String.class, insLoc);
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
    if (insStruct.getFieldNames().contains("dependency_attributes")) {
      dependencyAttributes =
          cast("dependency_attributes", insStruct, SkylarkList.class, String.class, insLoc);
    }
    List<String> sourceAttributes = Collections.emptyList();
    if (insStruct.getFieldNames().contains("source_attributes")) {
      sourceAttributes =
          cast("source_attributes", insStruct, SkylarkList.class, String.class, insLoc);
    }
    InstrumentationSpec instrumentationSpec =
        new InstrumentationSpec(fileTypeSet)
            .withSourceAttributes(sourceAttributes.toArray(new String[0]))
            .withDependencyAttributes(dependencyAttributes.toArray(new String[0]));
    InstrumentedFilesInfo instrumentedFilesProvider =
        InstrumentedFilesCollector.collect(
            ruleContext,
            instrumentationSpec,
            InstrumentedFilesCollector.NO_METADATA_COLLECTOR,
            /* rootFiles= */ Collections.emptySet(),
            /* reportedToActualSources= */ NestedSetBuilder.create(Order.STABLE_ORDER));
    builder.addNativeDeclaredProvider(instrumentedFilesProvider);
  }

  public static NestedSet<Artifact> convertToOutputGroupValue(Location loc, String outputGroup,
      Object objects) throws EvalException {
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
    return artifacts;
  }

  private static void addProviders(
      SkylarkRuleContext context, RuleConfiguredTargetBuilder builder, Object target, Location loc)
      throws EvalException {

    StructImpl oldStyleProviders = StructProvider.STRUCT.createEmpty(loc);
    Map<Provider.Key, InfoInterface> declaredProviders = new LinkedHashMap<>();

    if (target instanceof InfoInterface) {
      // Either an old-style struct or a single declared provider (not in a list)
      InfoInterface info = (InfoInterface) target;
      // Use the creation location of this struct as a better reference in error messages
      loc = info.getCreationLoc();
      if (info.getProvider().getKey().equals(StructProvider.STRUCT.getKey())) {
        // Old-style struct, but it may contain declared providers
        StructImpl struct = (StructImpl) target;
        oldStyleProviders = struct;

        if (struct.hasField("providers")) {
          Iterable iterable = cast("providers", struct, Iterable.class, loc);
          for (Object o : iterable) {
            InfoInterface declaredProvider =
                SkylarkType.cast(
                    o,
                    InfoInterface.class,
                    loc,
                    "The value of 'providers' should be a sequence of declared providers");
            Provider.Key providerKey = declaredProvider.getProvider().getKey();
            if (declaredProviders.put(providerKey, declaredProvider) != null) {
              context
                  .getRuleContext()
                  .ruleError("Multiple conflicting returned providers with key " + providerKey);
            }
          }
        }
      } else {
        Provider.Key providerKey = info.getProvider().getKey();
        // Single declared provider
        declaredProviders.put(providerKey, info);
      }
    } else if (target instanceof Iterable) {
      // Sequence of declared providers
      for (Object o : (Iterable) target) {
        InfoInterface declaredProvider =
            SkylarkType.cast(
                o,
                InfoInterface.class,
                loc,
                "A return value of a rule implementation function should be "
                    + "a sequence of declared providers");
        Provider.Key providerKey = declaredProvider.getProvider().getKey();
        if (declaredProviders.put(providerKey, declaredProvider)  != null) {
          context
              .getRuleContext()
              .ruleError("Multiple conflicting returned providers with key " + providerKey);
        }
      }
    }

    boolean defaultProviderProvidedExplicitly = false;

    for (InfoInterface declaredProvider : declaredProviders.values()) {
      if (declaredProvider
          .getProvider()
          .getKey()
          .equals(DefaultInfo.PROVIDER.getKey())) {
        parseDefaultProviderFields((DefaultInfo) declaredProvider, context, builder);
        defaultProviderProvidedExplicitly = true;
      } else {
        builder.addSkylarkDeclaredProvider(declaredProvider);
      }
    }

    if (!defaultProviderProvidedExplicitly) {
      parseDefaultProviderFields(oldStyleProviders, context, builder);
    }

    for (String field : oldStyleProviders.getFieldNames()) {
      if (DEFAULT_PROVIDER_FIELDS.contains(field)) {
        // These fields have already been parsed above.
        // If a default provider has been provided explicitly then it's an error that they also
        // occur here.
        if (defaultProviderProvidedExplicitly) {
          throw new EvalException(
              loc,
              "Provider '"
                  + field
                  + "' should be specified in DefaultInfo if it's provided explicitly.");
        }
      } else if (field.equals("output_groups")) {
        addOutputGroups(oldStyleProviders.getValue(field), loc, builder);
      } else if (field.equals("instrumented_files")) {
        StructImpl insStruct = cast("instrumented_files", oldStyleProviders, StructImpl.class, loc);
        addInstrumentedFiles(insStruct, context.getRuleContext(), builder);
      } else if (isNativeDeclaredProviderWithLegacySkylarkName(oldStyleProviders.getValue(field))) {
        builder.addNativeDeclaredProvider((InfoInterface) oldStyleProviders.getValue(field));
      } else if (!field.equals("providers")) {
        // We handled providers already.
        builder.addSkylarkTransitiveInfo(field, oldStyleProviders.getValue(field), loc);
      }
    }
  }

  private static boolean isNativeDeclaredProviderWithLegacySkylarkName(Object value) {
    if (!(value instanceof InfoInterface)) {
      return false;
    }
    return ((InfoInterface) value).getProvider() instanceof NativeProvider.WithLegacySkylarkName;
  }

  /**
   * Parses fields of (not necessarily a default) provider. If it is an actual default provider,
   * throws an {@link EvalException} if there are unknown fields.
   */
  private static void parseDefaultProviderFields(
      StructImpl provider, SkylarkRuleContext context, RuleConfiguredTargetBuilder builder)
      throws EvalException {
    SkylarkNestedSet files = null;
    Runfiles statelessRunfiles = null;
    Runfiles dataRunfiles = null;
    Runfiles defaultRunfiles = null;
    Artifact executable = null;

    Location loc = provider.getCreationLoc();

    if (provider
        .getProvider()
        .getKey()
        .equals(DefaultInfo.PROVIDER.getKey())) {
      DefaultInfo defaultInfo = (DefaultInfo) provider;

      files = defaultInfo.getFiles();
      statelessRunfiles = defaultInfo.getStatelessRunfiles();
      dataRunfiles = defaultInfo.getDataRunfiles();
      defaultRunfiles = defaultInfo.getDefaultRunfiles();
      executable = defaultInfo.getExecutable();

    } else {
      // Rule implementations aren't reqiured to return default-info fields via a DefaultInfo
      // provider. They can return them as fields on the returned struct. For example,
      // 'return struct(executable = foo)' instead of 'return DefaultInfo(executable = foo)'.
      // TODO(cparsons): Look into deprecating this option.
      for (String field : provider.getFieldNames()) {
        if (field.equals("files")) {
          files = cast("files", provider, SkylarkNestedSet.class, Artifact.class, loc);
        } else if (field.equals("runfiles")) {
          statelessRunfiles = cast("runfiles", provider, Runfiles.class, loc);
        } else if (field.equals("data_runfiles")) {
          dataRunfiles = cast("data_runfiles", provider, Runfiles.class, loc);
        } else if (field.equals("default_runfiles")) {
          defaultRunfiles = cast("default_runfiles", provider, Runfiles.class, loc);
        } else if (field.equals("executable") && provider.getValue("executable") != null) {
          executable = cast("executable", provider, Artifact.class, loc);
        }
      }

      if ((statelessRunfiles != null) && (dataRunfiles != null || defaultRunfiles != null)) {
        throw new EvalException(loc, "Cannot specify the provider 'runfiles' "
            + "together with 'data_runfiles' or 'default_runfiles'");
      }
    }

    if (executable != null
        && !executable.getArtifactOwner().equals(context.getRuleContext().getOwner())) {
      throw new EvalException(
          loc,
          String.format(
              "'executable' provided by an executable rule '%s' should be created "
                  + "by the same rule.",
              context.getRuleContext().getRule().getRuleClass()));
    }

    if (executable != null && context.isExecutable() && context.isDefaultExecutableCreated()) {
        Artifact defaultExecutable = context.getRuleContext().createOutputArtifact();
        if (!executable.equals(defaultExecutable)) {
          throw new EvalException(loc,
              String.format(
                  "The rule '%s' both accesses 'ctx.outputs.executable' and provides "
                      + "a different executable '%s'. Do not use 'ctx.output.executable'.",
                  context.getRuleContext().getRule().getRuleClass(),
                  executable.getRootRelativePathString())
          );
        }
    }

    if (context.getRuleContext().getRule().isAnalysisTest()) {
      // The Starlark Build API should already throw exception if the rule implementation attempts
      // to register any actions. This is just a sanity check of this invariant.
      Preconditions.checkState(
          context.getRuleContext().getAnalysisEnvironment().getRegisteredActions().isEmpty(),
          "%s", context.getRuleContext().getLabel());

      executable = context.getRuleContext().createOutputArtifactScript();
    }

    if (executable == null && context.isExecutable()) {
      if (context.isDefaultExecutableCreated()) {
        // This doesn't actually create a new Artifact just returns the one
        // created in SkylarkRuleContext.
        executable = context.getRuleContext().createOutputArtifact();
      } else {
        throw new EvalException(loc,
            String.format("The rule '%s' is executable. "
                    + "It needs to create an executable File and pass it as the 'executable' "
                    + "parameter to the DefaultInfo it returns.",
                context.getRuleContext().getRule().getRuleClass()));
      }
    }

    addSimpleProviders(
        builder,
        context.getRuleContext(),
        loc,
        executable,
        files,
        statelessRunfiles,
        dataRunfiles,
        defaultRunfiles);
  }

  private static void addSimpleProviders(
      RuleConfiguredTargetBuilder builder,
      RuleContext ruleContext,
      Location loc,
      Artifact executable,
      @Nullable SkylarkNestedSet files,
      Runfiles statelessRunfiles,
      Runfiles dataRunfiles,
      Runfiles defaultRunfiles)
      throws EvalException {

    // TODO(bazel-team) if both 'files' and 'executable' are provided 'files' override 'executalbe'
    NestedSetBuilder<Artifact> filesToBuild =
        NestedSetBuilder.<Artifact>stableOrder().addAll(ruleContext.getOutputArtifacts());
    if (executable != null) {
      filesToBuild.add(executable);
    }
    builder.setFilesToBuild(filesToBuild.build());

    if (files != null) {
      // If we specify files_to_build we don't have the executable in it by default.
      builder.setFilesToBuild(files.getSet(Artifact.class));
    }

    if (statelessRunfiles == null && dataRunfiles == null && defaultRunfiles == null) {
      // No runfiles specified, set default
      statelessRunfiles = Runfiles.EMPTY;
    }

    RunfilesProvider runfilesProvider =
        statelessRunfiles != null
            ? RunfilesProvider.simple(mergeFiles(statelessRunfiles, executable, ruleContext))
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

    if (ruleContext.getRule().getRuleClassObject().isSkylarkTestable()) {
      InfoInterface actions =
          ActionsProvider.create(ruleContext.getAnalysisEnvironment().getRegisteredActions());
      builder.addSkylarkDeclaredProvider(actions);
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

  private static Runfiles mergeFiles(
      Runfiles runfiles, Artifact executable, RuleContext ruleContext) {
    if (executable == null) {
      return runfiles;
    }
    return new Runfiles.Builder(
        ruleContext.getWorkspaceName(), ruleContext.getConfiguration().legacyExternalRunfiles())
        .addArtifact(executable)
        .merge(runfiles).build();
  }
}
