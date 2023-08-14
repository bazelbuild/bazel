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

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkState;
import static com.google.devtools.build.lib.analysis.BaseRuleClasses.RUN_UNDER;
import static com.google.devtools.build.lib.analysis.BaseRuleClasses.TIMEOUT_DEFAULT;
import static com.google.devtools.build.lib.analysis.BaseRuleClasses.getTestRuntimeLabelList;
import static com.google.devtools.build.lib.analysis.test.ExecutionInfo.DEFAULT_TEST_RUNNER_EXEC_GROUP;
import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.BuildType.LABEL;
import static com.google.devtools.build.lib.packages.BuildType.LABEL_LIST;
import static com.google.devtools.build.lib.packages.BuildType.LICENSE;
import static com.google.devtools.build.lib.packages.Type.BOOLEAN;
import static com.google.devtools.build.lib.packages.Type.INTEGER;
import static com.google.devtools.build.lib.packages.Type.STRING;
import static com.google.devtools.build.lib.packages.Type.STRING_LIST;

import com.github.benmanes.caffeine.cache.Caffeine;
import com.github.benmanes.caffeine.cache.LoadingCache;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.analysis.Allowlist;
import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.analysis.TemplateVariableInfo;
import com.google.devtools.build.lib.analysis.config.ExecutionTransitionFactory;
import com.google.devtools.build.lib.analysis.config.StarlarkDefinedConfigTransition;
import com.google.devtools.build.lib.analysis.config.ToolchainTypeRequirement;
import com.google.devtools.build.lib.analysis.config.transitions.PatchTransition;
import com.google.devtools.build.lib.analysis.config.transitions.StarlarkExposedRuleTransitionFactory;
import com.google.devtools.build.lib.analysis.config.transitions.TransitionFactory;
import com.google.devtools.build.lib.analysis.config.transitions.TransitionFactory.TransitionType;
import com.google.devtools.build.lib.analysis.platform.ConstraintValueInfo;
import com.google.devtools.build.lib.analysis.starlark.StarlarkAttrModule.Descriptor;
import com.google.devtools.build.lib.analysis.test.TestConfiguration;
import com.google.devtools.build.lib.cmdline.BazelModuleContext;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.packages.AllowlistChecker;
import com.google.devtools.build.lib.packages.AspectsListBuilder.AspectDetails;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.Attribute.StarlarkComputedDefaultTemplate;
import com.google.devtools.build.lib.packages.AttributeValueSource;
import com.google.devtools.build.lib.packages.BazelStarlarkContext;
import com.google.devtools.build.lib.packages.BuildSetting;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.BzlInitThreadContext;
import com.google.devtools.build.lib.packages.ConfigurationFragmentPolicy.MissingFragmentPolicy;
import com.google.devtools.build.lib.packages.ExecGroup;
import com.google.devtools.build.lib.packages.FunctionSplitTransitionAllowlist;
import com.google.devtools.build.lib.packages.ImplicitOutputsFunction.StarlarkImplicitOutputsFunctionWithCallback;
import com.google.devtools.build.lib.packages.ImplicitOutputsFunction.StarlarkImplicitOutputsFunctionWithMap;
import com.google.devtools.build.lib.packages.LabelConverter;
import com.google.devtools.build.lib.packages.Package.NameConflictException;
import com.google.devtools.build.lib.packages.PackageFactory.PackageContext;
import com.google.devtools.build.lib.packages.PredicateWithMessage;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder.RuleClassType;
import com.google.devtools.build.lib.packages.RuleFactory;
import com.google.devtools.build.lib.packages.RuleFactory.BuildLangTypedAttributeValuesMap;
import com.google.devtools.build.lib.packages.RuleFactory.InvalidRuleException;
import com.google.devtools.build.lib.packages.RuleFunction;
import com.google.devtools.build.lib.packages.RuleTransitionData;
import com.google.devtools.build.lib.packages.StarlarkAspect;
import com.google.devtools.build.lib.packages.StarlarkCallbackHelper;
import com.google.devtools.build.lib.packages.StarlarkDefinedAspect;
import com.google.devtools.build.lib.packages.StarlarkExportable;
import com.google.devtools.build.lib.packages.StarlarkProvider;
import com.google.devtools.build.lib.packages.StarlarkProviderIdentifier;
import com.google.devtools.build.lib.packages.TargetUtils;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.SerializationConstant;
import com.google.devtools.build.lib.starlarkbuildapi.StarlarkRuleFunctionsApi;
import com.google.devtools.build.lib.util.FileTypeSet;
import com.google.devtools.build.lib.util.Pair;
import com.google.errorprone.annotations.FormatMethod;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import javax.annotation.Nullable;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Printer;
import net.starlark.java.eval.Sequence;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkCallable;
import net.starlark.java.eval.StarlarkFunction;
import net.starlark.java.eval.StarlarkInt;
import net.starlark.java.eval.StarlarkSemantics;
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.eval.Tuple;
import net.starlark.java.syntax.Identifier;
import net.starlark.java.syntax.Location;

/** A helper class to provide an easier API for Starlark rule definitions. */
public class StarlarkRuleClassFunctions implements StarlarkRuleFunctionsApi {
  // A cache for base rule classes (especially tests).
  private static final LoadingCache<String, Label> labelCache =
      Caffeine.newBuilder().build(Label::parseCanonical);

  // TODO(bazel-team): Remove the code duplication (BaseRuleClasses and this class).
  /** Parent rule class for non-executable non-test Starlark rules. */
  public static final RuleClass baseRule =
      BaseRuleClasses.commonCoreAndStarlarkAttributes(
              new RuleClass.Builder("$base_rule", RuleClassType.ABSTRACT, true)
                  .add(attr("expect_failure", STRING)))
          // TODO(skylark-team): Allow Starlark rules to extend native rules and remove duplication.
          .add(
              attr("toolchains", LABEL_LIST)
                  .allowedFileTypes(FileTypeSet.NO_FILE)
                  .mandatoryProviders(ImmutableList.of(TemplateVariableInfo.PROVIDER.id()))
                  .dontCheckConstraints())
          .add(attr(RuleClass.EXEC_PROPERTIES_ATTR, Type.STRING_DICT).value(ImmutableMap.of()))
          .add(
              attr(RuleClass.EXEC_COMPATIBLE_WITH_ATTR, BuildType.LABEL_LIST)
                  .allowedFileTypes()
                  .nonconfigurable("Used in toolchain resolution")
                  .tool(
                      "exec_compatible_with exists for constraint checking, not to create an"
                          + " actual dependency")
                  .value(ImmutableList.of()))
          .add(
              attr(RuleClass.TARGET_COMPATIBLE_WITH_ATTR, LABEL_LIST)
                  .mandatoryProviders(ConstraintValueInfo.PROVIDER.id())
                  // This should be configurable to allow for complex types of restrictions.
                  .tool(
                      "target_compatible_with exists for constraint checking, not to create an"
                          + " actual dependency")
                  .allowedFileTypes(FileTypeSet.NO_FILE))
          .build();

  /** Parent rule class for executable non-test Starlark rules. */
  private static final RuleClass binaryBaseRule =
      new RuleClass.Builder("$binary_base_rule", RuleClassType.ABSTRACT, true, baseRule)
          .add(attr("args", STRING_LIST))
          .add(attr("output_licenses", LICENSE))
          .build();

  /** Parent rule class for test Starlark rules. */
  public static RuleClass getTestBaseRule(RuleDefinitionEnvironment env) {
    RepositoryName toolsRepository = env.getToolsRepository();
    RuleClass.Builder builder =
        new RuleClass.Builder("$test_base_rule", RuleClassType.ABSTRACT, true, baseRule)
            .requiresConfigurationFragments(TestConfiguration.class)
            // TestConfiguration only needed to create TestAction and TestProvider
            // Only necessary at top-level and can be skipped if trimmed.
            .setMissingFragmentPolicy(TestConfiguration.class, MissingFragmentPolicy.IGNORE)
            .add(
                attr("size", STRING)
                    .value("medium")
                    .taggable()
                    .nonconfigurable("used in loading phase rule validation logic"))
            .add(
                attr("timeout", STRING)
                    .taggable()
                    .nonconfigurable("policy decision: should be consistent across configurations")
                    .value(TIMEOUT_DEFAULT))
            .add(
                attr("flaky", BOOLEAN)
                    .value(false)
                    .taggable()
                    .nonconfigurable("taggable - called in Rule.getRuleTags"))
            .add(attr("shard_count", INTEGER).value(StarlarkInt.of(-1)))
            .add(
                attr("local", BOOLEAN)
                    .value(false)
                    .taggable()
                    .nonconfigurable(
                        "policy decision: this should be consistent across configurations"))
            .add(attr("args", STRING_LIST))
            // Input files for every test action
            .add(
                attr("$test_wrapper", LABEL)
                    .cfg(ExecutionTransitionFactory.createFactory())
                    .singleArtifact()
                    .value(labelCache.get(toolsRepository + "//tools/test:test_wrapper")))
            .add(
                attr("$xml_writer", LABEL)
                    .cfg(ExecutionTransitionFactory.createFactory())
                    .singleArtifact()
                    .value(labelCache.get(toolsRepository + "//tools/test:xml_writer")))
            .add(
                attr("$test_runtime", LABEL_LIST)
                    .cfg(ExecutionTransitionFactory.createFactory())
                    // Getting this default value through the getTestRuntimeLabelList helper ensures
                    // we reuse the same ImmutableList<Label> instance for each $test_runtime attr.
                    .value(getTestRuntimeLabelList(env)))
            .add(
                attr("$test_setup_script", LABEL)
                    .cfg(ExecutionTransitionFactory.createFactory())
                    .singleArtifact()
                    .value(labelCache.get(toolsRepository + "//tools/test:test_setup")))
            .add(
                attr("$xml_generator_script", LABEL)
                    .cfg(ExecutionTransitionFactory.createFactory())
                    .singleArtifact()
                    .value(labelCache.get(toolsRepository + "//tools/test:test_xml_generator")))
            .add(
                attr("$collect_coverage_script", LABEL)
                    .cfg(ExecutionTransitionFactory.createFactory())
                    .singleArtifact()
                    .value(labelCache.get(toolsRepository + "//tools/test:collect_coverage")))
            // Input files for test actions collecting code coverage
            .add(
                attr(":coverage_support", LABEL)
                    .cfg(ExecutionTransitionFactory.createFactory())
                    .value(
                        BaseRuleClasses.coverageSupportAttribute(
                            labelCache.get(
                                toolsRepository + BaseRuleClasses.DEFAULT_COVERAGE_SUPPORT_VALUE))))
            // Used in the one-per-build coverage report generation action.
            .add(
                attr(":coverage_report_generator", LABEL)
                    .cfg(ExecutionTransitionFactory.createFactory())
                    .value(
                        BaseRuleClasses.coverageReportGeneratorAttribute(
                            labelCache.get(
                                toolsRepository
                                    + BaseRuleClasses.DEFAULT_COVERAGE_REPORT_GENERATOR_VALUE))))
            .add(attr(":run_under", LABEL).value(RUN_UNDER));

    env.getNetworkAllowlistForTests()
        .ifPresent(
            label ->
                builder.add(
                    Allowlist.getAttributeFromAllowlistName("external_network").value(label)));

    return builder.build();
  }

  @Override
  public Object provider(Object doc, Object fields, Object init, StarlarkThread thread)
      throws EvalException {
    StarlarkProvider.Builder builder = StarlarkProvider.builder(thread.getCallerLocation());
    Starlark.toJavaOptional(doc, String.class)
        .map(Starlark::trimDocString)
        .ifPresent(builder::setDocumentation);
    if (fields instanceof Sequence) {
      builder.setSchema(Sequence.cast(fields, String.class, "fields"));
    } else if (fields instanceof Dict) {
      builder.setSchema(
          Maps.transformValues(
              Dict.cast(fields, String.class, String.class, "fields"), Starlark::trimDocString));
    }
    if (init == Starlark.NONE) {
      return builder.build();
    } else {
      if (init instanceof StarlarkCallable) {
        builder.setInit((StarlarkCallable) init);
      } else {
        throw Starlark.errorf("got %s for init, want callable value", Starlark.type(init));
      }
      StarlarkProvider provider = builder.build();
      return Tuple.of(provider, provider.createRawConstructor());
    }
  }

  // TODO(bazel-team): implement attribute copy and other rule properties
  @Override
  public StarlarkRuleFunction rule(
      StarlarkFunction implementation,
      Boolean test,
      Object attrs,
      Object implicitOutputs,
      Boolean executable,
      Boolean outputToGenfiles,
      Sequence<?> fragments,
      Sequence<?> hostFragments,
      Boolean starlarkTestable,
      Sequence<?> toolchains,
      boolean useToolchainTransition,
      Object doc,
      Sequence<?> providesArg,
      Sequence<?> execCompatibleWith,
      Object analysisTest,
      Object buildSetting,
      Object cfg,
      Object execGroups,
      StarlarkThread thread)
      throws EvalException {
    // Ensure we're initializing a .bzl file, which also means we have a RuleDefinitionEnvironment.
    BzlInitThreadContext bazelContext = BzlInitThreadContext.fromOrFail(thread, "rule()");

    // Get the callstack, sans the last entry, which is the builtin 'rule' callable itself.
    ImmutableList<StarlarkThread.CallStackEntry> callStack = thread.getCallStack();
    callStack = callStack.subList(0, callStack.size() - 1);

    LabelConverter labelConverter = LabelConverter.forBzlEvaluatingThread(thread);

    return createRule(
        // Contextual parameters.
        bazelContext,
        thread.getCallerLocation(),
        callStack,
        bazelContext.getBzlFile(),
        bazelContext.getTransitiveDigest(),
        labelConverter,
        thread.getSemantics(),
        // rule() parameters
        implementation,
        test,
        attrs,
        implicitOutputs,
        executable,
        outputToGenfiles,
        fragments,
        starlarkTestable,
        toolchains,
        doc,
        providesArg,
        execCompatibleWith,
        analysisTest,
        buildSetting,
        cfg,
        execGroups);
  }

  /**
   * Returns a new function representing a Starlark-defined rule.
   *
   * <p>This is public for the benefit of {@link StarlarkTestingModule}, which has the unusual use
   * case of creating new rule types to house analysis-time test assertions ({@code analysis_test}).
   * It's probably not a good idea to add new callers of this method.
   *
   * <p>Note that the bzlFile and transitiveDigest params correspond to the outermost .bzl file
   * being evaluated, not the one in which rule() is called.
   */
  public static StarlarkRuleFunction createRule(
      // Contextual parameters.
      RuleDefinitionEnvironment ruleDefinitionEnvironment,
      Location loc,
      ImmutableList<StarlarkThread.CallStackEntry> definitionCallstack,
      Label bzlFile,
      byte[] transitiveDigest,
      LabelConverter labelConverter,
      StarlarkSemantics starlarkSemantics,
      // Parameters that come from rule().
      StarlarkFunction implementation,
      boolean test,
      Object attrs,
      Object implicitOutputs,
      boolean executable,
      boolean outputToGenfiles,
      Sequence<?> fragments,
      boolean starlarkTestable,
      Sequence<?> toolchains,
      Object doc,
      Sequence<?> providesArg,
      Sequence<?> execCompatibleWith,
      Object analysisTest,
      Object buildSetting,
      Object cfg,
      Object execGroups)
      throws EvalException {

    // analysis_test=true implies test=true.
    test |= Boolean.TRUE.equals(analysisTest);

    RuleClassType type = test ? RuleClassType.TEST : RuleClassType.NORMAL;
    RuleClass parent =
        test
            ? getTestBaseRule(ruleDefinitionEnvironment)
            : (executable ? binaryBaseRule : baseRule);

    // We'll set the name later, pass the empty string for now.
    RuleClass.Builder builder = new RuleClass.Builder("", type, true, parent);

    builder.setCallStack(definitionCallstack);

    ImmutableList<Pair<String, StarlarkAttrModule.Descriptor>> attributes =
        attrObjectToAttributesList(attrs);

    if (starlarkTestable) {
      builder.setStarlarkTestable();
    }
    if (Boolean.TRUE.equals(analysisTest)) {
      builder.setIsAnalysisTest();
    }

    if (executable || test) {
      builder.addAttribute(
          attr("$is_executable", BOOLEAN)
              .value(true)
              .nonconfigurable("Called from RunCommand.isExecutable, which takes a Target")
              .build());
      builder.setExecutableStarlark();
    }

    if (implicitOutputs != Starlark.NONE) {
      if (implicitOutputs instanceof StarlarkFunction) {
        StarlarkCallbackHelper callback =
            new StarlarkCallbackHelper((StarlarkFunction) implicitOutputs, starlarkSemantics);
        builder.setImplicitOutputsFunction(
            new StarlarkImplicitOutputsFunctionWithCallback(callback));
      } else {
        builder.setImplicitOutputsFunction(
            new StarlarkImplicitOutputsFunctionWithMap(
                ImmutableMap.copyOf(
                    Dict.cast(
                        implicitOutputs,
                        String.class,
                        String.class,
                        "implicit outputs of the rule class"))));
      }
    }

    if (outputToGenfiles) {
      builder.setOutputToGenfiles();
    }

    builder.requiresConfigurationFragmentsByStarlarkModuleName(
        Sequence.cast(fragments, String.class, "fragments"));
    builder.setConfiguredTargetFunction(implementation);

    // The rule definition's label and transitive digest typically come from the context of the .bzl
    // file being initialized.
    //
    // Note that if rule() was called via a helper function (a meta-macro), the label and digest of
    // the .bzl file of the innermost stack frame might not be the same as that of the outermost
    // frame. In this case we really do want the outermost, in order to ensure that the digest
    // includes the code that determines the helper function's argument values.
    builder.setRuleDefinitionEnvironmentLabelAndDigest(bzlFile, transitiveDigest);

    builder.addToolchainTypes(parseToolchainTypes(toolchains, labelConverter));

    if (execGroups != Starlark.NONE) {
      Map<String, ExecGroup> execGroupDict =
          Dict.cast(execGroups, String.class, ExecGroup.class, "exec_group");
      for (String group : execGroupDict.keySet()) {
        // TODO(b/151742236): document this in the param documentation.
        if (!StarlarkExecGroupCollection.isValidGroupName(group)) {
          throw Starlark.errorf("Exec group name '%s' is not a valid name.", group);
        }
      }
      builder.addExecGroups(execGroupDict);
    }
    if (test && !builder.hasExecGroup(DEFAULT_TEST_RUNNER_EXEC_GROUP)) {
      builder.addExecGroup(DEFAULT_TEST_RUNNER_EXEC_GROUP);
    }

    if (!buildSetting.equals(Starlark.NONE) && !cfg.equals(Starlark.NONE)) {
      throw Starlark.errorf(
          "Build setting rules cannot use the `cfg` param to apply transitions to themselves.");
    }
    if (!buildSetting.equals(Starlark.NONE)) {
      builder.setBuildSetting((BuildSetting) buildSetting);
    }
    if (!cfg.equals(Starlark.NONE)) {
      if (cfg instanceof StarlarkDefinedConfigTransition) {
        StarlarkDefinedConfigTransition starlarkDefinedConfigTransition =
            (StarlarkDefinedConfigTransition) cfg;
        builder.cfg(new StarlarkRuleTransitionProvider(starlarkDefinedConfigTransition));
        builder.setHasStarlarkRuleTransition();
      } else if (cfg instanceof PatchTransition) {
        builder.cfg((PatchTransition) cfg);
      } else if (cfg instanceof StarlarkExposedRuleTransitionFactory) {
        StarlarkExposedRuleTransitionFactory transition =
            (StarlarkExposedRuleTransitionFactory) cfg;
        builder.cfg(transition);
        transition.addToStarlarkRule(ruleDefinitionEnvironment, builder);
      } else if (cfg instanceof TransitionFactory) {
        // This may be redundant with StarlarkExposedRuleTransitionFactory infra
        TransitionFactory<? extends TransitionFactory.Data> transitionFactory =
            (TransitionFactory<? extends TransitionFactory.Data>) cfg;
        if (transitionFactory.transitionType().isCompatibleWith(TransitionType.RULE)) {
          @SuppressWarnings("unchecked") // Actually checked due to above isCompatibleWith call.
          TransitionFactory<RuleTransitionData> ruleTransitionFactory =
              (TransitionFactory<RuleTransitionData>) transitionFactory;
          builder.cfg(ruleTransitionFactory);
        } else {
          throw Starlark.errorf(
              "`cfg` must be set to a transition appropriate for a rule, not an attribute-specific"
                  + " transition.");
        }
      } else {
        throw Starlark.errorf(
            "`cfg` must be set to a transition object initialized by the transition() function.");
      }
    }

    for (Object o : providesArg) {
      if (!StarlarkAttrModule.isProvider(o)) {
        throw Starlark.errorf(
            "Illegal argument: element in 'provides' is of unexpected type. "
                + "Should be list of providers, but got item of type %s.",
            Starlark.type(o));
      }
    }
    for (StarlarkProviderIdentifier starlarkProvider :
        StarlarkAttrModule.getStarlarkProviderIdentifiers(providesArg)) {
      builder.advertiseStarlarkProvider(starlarkProvider);
    }

    if (!execCompatibleWith.isEmpty()) {
      builder.addExecutionPlatformConstraints(
          parseExecCompatibleWith(execCompatibleWith, labelConverter));
    }

    return new StarlarkRuleFunction(
        builder,
        type,
        attributes,
        loc,
        Starlark.toJavaOptional(doc, String.class).map(Starlark::trimDocString));
  }

  private static void checkAttributeName(String name) throws EvalException {
    if (!Identifier.isValid(name)) {
      throw Starlark.errorf("attribute name `%s` is not a valid identifier.", name);
    }
  }

  private static ImmutableList<Pair<String, StarlarkAttrModule.Descriptor>>
      attrObjectToAttributesList(Object attrs) throws EvalException {
    ImmutableList.Builder<Pair<String, StarlarkAttrModule.Descriptor>> attributes =
        ImmutableList.builder();

    if (attrs != Starlark.NONE) {
      for (Map.Entry<String, Descriptor> attr :
          Dict.cast(attrs, String.class, Descriptor.class, "attrs").entrySet()) {
        Descriptor attrDescriptor = attr.getValue();
        AttributeValueSource source = attrDescriptor.getValueSource();
        checkAttributeName(attr.getKey());
        String attrName = source.convertToNativeName(attr.getKey());
        attributes.add(Pair.of(attrName, attrDescriptor));
      }
    }
    return attributes.build();
  }

  private static ImmutableSet<Label> parseExecCompatibleWith(
      Sequence<?> inputs, LabelConverter labelConverter) throws EvalException {
    ImmutableSet.Builder<Label> parsedLabels = new ImmutableSet.Builder<>();
    for (String input : Sequence.cast(inputs, String.class, "exec_compatible_with")) {
      try {
        Label label = labelConverter.convert(input);
        parsedLabels.add(label);
      } catch (LabelSyntaxException e) {
        throw Starlark.errorf("Unable to parse constraint label '%s': %s", input, e.getMessage());
      }
    }
    return parsedLabels.build();
  }

  @Override
  public StarlarkAspect aspect(
      StarlarkFunction implementation,
      Sequence<?> attributeAspects,
      Object attrs,
      Sequence<?> requiredProvidersArg,
      Sequence<?> requiredAspectProvidersArg,
      Sequence<?> providesArg,
      Sequence<?> requiredAspects,
      Sequence<?> fragments,
      Sequence<?> hostFragments,
      Sequence<?> toolchains,
      boolean useToolchainTransition,
      Object doc,
      Boolean applyToGeneratingRules,
      Sequence<?> rawExecCompatibleWith,
      Object rawExecGroups,
      StarlarkThread thread)
      throws EvalException {
    LabelConverter labelConverter = LabelConverter.forBzlEvaluatingThread(thread);

    ImmutableList.Builder<String> attrAspects = ImmutableList.builder();
    for (Object attributeAspect : attributeAspects) {
      String attrName = STRING.convert(attributeAspect, "attr_aspects");

      if (attrName.equals("*") && attributeAspects.size() != 1) {
        throw new EvalException("'*' must be the only string in 'attr_aspects' list");
      }

      if (!attrName.startsWith("_")) {
        attrAspects.add(attrName);
      } else {
        // Implicit attribute names mean either implicit or late-bound attributes
        // (``$attr`` or ``:attr``). Depend on both.
        attrAspects.add(AttributeValueSource.COMPUTED_DEFAULT.convertToNativeName(attrName));
        attrAspects.add(AttributeValueSource.LATE_BOUND.convertToNativeName(attrName));
      }
    }

    ImmutableList<Pair<String, StarlarkAttrModule.Descriptor>> descriptors =
        attrObjectToAttributesList(attrs);
    ImmutableList.Builder<Attribute> attributes = ImmutableList.builder();
    ImmutableSet.Builder<String> requiredParams = ImmutableSet.builder();
    for (Pair<String, Descriptor> nameDescriptorPair : descriptors) {
      String nativeName = nameDescriptorPair.first;
      boolean hasDefault = nameDescriptorPair.second.hasDefault();
      Attribute attribute = nameDescriptorPair.second.build(nameDescriptorPair.first);

      if (!Attribute.isImplicit(nativeName) && !Attribute.isLateBound(nativeName)) {
        if (attribute.getType() == Type.STRING) {
          // isValueSet() is always true for attr.string as default value is "" by default.
          hasDefault = !Objects.equals(attribute.getDefaultValue(null), "");
        } else if (attribute.getType() == Type.INTEGER) {
          // isValueSet() is always true for attr.int as default value is 0 by default.
          hasDefault = !Objects.equals(attribute.getDefaultValue(null), StarlarkInt.of(0));
        } else if (attribute.getType() == Type.BOOLEAN) {
          hasDefault = !Objects.equals(attribute.getDefaultValue(null), false);
        } else {
          throw Starlark.errorf(
              "Aspect parameter attribute '%s' must have type 'bool', 'int' or 'string'.",
              nativeName);
        }

        if (hasDefault && attribute.checkAllowedValues()) {
          PredicateWithMessage<Object> allowed = attribute.getAllowedValues();
          Object defaultVal = attribute.getDefaultValue(null);
          if (!allowed.apply(defaultVal)) {
            throw Starlark.errorf(
                "Aspect parameter attribute '%s' has a bad default value: %s",
                nativeName, allowed.getErrorReason(defaultVal));
          }
        }
        if (!hasDefault || attribute.isMandatory()) {
          requiredParams.add(nativeName);
        }
      } else if (!hasDefault) { // Implicit or late bound attribute
        String starlarkName = "_" + nativeName.substring(1);
        if (attribute.isLateBound()
            && !(attribute.getLateBoundDefault() instanceof StarlarkLateBoundDefault)) {
          // Code elsewhere assumes that a late-bound attribute of a Starlark-defined aspects can
          // exist in Java-land only as a StarlarkLateBoundDefault.
          throw Starlark.errorf(
              "Starlark aspect attribute '%s' is late-bound but somehow is not defined in Starlark."
                  + " This violates an invariant inside of Bazel. Please file a bug with"
                  + " instructions for reproducing this. Thanks!",
              starlarkName);
        }
        throw Starlark.errorf("Aspect attribute '%s' has no default value.", starlarkName);
      }
      if (attribute.getDefaultValueUnchecked() instanceof StarlarkComputedDefaultTemplate) {
        // Attributes specifying dependencies using computed value are currently not supported.
        // The limitation is in place because:
        //  - blaze query requires that all possible values are knowable without BuildConguration
        //  - aspects can attach to any rule
        // Current logic in StarlarkComputedDefault is not enough,
        // however {Conservative,Precise}AspectResolver can probably be improved to make that work.
        String starlarkName = "_" + nativeName.substring(1);
        throw Starlark.errorf(
            "Aspect attribute '%s' (%s) with computed default value is unsupported.",
            starlarkName, attribute.getType());
      }
      attributes.add(attribute);
    }

    for (Object o : providesArg) {
      if (!StarlarkAttrModule.isProvider(o)) {
        throw Starlark.errorf(
            "Illegal argument: element in 'provides' is of unexpected type. "
                + "Should be list of providers, but got item of type %s. ",
            Starlark.type(o));
      }
    }

    if (applyToGeneratingRules && !requiredProvidersArg.isEmpty()) {
      throw Starlark.errorf(
          "An aspect cannot simultaneously have required providers and apply to generating rules.");
    }

    ImmutableSet<Label> execCompatibleWith = ImmutableSet.of();
    if (!rawExecCompatibleWith.isEmpty()) {
      execCompatibleWith = parseExecCompatibleWith(rawExecCompatibleWith, labelConverter);
    }

    ImmutableMap<String, ExecGroup> execGroups = ImmutableMap.of();
    if (rawExecGroups != Starlark.NONE) {
      execGroups =
          ImmutableMap.copyOf(
              Dict.cast(rawExecGroups, String.class, ExecGroup.class, "exec_group"));
      for (String group : execGroups.keySet()) {
        // TODO(b/151742236): document this in the param documentation.
        if (!StarlarkExecGroupCollection.isValidGroupName(group)) {
          throw Starlark.errorf("Exec group name '%s' is not a valid name.", group);
        }
      }
    }

    return new StarlarkDefinedAspect(
        implementation,
        Starlark.toJavaOptional(doc, String.class).map(Starlark::trimDocString),
        attrAspects.build(),
        attributes.build(),
        StarlarkAttrModule.buildProviderPredicate(requiredProvidersArg, "required_providers"),
        StarlarkAttrModule.buildProviderPredicate(
            requiredAspectProvidersArg, "required_aspect_providers"),
        StarlarkAttrModule.getStarlarkProviderIdentifiers(providesArg),
        requiredParams.build(),
        ImmutableSet.copyOf(Sequence.cast(requiredAspects, StarlarkAspect.class, "requires")),
        ImmutableSet.copyOf(Sequence.cast(fragments, String.class, "fragments")),
        parseToolchainTypes(toolchains, labelConverter),
        applyToGeneratingRules,
        execCompatibleWith,
        execGroups);
  }

  /**
   * The implementation for the magic function "rule" that creates Starlark rule classes.
   *
   * <p>Exactly one of {@link #builder} or {@link #ruleClass} is null except inside {@link #export}.
   */
  public static final class StarlarkRuleFunction implements StarlarkExportable, RuleFunction {
    private RuleClass.Builder builder;

    private RuleClass ruleClass;
    private final RuleClassType type;
    private ImmutableList<Pair<String, StarlarkAttrModule.Descriptor>> attributes;
    private final Location definitionLocation;
    @Nullable private final String documentation;
    private Label starlarkLabel;

    // TODO(adonovan): merge {Starlark,Builtin}RuleFunction and RuleClass,
    // making the latter a callable, StarlarkExportable value.
    // (Making RuleClasses first-class values will help us to build a
    // rich query output mode that includes values from loaded .bzl files.)
    public StarlarkRuleFunction(
        RuleClass.Builder builder,
        RuleClassType type,
        ImmutableList<Pair<String, StarlarkAttrModule.Descriptor>> attributes,
        Location definitionLocation,
        Optional<String> documentation) {
      this.builder = builder;
      // For documentation generation, we need to distinguish Starlark-defined attributes passed via
      // `rule(attrs=...) from implicitly added attributes such as "tags" or "testonly".
      checkArgument(
          builder.getAttributes().stream().noneMatch(Attribute::starlarkDefined),
          "Implicitly added attributes are expected to be built-in, not Starlark-defined");
      this.type = type;
      this.attributes = attributes;
      this.definitionLocation = definitionLocation;
      this.documentation = documentation.orElse(null);
    }

    @Override
    public String getName() {
      return ruleClass != null ? ruleClass.getName() : "unexported rule";
    }

    /**
     * Returns the value of the doc parameter passed to {@code rule()} in Starlark, or an empty
     * Optional if a doc string was not provided.
     */
    public Optional<String> getDocumentation() {
      return Optional.ofNullable(documentation);
    }

    /**
     * Returns the label of the .bzl module where rule() was called, or null if the rule has not
     * been exported yet.
     */
    @Nullable
    public Label getExtensionLabel() {
      return starlarkLabel;
    }

    @Override
    public Object call(StarlarkThread thread, Tuple args, Dict<String, Object> kwargs)
        throws EvalException, InterruptedException {
      if (!args.isEmpty()) {
        throw new EvalException("Unexpected positional arguments");
      }
      try {
        BazelStarlarkContext.from(thread).checkLoadingPhase(getName());
      } catch (IllegalStateException unused) {
        throw new EvalException(
            "A rule can only be instantiated in a BUILD file, or a macro "
                + "invoked from a BUILD file");
      }
      if (ruleClass == null) {
        throw new EvalException("Invalid rule class hasn't been exported by a bzl file");
      }

      validateRulePropagatedAspects(ruleClass);

      BuildLangTypedAttributeValuesMap attributeValues =
          new BuildLangTypedAttributeValuesMap(kwargs);
      try {
        PackageContext pkgContext = thread.getThreadLocal(PackageContext.class);
        if (pkgContext == null) {
          throw new EvalException(
              "Cannot instantiate a rule when loading a .bzl file. "
                  + "Rules may be instantiated only in a BUILD thread.");
        }
        RuleFactory.createAndAddRule(
            pkgContext.getBuilder(),
            ruleClass,
            attributeValues,
            pkgContext.getEventHandler(),
            thread.getCallStack());
      } catch (InvalidRuleException | NameConflictException e) {
        throw new EvalException(e);
      }
      return Starlark.NONE;
    }

    private static void validateRulePropagatedAspects(RuleClass ruleClass) throws EvalException {
      for (Attribute attribute : ruleClass.getAttributes()) {
        for (AspectDetails<?> aspect : attribute.getAspectsDetails()) {
          ImmutableSet<String> requiredAspectParameters = aspect.getRequiredParameters();
          for (Attribute aspectAttribute : aspect.getAspectAttributes()) {
            String aspectAttrName = aspectAttribute.getPublicName();
            Type<?> aspectAttrType = aspectAttribute.getType();

            // When propagated from a rule, explicit aspect attributes must be of type boolean, int
            // or string. Integer and string attributes must have the `values` restriction.
            if (!aspectAttribute.isImplicit() && !aspectAttribute.isLateBound()) {
              if (aspectAttrType != Type.BOOLEAN && !aspectAttribute.checkAllowedValues()) {
                throw Starlark.errorf(
                    "Aspect %s: Aspect parameter attribute '%s' must use the 'values' restriction.",
                    aspect.getName(), aspectAttrName);
              }
            }

            // Required aspect parameters must be specified by the rule propagating the aspect with
            // the same parameter type.
            if (requiredAspectParameters.contains(aspectAttrName)) {
              if (!ruleClass.hasAttr(aspectAttrName, aspectAttrType)) {
                throw Starlark.errorf(
                    "Aspect %s requires rule %s to specify attribute '%s' with type %s.",
                    aspect.getName(), ruleClass.getName(), aspectAttrName, aspectAttrType);
              }
            }
          }
        }
      }
    }

    /** Export a RuleFunction from a Starlark file with a given name. */
    @Override
    public void export(EventHandler handler, Label starlarkLabel, String ruleClassName) {
      checkState(ruleClass == null && builder != null);
      this.starlarkLabel = starlarkLabel;
      if (type == RuleClassType.TEST != TargetUtils.isTestRuleName(ruleClassName)) {
        errorf(
            handler,
            "Invalid rule class name '%s', test rule class names must end with '_test' and other"
                + " rule classes must not",
            ruleClassName);
        return;
      }
      // Thus far, we only know if we have a rule transition. While iterating through attributes,
      // check if we have an attribute transition.
      boolean hasStarlarkDefinedTransition = builder.hasStarlarkRuleTransition();
      boolean hasFunctionTransitionAllowlist = false;
      for (Pair<String, StarlarkAttrModule.Descriptor> attribute : attributes) {
        String name = attribute.getFirst();
        StarlarkAttrModule.Descriptor descriptor = attribute.getSecond();

        Attribute attr = descriptor.build(name);

        hasStarlarkDefinedTransition |= attr.hasStarlarkDefinedTransition();
        if (attr.hasAnalysisTestTransition()) {
          if (!builder.isAnalysisTest()) {
            errorf(
                handler,
                "Only rule definitions with analysis_test=True may have attributes with"
                    + " analysis_test_transition transitions");
            continue;
          }
          builder.setHasAnalysisTestTransition();
        }
        // Check for existence of the function transition allowlist attribute.
        // TODO(b/121385274): remove when we stop allowlisting starlark transitions
        if (name.equals(FunctionSplitTransitionAllowlist.ATTRIBUTE_NAME)
            || name.equals(FunctionSplitTransitionAllowlist.LEGACY_ATTRIBUTE_NAME)) {
          if (!BuildType.isLabelType(attr.getType())) {
            errorf(handler, "_allowlist_function_transition attribute must be a label type");
            continue;
          }
          if (attr.getDefaultValueUnchecked() == null) {
            errorf(handler, "_allowlist_function_transition attribute must have a default value");
            continue;
          }
          Label defaultLabel = (Label) attr.getDefaultValueUnchecked();
          // Check the label value for package and target name, to make sure this works properly
          // in Bazel where it is expected to be found under @bazel_tools.
          if (!(defaultLabel
                      .getPackageName()
                      .equals(FunctionSplitTransitionAllowlist.LABEL.getPackageName())
                  && defaultLabel
                      .getName()
                      .equals(FunctionSplitTransitionAllowlist.LABEL.getName()))
              && !(defaultLabel
                      .getPackageName()
                      .equals(FunctionSplitTransitionAllowlist.LEGACY_LABEL.getPackageName())
                  && defaultLabel
                      .getName()
                      .equals(FunctionSplitTransitionAllowlist.LEGACY_LABEL.getName()))) {
            errorf(
                handler,
                "_allowlist_function_transition attribute (%s) does not have the expected value %s",
                defaultLabel,
                FunctionSplitTransitionAllowlist.LABEL);
            continue;
          }
          hasFunctionTransitionAllowlist = true;
        }

        try {
          builder.addAttribute(attr);
        } catch (IllegalStateException ex) {
          // TODO(bazel-team): stop using unchecked exceptions in this way.
          errorf(handler, "cannot add attribute: %s", ex.getMessage());
        }
      }
      // TODO(b/121385274): remove when we stop allowlisting starlark transitions
      if (hasStarlarkDefinedTransition) {
        if (!starlarkLabel.getRepository().getNameWithAt().equals("@_builtins")) {
          if (!hasFunctionTransitionAllowlist) {
            errorf(
                handler,
                "Use of Starlark transition without allowlist attribute"
                    + " '_allowlist_function_transition'. See Starlark transitions documentation"
                    + " for details and usage: %s %s",
                builder.getRuleDefinitionEnvironmentLabel(),
                builder.getType());
            return;
          }
          builder.addAllowlistChecker(FUNCTION_TRANSITION_ALLOWLIST_CHECKER);
        }
      } else {
        if (hasFunctionTransitionAllowlist) {
          errorf(
              handler,
              "Unused function-based split transition allowlist: %s %s",
              builder.getRuleDefinitionEnvironmentLabel(),
              builder.getType());
          return;
        }
      }

      try {
        this.ruleClass = builder.build(ruleClassName, starlarkLabel + "%" + ruleClassName);
      } catch (IllegalArgumentException | IllegalStateException ex) {
        // TODO(adonovan): this catch statement is an abuse of exceptions. Be more specific.
        String msg = ex.getMessage();
        errorf(handler, "%s", msg != null ? msg : ex.toString());
      }

      this.builder = null;
      this.attributes = null;
    }

    @FormatMethod
    private void errorf(EventHandler handler, String format, Object... args) {
      handler.handle(Event.error(definitionLocation, String.format(format, args)));
    }

    @Override
    public RuleClass getRuleClass() {
      checkState(ruleClass != null && builder == null);
      return ruleClass;
    }

    @Override
    public boolean isExported() {
      return starlarkLabel != null;
    }

    @Override
    public void repr(Printer printer) {
      if (isExported()) {
        printer.append("<rule ").append(getRuleClass().getName()).append(">");
      } else {
        printer.append("<rule>");
      }
    }

    @Override
    public String toString() {
      return "rule(...)";
    }

    @Override
    public boolean isImmutable() {
      return true;
    }
  }

  @SerializationConstant
  static final AllowlistChecker FUNCTION_TRANSITION_ALLOWLIST_CHECKER =
      AllowlistChecker.builder()
          .setAllowlistAttr(FunctionSplitTransitionAllowlist.NAME)
          .setErrorMessage("Non-allowlisted use of Starlark transition")
          .setLocationCheck(AllowlistChecker.LocationCheck.INSTANCE_OR_DEFINITION)
          .build();

  @Override
  public Label label(Object input, StarlarkThread thread) throws EvalException {
    if (input instanceof Label) {
      return (Label) input;
    }
    // The label string is interpreted with respect to the .bzl module containing the call to
    // `Label()`. An alternative to this approach that avoids stack inspection is to have each .bzl
    // module define its own copy of the `Label()` builtin embedding the module's own name. This
    // would lead to peculiarities like foo.bzl being able to call bar.bzl's `Label()` symbol to
    // resolve strings as if it were bar.bzl. It also would prevent sharing the same builtins
    // environment across .bzl files. Hence, we opt for stack inspection.
    BazelModuleContext moduleContext = BazelModuleContext.ofInnermostBzlOrFail(thread, "Label()");
    try {
      return Label.parseWithPackageContext((String) input, moduleContext.packageContext());
    } catch (LabelSyntaxException e) {
      throw Starlark.errorf("invalid label in Label(): %s", e.getMessage());
    }
  }

  @Override
  public ExecGroup execGroup(
      Sequence<?> toolchains, Sequence<?> execCompatibleWith, StarlarkThread thread)
      throws EvalException {
    LabelConverter labelConverter = LabelConverter.forBzlEvaluatingThread(thread);
    ImmutableSet<ToolchainTypeRequirement> toolchainTypes =
        parseToolchainTypes(toolchains, labelConverter);
    ImmutableSet<Label> constraints = parseExecCompatibleWith(execCompatibleWith, labelConverter);
    return ExecGroup.builder()
        .toolchainTypes(toolchainTypes)
        .execCompatibleWith(constraints)
        .copyFrom(null)
        .build();
  }

  private static ImmutableSet<ToolchainTypeRequirement> parseToolchainTypes(
      Sequence<?> rawToolchains, LabelConverter labelConverter) throws EvalException {
    Map<Label, ToolchainTypeRequirement> toolchainTypes = new HashMap<>();

    for (Object rawToolchain : rawToolchains) {
      ToolchainTypeRequirement toolchainType = parseToolchainType(rawToolchain, labelConverter);
      Label typeLabel = toolchainType.toolchainType();
      ToolchainTypeRequirement previous = toolchainTypes.get(typeLabel);
      if (previous != null) {
        // Keep the one with the strictest requirements.
        toolchainType = ToolchainTypeRequirement.strictest(previous, toolchainType);
      }
      toolchainTypes.put(typeLabel, toolchainType);
    }

    return ImmutableSet.copyOf(toolchainTypes.values());
  }

  private static ToolchainTypeRequirement parseToolchainType(
      Object rawToolchain, LabelConverter labelConverter) throws EvalException {
    // Handle actual ToolchainTypeRequirement objects.
    if (rawToolchain instanceof ToolchainTypeRequirement) {
      return (ToolchainTypeRequirement) rawToolchain;
    }

    // Handle Label-like objects.
    Label toolchainLabel = null;
    if (rawToolchain instanceof Label) {
      toolchainLabel = (Label) rawToolchain;
    } else if (rawToolchain instanceof String) {
      try {
        toolchainLabel = labelConverter.convert((String) rawToolchain);
      } catch (LabelSyntaxException e) {
        throw Starlark.errorf(
            "Unable to parse toolchain_type label '%s': %s", rawToolchain, e.getMessage());
      }
    }

    if (toolchainLabel != null) {
      return ToolchainTypeRequirement.builder(toolchainLabel).mandatory(true).build();
    }

    // It's not a valid type.
    throw Starlark.errorf(
        "'toolchains' takes a toolchain_type, Label, or String, but instead got a %s",
        rawToolchain.getClass().getSimpleName());
  }
}
