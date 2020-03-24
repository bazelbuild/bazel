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

import static com.google.devtools.build.lib.analysis.BaseRuleClasses.RUN_UNDER;
import static com.google.devtools.build.lib.analysis.BaseRuleClasses.getTestRuntimeLabelList;
import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.BuildType.LABEL;
import static com.google.devtools.build.lib.packages.BuildType.LABEL_LIST;
import static com.google.devtools.build.lib.packages.BuildType.LICENSE;
import static com.google.devtools.build.lib.packages.Type.BOOLEAN;
import static com.google.devtools.build.lib.packages.Type.INTEGER;
import static com.google.devtools.build.lib.packages.Type.STRING;
import static com.google.devtools.build.lib.packages.Type.STRING_LIST;
import static com.google.devtools.build.lib.syntax.SkylarkType.castMap;

import com.google.common.base.Preconditions;
import com.google.common.cache.CacheBuilder;
import com.google.common.cache.CacheLoader;
import com.google.common.cache.LoadingCache;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.RuleDefinitionContext;
import com.google.devtools.build.lib.analysis.TemplateVariableInfo;
import com.google.devtools.build.lib.analysis.config.ConfigAwareRuleClassBuilder;
import com.google.devtools.build.lib.analysis.config.HostTransition;
import com.google.devtools.build.lib.analysis.config.StarlarkDefinedConfigTransition;
import com.google.devtools.build.lib.analysis.skylark.SkylarkAttr.Descriptor;
import com.google.devtools.build.lib.analysis.test.TestConfiguration;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.cmdline.LabelValidator;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.AttributeContainer;
import com.google.devtools.build.lib.packages.AttributeMap;
import com.google.devtools.build.lib.packages.AttributeValueSource;
import com.google.devtools.build.lib.packages.BazelStarlarkContext;
import com.google.devtools.build.lib.packages.BuildSetting;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.FunctionSplitTransitionWhitelist;
import com.google.devtools.build.lib.packages.ImplicitOutputsFunction.SkylarkImplicitOutputsFunctionWithCallback;
import com.google.devtools.build.lib.packages.ImplicitOutputsFunction.SkylarkImplicitOutputsFunctionWithMap;
import com.google.devtools.build.lib.packages.Package.NameConflictException;
import com.google.devtools.build.lib.packages.PackageFactory.PackageContext;
import com.google.devtools.build.lib.packages.PredicateWithMessage;
import com.google.devtools.build.lib.packages.Provider;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder.RuleClassType;
import com.google.devtools.build.lib.packages.RuleFactory;
import com.google.devtools.build.lib.packages.RuleFactory.BuildLangTypedAttributeValuesMap;
import com.google.devtools.build.lib.packages.RuleFactory.InvalidRuleException;
import com.google.devtools.build.lib.packages.RuleFunction;
import com.google.devtools.build.lib.packages.SkylarkAspect;
import com.google.devtools.build.lib.packages.SkylarkDefinedAspect;
import com.google.devtools.build.lib.packages.SkylarkExportable;
import com.google.devtools.build.lib.packages.SkylarkProvider;
import com.google.devtools.build.lib.packages.SkylarkProviderIdentifier;
import com.google.devtools.build.lib.packages.StarlarkCallbackHelper;
import com.google.devtools.build.lib.packages.TargetUtils;
import com.google.devtools.build.lib.packages.TestSize;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.packages.Type.ConversionException;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skylarkbuildapi.SkylarkRuleFunctionsApi;
import com.google.devtools.build.lib.syntax.BaseFunction;
import com.google.devtools.build.lib.syntax.Dict;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.EvalUtils;
import com.google.devtools.build.lib.syntax.FunctionSignature;
import com.google.devtools.build.lib.syntax.Identifier;
import com.google.devtools.build.lib.syntax.Module;
import com.google.devtools.build.lib.syntax.Printer;
import com.google.devtools.build.lib.syntax.Sequence;
import com.google.devtools.build.lib.syntax.SkylarkType;
import com.google.devtools.build.lib.syntax.Starlark;
import com.google.devtools.build.lib.syntax.StarlarkFunction;
import com.google.devtools.build.lib.syntax.StarlarkThread;
import com.google.devtools.build.lib.syntax.Tuple;
import com.google.devtools.build.lib.util.FileTypeSet;
import com.google.devtools.build.lib.util.Pair;
import java.util.Collection;
import java.util.Map;
import java.util.concurrent.ExecutionException;

/** A helper class to provide an easier API for Skylark rule definitions. */
public class SkylarkRuleClassFunctions implements SkylarkRuleFunctionsApi<Artifact> {

  // TODO(bazel-team): Copied from ConfiguredRuleClassProvider for the transition from built-in
  // rules to skylark extensions. Using the same instance would require a large refactoring.
  // If we don't want to support old built-in rules and Skylark simultaneously
  // (except for transition phase) it's probably OK.
  private static final LoadingCache<String, Label> labelCache =
      CacheBuilder.newBuilder()
          .build(
              new CacheLoader<String, Label>() {
                @Override
                public Label load(String from) throws Exception {
                  try {
                    return Label.parseAbsolute(
                        from,
                        /* defaultToMain=*/ false,
                        /* repositoryMapping= */ ImmutableMap.of());
                  } catch (LabelSyntaxException e) {
                    throw new Exception(from);
                  }
                }
              });

  // TODO(bazel-team): Remove the code duplication (BaseRuleClasses and this class).
  /** Parent rule class for non-executable non-test Skylark rules. */
  public static final RuleClass baseRule =
      BaseRuleClasses.commonCoreAndSkylarkAttributes(
              BaseRuleClasses.nameAttribute(
                      new RuleClass.Builder("$base_rule", RuleClassType.ABSTRACT, true))
                  .add(attr("expect_failure", STRING)))
          // TODO(skylark-team): Allow Skylark rules to extend native rules and remove duplication.
          .add(
              attr("toolchains", LABEL_LIST)
                  .allowedFileTypes(FileTypeSet.NO_FILE)
                  .mandatoryProviders(ImmutableList.of(TemplateVariableInfo.PROVIDER.id()))
                  .dontCheckConstraints())
          .add(attr(RuleClass.EXEC_PROPERTIES, Type.STRING_DICT).value(ImmutableMap.of()))
          .add(
              attr(RuleClass.EXEC_COMPATIBLE_WITH_ATTR, BuildType.LABEL_LIST)
                  .allowedFileTypes()
                  .nonconfigurable("Used in toolchain resolution")
                  .value(ImmutableList.of()))
          .build();

  /** Parent rule class for executable non-test Skylark rules. */
  public static final RuleClass binaryBaseRule =
      new RuleClass.Builder("$binary_base_rule", RuleClassType.ABSTRACT, true, baseRule)
          .add(attr("args", STRING_LIST))
          .add(attr("output_licenses", LICENSE))
          .build();

  /** Parent rule class for test Skylark rules. */
  public static final RuleClass getTestBaseRule(RuleDefinitionContext env) {
    String toolsRepository = env.getToolsRepository();
    return new RuleClass.Builder("$test_base_rule", RuleClassType.ABSTRACT, true, baseRule)
        .requiresConfigurationFragments(TestConfiguration.class)
        .add(
            attr("size", STRING)
                .value("medium")
                .taggable()
                .nonconfigurable("used in loading phase rule validation logic"))
        .add(
            attr("timeout", STRING)
                .taggable()
                .nonconfigurable("used in loading phase rule validation logic")
                .value(timeoutAttribute))
        .add(
            attr("flaky", BOOLEAN)
                .value(false)
                .taggable()
                .nonconfigurable("taggable - called in Rule.getRuleTags"))
        .add(attr("shard_count", INTEGER).value(-1))
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
                .cfg(HostTransition.createFactory())
                .singleArtifact()
                .value(labelCache.getUnchecked(toolsRepository + "//tools/test:test_wrapper")))
        .add(
            attr("$xml_writer", LABEL)
                .cfg(HostTransition.createFactory())
                .singleArtifact()
                .value(labelCache.getUnchecked(toolsRepository + "//tools/test:xml_writer")))
        .add(
            attr("$test_runtime", LABEL_LIST)
                .cfg(HostTransition.createFactory())
                // Getting this default value through the getTestRuntimeLabelList helper ensures we
                // reuse the same ImmutableList<Label> instance for each $test_runtime attr.
                .value(getTestRuntimeLabelList(env)))
        .add(
            attr("$test_setup_script", LABEL)
                .cfg(HostTransition.createFactory())
                .singleArtifact()
                .value(labelCache.getUnchecked(toolsRepository + "//tools/test:test_setup")))
        .add(
            attr("$xml_generator_script", LABEL)
                .cfg(HostTransition.createFactory())
                .singleArtifact()
                .value(
                    labelCache.getUnchecked(toolsRepository + "//tools/test:test_xml_generator")))
        .add(
            attr("$collect_coverage_script", LABEL)
                .cfg(HostTransition.createFactory())
                .singleArtifact()
                .value(labelCache.getUnchecked(toolsRepository + "//tools/test:collect_coverage")))
        // Input files for test actions collecting code coverage
        .add(
            attr(":coverage_support", LABEL)
                .cfg(HostTransition.createFactory())
                .value(
                    BaseRuleClasses.coverageSupportAttribute(
                        labelCache.getUnchecked(
                            toolsRepository + BaseRuleClasses.DEFAULT_COVERAGE_SUPPORT_VALUE))))
        // Used in the one-per-build coverage report generation action.
        .add(
            attr(":coverage_report_generator", LABEL)
                .cfg(HostTransition.createFactory())
                .value(
                    BaseRuleClasses.coverageReportGeneratorAttribute(
                        labelCache.getUnchecked(
                            toolsRepository
                                + BaseRuleClasses.DEFAULT_COVERAGE_REPORT_GENERATOR_VALUE))))
        .add(attr(":run_under", LABEL).value(RUN_UNDER))
        .build();
  }

  @AutoCodec @AutoCodec.VisibleForSerialization
  static final Attribute.ComputedDefault timeoutAttribute =
      new Attribute.ComputedDefault() {
        @Override
        public Object getDefault(AttributeMap rule) {
          TestSize size = TestSize.getTestSize(rule.get("size", Type.STRING));
          if (size != null) {
            String timeout = size.getDefaultTimeout().toString();
            if (timeout != null) {
              return timeout;
            }
          }
          return "illegal";
        }
      };

  @Override
  public Provider provider(String doc, Object fields, StarlarkThread thread) throws EvalException {
    Collection<String> fieldNames = null;
    if (fields instanceof Sequence) {
      @SuppressWarnings("unchecked")
      Sequence<String> list =
          (Sequence<String>)
              SkylarkType.cast(
                  fields,
                  Sequence.class,
                  String.class,
                  null,
                  "Expected list of strings or dictionary of string -> string for 'fields'");
      fieldNames = list;
    } else if (fields instanceof Dict) {
      Map<String, String> dict =
          SkylarkType.castMap(
              fields,
              String.class,
              String.class,
              "Expected list of strings or dictionary of string -> string for 'fields'");
      fieldNames = dict.keySet();
    }
    return SkylarkProvider.createUnexportedSchemaful(fieldNames, thread.getCallerLocation());
  }

  // TODO(bazel-team): implement attribute copy and other rule properties
  @Override
  public BaseFunction rule(
      StarlarkFunction implementation,
      Boolean test,
      Object attrs,
      Object implicitOutputs,
      Boolean executable,
      Boolean outputToGenfiles,
      Sequence<?> fragments,
      Sequence<?> hostFragments,
      Boolean skylarkTestable,
      Sequence<?> toolchains,
      String doc,
      Sequence<?> providesArg,
      Sequence<?> execCompatibleWith,
      Object analysisTest,
      Object buildSetting,
      Object cfg,
      Object execGroups,
      StarlarkThread thread)
      throws EvalException {
    BazelStarlarkContext bazelContext = BazelStarlarkContext.from(thread);
    bazelContext.checkLoadingOrWorkspacePhase("rule");
    // analysis_test=true implies test=true.
    test |= Boolean.TRUE.equals(analysisTest);

    RuleClassType type = test ? RuleClassType.TEST : RuleClassType.NORMAL;
    RuleClass parent =
        test ? getTestBaseRule(bazelContext) : (executable ? binaryBaseRule : baseRule);

    // We'll set the name later, pass the empty string for now.
    RuleClass.Builder builder = new RuleClass.Builder("", type, true, parent);

    ImmutableList<StarlarkThread.CallStackEntry> callstack = thread.getCallStack();
    builder.setCallStack(callstack.subList(0, callstack.size() - 1)); // pop 'rule' itself

    ImmutableList<Pair<String, SkylarkAttr.Descriptor>> attributes =
        attrObjectToAttributesList(attrs);

    if (skylarkTestable) {
      builder.setSkylarkTestable();
    }
    if (Boolean.TRUE.equals(analysisTest)) {
      builder.setIsAnalysisTest();
    }

    if (executable || test) {
      addAttribute(
          builder,
          attr("$is_executable", BOOLEAN)
              .value(true)
              .nonconfigurable("Called from RunCommand.isExecutable, which takes a Target")
              .build());
      builder.setExecutableSkylark();
    }

    if (implicitOutputs != Starlark.NONE) {
      if (implicitOutputs instanceof StarlarkFunction) {
        StarlarkCallbackHelper callback =
            new StarlarkCallbackHelper(
                (StarlarkFunction) implicitOutputs, thread.getSemantics(), bazelContext);
        builder.setImplicitOutputsFunction(
            new SkylarkImplicitOutputsFunctionWithCallback(callback, thread.getCallerLocation()));
      } else {
        builder.setImplicitOutputsFunction(
            new SkylarkImplicitOutputsFunctionWithMap(
                ImmutableMap.copyOf(
                    castMap(
                        implicitOutputs,
                        String.class,
                        String.class,
                        "implicit outputs of the rule class"))));
      }
    }

    if (outputToGenfiles) {
      builder.setOutputToGenfiles();
    }

    builder.requiresConfigurationFragmentsBySkylarkModuleName(
        fragments.getContents(String.class, "fragments"));
    ConfigAwareRuleClassBuilder.of(builder)
        .requiresHostConfigurationFragmentsBySkylarkModuleName(
            hostFragments.getContents(String.class, "host_fragments"));
    builder.setConfiguredTargetFunction(implementation);
    builder.setRuleDefinitionEnvironmentLabelAndHashCode(
        (Label) Module.ofInnermostEnclosingStarlarkFunction(thread).getLabel(),
        thread.getTransitiveContentHashCode());

    builder.addRequiredToolchains(
        collectToolchainLabels(
            toolchains.getContents(String.class, "toolchains"), bazelContext.getRepoMapping()));

    if (!buildSetting.equals(Starlark.NONE) && !cfg.equals(Starlark.NONE)) {
      throw Starlark.errorf(
          "Build setting rules cannot use the `cfg` param to apply transitions to themselves.");
    }
    if (!buildSetting.equals(Starlark.NONE)) {
      builder.setBuildSetting((BuildSetting) buildSetting);
    }
    if (!cfg.equals(Starlark.NONE)) {
      if (!(cfg instanceof StarlarkDefinedConfigTransition)) {
        throw Starlark.errorf(
            "`cfg` must be set to a transition object initialized by the transition() function.");
      }
      StarlarkDefinedConfigTransition starlarkDefinedConfigTransition =
          (StarlarkDefinedConfigTransition) cfg;
      builder.cfg(new StarlarkRuleTransitionProvider(starlarkDefinedConfigTransition));
      builder.setHasStarlarkRuleTransition();
    }

    for (Object o : providesArg) {
      if (!SkylarkAttr.isProvider(o)) {
        throw Starlark.errorf(
            "Illegal argument: element in 'provides' is of unexpected type. "
                + "Should be list of providers, but got item of type %s.",
            EvalUtils.getDataTypeName(o, true));
      }
    }
    for (SkylarkProviderIdentifier skylarkProvider :
        SkylarkAttr.getSkylarkProviderIdentifiers(providesArg)) {
      builder.advertiseSkylarkProvider(skylarkProvider);
    }

    if (!execCompatibleWith.isEmpty()) {
      builder.addExecutionPlatformConstraints(
          collectConstraintLabels(
              execCompatibleWith.getContents(String.class, "exec_compatile_with"),
              bazelContext.getRepoMapping()));
    }

    return new SkylarkRuleFunction(builder, type, attributes, thread.getCallerLocation());
  }

  private static void checkAttributeName(String name) throws EvalException {
    if (!Identifier.isValid(name)) {
      throw Starlark.errorf("attribute name `%s` is not a valid identifier.", name);
    }
  }

  private static ImmutableList<Pair<String, Descriptor>> attrObjectToAttributesList(Object attrs)
      throws EvalException {
    ImmutableList.Builder<Pair<String, Descriptor>> attributes = ImmutableList.builder();

    if (attrs != Starlark.NONE) {
      for (Map.Entry<String, Descriptor> attr :
          castMap(attrs, String.class, Descriptor.class, "attrs").entrySet()) {
        Descriptor attrDescriptor = attr.getValue();
        AttributeValueSource source = attrDescriptor.getValueSource();
        checkAttributeName(attr.getKey());
        String attrName = source.convertToNativeName(attr.getKey());
        attributes.add(Pair.of(attrName, attrDescriptor));
      }
    }
    return attributes.build();
  }

  private static void addAttribute(RuleClass.Builder builder, Attribute attribute)
      throws EvalException {
    try {
      builder.addOrOverrideAttribute(attribute);
    } catch (IllegalArgumentException ex) {
      throw new EvalException(null, ex);
    }
  }

  private static ImmutableList<Label> collectToolchainLabels(
      Iterable<String> rawLabels, ImmutableMap<RepositoryName, RepositoryName> mapping)
      throws EvalException {
    ImmutableList.Builder<Label> requiredToolchains = new ImmutableList.Builder<>();
    for (String rawLabel : rawLabels) {
      try {
        Label toolchainLabel = Label.parseAbsolute(rawLabel, mapping);
        requiredToolchains.add(toolchainLabel);
      } catch (LabelSyntaxException e) {
        throw new EvalException(
            null, String.format("Unable to parse toolchain %s: %s", rawLabel, e.getMessage()), e);
      }
    }

    return requiredToolchains.build();
  }

  private static ImmutableList<Label> collectConstraintLabels(
      Iterable<String> rawLabels, ImmutableMap<RepositoryName, RepositoryName> mapping)
      throws EvalException {
    ImmutableList.Builder<Label> constraintLabels = new ImmutableList.Builder<>();
    for (String rawLabel : rawLabels) {
      try {
        Label constraintLabel = Label.parseAbsolute(rawLabel, mapping);
        constraintLabels.add(constraintLabel);
      } catch (LabelSyntaxException e) {
        throw Starlark.errorf("Unable to parse constraint %s: %s", rawLabel, e.getMessage());
      }
    }

    return constraintLabels.build();
  }

  @Override
  public SkylarkAspect aspect(
      StarlarkFunction implementation,
      Sequence<?> attributeAspects,
      Object attrs,
      Sequence<?> requiredAspectProvidersArg,
      Sequence<?> providesArg,
      Sequence<?> fragments,
      Sequence<?> hostFragments,
      Sequence<?> toolchains,
      String doc,
      Boolean applyToGeneratingRules,
      StarlarkThread thread)
      throws EvalException {
    ImmutableList.Builder<String> attrAspects = ImmutableList.builder();
    for (Object attributeAspect : attributeAspects) {
      String attrName = STRING.convert(attributeAspect, "attr_aspects");

      if (attrName.equals("*") && attributeAspects.size() != 1) {
        throw new EvalException(null, "'*' must be the only string in 'attr_aspects' list");
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

    ImmutableList<Pair<String, SkylarkAttr.Descriptor>> descriptors =
        attrObjectToAttributesList(attrs);
    ImmutableList.Builder<Attribute> attributes = ImmutableList.builder();
    ImmutableSet.Builder<String> requiredParams = ImmutableSet.builder();
    for (Pair<String, Descriptor> nameDescriptorPair : descriptors) {
      String nativeName = nameDescriptorPair.first;
      boolean hasDefault = nameDescriptorPair.second.hasDefault();
      Attribute attribute = nameDescriptorPair.second.build(nameDescriptorPair.first);
      if (attribute.getType() == Type.STRING
          && ((String) attribute.getDefaultValue(null)).isEmpty()) {
        hasDefault = false; // isValueSet() is always true for attr.string.
      }
      if (!Attribute.isImplicit(nativeName) && !Attribute.isLateBound(nativeName)) {
        if (!attribute.checkAllowedValues() || attribute.getType() != Type.STRING) {
          throw new EvalException(
              null,
              String.format(
                  "Aspect parameter attribute '%s' must have type 'string' and use the "
                      + "'values' restriction.",
                  nativeName));
        }
        if (!hasDefault) {
          requiredParams.add(nativeName);
        } else {
          PredicateWithMessage<Object> allowed = attribute.getAllowedValues();
          Object defaultVal = attribute.getDefaultValue(null);
          if (!allowed.apply(defaultVal)) {
            throw new EvalException(
                null,
                String.format(
                    "Aspect parameter attribute '%s' has a bad default value: %s",
                    nativeName, allowed.getErrorReason(defaultVal)));
          }
        }
      } else if (!hasDefault) { // Implicit or late bound attribute
        String skylarkName = "_" + nativeName.substring(1);
        throw new EvalException(
            null, String.format("Aspect attribute '%s' has no default value.", skylarkName));
      }
      attributes.add(attribute);
    }

    for (Object o : providesArg) {
      if (!SkylarkAttr.isProvider(o)) {
        throw new EvalException(
            null,
            String.format(
                "Illegal argument: element in 'provides' is of unexpected type. "
                    + "Should be list of providers, but got item of type %s. ",
                EvalUtils.getDataTypeName(o, true)));
      }
    }
    return new SkylarkDefinedAspect(
        implementation,
        attrAspects.build(),
        attributes.build(),
        SkylarkAttr.buildProviderPredicate(requiredAspectProvidersArg, "required_aspect_providers"),
        SkylarkAttr.getSkylarkProviderIdentifiers(providesArg),
        requiredParams.build(),
        ImmutableSet.copyOf(fragments.getContents(String.class, "fragments")),
        HostTransition.INSTANCE,
        ImmutableSet.copyOf(hostFragments.getContents(String.class, "host_fragments")),
        collectToolchainLabels(
            toolchains.getContents(String.class, "toolchains"),
            BazelStarlarkContext.from(thread).getRepoMapping()),
        applyToGeneratingRules);
  }

  /**
   * The implementation for the magic function "rule" that creates Skylark rule classes.
   *
   * <p>Exactly one of {@link #builder} or {@link #ruleClass} is null except inside {@link #export}.
   */
  public static final class SkylarkRuleFunction extends BaseFunction
      implements SkylarkExportable, RuleFunction {
    private RuleClass.Builder builder;

    private RuleClass ruleClass;
    private final RuleClassType type;
    private ImmutableList<Pair<String, SkylarkAttr.Descriptor>> attributes;
    private final Location definitionLocation;
    private Label skylarkLabel;

    public SkylarkRuleFunction(
        RuleClass.Builder builder,
        RuleClassType type,
        ImmutableList<Pair<String, SkylarkAttr.Descriptor>> attributes,
        Location definitionLocation) {
      this.builder = builder;
      this.type = type;
      this.attributes = attributes;
      this.definitionLocation = definitionLocation;
    }

    /** This is for post-export reconstruction for serialization. */
    private SkylarkRuleFunction(
        RuleClass ruleClass, RuleClassType type, Location definitionLocation, Label skylarkLabel) {
      Preconditions.checkNotNull(
          ruleClass,
          "RuleClass must be non-null as this SkylarkRuleFunction should have been exported.");
      Preconditions.checkNotNull(
          skylarkLabel,
          "Label must be non-null as this SkylarkRuleFunction should have been exported.");
      this.ruleClass = ruleClass;
      this.type = type;
      this.definitionLocation = definitionLocation;
      this.skylarkLabel = skylarkLabel;
    }

    @Override
    public String getName() {
      return "rule";
    }

    @Override
    public FunctionSignature getSignature() {
      return FunctionSignature.KWARGS; // just for documentation
    }

    @Override
    public Object call(StarlarkThread thread, Tuple<Object> args, Dict<String, Object> kwargs)
        throws EvalException, InterruptedException, ConversionException {
      if (!args.isEmpty()) {
        throw new EvalException(null, "unexpected positional arguments");
      }
      BazelStarlarkContext.from(thread).checkLoadingPhase(getName());
      if (ruleClass == null) {
        throw new EvalException(null, "Invalid rule class hasn't been exported by a bzl file");
      }

      for (Attribute attribute : ruleClass.getAttributes()) {
        // TODO(dslomov): If a Skylark parameter extractor is specified for this aspect, its
        // attributes may not be required.
        for (Map.Entry<String, ImmutableSet<String>> attrRequirements :
            attribute.getRequiredAspectParameters().entrySet()) {
          for (String required : attrRequirements.getValue()) {
            if (!ruleClass.hasAttr(required, Type.STRING)) {
              throw Starlark.errorf(
                  "Aspect %s requires rule %s to specify attribute '%s' with type string.",
                  attrRequirements.getKey(), ruleClass.getName(), required);
            }
          }
        }
      }

      BuildLangTypedAttributeValuesMap attributeValues =
          new BuildLangTypedAttributeValuesMap(kwargs);
      try {
        PackageContext pkgContext = thread.getThreadLocal(PackageContext.class);
        if (pkgContext == null) {
          throw new EvalException(
              null,
              "Cannot instantiate a rule when loading a .bzl file. "
                  + "Rules may be instantiated only in a BUILD thread.");
        }
        RuleFactory.createAndAddRule(
            pkgContext,
            ruleClass,
            attributeValues,
            thread.getSemantics(),
            thread.getCallStack(),
            new AttributeContainer(ruleClass));
      } catch (InvalidRuleException | NameConflictException e) {
        throw new EvalException(null, e.getMessage());
      }
      return Starlark.NONE;
    }

    /** Export a RuleFunction from a Skylark file with a given name. */
    public void export(Label skylarkLabel, String ruleClassName) throws EvalException {
      Preconditions.checkState(ruleClass == null && builder != null);
      this.skylarkLabel = skylarkLabel;
      if (type == RuleClassType.TEST != TargetUtils.isTestRuleName(ruleClassName)) {
        throw new EvalException(
            definitionLocation,
            "Invalid rule class name '"
                + ruleClassName
                + "', test rule class names must end with '_test' and other rule classes must not");
      }
      // Thus far, we only know if we have a rule transition. While iterating through attributes,
      // check if we have an attribute transition.
      boolean hasStarlarkDefinedTransition = builder.hasStarlarkRuleTransition();
      boolean hasFunctionTransitionWhitelist = false;
      for (Pair<String, SkylarkAttr.Descriptor> attribute : attributes) {
        String name = attribute.getFirst();
        SkylarkAttr.Descriptor descriptor = attribute.getSecond();

        Attribute attr = descriptor.build(name);

        hasStarlarkDefinedTransition |= attr.hasStarlarkDefinedTransition();
        if (attr.hasAnalysisTestTransition()) {
          if (!builder.isAnalysisTest()) {
            throw new EvalException(
                getLocation(),
                "Only rule definitions with analysis_test=True may have attributes with "
                    + "analysis_test_transition transitions");
          }
          builder.setHasAnalysisTestTransition();
        }
        // Check for existence of the function transition whitelist attribute.
        // TODO(b/121385274): remove when we stop whitelisting starlark transitions
        if (name.equals(FunctionSplitTransitionWhitelist.WHITELIST_ATTRIBUTE_NAME)) {
          if (!BuildType.isLabelType(attr.getType())) {
            throw new EvalException(
                getLocation(), "_whitelist_function_transition attribute must be a label type");
          }
          if (attr.getDefaultValueUnchecked() == null) {
            throw new EvalException(
                getLocation(),
                "_whitelist_function_transition attribute must have a default value");
          }
          Label defaultLabel = (Label) attr.getDefaultValueUnchecked();
          // Check the label value for package and target name, to make sure this works properly
          // in Bazel where it is expected to be found under @bazel_tools.
          if (!defaultLabel
                  .getPackageName()
                  .equals(FunctionSplitTransitionWhitelist.WHITELIST_LABEL.getPackageName())
              || !defaultLabel
                  .getName()
                  .equals(FunctionSplitTransitionWhitelist.WHITELIST_LABEL.getName())) {
            throw new EvalException(
                getLocation(),
                "_whitelist_function_transition attribute ("
                    + defaultLabel
                    + ") does not have the expected value "
                    + FunctionSplitTransitionWhitelist.WHITELIST_LABEL);
          }
          hasFunctionTransitionWhitelist = true;
          builder.setHasFunctionTransitionWhitelist();
        }
        addAttribute(builder, attr);
      }
      // TODO(b/121385274): remove when we stop whitelisting starlark transitions
      if (hasStarlarkDefinedTransition) {
        if (!hasFunctionTransitionWhitelist) {
          throw new EvalException(
              getLocation(),
              String.format(
                  "Use of Starlark transition without whitelist attribute"
                      + " '_whitelist_function_transition'. See Starlark transitions documentation"
                      + " for details and usage: %s %s",
                  builder.getRuleDefinitionEnvironmentLabel(), builder.getType()));
        }
      } else {
        if (hasFunctionTransitionWhitelist) {
          throw new EvalException(
              getLocation(),
              String.format(
                  "Unused function-based split transition whitelist: %s %s",
                  builder.getRuleDefinitionEnvironmentLabel(), builder.getType()));
        }
      }

      try {
        this.ruleClass = builder.build(ruleClassName, skylarkLabel + "%" + ruleClassName);
      } catch (IllegalArgumentException | IllegalStateException ex) {
        throw new EvalException(getLocation(), ex);
      }

      this.builder = null;
      this.attributes = null;
    }

    public RuleClass getRuleClass() {
      Preconditions.checkState(ruleClass != null && builder == null);
      return ruleClass;
    }

    @Override
    public boolean isExported() {
      return skylarkLabel != null;
    }

    @Override
    public void repr(Printer printer) {
      printer.append("<rule>");
    }
  }

  @Override
  public Label label(String labelString, Boolean relativeToCallerRepository, StarlarkThread thread)
      throws EvalException {
    BazelStarlarkContext context = BazelStarlarkContext.from(thread);

    // This function is surprisingly complex.
    //
    // Doc:
    // "When relative_to_caller_repository is True and the calling thread is a
    // rule's implementation function, then a repo-relative label //foo:bar is
    // resolved relative to the rule's repository. For calls to Label from any
    // other thread, or calls in which the relative_to_caller_repository flag is
    // False, a repo-relative label is resolved relative to the file in which the
    // Label() call appears.)"
    //
    // - The "and" conjunction in first line of the doc above doesn't match the code.
    //   There are three cases to consider, not two, as parentLabel can be null or
    //   in the relativeToCallerRepository branch.
    //   Thus in a loading phase thread with relativeToCallerRepository=True,
    //   the repo mapping is (I suspect) erroneously skipped.
    //   TODO(adonovan): verify, and file a doc bug if so.
    //
    // - The deprecated relative_to_caller_repository semantics can be explained
    //   as thread-local state, something we've embraced elsewhere in the build language.
    //   (For example, in the loading phase, calling cc_binary creates a rule in the
    //   package associated with the calling thread.)
    //
    //   By contrast, the default relative_to_caller_repository=False semantics
    //   are more magical, using dynamic scope: introspection on the call stack.
    //   This is an obstacle to removing GlobalFrame.
    //
    //   An alternative way to implement that would be to say that each BUILD/.bzl file
    //   has its own function value called Label that is a closure over the current
    //   file label. (That would mean that if you export the Label function from a.bzl
    //   file and load it into b.bzl, it would behave differently from the Label function
    //   predeclared in b.bzl, so the choice of implementation strategy is observable.
    //   However this case is not important in practice.)
    //   TODO(adonovan): use this alternative implementation.
    //
    // - Logically all we really need from this process is a RepoID, not a Label
    //   or PackageID, but the Label class doesn't yet have the necessary primitives.
    //   TODO(adonovan): augment the Label class.
    //
    // - When repository mapping does occur, the result is converted back to a string
    //   "unambiguous" canonical form and then parsed again by the cache, with
    //   no repo mapping.
    //   TODO(adonovan): augment the Label class so that we can validate, remap,
    //   and cache without needing four allocations (parseAbsoluteLabel,
    //   getRelativeWithRemapping, getUnambiguousCanonicalForm, parseAbsoluteLabel
    //   in labelCache)

    Label parentLabel;
    if (relativeToCallerRepository) {
      // This is the label of the rule, if this is an analysis-phase
      // rule or aspect implementation thread, or null otherwise.
      parentLabel = context.getAnalysisRuleLabel();
    } else {
      // This is the label of the innermost BUILD/.bzl file on the current call stack.
      parentLabel = (Label) Module.ofInnermostEnclosingStarlarkFunction(thread).getLabel();
    }

    try {
      if (parentLabel != null) {
        LabelValidator.parseAbsoluteLabel(labelString);
        labelString =
            parentLabel
                .getRelativeWithRemapping(labelString, context.getRepoMapping())
                .getUnambiguousCanonicalForm();
      }
      return labelCache.get(labelString);
    } catch (LabelValidator.BadLabelException | LabelSyntaxException | ExecutionException e) {
      throw Starlark.errorf("Illegal absolute label syntax: %s", labelString);
    }
  }
}
