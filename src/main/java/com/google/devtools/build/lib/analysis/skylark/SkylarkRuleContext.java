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

import static com.google.devtools.build.lib.analysis.config.transitions.ConfigurationTransition.PATCH_TRANSITION_KEY;
import static com.google.devtools.build.lib.packages.RuleClass.Builder.SKYLARK_BUILD_SETTING_DEFAULT_ATTR_NAME;

import com.google.common.base.Function;
import com.google.common.base.Optional;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Ordering;
import com.google.common.hash.Hashing;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.analysis.ActionsProvider;
import com.google.devtools.build.lib.analysis.AliasProvider;
import com.google.devtools.build.lib.analysis.BashCommandConstructor;
import com.google.devtools.build.lib.analysis.CommandHelper;
import com.google.devtools.build.lib.analysis.ConfigurationMakeVariableContext;
import com.google.devtools.build.lib.analysis.DefaultInfo;
import com.google.devtools.build.lib.analysis.FileProvider;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.LabelExpander;
import com.google.devtools.build.lib.analysis.LabelExpander.NotUniqueExpansionException;
import com.google.devtools.build.lib.analysis.LocationExpander;
import com.google.devtools.build.lib.analysis.ResolvedToolchainContext;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.Runfiles;
import com.google.devtools.build.lib.analysis.RunfilesProvider;
import com.google.devtools.build.lib.analysis.ShToolchain;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.FragmentCollection;
import com.google.devtools.build.lib.analysis.config.HostTransition;
import com.google.devtools.build.lib.analysis.config.transitions.NoTransition;
import com.google.devtools.build.lib.analysis.configuredtargets.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.analysis.platform.ConstraintValueInfo;
import com.google.devtools.build.lib.analysis.stringtemplate.ExpansionException;
import com.google.devtools.build.lib.analysis.test.InstrumentedFilesCollector;
import com.google.devtools.build.lib.analysis.test.InstrumentedFilesInfo;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.packages.Aspect;
import com.google.devtools.build.lib.packages.AspectDescriptor;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.ImplicitOutputsFunction;
import com.google.devtools.build.lib.packages.ImplicitOutputsFunction.SkylarkImplicitOutputsFunction;
import com.google.devtools.build.lib.packages.OutputFile;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.Provider;
import com.google.devtools.build.lib.packages.RawAttributeMapper;
import com.google.devtools.build.lib.packages.RuleClass.ConfiguredTargetFactory.RuleErrorException;
import com.google.devtools.build.lib.packages.StructImpl;
import com.google.devtools.build.lib.packages.StructProvider;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.packages.Type.ConversionException;
import com.google.devtools.build.lib.packages.Type.LabelClass;
import com.google.devtools.build.lib.shell.ShellUtils;
import com.google.devtools.build.lib.shell.ShellUtils.TokenizationException;
import com.google.devtools.build.lib.skylarkbuildapi.FileApi;
import com.google.devtools.build.lib.skylarkbuildapi.SkylarkRuleContextApi;
import com.google.devtools.build.lib.syntax.ClassObject;
import com.google.devtools.build.lib.syntax.Depset;
import com.google.devtools.build.lib.syntax.Dict;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.EvalUtils;
import com.google.devtools.build.lib.syntax.NoneType;
import com.google.devtools.build.lib.syntax.Printer;
import com.google.devtools.build.lib.syntax.Sequence;
import com.google.devtools.build.lib.syntax.Starlark;
import com.google.devtools.build.lib.syntax.StarlarkList;
import com.google.devtools.build.lib.syntax.StarlarkSemantics;
import com.google.devtools.build.lib.syntax.StarlarkThread;
import com.google.devtools.build.lib.syntax.StarlarkValue;
import com.google.devtools.build.lib.syntax.Tuple;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import javax.annotation.Nullable;

/**
 * A Skylark API for the ruleContext.
 *
 * <p>"This object becomes featureless once the rule implementation function that it was created for
 * has completed. To achieve this, the {@link #nullify()} should be called once the evaluation of
 * the function is completed. The method both frees memory by deleting all significant fields of the
 * object and makes it impossible to accidentally use this object where it's not supposed to be used
 * (such attempts will result in {@link EvalException}s).
 */
public final class SkylarkRuleContext implements SkylarkRuleContextApi<ConstraintValueInfo> {

  public static final Function<Attribute, Object> ATTRIBUTE_VALUE_EXTRACTOR_FOR_ASPECT =
      new Function<Attribute, Object>() {
        @Nullable
        @Override
        public Object apply(Attribute attribute) {
          return attribute.getDefaultValue(null);
        }
      };
  public static final String EXECUTABLE_OUTPUT_NAME = "executable";

  // This field is a copy of the info from ruleContext, stored separately so it can be accessed
  // after this object has been nullified.
  private final String ruleLabelCanonicalName;

  private final boolean isForAspect;

  private final SkylarkActionFactory actionFactory;

  // The fields below intended to be final except that they can be cleared by calling `nullify()`
  // when the object becomes featureless.
  private RuleContext ruleContext;
  private FragmentCollection fragments;
  private FragmentCollection hostFragments;
  private AspectDescriptor aspectDescriptor;
  private final StarlarkSemantics starlarkSemantics;

  private Dict<String, String> makeVariables;
  private SkylarkAttributesCollection attributesCollection;
  private SkylarkAttributesCollection ruleAttributesCollection;
  private StructImpl splitAttributes;

  // TODO(bazel-team): we only need this because of the css_binary rule.
  private ImmutableMap<Artifact, Label> artifactsLabelMap;
  private Outputs outputsObject;

  /**
   * Creates a new SkylarkRuleContext using ruleContext.
   *
   * @param aspectDescriptor aspect for which the context is created, or <code>null</code> if it is
   *     for a rule.
   * @throws InterruptedException
   */
  public SkylarkRuleContext(
      RuleContext ruleContext,
      @Nullable AspectDescriptor aspectDescriptor,
      StarlarkSemantics starlarkSemantics)
      throws EvalException, InterruptedException, RuleErrorException {
    this.actionFactory = new SkylarkActionFactory(this, starlarkSemantics, ruleContext);
    this.ruleContext = Preconditions.checkNotNull(ruleContext);
    this.ruleLabelCanonicalName = ruleContext.getLabel().getCanonicalForm();
    this.fragments = new FragmentCollection(ruleContext, NoTransition.INSTANCE);
    this.hostFragments = new FragmentCollection(ruleContext, HostTransition.INSTANCE);
    this.aspectDescriptor = aspectDescriptor;
    this.starlarkSemantics = starlarkSemantics;

    if (aspectDescriptor == null) {
      this.isForAspect = false;
      Collection<Attribute> attributes = ruleContext.getRule().getAttributes();
      Outputs outputs = new Outputs(this);

      ImplicitOutputsFunction implicitOutputsFunction =
          ruleContext.getRule().getImplicitOutputsFunction();

      if (implicitOutputsFunction instanceof SkylarkImplicitOutputsFunction) {
        SkylarkImplicitOutputsFunction func =
            (SkylarkImplicitOutputsFunction) implicitOutputsFunction;
        for (Map.Entry<String, String> entry :
            func.calculateOutputs(
                    ruleContext.getAnalysisEnvironment().getEventHandler(),
                    RawAttributeMapper.of(ruleContext.getRule()))
                .entrySet()) {
          outputs.addOutput(
              entry.getKey(),
              ruleContext.getImplicitOutputArtifact(entry.getValue()));
        }
      }

      ImmutableMap.Builder<Artifact, Label> artifactLabelMapBuilder = ImmutableMap.builder();
      for (Attribute a : attributes) {
        String attrName = a.getName();
        Type<?> type = a.getType();
        if (type.getLabelClass() != LabelClass.OUTPUT) {
          continue;
        }
        ImmutableList.Builder<Artifact> artifactsBuilder = ImmutableList.builder();
        for (OutputFile outputFile : ruleContext.getRule().getOutputFileMap().get(attrName)) {
          Artifact artifact = ruleContext.createOutputArtifact(outputFile);
          artifactsBuilder.add(artifact);
          artifactLabelMapBuilder.put(artifact, outputFile.getLabel());
        }
        ImmutableList<Artifact> artifacts = artifactsBuilder.build();

        if (type == BuildType.OUTPUT) {
          if (artifacts.size() == 1) {
            outputs.addOutput(attrName, Iterables.getOnlyElement(artifacts));
          } else {
            outputs.addOutput(attrName, Starlark.NONE);
          }
        } else if (type == BuildType.OUTPUT_LIST) {
          outputs.addOutput(attrName, StarlarkList.immutableCopyOf(artifacts));
        } else {
          throw new IllegalArgumentException(
              "Type of " + attrName + "(" + type + ") is not output type ");
        }
      }

      this.artifactsLabelMap = artifactLabelMapBuilder.build();
      this.outputsObject = outputs;

      SkylarkAttributesCollection.Builder builder = SkylarkAttributesCollection.builder(this);
      for (Attribute attribute : ruleContext.getRule().getAttributes()) {
        Object value = ruleContext.attributes().get(attribute.getName(), attribute.getType());
        builder.addAttribute(attribute, value);
      }

      this.attributesCollection = builder.build();
      this.splitAttributes = buildSplitAttributeInfo(attributes, ruleContext);
      this.ruleAttributesCollection = null;
    } else { // ASPECT
      this.isForAspect = true;
      this.artifactsLabelMap = ImmutableMap.of();
      this.outputsObject = null;

      ImmutableCollection<Attribute> attributes =
          ruleContext.getMainAspect().getDefinition().getAttributes().values();
      SkylarkAttributesCollection.Builder aspectBuilder = SkylarkAttributesCollection.builder(this);
      for (Attribute attribute : attributes) {
        aspectBuilder.addAttribute(attribute, attribute.getDefaultValue(null));
      }
      this.attributesCollection = aspectBuilder.build();

      this.splitAttributes = null;
      SkylarkAttributesCollection.Builder ruleBuilder = SkylarkAttributesCollection.builder(this);

      for (Attribute attribute : ruleContext.getRule().getAttributes()) {
        Object value = ruleContext.attributes().get(attribute.getName(), attribute.getType());
        ruleBuilder.addAttribute(attribute, value);
      }
      for (Aspect aspect : ruleContext.getAspects()) {
        if (aspect.equals(ruleContext.getMainAspect())) {
          // Aspect's own attributes are in <code>attributesCollection</code>.
          continue;
        }
        for (Attribute attribute : aspect.getDefinition().getAttributes().values()) {
          ruleBuilder.addAttribute(attribute, attribute.getDefaultValue(null));
        }
      }

      this.ruleAttributesCollection = ruleBuilder.build();
    }

    try {
      makeVariables = ruleContext.getConfigurationMakeVariableContext().collectMakeVariables();
    } catch (ExpansionException e) {
      throw ruleContext.throwWithRuleError(e);
    }
  }

  /**
   * Represents `ctx.outputs`.
   *
   * <p>A {@link ClassObject} (struct-like data structure) with "executable" field created lazily
   * on-demand.
   *
   * <p>Note: There is only one {@code Outputs} object per rule context, so default (object
   * identity) equals and hashCode suffice.
   */
  private static class Outputs implements ClassObject, StarlarkValue {
    private final Map<String, Object> outputs;
    private final SkylarkRuleContext context;
    private boolean executableCreated = false;

    public Outputs(SkylarkRuleContext context) {
      this.outputs = new LinkedHashMap<>();
      this.context = context;
    }

    private void addOutput(String key, Object value)
        throws EvalException {
      Preconditions.checkState(!context.isImmutable(),
          "Cannot add outputs to immutable Outputs object");
      if (outputs.containsKey(key)
          || (context.isExecutable() && EXECUTABLE_OUTPUT_NAME.equals(key))) {
        throw new EvalException(null, "Multiple outputs with the same key: " + key);
      }
      outputs.put(key, value);
    }

    @Override
    public boolean isImmutable() {
      return context.isImmutable();
    }

    @Override
    public ImmutableCollection<String> getFieldNames() {
      ImmutableList.Builder<String> result = ImmutableList.builder();
      if (context.isExecutable() && executableCreated) {
        result.add(EXECUTABLE_OUTPUT_NAME);
      }
      result.addAll(outputs.keySet());
      return result.build();
    }

    @Nullable
    @Override
    public Object getValue(String name) throws EvalException {
      checkMutable();
      if (context.isExecutable() && EXECUTABLE_OUTPUT_NAME.equals(name)) {
        executableCreated = true;
        // createOutputArtifact() will cache the created artifact.
        return context.getRuleContext().createOutputArtifact();
      }

      return outputs.get(name);
    }

    @Nullable
    @Override
    public String getErrorMessageForUnknownField(String name) {
      return String.format(
          "No attribute '%s' in outputs. Make sure you declared a rule output with this name.",
          name);
    }

    @Override
    public void repr(Printer printer) {
      if (isImmutable()) {
        printer.append("ctx.outputs(for ");
        printer.append(context.ruleLabelCanonicalName);
        printer.append(")");
        return;
      }
      boolean first = true;
      printer.append("ctx.outputs(");
      // Sort by field name to ensure deterministic output.
      try {
        for (String field : Ordering.natural().sortedCopy(getFieldNames())) {
          if (!first) {
            printer.append(", ");
          }
          first = false;
          printer.append(field);
          printer.append(" = ");
          printer.repr(getValue(field));
        }
        printer.append(")");
      } catch (EvalException e) {
        throw new AssertionError("mutable ctx.outputs should not throw", e);
      }
    }

    private void checkMutable() throws EvalException {
      if (isImmutable()) {
        throw new EvalException(
            null,
            String.format(
                "cannot access outputs of rule '%s' outside of its own "
                    + "rule implementation function",
                context.ruleLabelCanonicalName));
      }
    }

  }

  public boolean isExecutable() {
    return ruleContext.getRule().getRuleClassObject().isExecutableSkylark();
  }

  public boolean isDefaultExecutableCreated() {
    return this.outputsObject.executableCreated;
  }


  /**
   * Nullifies fields of the object when it's not supposed to be used anymore to free unused memory
   * and to make sure this object is not accessed when it's not supposed to (after the corresponding
   * rule implementation function has exited).
   */
  public void nullify() {
    actionFactory.nullify();
    ruleContext = null;
    fragments = null;
    hostFragments = null;
    aspectDescriptor = null;
    makeVariables = null;
    attributesCollection = null;
    ruleAttributesCollection = null;
    splitAttributes = null;
    artifactsLabelMap = null;
    outputsObject = null;
  }

  public void checkMutable(String attrName) throws EvalException {
    if (isImmutable()) {
      throw new EvalException(null, String.format(
          "cannot access field or method '%s' of rule context for '%s' outside of its own rule " 
              + "implementation function", attrName, ruleLabelCanonicalName));
    }
  }

  @Nullable
  public AspectDescriptor getAspectDescriptor() {
    return aspectDescriptor;
  }

  public String getRuleLabelCanonicalName() {
    return ruleLabelCanonicalName;
  }

  private static StructImpl buildSplitAttributeInfo(
      Collection<Attribute> attributes, RuleContext ruleContext) {

    ImmutableMap.Builder<String, Object> splitAttrInfos = ImmutableMap.builder();
    for (Attribute attr : attributes) {
      if (!attr.getTransitionFactory().isSplit()) {
        continue;
      }
      Map<Optional<String>, ? extends List<? extends TransitiveInfoCollection>> splitPrereqs =
          ruleContext.getSplitPrerequisites(attr.getName());

      Map<Object, Object> splitPrereqsMap = new LinkedHashMap<>();
      for (Map.Entry<Optional<String>, ? extends List<? extends TransitiveInfoCollection>>
          splitPrereq : splitPrereqs.entrySet()) {

        // Skip a split with an empty dependency list.
        // TODO(jungjw): Figure out exactly which cases trigger this and see if this can be made
        // more error-proof.
        if (splitPrereq.getValue().isEmpty()) {
          continue;
        }

        Object value;
        if (attr.getType() == BuildType.LABEL) {
          Preconditions.checkState(splitPrereq.getValue().size() == 1);
          value = splitPrereq.getValue().get(0);
        } else {
          // BuildType.LABEL_LIST
          value = StarlarkList.immutableCopyOf(splitPrereq.getValue());
        }

        if (splitPrereq.getKey().isPresent()
            && !splitPrereq.getKey().get().equals(PATCH_TRANSITION_KEY)) {
          splitPrereqsMap.put(splitPrereq.getKey().get(), value);
        } else {
          // If the split transition is not in effect, then the key will be missing since there's
          // nothing to key on because the dependencies aren't split and getSplitPrerequisites()
          // behaves like getPrerequisites(). This also means there should be only one entry in
          // the map. Use None in Skylark to represent this.
          Preconditions.checkState(splitPrereqs.size() == 1);
          splitPrereqsMap.put(Starlark.NONE, value);
        }
      }

      splitAttrInfos.put(attr.getPublicName(), Dict.copyOf(null, splitPrereqsMap));
    }

    return StructProvider.STRUCT.create(
        splitAttrInfos.build(),
        "No attribute '%s' in split_attr."
            + " This attribute is not defined with a split configuration.");
  }

  @Override
  public boolean isImmutable() {
    return ruleContext == null;
  }

  @Override
  public void repr(Printer printer) {
    if (isForAspect) {
      printer.append("<aspect context for " + ruleLabelCanonicalName + ">");
    } else {
      printer.append("<rule context for " + ruleLabelCanonicalName + ">");
    }
  }

  /**
   * Returns the original ruleContext.
   */
  public RuleContext getRuleContext() {
    return ruleContext;
  }

  @Override
  public Provider getDefaultProvider() {
    return DefaultInfo.PROVIDER;
  }

  @Override
  public SkylarkActionFactory actions() {
    return actionFactory;
  }

  @Override
  public StarlarkValue createdActions() throws EvalException {
    checkMutable("created_actions");
    if (ruleContext.getRule().getRuleClassObject().isSkylarkTestable()) {
      return ActionsProvider.create(
          ruleContext.getAnalysisEnvironment().getRegisteredActions());
    } else {
      return Starlark.NONE;
    }
  }

  @Override
  public StructImpl getAttr() throws EvalException {
    checkMutable("attr");
    return attributesCollection.getAttr();
  }

  @Override
  public StructImpl getSplitAttr() throws EvalException {
    checkMutable("split_attr");
    if (splitAttributes == null) {
      throw new EvalException(
          Location.BUILTIN, "'split_attr' is available only in rule implementations");
    }
    return splitAttributes;
  }

  /** See {@link RuleContext#getExecutablePrerequisite(String, Mode)}. */
  @Override
  public StructImpl getExecutable() throws EvalException {
    checkMutable("executable");
    return attributesCollection.getExecutable();
  }

  /** See {@link RuleContext#getPrerequisiteArtifact(String, Mode)}. */
  @Override
  public StructImpl getFile() throws EvalException {
    checkMutable("file");
    return attributesCollection.getFile();
  }

  /** See {@link RuleContext#getPrerequisiteArtifacts(String, Mode)}. */
  @Override
  public StructImpl getFiles() throws EvalException {
    checkMutable("files");
    return attributesCollection.getFiles();
  }

  @Override
  public String getWorkspaceName() throws EvalException {
    checkMutable("workspace_name");
    return ruleContext.getWorkspaceName();
  }

  @Override
  public Label getLabel() throws EvalException {
    checkMutable("label");
    return ruleContext.getLabel();
  }

  @Override
  public FragmentCollection getFragments() throws EvalException {
    checkMutable("fragments");
    return fragments;
  }

  @Override
  public FragmentCollection getHostFragments() throws EvalException {
    checkMutable("host_fragments");
    return hostFragments;
  }

  @Override
  public BuildConfiguration getConfiguration() throws EvalException {
    checkMutable("configuration");
    return ruleContext.getConfiguration();
  }

  @Override
  public BuildConfiguration getHostConfiguration() throws EvalException {
    checkMutable("host_configuration");
    return ruleContext.getHostConfiguration();
  }

  @Override
  @Nullable
  public Object getBuildSettingValue() throws EvalException {
    if (ruleContext.getRule().getRuleClassObject().getBuildSetting() == null) {
      throw new EvalException(
          Location.BUILTIN,
          String.format(
              "attempting to access 'build_setting_value' of non-build setting %s",
              ruleLabelCanonicalName));
    }
    ImmutableMap<Label, Object> skylarkFlagSettings =
        ruleContext.getConfiguration().getOptions().getStarlarkOptions();

    Type<?> buildSettingType =
        ruleContext.getRule().getRuleClassObject().getBuildSetting().getType();
    if (skylarkFlagSettings.containsKey(ruleContext.getLabel())) {
      return skylarkFlagSettings.get(ruleContext.getLabel());
    } else {
      return ruleContext
          .attributes()
          .get(SKYLARK_BUILD_SETTING_DEFAULT_ATTR_NAME, buildSettingType);
    }
  }

  @Override
  public boolean instrumentCoverage(Object targetUnchecked) throws EvalException {
    checkMutable("coverage_instrumented");
    BuildConfiguration config = ruleContext.getConfiguration();
    if (!config.isCodeCoverageEnabled()) {
      return false;
    }
    if (targetUnchecked == Starlark.NONE) {
      return InstrumentedFilesCollector.shouldIncludeLocalSources(
          ruleContext.getConfiguration(), ruleContext.getLabel(), ruleContext.isTestTarget());
    }
    TransitiveInfoCollection target = (TransitiveInfoCollection) targetUnchecked;
    return (target.get(InstrumentedFilesInfo.SKYLARK_CONSTRUCTOR) != null)
        && InstrumentedFilesCollector.shouldIncludeLocalSources(config, target);
  }

  @Override
  public ImmutableList<String> getFeatures() throws EvalException {
    checkMutable("features");
    return ImmutableList.copyOf(ruleContext.getFeatures());
  }

  @Override
  public ImmutableList<String> getDisabledFeatures() throws EvalException {
    checkMutable("disabled_features");
    return ImmutableList.copyOf(ruleContext.getDisabledFeatures());
  }

  @Override
  public ArtifactRoot getBinDirectory() throws EvalException {
    checkMutable("bin_dir");
    return getConfiguration().getBinDirectory(ruleContext.getRule().getRepository());
  }

  @Override
  public ArtifactRoot getGenfilesDirectory() throws EvalException {
    checkMutable("genfiles_dir");
    return getConfiguration().getGenfilesDirectory(ruleContext.getRule().getRepository());
  }

  @Override
  public ClassObject outputs() throws EvalException {
    checkMutable("outputs");
    if (outputsObject == null) {
      throw new EvalException(Location.BUILTIN, "'outputs' is not defined");
    }
    return outputsObject;
  }

  @Override
  public SkylarkAttributesCollection rule() throws EvalException {
    checkMutable("rule");
    if (!isForAspect) {
      throw new EvalException(
          Location.BUILTIN, "'rule' is only available in aspect implementations");
    }
    return ruleAttributesCollection;
  }

  @Override
  public ImmutableList<String> aspectIds() throws EvalException {
    checkMutable("aspect_ids");
    if (!isForAspect) {
      throw new EvalException(
          Location.BUILTIN, "'aspect_ids' is only available in aspect implementations");
    }

    ImmutableList.Builder<String> result = ImmutableList.builder();
    for (AspectDescriptor descriptor : ruleContext.getAspectDescriptors()) {
      result.add(descriptor.getDescription());
    }
    return result.build();
  }

  @Override
  public Dict<String, String> var() throws EvalException {
    checkMutable("var");
    return makeVariables;
  }

  @Override
  public ResolvedToolchainContext toolchains() throws EvalException {
    checkMutable("toolchains");
    return ruleContext.getToolchainContext();
  }

  @Override
  public boolean targetPlatformHasConstraint(ConstraintValueInfo constraintValue) {
    return ruleContext.targetPlatformHasConstraint(constraintValue);
  }

  @Override
  public String toString() {
    return ruleLabelCanonicalName;
  }

  @Override
  public Sequence<String> tokenize(String optionString) throws EvalException {
    checkMutable("tokenize");
    List<String> options = new ArrayList<>();
    try {
      ShellUtils.tokenize(options, optionString);
    } catch (TokenizationException e) {
      throw new EvalException(null, e.getMessage() + " while tokenizing '" + optionString + "'");
    }
    return StarlarkList.immutableCopyOf(options);
  }

  @Override
  public String expand(
      @Nullable String expression,
      Sequence<?> artifacts, // <Artifact>
      Label labelResolver)
      throws EvalException {
    checkMutable("expand");
    try {
      Map<Label, Iterable<Artifact>> labelMap = new HashMap<>();
      for (Artifact artifact : artifacts.getContents(Artifact.class, "artifacts")) {
        labelMap.put(artifactsLabelMap.get(artifact), ImmutableList.of(artifact));
      }
      return LabelExpander.expand(expression, labelMap, labelResolver);
    } catch (NotUniqueExpansionException e) {
      throw new EvalException(null, e.getMessage() + " while expanding '" + expression + "'");
    }
  }

  boolean isForAspect() {
    return isForAspect;
  }

  @Override
  public Artifact newFile(Object var1, Object var2, Object fileSuffix) throws EvalException {
    checkDeprecated("ctx.actions.declare_file", "ctx.new_file", starlarkSemantics);
    checkMutable("new_file");

    // Determine which of new_file's four signatures is being used. Yes, this is terrible.
    // It's one major reason that this method is deprecated.
    if (fileSuffix != Starlark.UNBOUND) {
      // new_file(file_root, sibling_file, suffix)
      ArtifactRoot root =
          assertTypeForNewFile(
              var1, ArtifactRoot.class, "expected first param to be of type 'root'");
      Artifact siblingFile =
          assertTypeForNewFile(var2, Artifact.class, "expected second param to be of type 'File'");
      PathFragment original = siblingFile.getRootRelativePath();
      PathFragment fragment = original.replaceName(original.getBaseName() + fileSuffix);
      return ruleContext.getDerivedArtifact(fragment, root);

    } else if (var2 == Starlark.UNBOUND) {
      // new_file(filename)
      String filename =
          assertTypeForNewFile(var1, String.class, "expected first param to be of type 'string'");
      return actionFactory.declareFile(filename, Starlark.NONE);

    } else {
      String filename =
          assertTypeForNewFile(var2, String.class, "expected second param to be of type 'string'");
      if (var1 instanceof ArtifactRoot) {
        // new_file(root, filename)
        ArtifactRoot root = (ArtifactRoot) var1;

        return ruleContext.getPackageRelativeArtifact(filename, root);
      } else {
        // new_file(sibling_file, filename)
        Artifact siblingFile =
            assertTypeForNewFile(
                var1, Artifact.class, "expected first param to be of type 'File' or 'root'");

        return actionFactory.declareFile(filename, siblingFile);
      }
    }
  }

  private static <T> T assertTypeForNewFile(Object obj, Class<T> type, String errorMessage)
      throws EvalException {
    if (type.isInstance(obj)) {
      return type.cast(obj);
    } else {
      throw new EvalException(null, errorMessage);
    }
  }

  @Override
  public Artifact newDirectory(String name, Object siblingArtifactUnchecked) throws EvalException {
    checkDeprecated(
        "ctx.actions.declare_directory", "ctx.experimental_new_directory", starlarkSemantics);
    checkMutable("experimental_new_directory");
    return actionFactory.declareDirectory(name, siblingArtifactUnchecked);
  }

  @Override
  public boolean checkPlaceholders(String template, Sequence<?> allowedPlaceholders) // <String>
      throws EvalException {
    checkMutable("check_placeholders");
    List<String> actualPlaceHolders = new LinkedList<>();
    Set<String> allowedPlaceholderSet =
        ImmutableSet.copyOf(allowedPlaceholders.getContents(String.class, "allowed_placeholders"));
    ImplicitOutputsFunction.createPlaceholderSubstitutionFormatString(template, actualPlaceHolders);
    for (String placeholder : actualPlaceHolders) {
      if (!allowedPlaceholderSet.contains(placeholder)) {
        return false;
      }
    }
    return true;
  }

  @Override
  public String expandMakeVariables(
      String attributeName, String command, Dict<?, ?> additionalSubstitutions) // <String, String>
      throws EvalException {
    checkMutable("expand_make_variables");
    final Map<String, String> additionalSubstitutionsMap =
        additionalSubstitutions.getContents(String.class, String.class, "additional_substitutions");
    return expandMakeVariables(attributeName, command, additionalSubstitutionsMap);
  }

  private String expandMakeVariables(
      String attributeName, String command, final Map<String, String> additionalSubstitutionsMap) {
    ConfigurationMakeVariableContext makeVariableContext =
        new ConfigurationMakeVariableContext(
            this.getRuleContext(),
            ruleContext.getRule().getPackage(),
            ruleContext.getConfiguration(),
            ImmutableList.of()) {
          @Override
          public String lookupVariable(String variableName) throws ExpansionException {
            if (additionalSubstitutionsMap.containsKey(variableName)) {
              return additionalSubstitutionsMap.get(variableName);
            } else {
              return super.lookupVariable(variableName);
            }
          }
        };
    return ruleContext.getExpander(makeVariableContext).expand(attributeName, command);
  }

  FilesToRunProvider getExecutableRunfiles(Artifact executable) {
    return attributesCollection.getExecutableRunfilesMap().get(executable);
  }

  @Override
  public Artifact getStableWorkspaceStatus() throws InterruptedException, EvalException {
    checkMutable("info_file");
    return ruleContext.getAnalysisEnvironment().getStableWorkspaceStatusArtifact();
  }

  @Override
  public Artifact getVolatileWorkspaceStatus() throws InterruptedException, EvalException {
    checkMutable("version_file");
    return ruleContext.getAnalysisEnvironment().getVolatileWorkspaceStatusArtifact();
  }

  @Override
  public String getBuildFileRelativePath() throws EvalException {
    checkMutable("build_file_path");
    Package pkg = ruleContext.getRule().getPackage();
    return pkg.getSourceRoot().relativize(pkg.getBuildFile().getPath()).getPathString();
  }

  /**
   * A Skylark built-in function to create and register a SpawnAction using a dictionary of
   * parameters: action( inputs = [input1, input2, ...], outputs = [output1, output2, ...],
   * executable = executable, arguments = [argument1, argument2, ...], mnemonic = 'Mnemonic',
   * command = 'command', )
   */
  @Override
  public NoneType action(
      Sequence<?> outputs,
      Object inputs,
      Object executableUnchecked,
      Object toolsUnchecked,
      Object arguments,
      Object mnemonicUnchecked,
      Object commandUnchecked,
      Object progressMessage,
      Boolean useDefaultShellEnv,
      Object envUnchecked,
      Object executionRequirementsUnchecked,
      Object inputManifestsUnchecked,
      StarlarkThread thread)
      throws EvalException {
    checkDeprecated(
        "ctx.actions.run or ctx.actions.run_shell", "ctx.action", thread.getSemantics());
    checkMutable("action");
    if ((commandUnchecked == Starlark.NONE) == (executableUnchecked == Starlark.NONE)) {
      throw Starlark.errorf("You must specify either 'command' or 'executable' argument");
    }
    boolean hasCommand = commandUnchecked != Starlark.NONE;
    if (!hasCommand) {
      actions()
          .run(
              outputs,
              inputs,
              /*unusedInputsList=*/ Starlark.NONE,
              executableUnchecked,
              toolsUnchecked,
              arguments,
              mnemonicUnchecked,
              progressMessage,
              useDefaultShellEnv,
              envUnchecked,
              executionRequirementsUnchecked,
              inputManifestsUnchecked);

    } else {
      actions()
          .runShell(
              outputs,
              inputs,
              toolsUnchecked,
              arguments,
              mnemonicUnchecked,
              commandUnchecked,
              progressMessage,
              useDefaultShellEnv,
              envUnchecked,
              executionRequirementsUnchecked,
              inputManifestsUnchecked,
              thread);
    }
    return Starlark.NONE;
  }

  @Override
  public String expandLocation(String input, Sequence<?> targets, StarlarkThread thread)
      throws EvalException {
    checkMutable("expand_location");
    try {
      return LocationExpander.withExecPaths(
              getRuleContext(),
              makeLabelMap(targets.getContents(TransitiveInfoCollection.class, "targets")))
          .expand(input);
    } catch (IllegalStateException ise) {
      throw new EvalException(null, ise);
    }
  }

  @Override
  public NoneType fileAction(
      FileApi output, String content, Boolean executable, StarlarkThread thread)
      throws EvalException {
    checkDeprecated("ctx.actions.write", "ctx.file_action", thread.getSemantics());
    checkMutable("file_action");
    actions().write(output, content, executable);
    return Starlark.NONE;
  }

  @Override
  public NoneType emptyAction(String mnemonic, Object inputs, StarlarkThread thread)
      throws EvalException {
    checkDeprecated("ctx.actions.do_nothing", "ctx.empty_action", thread.getSemantics());
    checkMutable("empty_action");
    actions().doNothing(mnemonic, inputs);
    return Starlark.NONE;
  }

  @Override
  public NoneType templateAction(
      FileApi template,
      FileApi output,
      Dict<?, ?> substitutionsUnchecked,
      Boolean executable,
      StarlarkThread thread)
      throws EvalException {
    checkDeprecated("ctx.actions.expand_template", "ctx.template_action", thread.getSemantics());
    checkMutable("template_action");
    actions().expandTemplate(template, output, substitutionsUnchecked, executable);
    return Starlark.NONE;
  }

  @Override
  public Runfiles runfiles(
      Sequence<?> files,
      Object transitiveFiles,
      Boolean collectData,
      Boolean collectDefault,
      Dict<?, ?> symlinks,
      Dict<?, ?> rootSymlinks)
      throws EvalException, ConversionException {
    checkMutable("runfiles");
    Runfiles.Builder builder =
        new Runfiles.Builder(
            getRuleContext().getWorkspaceName(), getConfiguration().legacyExternalRunfiles());
    boolean checkConflicts = false;
    if (Starlark.truth(collectData)) {
      builder.addRunfiles(getRuleContext(), RunfilesProvider.DATA_RUNFILES);
    }
    if (Starlark.truth(collectDefault)) {
      builder.addRunfiles(getRuleContext(), RunfilesProvider.DEFAULT_RUNFILES);
    }
    if (!files.isEmpty()) {
      builder.addArtifacts(files.getContents(Artifact.class, "files"));
    }
    if (transitiveFiles != Starlark.NONE) {
      builder.addTransitiveArtifacts(
          ((Depset) transitiveFiles).getSetFromParam(Artifact.class, "transitive_files"));
    }
    if (!symlinks.isEmpty()) {
      // If Skylark code directly manipulates symlinks, activate more stringent validity checking.
      checkConflicts = true;
      for (Map.Entry<String, Artifact> entry :
          symlinks.getContents(String.class, Artifact.class, "symlinks").entrySet()) {
        builder.addSymlink(PathFragment.create(entry.getKey()), entry.getValue());
      }
    }
    if (!rootSymlinks.isEmpty()) {
      checkConflicts = true;
      for (Map.Entry<String, Artifact> entry :
          rootSymlinks.getContents(String.class, Artifact.class, "root_symlinks").entrySet()) {
        builder.addRootSymlink(PathFragment.create(entry.getKey()), entry.getValue());
      }
    }
    Runfiles runfiles = builder.build();
    if (checkConflicts) {
      runfiles.setConflictPolicy(Runfiles.ConflictPolicy.ERROR);
    }
    return runfiles;
  }

  @Override
  public Tuple<Object> resolveCommand(
      String command,
      Object attributeUnchecked,
      Boolean expandLocations,
      Object makeVariablesUnchecked,
      Sequence<?> tools,
      Dict<?, ?> labelDictUnchecked,
      Dict<?, ?> executionRequirementsUnchecked,
      StarlarkThread thread)
      throws ConversionException, EvalException {
    checkMutable("resolve_command");
    Label ruleLabel = getLabel();
    Map<Label, Iterable<Artifact>> labelDict = checkLabelDict(labelDictUnchecked);
    // The best way to fix this probably is to convert CommandHelper to Skylark.
    CommandHelper helper =
        CommandHelper.builder(getRuleContext())
            .addToolDependencies(tools.getContents(TransitiveInfoCollection.class, "tools"))
            .addLabelMap(labelDict)
            .build();
    String attribute = Type.STRING.convertOptional(attributeUnchecked, "attribute", ruleLabel);
    if (expandLocations) {
      command =
          helper.resolveCommandAndExpandLabels(command, attribute, /*allowDataInLabel=*/ false);
    }
    if (!EvalUtils.isNullOrNone(makeVariablesUnchecked)) {
      Map<String, String> makeVariables =
          Type.STRING_DICT.convert(makeVariablesUnchecked, "make_variables", ruleLabel);
      command = expandMakeVariables(attribute, command, makeVariables);
    }
    List<Artifact> inputs = new ArrayList<>();
    // TODO(lberki): This flattens a NestedSet.
    // However, we can't turn this into a Depset because it's an incompatible change to
    // Skylark.
    inputs.addAll(helper.getResolvedTools().toList());

    ImmutableMap<String, String> executionRequirements =
        ImmutableMap.copyOf(
            Dict.castSkylarkDictOrNoneToDict(
                executionRequirementsUnchecked,
                String.class,
                String.class,
                "execution_requirements"));
    PathFragment shExecutable = ShToolchain.getPathOrError(ruleContext);

    BashCommandConstructor constructor =
        CommandHelper.buildBashCommandConstructor(
            executionRequirements,
            shExecutable,
            // Hash the command-line to prevent multiple actions from the same rule invocation
            // conflicting with each other.
            "." + Hashing.murmur3_32().hashUnencodedChars(command).toString() + SCRIPT_SUFFIX);
    List<String> argv = helper.buildCommandLine(command, inputs, constructor);
    return Tuple.<Object>of(
        StarlarkList.copyOf(thread.mutability(), inputs),
        StarlarkList.copyOf(thread.mutability(), argv),
        helper.getToolsRunfilesSuppliers());
  }

  @Override
  public Tuple<Object> resolveTools(Sequence<?> tools) throws EvalException {
    checkMutable("resolve_tools");
    CommandHelper helper =
        CommandHelper.builder(getRuleContext())
            .addToolDependencies(tools.getContents(TransitiveInfoCollection.class, "tools"))
            .build();
    return Tuple.<Object>of(
        Depset.of(Artifact.TYPE, helper.getResolvedTools()), helper.getToolsRunfilesSuppliers());
  }

  public StarlarkSemantics getSkylarkSemantics() {
    return starlarkSemantics;
  }

  /**
   * Ensures the given {@link Map} has keys that have {@link Label} type and values that have either
   * {@link Iterable} or {@link Depset} type, and raises {@link EvalException} otherwise. Returns a
   * corresponding map where any sets are replaced by iterables.
   */
  // TODO(bazel-team): find a better way to typecheck this argument.
  private static Map<Label, Iterable<Artifact>> checkLabelDict(Map<?, ?> labelDict)
      throws EvalException {
    Map<Label, Iterable<Artifact>> convertedMap = new HashMap<>();
    for (Map.Entry<?, ?> entry : labelDict.entrySet()) {
      Object key = entry.getKey();
      if (!(key instanceof Label)) {
        throw Starlark.errorf("invalid key %s in 'label_dict'", Starlark.repr(key));
      }
      ImmutableList.Builder<Artifact> files = ImmutableList.builder();
      Object val = entry.getValue();
      Iterable<?> valIter;
      if (val instanceof Iterable) {
        valIter = (Iterable<?>) val;
      } else {
        throw Starlark.errorf(
            "invalid value %s in 'label_dict': expected iterable, but got '%s'",
            Starlark.repr(val), EvalUtils.getDataTypeName(val));
      }
      for (Object file : valIter) {
        if (!(file instanceof Artifact)) {
          throw Starlark.errorf("invalid value %s in 'label_dict'", Starlark.repr(val));
        }
        files.add((Artifact) file);
      }
      convertedMap.put((Label) key, files.build());
    }
    return convertedMap;
  }

  /** suffix of script to be used in case the command is too long to fit on a single line */
  private static final String SCRIPT_SUFFIX = ".script.sh";

  private static void checkDeprecated(String newApi, String oldApi, StarlarkSemantics semantics)
      throws EvalException {
    if (semantics.incompatibleNewActionsApi()) {
      throw Starlark.errorf(
          "Use %s instead of %s. \n"
              + "Use --incompatible_new_actions_api=false to temporarily disable this check.",
          newApi, oldApi);
    }
  }

  /**
   * Builds a map: Label -> List of files from the given labels
   *
   * @param knownLabels List of known labels
   * @return Immutable map with immutable collections as values
   */
  private static ImmutableMap<Label, ImmutableCollection<Artifact>> makeLabelMap(
      Iterable<TransitiveInfoCollection> knownLabels) {
    ImmutableMap.Builder<Label, ImmutableCollection<Artifact>> builder = ImmutableMap.builder();

    for (TransitiveInfoCollection current : knownLabels) {
      builder.put(
          AliasProvider.getDependencyLabel(current),
          current.getProvider(FileProvider.class).getFilesToBuild().toList());
    }

    return builder.build();
  }
}
