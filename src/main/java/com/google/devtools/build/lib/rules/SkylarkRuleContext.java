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
import com.google.common.base.Preconditions;
import com.google.common.cache.CacheBuilder;
import com.google.common.cache.CacheLoader;
import com.google.common.cache.LoadingCache;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Root;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.Attribute.ConfigurationTransition;
import com.google.devtools.build.lib.packages.ImplicitOutputsFunction;
import com.google.devtools.build.lib.packages.ImplicitOutputsFunction.SkylarkImplicitOutputsFunction;
import com.google.devtools.build.lib.packages.OutputFile;
import com.google.devtools.build.lib.packages.RawAttributeMapper;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.shell.ShellUtils;
import com.google.devtools.build.lib.shell.ShellUtils.TokenizationException;
import com.google.devtools.build.lib.syntax.ClassObject;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.FuncallExpression.FuncallException;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.syntax.SkylarkCallable;
import com.google.devtools.build.lib.syntax.SkylarkModule;
import com.google.devtools.build.lib.util.FileType;
import com.google.devtools.build.lib.util.FileTypeSet;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.view.AnalysisUtils;
import com.google.devtools.build.lib.view.ConfigurationMakeVariableContext;
import com.google.devtools.build.lib.view.FilesToRunProvider;
import com.google.devtools.build.lib.view.LabelExpander;
import com.google.devtools.build.lib.view.LabelExpander.NotUniqueExpansionException;
import com.google.devtools.build.lib.view.MakeVariableExpander.ExpansionException;
import com.google.devtools.build.lib.view.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.view.RuleContext;
import com.google.devtools.build.lib.view.TransitiveInfoCollection;
import com.google.devtools.build.lib.view.TransitiveInfoProvider;

import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ExecutionException;

import javax.annotation.Nullable;

/**
 * A Skylark API for the ruleContext.
 */
@SkylarkModule(name = "ctx", doc = "The Skylark rule context.")
public final class SkylarkRuleContext implements ClassObject {

  public static final String PROVIDER_CLASS_PREFIX = "com.google.devtools.build.lib.view.";

  static final LoadingCache<String, Class<?>> classCache = CacheBuilder.newBuilder()
      .initialCapacity(10)
      .maximumSize(100)
      .build(new CacheLoader<String, Class<?>>() {

      @Override
      public Class<?> load(String key) throws Exception {
        String classPath = SkylarkRuleContext.PROVIDER_CLASS_PREFIX + key;
        return Class.forName(classPath);
      }
    });

  private final RuleContext ruleContext;

  // TODO(bazel-team): support configurable attributes.
  private final ClassObject attrObject;

  private final ImmutableMap<String, Artifact> implicitOutputs;

  private final ImmutableMap<String, ImmutableMap<Label, Artifact>> explicitOutputs;

  /**
   * Creates a new SkylarkRuleContext using ruleContext.
   */
  public SkylarkRuleContext(RuleContext ruleContext) throws EvalException {
    this.ruleContext = Preconditions.checkNotNull(ruleContext);

    ImmutableMap.Builder<String, Object> builder = new ImmutableMap.Builder<>();
    for (Attribute a : ruleContext.getRule().getAttributes()) {
      Object val = ruleContext.attributes().get(a.getName(), a.getType());
      builder.put(a.getName(), val == null ? Environment.NONE : val);
    }
    attrObject = new SkylarkClassObject(builder.build());

    ImplicitOutputsFunction implicitOutputsFunction =
        ruleContext.getRule().getRuleClassObject().getImplicitOutputsFunction();
    ImmutableMap.Builder<String, Artifact> implicitOutputsBuilder = ImmutableMap.builder();
    if (implicitOutputsFunction instanceof SkylarkImplicitOutputsFunction) {
      SkylarkImplicitOutputsFunction func = (SkylarkImplicitOutputsFunction)
          ruleContext.getRule().getRuleClassObject().getImplicitOutputsFunction();
      for (Map.Entry<String, String> entry : func.calculateOutputs(
          RawAttributeMapper.of(ruleContext.getRule())).entrySet()) {
        implicitOutputsBuilder.put(
            entry.getKey(), ruleContext.getImplicitOutputArtifact(entry.getValue()));
      }
    }
    implicitOutputs = implicitOutputsBuilder.build();

    ImmutableMap.Builder<String, ImmutableMap<Label, Artifact>> explicitOutputsBuilder =
        ImmutableMap.builder();
    for (Map.Entry<String, Collection<OutputFile>> entry
        : ruleContext.getRule().getOutputFileMap().asMap().entrySet()) {
      ImmutableMap.Builder<Label, Artifact> labelArtifactBuilder = ImmutableMap.builder();
      for (OutputFile outputFile : entry.getValue()) {
        Artifact artifact = ruleContext.createOutputArtifact(outputFile);
        labelArtifactBuilder.put(outputFile.getLabel(), artifact);
      }
      explicitOutputsBuilder.put(entry.getKey(), labelArtifactBuilder.build());
    }
    explicitOutputs = explicitOutputsBuilder.build();
  }

  /**
   * Returns the original ruleContext.
   */
  public RuleContext getRuleContext() {
    return ruleContext;
  }

  /**
   * See {@link RuleContext#getPrerequisiteArtifacts(String, Mode)}.
   */
  @SkylarkCallable(
      doc = "Returns the immutable list of files for the specified attribute and mode.")
  public ImmutableList<Artifact> files(String attributeName, String mode)
      throws FuncallException {
    return ruleContext.getPrerequisiteArtifacts(attributeName, convertMode(mode));
  }

  /**
   * See {@link RuleContext#getPrerequisiteArtifacts(String, Mode, FileTypeSet)}.
   */
  @SkylarkCallable(doc = "")
  public ImmutableList<Artifact> files(
      String attributeName, String mode, List<?> fileTypes) throws FuncallException {
    return ruleContext.getPrerequisiteArtifacts(attributeName, convertMode(mode), FileTypeSet.of(
        Iterables.transform(fileTypes, new Function<Object, FileType>() {
          @Override
          public FileType apply(Object input) {
            Preconditions.checkArgument(input instanceof String, "File types have to be strings.");
            return FileType.of((String) input);
          }
        })));
  }

  /**
   * See {@link RuleContext#getPrerequisites(String, Mode)}.
   */
  @SkylarkCallable(doc = "")
  public Iterable<? extends TransitiveInfoCollection> targets(String attributeName, String mode)
      throws FuncallException {
    return ruleContext.getPrerequisites(attributeName, convertMode(mode));
  }

  // TODO(bazel-team): of course this is a temporary solution. Eventually the Transitive
  // Info Providers too have to be implemented using the Build Extension Language.
  /**
   * Returns all the providers of the given type. Type has to be the Transitive Info Provider's
   * canonical name after 'com.google.devtools.build.lib.view.', e.g.
   * 'go.GoContextProvider'.
   *
   * <p>See {@link RuleContext#getPrerequisites(String, Mode, Class)}.
   */
  @SkylarkCallable(doc = "")
  public Iterable<? extends TransitiveInfoProvider> targets(
      String attributeName, String mode, String type) throws FuncallException {
    try {
      Class<? extends TransitiveInfoProvider> convertedClass =
          classCache.get(type).asSubclass(TransitiveInfoProvider.class);
      return ruleContext.getPrerequisites(attributeName, convertMode(mode), convertedClass);
    } catch (ExecutionException e) {
      throw new FuncallException("Unknown Transitive Info Provider " + type);
    }
  }

  /**
   * See {@link RuleContext#getPrerequisiteArtifact(String, Mode)}.
   */
  @SkylarkCallable(doc =
      "Returns the file for the specified attribute and mode. "
      + "An error is raised if the attribute refers to 0 or multiple files.")
  public Artifact file(String attributeName, String mode) throws FuncallException {
    return ruleContext.getPrerequisiteArtifact(attributeName, convertMode(mode));
  }

  /**
   * <p>See {@link RuleContext#getExecutablePrerequisite(String, Mode)}.
   */
  @SkylarkCallable(doc = "")
  public FilesToRunProvider executable(String attributeName, String mode)
      throws FuncallException {
    return ruleContext.getExecutablePrerequisite(attributeName, convertMode(mode));
  }

  private Mode convertMode(String mode) throws FuncallException {
    try {
      return Mode.valueOf(mode);
    } catch (IllegalArgumentException e) {
      throw new FuncallException("Unknown mode " + mode);
    }
  }

  /**
   * <p>See {@link RuleContext#getCompiler(boolean)}.
   */
  @SkylarkCallable(doc =
      "Convenience method to return a host configured target for the \"compiler\" "
    + "attribute. Allows caller to decide whether a warning should be printed if "
    + "the \"compiler\" attribute is not set to the default value.<br> "
    + "If argument is true, print a warning if the value for the \"compiler\" attribute "
    + "is set to something other than the default")
  public FilesToRunProvider compiler(Boolean warnIfNotDefault) {
    return ruleContext.getCompiler(warnIfNotDefault);
  }

  @Override
  public Object getValue(String name) {
    if (name.equals("attr")) {
      return attrObject;
    } else if (name.equals("label")) {
      return ruleContext.getLabel();
    } else if (name.equals("action_owner")) {
      return ruleContext.getActionOwner();
    } else if (name.equals("outputs")) {
      return implicitOutputs;
    } else if (name.equals("configuration")) {
      return ruleContext.getConfiguration();
    } else if (name.equals("host_configuration")) {
      return ruleContext.getHostConfiguration();
    } else if (name.equals("data_configuration")) {
      return ruleContext.getConfiguration().getConfiguration(ConfigurationTransition.DATA);
    }
    return null;
  }

  @SkylarkCallable(doc = "Registers an action, that will be executed at runtime if needed.")
  public void register(Action action) {
    ruleContext.getAnalysisEnvironment().registerAction(action);
  }

  @SkylarkCallable(doc =
      "Signals a rule error with the given message. This function does not return.")
  public void error(String message) throws FuncallException {
    throw new FuncallException(message);
  }

  @SkylarkCallable(doc =
      "Signals an attribute error with the given attribute and message.")
  public void error(String attrName, String message) throws FuncallException {
    throw new FuncallException("attribute " + attrName + ": " + message);
  }

  @SkylarkCallable(doc = "Signals a warning error with the given message.")
  public void warning(String message) {
    ruleContext.ruleWarning(message);
  }

  @SkylarkCallable(doc = "Signals an attribute warning with the given attribute and message.")
  public void warning(String attrName, String message) {
    ruleContext.attributeWarning(attrName, message);
  }

  @SkylarkCallable(doc = "Returns the implicit output map.")
  public ImmutableMap<String, Artifact> outputs() {
    return implicitOutputs;
  }

  @SkylarkCallable(
      doc = "Returns the dict of labels and the corresponding output files of "
          + "the output type attribute \"attr\".")
  public ImmutableMap<Label, Artifact> outputsWithLabel(String attr) throws FuncallException {
    if  (ruleContext.attributes().getAttributeType(attr) != Type.OUTPUT
        && ruleContext.attributes().getAttributeType(attr) != Type.OUTPUT_LIST) {
      throw new FuncallException("Attribute " + attr + " is not of output type");
    }
    ImmutableMap<Label, Artifact> map = explicitOutputs.get(attr);
    if (map == null) {
      // The attribute is output type but it\'s not in the map which means it's empty,
      // i.e. not defined or defined with an empty list.
      return ImmutableMap.<Label, Artifact>of();
    }
    return map;
  }

  @SkylarkCallable(doc = "Returns the list of output files of the output type attribute \"attr\".")
  public ImmutableList<Artifact> outputs(String attr) throws FuncallException {
    // We need to convert to ImmutableList from ImmutableCollection because Skylark doesn't
    // handle collections well.
    return ImmutableList.copyOf(outputsWithLabel(attr).values());
  }

  @Override
  public String toString() {
    return ruleContext.getLabel().toString();
  }

  @SkylarkCallable(doc = "Splits a shell command to a list of tokens.")
  public List<String> tokenize(String optionString) throws FuncallException {
    List<String> options = new ArrayList<String>();
    try {
      ShellUtils.tokenize(options, optionString);
    } catch (TokenizationException e) {
      throw new FuncallException(e.getMessage() + " while tokenizing '" + optionString + "'");
    }
    return ImmutableList.copyOf(options);
  }

  @SkylarkCallable(doc =
      "Expands all references to labels embedded within a string using the "
    + "provided expansion mapping from labels to files.")
  public <T extends Iterable<Artifact>> String expand(@Nullable String expression,
      Map<Label, T> labelMap, Label labelResolver) throws FuncallException {
    try {
      return LabelExpander.expand(expression, labelMap, labelResolver);
    } catch (NotUniqueExpansionException e) {
      throw new FuncallException(e.getMessage() + " while expanding '" + expression + "'");
    }
  }

  @SkylarkCallable(doc = "Creates a new file")
  public Artifact newFile(Root root, List<String> pathFragmentStrings) {
    PathFragment fragment = ruleContext.getLabel().getPackageFragment();
    for (String pathFragmentString : pathFragmentStrings) {
      fragment = fragment.getRelative(pathFragmentString);
    }
    return ruleContext.getAnalysisEnvironment().getDerivedArtifact(fragment, root);
  }

  @SkylarkCallable(doc =
      "Creates a new file, derived from the given file and suffix.")
  public Artifact newFile(Root root, Artifact baseArtifact, String suffix) {
    PathFragment original = baseArtifact.getRootRelativePath();
    PathFragment fragment = original.replaceName(original.getBaseName() + suffix);
    return ruleContext.getAnalysisEnvironment().getDerivedArtifact(fragment, root);
  }

  @SkylarkCallable(doc = "")
  public NestedSet<Artifact> middleMan(String attribute) {
    return AnalysisUtils.getMiddlemanFor(ruleContext, attribute);
  }

  @SkylarkCallable(doc = "")
  public boolean checkPlaceholders(String template, List<String> allowedPlaceholders) {
    List<String> actualPlaceHolders = new LinkedList<>();
    Set<String> allowedPlaceholderSet = ImmutableSet.copyOf(allowedPlaceholders);
    ImplicitOutputsFunction.createPlaceholderSubstitutionFormatString(template, actualPlaceHolders);
    for (String placeholder : actualPlaceHolders) {
      if (!allowedPlaceholderSet.contains(placeholder)) {
        return false;
      }
    }
    return true;
  }

  @SkylarkCallable(doc = "")
  public String expandMakeVariables(String attributeName, String command,
      final Map<String, String> additionalSubstitutions) {
    return ruleContext.expandMakeVariables(attributeName,
        command, new ConfigurationMakeVariableContext(ruleContext.getRule().getPackage(),
            ruleContext.getConfiguration()) {
          @Override
          public String lookupMakeVariable(String name) throws ExpansionException {
            if (additionalSubstitutions.containsKey(name)) {
              return additionalSubstitutions.get(name);
            } else {
              return super.lookupMakeVariable(name);
            }
          }
        });
  }
}
