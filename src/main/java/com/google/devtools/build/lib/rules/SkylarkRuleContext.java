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

import com.google.common.base.Preconditions;
import com.google.common.cache.CacheBuilder;
import com.google.common.cache.CacheLoader;
import com.google.common.cache.LoadingCache;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.ActionOwner;
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
import com.google.devtools.build.lib.syntax.ClassObject.SkylarkClassObject;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.FuncallExpression.FuncallException;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.syntax.SkylarkCallable;
import com.google.devtools.build.lib.syntax.SkylarkList;
import com.google.devtools.build.lib.syntax.SkylarkModule;
import com.google.devtools.build.lib.syntax.SkylarkType;
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
import com.google.devtools.build.lib.view.config.BuildConfiguration;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ExecutionException;

import javax.annotation.Nullable;

/**
 * A Skylark API for the ruleContext.
 */
@SkylarkModule(name = "ctx", doc = "The context of the rule containing helper functions and "
    + "information about attributes, depending targets and outputs. "
    + "You get a ctx object as an argument to the <code>implementation</code> function when "
    + "you create a rule.")
public final class SkylarkRuleContext {

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
  private final SkylarkClassObject attrObject;

  private final SkylarkClassObject outputsObject;

  private final SkylarkClassObject executableObject;

  private final SkylarkClassObject fileObject;

  private final SkylarkClassObject filesObject;

  private final SkylarkClassObject targetsObject;

  // TODO(bazel-team): we only need this because of the css_binary rule.
  private final ImmutableMap<Artifact, Label> artifactLabelMap;

  private final ImmutableMap<Artifact, FilesToRunProvider> executableRunfilesMap;

  /**
   * In native code, private values start with $.
   * In Skylark, private values start with _, because of the grammar.
   */
  private String attributeToSkylark(String oldName) {
    if (!oldName.isEmpty() && oldName.charAt(0) == '$') {
      return "_" + oldName.substring(1);
    }
    return oldName;
  }

  /**
   * Creates a new SkylarkRuleContext using ruleContext.
   */
  public SkylarkRuleContext(RuleContext ruleContext) throws EvalException {
    this.ruleContext = Preconditions.checkNotNull(ruleContext);

    ImmutableMap.Builder<String, Object> builder = new ImmutableMap.Builder<>();
    for (Attribute a : ruleContext.getRule().getAttributes()) {
      Object val = ruleContext.attributes().get(a.getName(), a.getType());
      builder.put(attributeToSkylark(a.getName()), val == null ? Environment.NONE
          // Attribute values should be type safe
          : SkylarkType.convertToSkylark(val, null));
    }
    attrObject = new SkylarkClassObject(builder.build(), "No such attribute '%s'.");

    HashMap<String, Object> outputsBuilder = new HashMap<>();
    if (ruleContext.getRule().getRuleClassObject().outputsDefaultExecutable()) {
      addOutput(outputsBuilder, "executable", ruleContext.createOutputArtifact());
    }
    ImplicitOutputsFunction implicitOutputsFunction =
        ruleContext.getRule().getRuleClassObject().getImplicitOutputsFunction();

    if (implicitOutputsFunction instanceof SkylarkImplicitOutputsFunction) {
      SkylarkImplicitOutputsFunction func = (SkylarkImplicitOutputsFunction)
          ruleContext.getRule().getRuleClassObject().getImplicitOutputsFunction();
      for (Map.Entry<String, String> entry : func.calculateOutputs(
          RawAttributeMapper.of(ruleContext.getRule())).entrySet()) {
        addOutput(outputsBuilder, entry.getKey(),
            ruleContext.getImplicitOutputArtifact(entry.getValue()));
      }
    }

    ImmutableMap.Builder<Artifact, Label> artifactLabelMapBuilder =
        ImmutableMap.builder();
    for (Map.Entry<String, Collection<OutputFile>> entry
        : ruleContext.getRule().getOutputFileMap().asMap().entrySet()) {
      String attrName = entry.getKey();
      ImmutableList.Builder<Artifact> artifactsBuilder = ImmutableList.builder();
      for (OutputFile outputFile : entry.getValue()) {
        Artifact artifact = ruleContext.createOutputArtifact(outputFile);
        artifactsBuilder.add(artifact);
        artifactLabelMapBuilder.put(artifact, outputFile.getLabel());
      }
      ImmutableList<Artifact> artifacts = artifactsBuilder.build();

      Type<?> attrType = ruleContext.attributes().getAttributeDefinition(attrName).getType();
      if (attrType == Type.OUTPUT) {
        addOutput(outputsBuilder, attrName, Iterables.getOnlyElement(artifacts));
      } else if (attrType == Type.OUTPUT_LIST) {
        addOutput(outputsBuilder, attrName,
            SkylarkList.list(artifacts, Artifact.class));
      } else {
        throw new IllegalArgumentException(
            "Type of " + attrName + "(" + attrType + ") is not output type ");
      }
    }
    artifactLabelMap = artifactLabelMapBuilder.build();
    outputsObject = new SkylarkClassObject(outputsBuilder, "No such output '%s'.");

    ImmutableMap.Builder<String, Object> executableBuilder = new ImmutableMap.Builder<>();
    ImmutableMap.Builder<Artifact, FilesToRunProvider> executableRunfilesbuilder =
        new ImmutableMap.Builder<>();
    ImmutableMap.Builder<String, Object> fileBuilder = new ImmutableMap.Builder<>();
    ImmutableMap.Builder<String, Object> filesBuilder = new ImmutableMap.Builder<>();
    ImmutableMap.Builder<String, Object> targetsBuilder = new ImmutableMap.Builder<>();
    for (Attribute a : ruleContext.getRule().getAttributes()) {
      Type<?> type = a.getType();
      if (type != Type.LABEL && type != Type.LABEL_LIST) {
        continue;
      }
      String skyname = attributeToSkylark(a.getName());
      Mode mode = getMode(a.getName());
      if (a.isExecutable()) {
        FilesToRunProvider provider = ruleContext.getExecutablePrerequisite(a.getName(), mode);
        if (provider != null && provider.getExecutable() != null) {
          Artifact executable = provider.getExecutable();
          executableBuilder.put(skyname, executable);
          executableRunfilesbuilder.put(executable, provider);
        }
      }
      if (a.isSingleArtifact()) {
        fileBuilder.put(skyname, ruleContext.getPrerequisiteArtifact(a.getName(), mode));
      }
      filesBuilder.put(skyname, ruleContext.getPrerequisiteArtifacts(a.getName(), mode));
      targetsBuilder.put(skyname, SkylarkList.list(
          ruleContext.getPrerequisites(a.getName(), mode), TransitiveInfoCollection.class));
    }
    executableObject = new SkylarkClassObject(executableBuilder.build(), "No such executable. "
        + "Make sure there is a '%s' label type attribute marked as 'executable'.");
    fileObject = new SkylarkClassObject(fileBuilder.build(),
        "No such file. Make sure there is a '%s' label type attribute marked as 'single_file'.");
    filesObject = new SkylarkClassObject(filesBuilder.build(),
        "No such files. Make sure there is a '%s' label or label_list type attribute.");
    targetsObject = new SkylarkClassObject(targetsBuilder.build(),
        "No such targets. Make sure there is a '%s' label or label_list type attribute.");
    executableRunfilesMap = executableRunfilesbuilder.build();
  }

  private void addOutput(HashMap<String, Object> outputsBuilder, String key, Object value)
      throws EvalException {
    if (outputsBuilder.containsKey(key)) {
      throw new EvalException(null, "Multiple outputs with the same key: " + key);
    }
    outputsBuilder.put(key, value);
  }

  /**
   * Returns the original ruleContext.
   */
  public RuleContext getRuleContext() {
    return ruleContext;
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
      String attributeName, String type) throws FuncallException {
    try {
      Class<? extends TransitiveInfoProvider> convertedClass =
          classCache.get(type).asSubclass(TransitiveInfoProvider.class);
      return ruleContext.getPrerequisites(attributeName, getMode(attributeName), convertedClass);
    } catch (ExecutionException e) {
      throw new FuncallException("Unknown Transitive Info Provider " + type);
    }
  }

  private Mode getMode(String attributeName) {
    return ruleContext.getAttributeMode(attributeName);
  }

  @SkylarkCallable(name = "attr", structField = true,
      doc = "A struct to access the values of the attributes. The values are provided by "
      + "the user (if not, a default value is used).")
  public SkylarkClassObject getAttr() {
    return attrObject;
  }

  /**
   * <p>See {@link RuleContext#getExecutablePrerequisite(String, Mode)}.
   */
  @SkylarkCallable(name = "executable", structField = true,
      doc = "Return the executable file corresponding to the attribute. "
      + "Requires <code>executable=True</code>.")
  public SkylarkClassObject getExecutable() {
    return executableObject;
  }

  /**
   * See {@link RuleContext#getPrerequisiteArtifact(String, Mode)}.
   */
  @SkylarkCallable(name = "file", structField = true,
      doc = "Returns the file for the specified attribute, or None if the label is not specified. "
      + "Requires <code>single_file=True</code>.")
  public SkylarkClassObject getFile() {
    return fileObject;
  }

  /**
   * See {@link RuleContext#getPrerequisiteArtifacts(String, Mode)}.
   */
  @SkylarkCallable(name = "files", structField = true,
      doc = "Returns the list of files for the specified attribute.")
  public SkylarkClassObject getFiles() {
    return filesObject;
  }

  /**
   * See {@link RuleContext#getPrerequisites(String, Mode)}.
   */
  @SkylarkCallable(name = "targets", structField = true, doc = "")
  public SkylarkClassObject getTargets() {
    return targetsObject;
  }

  @SkylarkCallable(name = "action_owner", structField = true, doc = "Deprecated, to be removed.")
  public ActionOwner getActionOwner() {
    return ruleContext.getActionOwner();
  }

  @SkylarkCallable(name = "label", structField = true, doc = "The label of this rule.")
  public Label getLabel() {
    return ruleContext.getLabel();
  }

  @SkylarkCallable(name = "configuration", structField = true, doc = "")
  public BuildConfiguration getConfiguration() {
    return ruleContext.getConfiguration();
  }

  @SkylarkCallable(name = "host_configuration", structField = true, doc = "")
  public BuildConfiguration getHostConfiguration() {
    return ruleContext.getHostConfiguration();
  }

  @SkylarkCallable(name = "data_configuration", structField = true, doc = "")
  public BuildConfiguration getDataConfiguration() {
    return ruleContext.getConfiguration().getConfiguration(ConfigurationTransition.DATA);
  }

  @SkylarkCallable(doc = "Signals a warning error with the given message.")
  public void warning(String message) {
    ruleContext.ruleWarning(message);
  }

  @SkylarkCallable(doc = "Signals an attribute warning with the given attribute and message.")
  public void warning(String attrName, String message) {
    ruleContext.attributeWarning(attrName, message);
  }

  @SkylarkCallable(doc = "a struct containing all the outputs", structField = true)
  public SkylarkClassObject outputs() {
    return outputsObject;
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
      "Expands all references to labels embedded within a string for all files using a mapping "
    + "from definition labels (i.e. the label in the output type attribute) to files.")
  public String expand(@Nullable String expression,
      List<Artifact> artifacts, Label labelResolver) throws FuncallException {
    try {
      Map<Label, Iterable<Artifact>> labelMap = new HashMap<>();
      for (Artifact artifact : artifacts) {
        labelMap.put(artifactLabelMap.get(artifact), ImmutableList.of(artifact));
      }
      return LabelExpander.expand(expression, labelMap, labelResolver);
    } catch (NotUniqueExpansionException e) {
      throw new FuncallException(e.getMessage() + " while expanding '" + expression + "'");
    }
  }

  @SkylarkCallable(doc =
      "Creates a temporary file. You must create an action that generates "
      + "the file. If the file should be publicly visible, declare a rule "
      + "output instead.")
  public Artifact newFile(Root root, List<String> pathFragmentStrings) {
    PathFragment fragment = ruleContext.getLabel().getPackageFragment();
    for (String pathFragmentString : pathFragmentStrings) {
      fragment = fragment.getRelative(pathFragmentString);
    }
    return ruleContext.getAnalysisEnvironment().getDerivedArtifact(fragment, root);
  }

  @SkylarkCallable(doc =
      "Creates a new file, derived from the given file and suffix. "
      + "You must create an action that generates "
      + "the file. If the file should be publicly visible, declare a rule "
      + "output instead.")
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

  FilesToRunProvider getExecutableRunfiles(Artifact executable) {
    return executableRunfilesMap.get(executable);
  }  
}
