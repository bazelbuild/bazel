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
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.ImplicitOutputsFunction;
import com.google.devtools.build.lib.packages.OutputFile;
import com.google.devtools.build.lib.shell.ShellUtils;
import com.google.devtools.build.lib.shell.ShellUtils.TokenizationException;
import com.google.devtools.build.lib.syntax.ClassObject;
import com.google.devtools.build.lib.syntax.FuncallExpression.FuncallException;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.syntax.SkylarkBuiltin;
import com.google.devtools.build.lib.syntax.SkylarkCallable;
import com.google.devtools.build.lib.syntax.UserDefinedFunction;
import com.google.devtools.build.lib.util.FileType;
import com.google.devtools.build.lib.util.FileTypeSet;
import com.google.devtools.build.lib.view.ConfigurationMakeVariableContext;
import com.google.devtools.build.lib.view.LabelExpander;
import com.google.devtools.build.lib.view.LabelExpander.NotUniqueExpansionException;
import com.google.devtools.build.lib.view.MakeVariableExpander.ExpansionException;
import com.google.devtools.build.lib.view.RuleConfiguredTarget.Mode;
import com.google.devtools.build.lib.view.RuleContext;
import com.google.devtools.build.lib.view.TransitiveInfoProvider;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ExecutionException;

import javax.annotation.Nullable;

/**
 * A Skylark API for the ruleContext.
 */
@SkylarkBuiltin(name = "ctx", doc = "The Skylark rule context.")
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

  private final ClassObject attrObject;

  /**
   * Creates a new SkylarkRuleContext using ruleContext.
   */
  public SkylarkRuleContext(RuleContext ruleContext) {
    this.ruleContext = Preconditions.checkNotNull(ruleContext);

    ImmutableMap.Builder<String, Object> builder = new ImmutableMap.Builder<>();
    for (Attribute a : ruleContext.getRule().getAttributes()) {
      Object val = ruleContext.getRule().getAttr(a);
      builder.put(a.getName(), val == null ? UserDefinedFunction.NONE : val);
    }
    attrObject = new ClassObject(builder.build());
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
  public Object getPrerequisiteArtifacts(String attributeName, String mode)
      throws FuncallException {
    return ruleContext.getPrerequisiteArtifacts(attributeName, convertMode(mode));
  }

  /**
   * See {@link RuleContext#getPrerequisiteArtifacts(String, Mode, FileTypeSet)}.
   */
  @SkylarkCallable(doc = "")
  public Object getPrerequisiteArtifacts(
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
  public Object getPrerequisites(String attributeName, String mode) throws FuncallException {
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
  public Iterable<? extends TransitiveInfoProvider> getPrerequisites(
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
  @SkylarkCallable(doc = "")
  public Object getPrerequisiteArtifact(String attributeName, String mode)
      throws FuncallException {
    return ruleContext.getPrerequisiteArtifact(attributeName, convertMode(mode));
  }

  /**
   * <p>See {@link RuleContext#getExecutablePrerequisite(String, Mode)}.
   */
  @SkylarkCallable(doc = "")
  public Object getExecutablePrerequisite(String attributeName, String mode)
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
  @SkylarkCallable(doc = "")
  public Object getCompiler(Boolean warnIfNotDefault) {
    return ruleContext.getCompiler(warnIfNotDefault);
  }

  /**
   * Returns the rule attribute field if exists.
   */
  @SkylarkCallable(doc = "")
  public Object get(String attributeName) throws FuncallException {
    try {
      return ruleContext.getRule().getAttr(attributeName);
    } catch (IllegalArgumentException e) {
      throw new FuncallException(e.getMessage());
    }
  }

  @SkylarkCallable(doc = "")
  public ClassObject attr() {
    return attrObject;
  }

  /**
   * Returns the rule's label.
   */
  @SkylarkCallable(doc = "")
  public Label getLabel() {
    return ruleContext.getLabel();
  }

  /**
   * Returns the rule's action owner.
   */
  @SkylarkCallable(doc = "")
  public Object getActionOwner() {
    return ruleContext.getActionOwner();
  }

  /**
   * Registers an action.
   */
  @SkylarkCallable(doc = "")
  public void registerAction(Action action) {
    ruleContext.getAnalysisEnvironment().registerAction(action);
  }

  /**
   * See {@link RuleContext#getConfiguration()}.
   */
  @SkylarkCallable(doc = "")
  public Object getConfiguration() {
    return ruleContext.getConfiguration();
  }

  /**
   * Signals a rule error with the given message.
   */
  @SkylarkCallable(doc = "")
  public void error(String message) throws FuncallException {
    throw new FuncallException(message);
  }

  /**
   * Signals an attribute error with the given attribute and message.
   */
  @SkylarkCallable(doc = "")
  public void error(String attrName, String message) throws FuncallException {
    throw new FuncallException("attribute " + attrName + ": " + message);
  }

  /**
   * Signals a warning error with the given message.
   */
  @SkylarkCallable(doc = "")
  public void warning(String message) {
    ruleContext.ruleWarning(message);
  }

  /**
   * Signals an attribute warning with the given attribute and message.
   */
  @SkylarkCallable(doc = "")
  public void warning(String attrName, String message) {
    ruleContext.attributeWarning(attrName, message);
  }

  /**
   * See {@link RuleContext#createOutputArtifact(OutputFile)}.
   */
  @SkylarkCallable(doc = "")
  public Object createOutputArtifact(OutputFile out) {
    return ruleContext.createOutputArtifact(out);
  }

  /**
   * See {@link RuleContext#getOutputArtifacts()}.
   */
  @SkylarkCallable(doc = "")
  public Object getOutputArtifacts() {
    return ruleContext.getOutputArtifacts();
  }

  /**
   * See {@link RuleContext#createOutputArtifact(OutputFile)}.
   */
  @SkylarkCallable(doc = "")
  public ImmutableList<OutputFile> getOutputFiles() {
    return ImmutableList.copyOf(ruleContext.getRule().getOutputFiles());
  }

  @Override
  @SkylarkCallable(doc = "")
  public String toString() {
    return ruleContext.getLabel().toString();
  }

  @SkylarkCallable(doc = "")
  public ImmutableMap<String, String> getDefaultShellEnvironment() {
    return ruleContext.getConfiguration().getDefaultShellEnvironment();
  }

  @SkylarkCallable(doc = "")
  public List<String> tokenize(String optionString) throws FuncallException {
    List<String> options = new ArrayList<String>();
    try {
      ShellUtils.tokenize(options, optionString);
    } catch (TokenizationException e) {
      throw new FuncallException(e.getMessage() + " while tokenizing '" + optionString + "'");
    }
    return ImmutableList.copyOf(options);
  }

  @SkylarkCallable(doc = "")
  public <T extends Iterable<Artifact>> String expand(@Nullable String expression,
      Map<Label, T> labelMap, Label labelResolver) throws FuncallException {
    try {
      return LabelExpander.expand(expression, labelMap, labelResolver);
    } catch (NotUniqueExpansionException e) {
      throw new FuncallException(e.getMessage() + " while expanding '" + expression + "'");
    }
  }

  @SkylarkCallable(doc = "")
  public ImmutableList<String> substitutePlaceholders(String template) {
    return ImplicitOutputsFunction.substitutePlaceholderIntoTemplate(
        template, ruleContext.getRule());
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
