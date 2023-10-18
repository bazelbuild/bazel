// Copyright 2023 The Bazel Authors. All rights reserved.
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

import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.common.collect.ImmutableSet.toImmutableSet;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.docgen.annot.DocCategory;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.analysis.BazelRuleAnalysisThreadContext;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.config.ToolchainTypeRequirement;
import com.google.devtools.build.lib.analysis.starlark.StarlarkActionFactory.StarlarkActionContext;
import com.google.devtools.build.lib.analysis.starlark.StarlarkAttrModule.Descriptor;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.AttributeValueSource;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.StarlarkExportable;
import com.google.devtools.build.lib.starlarkbuildapi.StarlarkActionFactoryApi;
import com.google.devtools.build.lib.starlarkbuildapi.StarlarkSubruleApi;
import com.google.devtools.build.lib.starlarkbuildapi.platform.ToolchainContextApi;
import com.google.devtools.build.lib.util.Pair;
import javax.annotation.Nullable;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Printer;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkCallable;
import net.starlark.java.eval.StarlarkFunction;
import net.starlark.java.eval.StarlarkSemantics;
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.eval.Tuple;

/**
 * Represents a subrule which can be invoked in a Starlark rule's implementation function.
 *
 * <p>The basic mechanism used is that a rule class declared a dependency on a set of subrules. The
 * (implicit) attributes of the subrule are lifted to the rule class, and thus, behave as if they
 * were directly declared on the rule class itself. The rule class also holds a reference to the set
 * of subrules. The latter is only used for validating that a rule invoking a subrule declared that
 * subrule as a dependency.
 */
public class StarlarkSubrule implements StarlarkExportable, StarlarkCallable, StarlarkSubruleApi {
  // TODO(hvd) this class is a WIP, will be implemented over many commits
  // TODO: b/293304174 - Fix all user-facing Starlark documentation

  private final StarlarkFunction implementation;
  private final ImmutableSet<ToolchainTypeRequirement> toolchains;

  // following fields are set on export
  @Nullable private String exportedName = null;
  private ImmutableList<SubruleAttribute> attributes;

  public StarlarkSubrule(
      StarlarkFunction implementation,
      ImmutableMap<String, Descriptor> attributes,
      ImmutableSet<ToolchainTypeRequirement> toolchains) {
    this.implementation = implementation;
    this.attributes = SubruleAttribute.from(attributes);
    this.toolchains = toolchains;
  }

  @Override
  public String getName() {
    if (isExported()) {
      return exportedName;
    } else {
      return "unexported subrule";
    }
  }

  @Override
  public void repr(Printer printer) {
    printer.append("<subrule ").append(getName()).append(">");
  }

  @Override
  public Object call(StarlarkThread thread, Tuple args, Dict<String, Object> kwargs)
      throws EvalException, InterruptedException {
    checkExported();
    StarlarkRuleContext ruleContext =
        BazelRuleAnalysisThreadContext.fromOrFail(thread, getName())
            .getRuleContext()
            .getStarlarkRuleContext();
    ImmutableSet<? extends StarlarkSubruleApi> declaredSubrules = ruleContext.getSubrules();
    if (!declaredSubrules.contains(this)) {
      throw getUndeclaredSubruleError(ruleContext);
    }
    ImmutableSet.Builder<FilesToRunProvider> runfilesFromDeps = ImmutableSet.builder();
    ImmutableMap.Builder<String, Object> namedArgs = ImmutableMap.builder();
    namedArgs.putAll(kwargs);
    for (SubruleAttribute attr : attributes) {
      // TODO: b/293304174 - maybe permit overriding?
      if (kwargs.containsKey(attr.attrName)) {
        throw Starlark.errorf(
            "got invalid named argument: '%s' is an implicit dependency and cannot be overridden",
            attr.attrName);
      }
      Attribute attribute =
          ruleContext
              .getRuleContext()
              .getRule()
              .getRuleClassObject()
              .getAttributeByName(attr.ruleAttrName);
      // We need to use the underlying RuleContext because the subrule attributes are hidden from
      // the rule ctx.attr
      Object value;
      if (attribute.isExecutable()) {
        FilesToRunProvider runfiles =
            ruleContext.getRuleContext().getExecutablePrerequisite(attribute.getName());
        runfilesFromDeps.add(runfiles);
        value = runfiles;
      } else if (attribute.getType() == BuildType.LABEL_LIST) {
        value = ruleContext.getRuleContext().getPrerequisites(attribute.getName());
      } else if (attribute.getType() == BuildType.LABEL) {
        value = ruleContext.getRuleContext().getPrerequisite(attribute.getName());
      } else {
        // this should never happen, we've already validated the type while evaluating the subrule
        throw new IllegalStateException("unexpected attribute type");
      }
      namedArgs.put(attr.attrName, value == null ? Starlark.NONE : value);
    }
    SubruleContext subruleContext =
        new SubruleContext(ruleContext, toolchains, runfilesFromDeps.build());
    ImmutableList<Object> positionals =
        ImmutableList.builder().add(subruleContext).addAll(args).build();
    try {
      ruleContext.setLockedForSubrule(true);
      return Starlark.call(
          thread, implementation, positionals, Dict.immutableCopyOf(namedArgs.buildOrThrow()));
    } finally {
      subruleContext.nullify();
      ruleContext.setLockedForSubrule(false);
    }
  }

  private EvalException getUndeclaredSubruleError(StarlarkRuleContext starlarkRuleContext) {
    if (starlarkRuleContext.isForAspect()) {
      return Starlark.errorf(
          "aspect '%s' must declare '%s' in 'subrules'",
          starlarkRuleContext.getRuleContext().getMainAspect().getAspectClass().getName(),
          this.getName());
    } else {
      return Starlark.errorf(
          "rule '%s' must declare '%s' in 'subrules'",
          starlarkRuleContext.getRuleContext().getRule().getRuleClass(), this.getName());
    }
  }

  /**
   * Returns the collection of attributes to be lifted to a rule that uses this {@code subrule}.
   *
   * @throws EvalException if this subrule is unexported
   */
  private ImmutableList<Pair<String, Descriptor>> attributesForRule() throws EvalException {
    checkExported();
    ImmutableList.Builder<Pair<String, Descriptor>> builder = ImmutableList.builder();
    for (SubruleAttribute attr : attributes) {
      builder.add(Pair.of(attr.ruleAttrName, attr.descriptor));
    }
    return builder.build();
  }

  private void checkExported() throws EvalException {
    if (!isExported()) {
      throw Starlark.errorf("Invalid subrule hasn't been exported by a bzl file");
    }
  }

  @Override
  public boolean isExported() {
    return this.exportedName != null;
  }

  @Override
  public void export(EventHandler handler, Label extensionLabel, String exportedName) {
    Preconditions.checkState(!isExported());
    this.exportedName = exportedName;
    this.attributes =
        SubruleAttribute.transformOnExport(attributes, extensionLabel, exportedName, handler);
  }

  /**
   * Returns all attributes to be lifted from the given subrules to a rule/aspect
   *
   * <p>Attributes are discovered transitively (if a subrule depends on another subrule) and those
   * from common, transitive dependencies are de-duped.
   *
   * @throws EvalException if any of the given subrules are unexported
   */
  static ImmutableList<Pair<String, Descriptor>> discoverAttributes(
      ImmutableList<? extends StarlarkSubruleApi> subrules) throws EvalException {
    ImmutableSet.Builder<StarlarkSubruleApi> uniqueSubrules = ImmutableSet.builder();
    for (StarlarkSubruleApi subrule : subrules) {
      // TODO: b/293304174 - use all transitive subrules once subrules can depend on other subrules
      uniqueSubrules.add(subrule);
    }
    ImmutableList.Builder<Pair<String, Descriptor>> attributes = ImmutableList.builder();
    for (StarlarkSubruleApi subrule : uniqueSubrules.build()) {
      if (subrule instanceof StarlarkSubrule) {
        attributes.addAll(((StarlarkSubrule) subrule).attributesForRule());
      }
    }
    return attributes.build();
  }

  /** Returns all toolchain types to be lifted from the given subrules to a rule/aspect */
  static ImmutableSet<ToolchainTypeRequirement> discoverToolchains(
      ImmutableList<? extends StarlarkSubruleApi> subrules) {
    ImmutableSet.Builder<StarlarkSubruleApi> uniqueSubrules = ImmutableSet.builder();
    for (StarlarkSubruleApi subrule : subrules) {
      // TODO: b/293304174 - use all transitive subrules once subrules can depend on other subrules
      uniqueSubrules.add(subrule);
    }
    ImmutableSet.Builder<ToolchainTypeRequirement> toolchains = ImmutableSet.builder();
    for (StarlarkSubruleApi subrule : uniqueSubrules.build()) {
      if (subrule instanceof StarlarkSubrule) {
        toolchains.addAll(((StarlarkSubrule) subrule).toolchains);
      }
    }
    return toolchains.build();
  }

  /**
   * The context object passed to the implementation function of a subrule.
   *
   * <p>This class exists to reduce the API surface visible to subrules and avoid leaking deprecated
   * or legacy APIs. It wraps the underlying rule's {@link StarlarkRuleContext} and either simply
   * delegates the operation to the latter, or has very similar behavior to it. Cases where behavior
   * differs is documented on the respective methods.
   */
  @StarlarkBuiltin(
      name = "subrule_ctx",
      category = DocCategory.BUILTIN,
      doc = "A context object passed to the implementation function of a subrule.")
  private static class SubruleContext implements StarlarkActionContext {
    // these fields are effectively final, set to null once this instance is no longer usable by
    // Starlark
    private StarlarkRuleContext starlarkRuleContext;
    private ImmutableSet<Label> requestedToolchains;
    private ImmutableSet<FilesToRunProvider> runfilesFromDeps;
    private StarlarkActionFactory actions;

    private SubruleContext(
        StarlarkRuleContext ruleContext,
        ImmutableSet<ToolchainTypeRequirement> requestedToolchains,
        ImmutableSet<FilesToRunProvider> runfilesFromDeps) {
      this.starlarkRuleContext = ruleContext;
      this.requestedToolchains =
          requestedToolchains.stream()
              .map(ToolchainTypeRequirement::toolchainType)
              .collect(toImmutableSet());
      this.runfilesFromDeps = runfilesFromDeps;
      this.actions = new StarlarkActionFactory(this);
    }

    @StarlarkMethod(
        name = "label",
        doc = "The label of the target currently being analyzed",
        structField = true)
    public Label getLabel() throws EvalException {
      checkMutable("label");
      // we use the underlying RuleContext to bypass the mutability check in
      // StarlarkRuleContext.getLabel() since it's locked
      return starlarkRuleContext.getRuleContext().getLabel();
    }

    // This is identical to the StarlarkActionFactory used by StarlarkRuleContext, and subrule
    // specific behaviour is triggered by the methods inherited from StarlarkActionContext
    @StarlarkMethod(
        name = "actions",
        doc = "Contains methods for declaring output files and the actions that produce them",
        structField = true)
    public StarlarkActionFactoryApi actions() throws EvalException {
      checkMutable("actions");
      return actions;
    }

    @StarlarkMethod(
        name = "toolchains",
        doc = "Contains methods for declaring output files and the actions that produce them",
        structField = true)
    public ToolchainContextApi toolchains() throws EvalException {
      checkMutable("toolchains");
      RuleContext ruleContext = starlarkRuleContext.getRuleContext();
      if (ruleContext.getToolchainContext() == null) {
        return StarlarkToolchainContext.TOOLCHAINS_NOT_VALID;
      }
      if (ruleContext.useAutoExecGroups()) {
        return StarlarkToolchainContext.create(
            /* targetDescription= */ ruleContext.getToolchainContext().targetDescription(),
            /* resolveToolchainInfoFunc= */ ruleContext::getToolchainInfo,
            /* resolvedToolchainTypeLabels= */ getAutomaticExecGroupLabels());
      } else {
        throw Starlark.errorf("subrules using toolchains must enable automatic exec-groups");
      }
    }

    private ImmutableSet<Label> getAutomaticExecGroupLabels() {
      return starlarkRuleContext.getAutomaticExecGroupLabels().stream()
          .filter(label -> requestedToolchains.contains(label))
          .collect(toImmutableSet());
    }

    @Override
    public ArtifactRoot newFileRoot() {
      return starlarkRuleContext.getRuleContext().getBinDirectory();
    }

    @Override
    public void checkMutable(String attrName) throws EvalException {
      // TODO: b/293304174 - check if subrule is locked once subrules can call other subrules
      if (isImmutable()) {
        throw Starlark.errorf(
            "cannot access field or method '%s' of subrule context outside of its own"
                + " implementation function",
            attrName);
      }
    }

    @Override
    public boolean isImmutable() {
      return starlarkRuleContext == null;
    }

    @Override
    @Nullable
    public FilesToRunProvider getExecutableRunfiles(Artifact executable, String what)
        throws EvalException {
      if (runfilesFromDeps.stream().anyMatch(dep -> executable.equals(dep.getExecutable()))) {
        // TODO: b/293304174 - maybe return the matched FilesToRunProvider instead of failing?
        throw Starlark.errorf("for '%s', expected FilesToRunProvider, got File", what);
      } else {
        // executable attributes of a subrule are passed to the implementation as FilesToRunProvider
        // so this should never happen unless this comes from somewhere else, in which case, we
        // can't resolve it anyway
        return null;
      }
    }

    @Override
    public boolean areRunfilesFromDeps(FilesToRunProvider executable) {
      return runfilesFromDeps.contains(executable);
    }

    @Override
    public RuleContext getRuleContext() {
      return starlarkRuleContext.getRuleContext();
    }

    @Override
    public StarlarkSemantics getStarlarkSemantics() {
      return starlarkRuleContext.getStarlarkSemantics();
    }

    @Override
    public Object maybeOverrideExecGroup(Object execGroupUnchecked) throws EvalException {
      if (execGroupUnchecked != Starlark.NONE) {
        throw Starlark.errorf("'exec_group' may not be specified in subrules");
      }
      // TODO: b/293304174 - return the correct exec group
      return execGroupUnchecked;
    }

    @Override
    public Object maybeOverrideToolchain(Object toolchainUnchecked) throws EvalException {
      if (toolchainUnchecked != Starlark.UNBOUND) {
        throw Starlark.errorf("'toolchain' may not be specified in subrules");
      }
      return requestedToolchains.isEmpty()
          ? toolchainUnchecked
          : Iterables.getOnlyElement(requestedToolchains);
    }

    private void nullify() {
      this.starlarkRuleContext = null;
      this.actions = null;
      this.requestedToolchains = null;
      this.runfilesFromDeps = null;
    }
  }

  private static class SubruleAttribute {

    private final String attrName;
    private final Descriptor descriptor;

    /**
     * This is the attribute name when lifted to a rule, see {@link #copyWithRuleAttributeName} and
     * is set only after the subrule is exported
     */
    @Nullable private final String ruleAttrName;

    private SubruleAttribute(
        String attrName, Descriptor descriptor, @Nullable String ruleAttrName) {
      this.attrName = attrName;
      this.descriptor = descriptor;
      this.ruleAttrName = ruleAttrName;
    }

    private static ImmutableList<SubruleAttribute> from(
        ImmutableMap<String, Descriptor> attributes) {
      return attributes.entrySet().stream()
          .map(e -> new SubruleAttribute(e.getKey(), e.getValue(), null))
          .collect(toImmutableList());
    }

    private static ImmutableList<SubruleAttribute> transformOnExport(
        ImmutableList<SubruleAttribute> attributes,
        Label label,
        String exportedName,
        EventHandler handler) {
      ImmutableList.Builder<SubruleAttribute> builder = ImmutableList.builder();
      for (SubruleAttribute attribute : attributes) {
        try {
          builder.add(attribute.copyWithRuleAttributeName(label, exportedName));
        } catch (EvalException e) {
          handler.handle(Event.error(e.getMessage()));
        }
      }
      return builder.build();
    }

    private SubruleAttribute copyWithRuleAttributeName(Label label, String exportedName)
        throws EvalException {
      String ruleAttrName =
          getRuleAttrName(label, exportedName, attrName, descriptor.getValueSource());
      return new SubruleAttribute(attrName, descriptor, ruleAttrName);
    }
  }

  @VisibleForTesting
  // _foo -> $//pkg:label%my_subrule%_foo
  static String getRuleAttrName(
      Label label, String exportedName, String attrName, AttributeValueSource valueSource)
      throws EvalException {
    return valueSource.convertToNativeName(
        "_" + label.getCanonicalForm() + "%" + exportedName + "%" + attrName);
  }
}
