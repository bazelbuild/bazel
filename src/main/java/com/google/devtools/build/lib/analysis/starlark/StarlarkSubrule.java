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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.docgen.annot.DocCategory;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.analysis.BazelRuleAnalysisThreadContext;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.starlark.StarlarkActionFactory.StarlarkActionContext;
import com.google.devtools.build.lib.analysis.starlark.StarlarkAttrModule.Descriptor;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.StarlarkExportable;
import com.google.devtools.build.lib.starlarkbuildapi.StarlarkActionFactoryApi;
import com.google.devtools.build.lib.starlarkbuildapi.StarlarkSubruleApi;
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

  // following fields are set on export
  @Nullable private String exportedName = null;
  private ImmutableList<SubruleAttribute> attributes;

  public StarlarkSubrule(
      StarlarkFunction implementation, ImmutableMap<String, Descriptor> attributes) {
    this.implementation = implementation;
    this.attributes = SubruleAttribute.from(attributes);
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
      namedArgs.put(attr.attrName, value);
    }
    SubruleContext subruleContext = new SubruleContext(ruleContext, runfilesFromDeps.build());
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
    this.attributes = SubruleAttribute.transformOnExport(attributes, extensionLabel, exportedName);
  }

  /**
   * Returns all attributes to be lifted from the given subrules to a rule
   *
   * <p>Attributes are discovered transitively (if a subrule depends on another subrule) and those
   * from common, transitive dependencies are de-duped.
   *
   * @throws EvalException if any of the given subrules are unexported
   */
  static ImmutableList<Pair<String, Descriptor>> discoverAttributesForRule(
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
    private StarlarkRuleContext ruleContext;
    private ImmutableSet<FilesToRunProvider> runfilesFromDeps;
    private StarlarkActionFactory actions;

    private SubruleContext(
        StarlarkRuleContext ruleContext, ImmutableSet<FilesToRunProvider> runfilesFromDeps) {
      this.ruleContext = ruleContext;
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
      return ruleContext.getRuleContext().getLabel();
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

    @Override
    public ArtifactRoot newFileRoot() {
      return ruleContext.getRuleContext().getBinDirectory();
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
      return ruleContext == null;
    }

    @Override
    public FilesToRunProvider getExecutableRunfiles(Artifact executable, String what)
        throws EvalException {
      throw Starlark.errorf("for '%s', expected FilesToRunProvider, got File", what);
    }

    @Override
    public boolean areRunfilesFromDeps(FilesToRunProvider executable) {
      return runfilesFromDeps.contains(executable);
    }

    @Override
    public RuleContext getRuleContext() {
      return ruleContext.getRuleContext();
    }

    @Override
    public StarlarkSemantics getStarlarkSemantics() {
      return ruleContext.getStarlarkSemantics();
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
      // TODO: b/293304174 - return the correct toolchain
      return toolchainUnchecked;
    }

    private void nullify() {
      this.ruleContext = null;
      this.actions = null;
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
        ImmutableList<SubruleAttribute> attributes, Label label, String exportedName) {
      return attributes.stream()
          .map(s -> s.copyWithRuleAttributeName(label, exportedName))
          .collect(toImmutableList());
    }

    private SubruleAttribute copyWithRuleAttributeName(Label label, String exportedName) {
      String ruleAttrName = getRuleAttrName(label, exportedName, attrName);
      return new SubruleAttribute(attrName, descriptor, ruleAttrName);
    }
  }

  @VisibleForTesting
  // _foo -> //pkg:label%my_subrule%_foo
  static String getRuleAttrName(Label label, String exportedName, String attrName) {
    return label.getCanonicalForm() + "%" + exportedName + "%" + attrName;
  }
}
