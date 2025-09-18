// Copyright 2025 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.analysis.actions;

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.base.Preconditions.checkState;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.ActionKeyComputer;
import com.google.devtools.build.lib.actions.ActionKeyContext;
import com.google.devtools.build.lib.actions.ActionLookupKey;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.ActionTemplate;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.CommandLineExpansionException;
import com.google.devtools.build.lib.actions.InputMetadataProvider;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.starlark.StarlarkActionFactory;
import com.google.devtools.build.lib.analysis.starlark.StarlarkActionFactory.StarlarkActionContext;
import com.google.devtools.build.lib.cmdline.BazelModuleContext;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.server.FailureDetails.PackageLoading;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import javax.annotation.Nullable;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Mutability;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkFunction;
import net.starlark.java.eval.StarlarkList;
import net.starlark.java.eval.StarlarkSemantics;
import net.starlark.java.eval.StarlarkThread;

/** An {@link ActionTemplate} that expands into {@link StarlarkAction}s at execution time. */
public final class StarlarkActionTemplate extends ActionKeyComputer
    implements ActionTemplate<StarlarkAction> {

  private static final String GUID = "c84d020f-9fb4-4f61-ab5a-582f3ece6782";
  private static final String INTERNAL_ACTION_TEMPLATE_MNEMONIC = "StarlarkActionTemplate";

  private final ActionOwner actionOwner;
  private final SpecialArtifact inputTreeArtifact;
  private final SpecialArtifact outputTreeArtifact;
  private final NestedSet<Artifact> allInputs;
  private final ImmutableList<Artifact> allOutputs;
  private ImmutableMap<String, String> executionInfo = ImmutableMap.of();
  private final Object execGroupUnchecked;
  private final Object toolchainUnchecked;
  private final StarlarkFunction implementation;
  private final Dict<String, Object> kwargs;
  private final StarlarkSemantics semantics;
  // TODO(b/130571505): Do not persist the RuleContext past the analysis phase.
  private final RuleContext ruleContext;

  private StarlarkActionTemplate(
      ActionOwner actionOwner,
      SpecialArtifact inputTreeArtifact,
      SpecialArtifact outputTreeArtifact,
      NestedSet<Artifact> additionalInputs,
      ImmutableSet<SpecialArtifact> additionalOutputs,
      ImmutableMap<String, String> executionInfo,
      Object execGroupUnchecked,
      Object toolchainUnchecked,
      StarlarkFunction implementation,
      Dict<String, Object> kwargs,
      StarlarkSemantics semantics,
      RuleContext ruleContext) {
    this.actionOwner = actionOwner;
    this.inputTreeArtifact = inputTreeArtifact;
    this.outputTreeArtifact = outputTreeArtifact;
    this.allInputs =
        NestedSetBuilder.<Artifact>stableOrder()
            .add(inputTreeArtifact)
            .addTransitive(additionalInputs)
            .build();
    this.allOutputs =
        ImmutableList.<Artifact>builder().add(outputTreeArtifact).addAll(additionalOutputs).build();
    this.executionInfo = executionInfo;
    this.execGroupUnchecked = execGroupUnchecked;
    this.toolchainUnchecked = toolchainUnchecked;
    this.implementation = implementation;
    this.kwargs = Dict.immutableCopyOf(kwargs);
    this.semantics = semantics;
    this.ruleContext = ruleContext;
  }

  @Override
  public ImmutableList<StarlarkAction> generateActionsForInputArtifacts(
      ImmutableList<TreeFileArtifact> inputTreeFileArtifacts, ActionLookupKey artifactOwner)
      throws ActionExecutionException, InterruptedException {
    StarlarkActionTemplateContext actionTemplateContext =
        new StarlarkActionTemplateContext(
            ruleContext,
            semantics,
            execGroupUnchecked,
            toolchainUnchecked,
            getOutputs(),
            artifactOwner);

    try (Mutability mutability = Mutability.create("action template")) {
      try {
        StarlarkThread thread =
            StarlarkThread.create(
                mutability,
                semantics,
                ruleContext.getLabel().toString(),
                ruleContext.getSymbolGenerator());

        ImmutableList<Object> positionals =
            ImmutableList.of(
                new StarlarkActionFactory(actionTemplateContext),
                StarlarkList.copyOf(thread.mutability(), inputTreeFileArtifacts),
                outputTreeArtifact);

        Starlark.call(thread, implementation, positionals, kwargs);
        return actionTemplateContext.getRegisteredActions();
      } catch (EvalException e) {
        // TODO(b/130571505): Figure out how to display the stack trace from the EvalException.
        throw new ActionExecutionException(
            e,
            this,
            /* catastrophe= */ true,
            DetailedExitCode.of(
                FailureDetail.newBuilder()
                    .setPackageLoading(
                        PackageLoading.newBuilder()
                            .setCode(PackageLoading.Code.STARLARK_EVAL_ERROR)
                            .build())
                    .build()));
      } finally {
        actionTemplateContext.close();
      }
    }
  }

  @Override
  public ImmutableList<SpecialArtifact> getInputTreeArtifacts() {
    return ImmutableList.of(inputTreeArtifact);
  }

  @Override
  public ImmutableSet<Artifact> getOutputs() {
    return ImmutableSet.copyOf(allOutputs);
  }

  @Override
  public String prettyPrint() {
    return getMnemonic() + " for directory: " + getPrimaryInput().prettyPrint();
  }

  @Override
  public boolean isShareable() {
    return true;
  }

  @Override
  public NestedSet<Artifact> getTools() {
    return NestedSetBuilder.emptySet(Order.STABLE_ORDER);
  }

  @Override
  public NestedSet<Artifact> getSchedulingDependencies() {
    return NestedSetBuilder.emptySet(Order.STABLE_ORDER);
  }

  @Override
  public ActionOwner getOwner() {
    return actionOwner;
  }

  @Override
  public NestedSet<Artifact> getOriginalInputs() {
    return getInputs();
  }

  @Override
  public String getMnemonic() {
    return INTERNAL_ACTION_TEMPLATE_MNEMONIC;
  }

  @Override
  public ImmutableSet<Artifact> getMandatoryOutputs() {
    return ImmutableSet.of();
  }

  @Override
  public NestedSet<Artifact> getMandatoryInputs() {
    return allInputs;
  }

  @Override
  public ImmutableList<String> getClientEnvironmentVariables() {
    // Users are able to define their own environment variables for each action generated in the
    // implementation function.
    return ImmutableList.of();
  }

  @Override
  public NestedSet<Artifact> getInputs() {
    return allInputs;
  }

  @Override
  public NestedSet<Artifact> getInputFilesForExtraAction(
      ActionExecutionContext actionExecutionContext) {
    return NestedSetBuilder.emptySet(Order.STABLE_ORDER);
  }

  @Override
  public String describe() {
    return String.format(
        "Executing %s action template for files in %s",
        getMnemonic(), inputTreeArtifact.prettyPrint());
  }

  @Override
  protected void computeKey(
      ActionKeyContext actionKeyContext,
      @Nullable InputMetadataProvider inputMetadataProvider,
      Fingerprint fp)
      throws CommandLineExpansionException, InterruptedException {
    fp.addString(GUID);
    fp.addInt(implementation.hashCode());
    fp.addBytes(BazelModuleContext.of(implementation.getModule()).bzlTransitiveDigest());
    fp.addPath(inputTreeArtifact.getExecPath());
    actionKeyContext.addNestedSetToFingerprint(fp, allInputs);
    for (Artifact output : allOutputs) {
      fp.addPath(output.getExecPath());
    }
    fp.addString(getMnemonic());
    fp.addStringMap(executionInfo);
  }

  /** Builder for {@link StarlarkActionTemplate} instances */
  public static class Builder {

    private final SpecialArtifact inputTreeArtifact;
    private final SpecialArtifact outputTreeArtifact;
    private final NestedSetBuilder<Artifact> additionalInputsBuilder =
        NestedSetBuilder.<Artifact>stableOrder();
    private final ImmutableSet.Builder<SpecialArtifact> additionalOutputsBuilder =
        ImmutableSet.builder();
    private StarlarkFunction implementation;
    private ImmutableMap<String, String> executionInfo = ImmutableMap.of();
    private Object execGroupUnchecked;
    private Object toolchainUnchecked;
    private Dict<String, Object> kwargs = Dict.<String, Object>empty();

    public Builder(SpecialArtifact inputTreeArtifact, SpecialArtifact outputTreeArtifact) {
      checkState(
          inputTreeArtifact.isTreeArtifact() && outputTreeArtifact.isTreeArtifact(),
          "Either %s or %s is not a TreeArtifact",
          inputTreeArtifact,
          outputTreeArtifact);
      this.inputTreeArtifact = inputTreeArtifact;
      this.outputTreeArtifact = outputTreeArtifact;
    }

    @CanIgnoreReturnValue
    public Builder addAdditionalOutput(SpecialArtifact output) {
      additionalOutputsBuilder.add(output);
      return this;
    }

    @CanIgnoreReturnValue
    public Builder addAdditionalInput(Artifact input) {
      additionalInputsBuilder.add(input);
      return this;
    }

    @CanIgnoreReturnValue
    public Builder addAdditionalInput(FilesToRunProvider input) {
      additionalInputsBuilder.addTransitive(input.getFilesToRun());
      return this;
    }

    @CanIgnoreReturnValue
    public Builder setImplementation(StarlarkFunction implementation) {
      this.implementation = implementation;
      return this;
    }

    @CanIgnoreReturnValue
    public Builder setKwargs(Dict<String, Object> kwargs) {
      this.kwargs = kwargs;
      return this;
    }

    @CanIgnoreReturnValue
    public Builder setExecutionInfo(ImmutableMap<String, String> executionInfo) {
      this.executionInfo = executionInfo;
      return this;
    }

    @CanIgnoreReturnValue
    public Builder setExecGroup(Object execGroupUnchecked) {
      this.execGroupUnchecked = execGroupUnchecked;
      return this;
    }

    @CanIgnoreReturnValue
    public Builder setToolchain(Object toolchainUnchecked) {
      this.toolchainUnchecked = toolchainUnchecked;
      return this;
    }

    public StarlarkActionTemplate build(StarlarkActionContext context, StarlarkSemantics semantics)
        throws EvalException {
      return new StarlarkActionTemplate(
          context.getRuleContext().getActionOwner(),
          checkNotNull(inputTreeArtifact),
          checkNotNull(outputTreeArtifact),
          additionalInputsBuilder.build(),
          additionalOutputsBuilder.build(),
          executionInfo,
          execGroupUnchecked,
          toolchainUnchecked,
          checkNotNull(implementation),
          kwargs,
          semantics,
          context.getRuleContext());
    }
  }

  /** A {@link StarlarkActionContext} for {@link StarlarkActionTemplate} implementations. */
  private static class StarlarkActionTemplateContext implements StarlarkActionContext {
    private ActionLookupKey artifactOwner;
    private ImmutableList.Builder<StarlarkAction> registeredActions =
        ImmutableList.<StarlarkAction>builder();
    // TODO(b/130571505): Do not persist the RuleContext past the analysis phase.
    private RuleContext ruleContext;
    private StarlarkSemantics semantics;
    private Object execGroupUnchecked;
    private Object toolchainUnchecked;
    private ImmutableSet<Artifact> outputTreeArtifacts;

    private StarlarkActionTemplateContext(
        RuleContext ruleContext,
        StarlarkSemantics semantics,
        Object execGroupUnchecked,
        Object toolchainUnchecked,
        ImmutableSet<Artifact> outputTreeArtifacts,
        ActionLookupKey artifactOwner) {
      this.ruleContext = ruleContext;
      this.semantics = semantics;
      this.execGroupUnchecked = execGroupUnchecked;
      this.toolchainUnchecked = toolchainUnchecked;
      this.outputTreeArtifacts = outputTreeArtifacts;
      this.artifactOwner = artifactOwner;
    }

    @Override
    public ArtifactRoot newFileRoot() {
      return ruleContext.getBinDirectory();
    }

    @Override
    public StarlarkSemantics getStarlarkSemantics() {
      return semantics;
    }

    @Override
    public RuleContext getRuleContext() {
      return ruleContext;
    }

    @Override
    public FilesToRunProvider getExecutableRunfiles(Artifact executable, String what) {
      // Users should pass the FilesToRunProvider directly instead of relying on Blaze to infer it.
      return null;
    }

    @Override
    public void checkMutable(String attrName) {
      checkNotNull(ruleContext);
    }

    @Override
    public boolean areRunfilesFromDeps(FilesToRunProvider executable) {
      return false;
    }

    @Override
    public void registerAction(ActionAnalysisMetadata action) throws EvalException {
      for (Artifact output : action.getOutputs()) {
        // This happens when an artifact declared outside of transform_directory is passed in as
        // an input to transform_directory, but then used as an output of an action declared
        // within transform_directory. This is an action conflict, but manifests as an output
        // having the wrong owner.
        if (!output.getArtifactOwner().equals(artifactOwner)) {
          throw new EvalException(
              String.format(
                  "`ctx.actions.transform_directory` function generated an action with an"
                      + " output %s that belongs to an external action.",
                  output));
        }
      }
      registeredActions.add((StarlarkAction) action);
    }

    private ImmutableList<StarlarkAction> getRegisteredActions() {
      return registeredActions.build();
    }

    @Override
    public Artifact declareFile(String filename, Object sibling) throws EvalException {
      throw Starlark.errorf(
          "Cannot declare a file outside of a directory within a ctx.actions.transform_directory"
              + " `implementation` function.");
    }

    @Override
    public Artifact declareTreeFileArtifact(String filename, SpecialArtifact treeArtifact)
        throws EvalException {
      if (!outputTreeArtifacts.contains(treeArtifact)) {
        throw Starlark.errorf("Cannot declare a directory file in a non-output directory.");
      }
      return TreeFileArtifact.createTemplateExpansionOutput(
          treeArtifact, PathFragment.create(filename), artifactOwner);
    }

    @Override
    public Object maybeOverrideExecGroup(Object execGroupUnchecked) throws EvalException {
      if (execGroupUnchecked != Starlark.NONE) {
        throw Starlark.errorf("'exec_group' may not be specified in subrules");
      }
      return this.execGroupUnchecked;
    }

    @Override
    public Object maybeOverrideToolchain(Object toolchainUnchecked) throws EvalException {
      if (toolchainUnchecked != Starlark.UNBOUND) {
        throw Starlark.errorf("'toolchain' may not be specified in subrules");
      }
      return this.toolchainUnchecked;
    }

    void close() {
      registeredActions = null;
      artifactOwner = null;
      ruleContext = null;
      semantics = null;
      execGroupUnchecked = null;
      toolchainUnchecked = null;
      outputTreeArtifacts = null;
    }
  }
}
