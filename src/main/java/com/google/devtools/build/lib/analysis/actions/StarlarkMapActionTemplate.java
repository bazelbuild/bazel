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

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.collect.ImmutableList.toImmutableList;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableListMultimap;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.AbstractAction;
import com.google.devtools.build.lib.actions.ActionConflictException;
import com.google.devtools.build.lib.actions.ActionEnvironment;
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
import com.google.devtools.build.lib.actions.CommandLineExpansionException;
import com.google.devtools.build.lib.actions.InputMetadataProvider;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.config.CoreOptions.OutputPathsMode;
import com.google.devtools.build.lib.analysis.starlark.StarlarkTemplateContext;
import com.google.devtools.build.lib.cmdline.BazelModuleContext;
import com.google.devtools.build.lib.cmdline.RepositoryMapping;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.server.FailureDetails;
import com.google.devtools.build.lib.starlarkbuildapi.ExpandedDirectoryApi;
import com.google.devtools.build.lib.starlarkbuildapi.FileApi;
import com.google.devtools.build.lib.supplier.InterruptibleSupplier;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.devtools.build.lib.util.Fingerprint;
import java.util.Map.Entry;
import javax.annotation.Nullable;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Mutability;
import net.starlark.java.eval.Printer;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkCallable;
import net.starlark.java.eval.StarlarkFunction;
import net.starlark.java.eval.StarlarkInt;
import net.starlark.java.eval.StarlarkList;
import net.starlark.java.eval.StarlarkSemantics;
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.eval.SymbolGenerator;

/** An {@link ActionTemplate} generated from Starlark's `ctx.actions.map_directory()` API. */
public final class StarlarkMapActionTemplate extends ActionKeyComputer
    implements ActionTemplate<AbstractAction> {

  private static final String INTERNAL_MAP_ACTION_TEMPLATE_MNEMONIC = "StarlarkMapActionTemplate";

  public static final String INPUT_DIRECTORIES_KEY = "input_directories";
  public static final String ADDITIONAL_INPUTS_KEY = "additional_inputs";
  public static final String OUTPUT_DIRECTORIES_KEY = "output_directories";
  public static final String TOOLS_KEY = "tools";
  public static final String ADDITIONAL_PARAMS_KEY = "additional_params";

  // The allowed classes for values for the different keys.
  private static final ImmutableSet<Class<?>> ADDITIONAL_INPUTS_CLASSES =
      ImmutableSet.of(Artifact.class, FilesToRunProvider.class, Depset.class);
  private static final ImmutableSet<Class<?>> ALLOWED_PRIMITIVE_CLASSES =
      ImmutableSet.of(Boolean.class, StarlarkInt.class, String.class);
  private static final ImmutableSet<Class<?>> DIRECTORY_CLASSES =
      ImmutableSet.of(SpecialArtifact.class);

  private final ActionOwner actionOwner;
  private final Dict<String, SpecialArtifact> inputDirectories;
  // Values in `additionalInputs` are either Artifact(s), FilesToRunProvider(s),
  // or NestedSet<Artifact>.
  private final Dict<String, Object> additionalInputs;
  private final NestedSet<Artifact> allInputs;
  private final Dict<String, SpecialArtifact> outputDirectories;
  private final Dict<String, Object> tools;
  private final NestedSet<Artifact> toolsNs;
  private final Dict<String, Object> additionalParams;
  // Comprises of the inputs that get passed along to each SpawnAction.
  private final SpawnAction.Builder spawnActionBuilder;
  private final ImmutableMap<String, String> executionRequirements;
  private final OutputPathsMode outputPathsMode;
  private final ActionEnvironment env;
  private final RepositoryMapping repoMapping;
  private final String expandedActionsMnemonic;
  private final StarlarkFunction implementation;
  private final StarlarkSemantics semantics;
  private final SymbolGenerator<?> symbolGenerator;

  public StarlarkMapActionTemplate(
      ActionOwner actionOwner,
      Dict<String, SpecialArtifact> inputDirectories,
      Dict<String, Object> additionalInputs,
      Dict<String, SpecialArtifact> outputDirectories,
      Dict<String, Object> tools,
      Dict<String, Object> additionalParams,
      SpawnAction.Builder spawnActionBuilder,
      ImmutableMap<String, String> executionRequirements,
      OutputPathsMode outputPathsMode,
      ActionEnvironment env,
      InterruptibleSupplier<RepositoryMapping> repoMappingSupplier,
      String expandedActionsMnemonic,
      StarlarkFunction implementation,
      StarlarkSemantics semantics,
      SymbolGenerator<?> symbolGenerator)
      throws EvalException, InterruptedException {
    NestedSetBuilder<Artifact> allInputsNsBuilder = NestedSetBuilder.<Artifact>stableOrder();
    NestedSetBuilder<Artifact> toolsNsBuilder = NestedSetBuilder.<Artifact>stableOrder();
    this.actionOwner = actionOwner;
    this.inputDirectories =
        validateDictValues(inputDirectories, INPUT_DIRECTORIES_KEY, DIRECTORY_CLASSES);
    addDictValuesToNestedSets(inputDirectories, INPUT_DIRECTORIES_KEY, allInputsNsBuilder);
    this.additionalInputs =
        validateDictValues(additionalInputs, ADDITIONAL_INPUTS_KEY, ADDITIONAL_INPUTS_CLASSES);
    addDictValuesToNestedSets(additionalInputs, ADDITIONAL_INPUTS_KEY, allInputsNsBuilder);
    this.outputDirectories =
        validateDictValues(outputDirectories, OUTPUT_DIRECTORIES_KEY, DIRECTORY_CLASSES);
    this.tools = validateDictValues(tools, TOOLS_KEY, ADDITIONAL_INPUTS_CLASSES);
    addDictValuesToNestedSets(tools, TOOLS_KEY, allInputsNsBuilder, toolsNsBuilder);
    this.allInputs = allInputsNsBuilder.build();
    this.toolsNs = toolsNsBuilder.build();
    // Only allow bool, int and string values in `additional_params` for now so that users do not
    // pass in arbitrary objects as inputs / outputs to expanded actions and circumvent the
    // `input_directories`, `output_directories` and `additional_inputs` fields.
    this.additionalParams =
        validateDictValues(additionalParams, ADDITIONAL_PARAMS_KEY, ALLOWED_PRIMITIVE_CLASSES);
    this.spawnActionBuilder = spawnActionBuilder;
    this.executionRequirements = executionRequirements;
    this.outputPathsMode = outputPathsMode;
    this.env = env;
    this.repoMapping = repoMappingSupplier.get();
    this.expandedActionsMnemonic = expandedActionsMnemonic;
    this.implementation = implementation;
    this.semantics = semantics;
    this.symbolGenerator = symbolGenerator;
  }

  private <V> Dict<String, V> validateDictValues(
      Dict<String, V> dict, String what, ImmutableSet<Class<?>> allowedClasses)
      throws EvalException {
    for (Entry<String, ?> entry : dict.entrySet()) {
      String keyedDebugString = String.format("%s['%s']", what, entry.getKey());
      boolean assignable = false;
      for (Class<?> allowedClass : allowedClasses) {
        if (allowedClass.isAssignableFrom(entry.getValue().getClass())) {
          assignable = true;
          break;
        }
      }
      if (!assignable) {
        throw Starlark.errorf(
            "Expected one of %s; but got %s in %s.",
            allowedClasses.stream().map(Starlark::classType).collect(toImmutableList()),
            Starlark.type(entry.getValue()),
            keyedDebugString);
      }
    }
    return Dict.<String, V>immutableCopyOf(dict);
  }

  private void addDictValuesToNestedSets(
      Dict<String, ?> dict, String what, NestedSetBuilder<Artifact>... builders)
      throws EvalException {
    for (Entry<String, ?> entry : dict.entrySet()) {
      String keyedDebugString = String.format("%s['%s']", what, entry.getKey());
      switch (entry.getValue()) {
        case Artifact artifact -> {
          for (NestedSetBuilder<Artifact> builder : builders) {
            builder.add(artifact);
          }
        }
        case FilesToRunProvider filesToRunProvider -> {
          for (NestedSetBuilder<Artifact> builder : builders) {
            builder.addTransitive(filesToRunProvider.getFilesToRun());
          }
        }
        case Depset depset -> {
          for (NestedSetBuilder<Artifact> builder : builders) {
            builder.addTransitive(Depset.cast(depset, Artifact.class, keyedDebugString));
          }
        }
        default -> {
          throw new IllegalStateException(
              String.format("Unexpected value %s in %s", entry.getValue(), what));
        }
      }
    }
  }

  @Override
  public ImmutableList<AbstractAction> generateActionsForInputArtifacts(
      ImmutableList<TreeFileArtifact> inputTreeFileArtifacts,
      ActionLookupKey artifactOwner,
      EventHandler eventHandler)
      throws ActionConflictException, ActionExecutionException, InterruptedException {

    ImmutableListMultimap<SpecialArtifact, TreeFileArtifact> inputTreeArtifactsToChildren =
        ActionTemplate.getInputTreeArtifactsToChildren(inputTreeFileArtifacts);

    StarlarkTemplateContext context =
        new StarlarkTemplateContext(
            semantics,
            actionOwner,
            artifactOwner,
            spawnActionBuilder,
            () -> repoMapping,
            ImmutableSet.copyOf(outputDirectories.values()),
            getExecutionInfo());

    ImmutableMap.Builder<String, ExpandedDirectory> expandedDirectories = ImmutableMap.builder();
    for (Entry<String, SpecialArtifact> entry : inputDirectories.entrySet()) {
      SpecialArtifact inputDirectory = entry.getValue();
      ImmutableList<TreeFileArtifact> children = inputTreeArtifactsToChildren.get(inputDirectory);
      expandedDirectories.put(entry.getKey(), new ExpandedDirectory(inputDirectory, children));
    }

    try (Mutability mu = Mutability.create("action template")) {
      StarlarkThread thread =
          StarlarkThread.create(mu, semantics, "map_directory implementation", symbolGenerator);
      thread.setPrintHandler(Event.makeDebugPrintHandler(eventHandler));
      StarlarkCallable.ArgumentProcessor argumentProcessor =
          Starlark.requestArgumentProcessor(thread, implementation);
      argumentProcessor.addPositionalArg(context);
      argumentProcessor.addNamedArg(
          INPUT_DIRECTORIES_KEY,
          Dict.<String, ExpandedDirectory>immutableCopyOf(expandedDirectories.buildOrThrow()));
      argumentProcessor.addNamedArg(OUTPUT_DIRECTORIES_KEY, outputDirectories);
      argumentProcessor.addNamedArg(ADDITIONAL_INPUTS_KEY, additionalInputs);
      argumentProcessor.addNamedArg(TOOLS_KEY, tools);
      argumentProcessor.addNamedArg(ADDITIONAL_PARAMS_KEY, additionalParams);

      Object returnValue =
          Starlark.callViaArgumentProcessor(thread, implementation, argumentProcessor);

      if (returnValue != Starlark.NONE) {
        throw Starlark.errorf(
            "actions.map_directory() implementation %s at %s may not return a non-None value (got"
                + " %s)",
            implementation.getName(), implementation.getLocation(), Starlark.repr(returnValue));
      }

      ImmutableList<AbstractAction> actions = context.getActions();
      checkActionOutputsArtifactOwner(actions, artifactOwner);
      return actions;
    } catch (EvalException e) {
      throw new ActionExecutionException(
          e, this, /* catastrophe= */ true, makeDetailedExitCode(e.getMessage()));
    } finally {
      context.close();
    }
  }

  @Override
  public ImmutableMap<String, String> getExecProperties() {
    return this.executionRequirements;
  }

  private void checkActionOutputsArtifactOwner(
      ImmutableList<AbstractAction> actions, ActionLookupKey artifactOwner)
      throws ActionConflictException {
    // This partially checks for action conflicts whereby files declared outside of this
    // Starlark implementation call are set as outputs of actions created within the implementation.
    // Files declared within this implementation should have the `artifactOwner` as their artifact
    // owner, and if the artifact owner any output artifact is different, it means that is is
    // is already output by some other action outside of this implementation, and hence is an action
    // conflict. The other typical checks for action conflicts are handled in the
    // ActionTemplateExpansionFunction.
    for (AbstractAction action : actions) {
      for (Artifact output : action.getOutputs()) {
        if (!output.getArtifactOwner().equals(artifactOwner)) {
          throw ActionConflictException.create(
              output,
              action,
              String.format(
                  "%s has conflicting output '%s' that is an output of another action, thus causing"
                      + " an action conflict. `template_ctx.run()` should only use outputs declared"
                      + " by `template_ctx.declare_file()` within the same Starlark implementation"
                      + " function.",
                  action.prettyPrint(), output.getExecPath()),
              /* isPrefixConflict= */ false);
        }
      }
    }
  }

  private static DetailedExitCode makeDetailedExitCode(String message) {
    return DetailedExitCode.of(
        FailureDetails.FailureDetail.newBuilder()
            .setMessage(message)
            .setExecution(
                FailureDetails.Execution.newBuilder()
                    .setCode(
                        FailureDetails.Execution.Code
                            .PERSISTENT_ACTION_OUTPUT_DIRECTORY_CREATION_FAILURE))
            .build());
  }

  @Override
  protected void computeKey(
      ActionKeyContext actionKeyContext,
      @Nullable InputMetadataProvider inputMetadataProvider,
      Fingerprint fp)
      throws CommandLineExpansionException, InterruptedException {
    // Already contains input_directories, additional_inputs and tools.
    actionKeyContext.addNestedSetToFingerprint(fp, allInputs);
    addMapToFingerprint(actionKeyContext, fp, outputDirectories);
    addMapToFingerprint(actionKeyContext, fp, additionalParams);
    fp.addStringMap(executionRequirements);
    fp.addString(getMnemonic());
    fp.addString(expandedActionsMnemonic);
    PathMappers.addToFingerprint(
        getMnemonic(),
        getExecutionInfo(),
        NestedSetBuilder.emptySet(Order.STABLE_ORDER),
        actionKeyContext,
        outputPathsMode,
        fp);
    env.addTo(fp);
    fp.addString(implementation.getName());
    fp.addBytes(BazelModuleContext.of(implementation.getModule()).bzlTransitiveDigest());
  }

  private void addMapToFingerprint(
      ActionKeyContext actionKeyContext, Fingerprint fp, Dict<String, ?> dict)
      throws CommandLineExpansionException, InterruptedException {
    try {
      for (Entry<String, ?> entry : dict.entrySet()) {
        fp.addString(entry.getKey());
        switch (entry.getValue()) {
          case Artifact artifact -> fp.addPath(artifact.getExecPath());
          case Depset depset ->
              actionKeyContext.addNestedSetToFingerprint(
                  fp, Depset.cast(depset, Artifact.class, "unused"));
          case Boolean bool -> fp.addBoolean(bool);
          case StarlarkInt starlarkInt -> fp.addInt(starlarkInt.toIntUnchecked());
          case String string -> fp.addString(string);
          default -> {
            throw new IllegalStateException(
                String.format(
                    "Expected Artifact or Depset; but got %s in %s.",
                    Starlark.type(entry.getValue()), entry.getKey()));
          }
        }
      }
    } catch (EvalException e) {
      // This should never happen, and should be validated / thrown in StarlarkActionFactory.
      throw new IllegalStateException(e);
    }
  }

  @Override
  public String prettyPrint() {
    return String.format(
        "StarlarkMapActionTemplate with output TreeArtifacts: %s", outputDirectories.values());
  }

  @Override
  public String getMnemonic() {
    return INTERNAL_MAP_ACTION_TEMPLATE_MNEMONIC;
  }

  @Override
  public boolean isShareable() {
    return true;
  }

  @Override
  public NestedSet<Artifact> getTools() {
    return toolsNs;
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
  public ImmutableSet<Artifact> getOutputs() {
    return ImmutableSet.copyOf(outputDirectories.values());
  }

  @Override
  public NestedSet<Artifact> getOriginalInputs() {
    return getInputs();
  }

  @Override
  public ImmutableSet<Artifact> getMandatoryOutputs() {
    return ImmutableSet.of();
  }

  @Override
  public NestedSet<Artifact> getMandatoryInputs() {
    return getInputs();
  }

  @Override
  public NestedSet<Artifact> getInputs() {
    return allInputs;
  }

  @Override
  public ImmutableList<SpecialArtifact> getInputTreeArtifacts() {
    return ImmutableList.copyOf(inputDirectories.values());
  }

  @Override
  public NestedSet<Artifact> getInputFilesForExtraAction(
      ActionExecutionContext actionExecutionContext) {
    return NestedSetBuilder.emptySet(Order.STABLE_ORDER);
  }

  @Override
  public ImmutableList<String> getClientEnvironmentVariables() {
    return ImmutableList.of();
  }

  @Override
  public String describe() {
    return String.format("Expanding %s into actions.", getInputTreeArtifacts());
  }

  /**
   * Represents a directory that has been expanded at execution time.
   *
   * <p>This is used to access the files within this directory that were previously generated by
   * other actions.
   *
   * <p>Implements {@link FileApi} and delegates all calls to the underlying {@link
   * SpecialArtifact}.
   */
  public static class ExpandedDirectory implements ExpandedDirectoryApi {

    private final SpecialArtifact directory;
    private final StarlarkList<FileApi> children;

    public ExpandedDirectory(SpecialArtifact directory, ImmutableList<TreeFileArtifact> children) {
      checkArgument(directory.isTreeArtifact());
      this.directory = directory;
      this.children = StarlarkList.immutableCopyOf(children);
    }

    @Override
    public StarlarkList<FileApi> children() {
      return children;
    }

    @Override
    public SpecialArtifact getDirectory() {
      return directory;
    }

    // TODO(b/130571505): Switch over to use the new repr(Printer, StarlarkSemantics) method.
    @Override
    public void repr(Printer printer) {
      printer.append("ExpandedDirectory(directory = ");
      directory.repr(printer);
      printer.append(", children = ");
      children.repr(printer);
      printer.append(")");
    }
  }
}
