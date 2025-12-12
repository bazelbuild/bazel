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
package com.google.devtools.build.lib.analysis.starlark;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.AbstractAction;
import com.google.devtools.build.lib.actions.ActionLookupKey;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifactType;
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.cmdline.RepositoryMapping;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.starlarkbuildapi.FileApi;
import com.google.devtools.build.lib.starlarkbuildapi.StarlarkTemplateContextApi;
import com.google.devtools.build.lib.supplier.InterruptibleSupplier;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.List;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Sequence;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkSemantics;
import net.starlark.java.eval.StarlarkThread;

/** Context object to be passed to the implementation of a ctx.actions.map_directory(). */
public final class StarlarkTemplateContext implements StarlarkTemplateContextApi {

  private final StarlarkSemantics semantics;
  private final ActionOwner actionOwner;
  private final ActionLookupKey artifactOwner;
  private final SpawnAction.Builder spawnActionBuilder;
  private final InterruptibleSupplier<RepositoryMapping> repoMappingSupplier;
  private final ImmutableSet<SpecialArtifact> outputDirectories;
  private final ImmutableMap<String, String> executionInfo;
  private ImmutableList.Builder<AbstractAction> actions = ImmutableList.builder();

  public StarlarkTemplateContext(
      StarlarkSemantics semantics,
      ActionOwner actionOwner,
      ActionLookupKey artifactOwner,
      SpawnAction.Builder spawnActionBuilder,
      InterruptibleSupplier<RepositoryMapping> repoMappingSupplier,
      ImmutableSet<SpecialArtifact> outputDirectories,
      ImmutableMap<String, String> executionInfo) {
    this.semantics = semantics;
    this.actionOwner = actionOwner;
    this.artifactOwner = artifactOwner;
    this.spawnActionBuilder = spawnActionBuilder;
    this.repoMappingSupplier = repoMappingSupplier;
    this.outputDirectories = outputDirectories;
    this.executionInfo = executionInfo;
  }

  @Override
  public void run(
      Sequence<?> outputs,
      Object inputs,
      Object executableUnchecked,
      Object toolsUnchecked,
      Sequence<?> arguments,
      Object progressMessage)
      throws EvalException, InterruptedException {

    SpawnAction.Builder builder =
        newSpawnActionBuilder().addOutputs(Sequence.cast(outputs, Artifact.class, "outputs"));

    // The only other type is NoneType, which if specified will use the default progress message of
    // an action.
    if (progressMessage instanceof String progressMessageString) {
      builder.setProgressMessageFromStarlark(progressMessageString);
    }

    StarlarkActionFactory.buildCommandLine(builder, arguments, repoMappingSupplier);

    switch (inputs) {
      case Sequence<?> sequence ->
          builder.addInputs(Sequence.cast(inputs, Artifact.class, "inputs"));
      case Depset depset ->
          builder.addTransitiveInputs(Depset.cast(depset, Artifact.class, "inputs"));
      default -> {
        throw Starlark.errorf("Expected a list or depset but got %s", Starlark.type(inputs));
      }
    }
    switch (executableUnchecked) {
      case Artifact executable -> builder.setExecutable(executable);
      case FilesToRunProvider filesToRun -> builder.setExecutable(filesToRun);
      default -> {
        throw Starlark.errorf(
            "Expected a File or FilesToRunProvider but got %s", Starlark.type(executableUnchecked));
      }
    }

    if (toolsUnchecked != Starlark.NONE) {
      List<?> tools =
          switch (toolsUnchecked) {
            case Sequence<?> sequence -> Sequence.cast(toolsUnchecked, Object.class, "tools");
            case Depset depset -> Depset.cast(depset, Artifact.class, "tools").toList();
            default ->
                throw Starlark.errorf(
                    "Expected a list or depset but got %s", Starlark.type(toolsUnchecked));
          };
      for (Object toolUnchecked : tools) {
        switch (toolUnchecked) {
          // We don't infer the associated FileToRunProvider(s) of the tool here, users should pass
          // it in explicitly.
          case Artifact artifact -> builder.addTool(artifact);
          case FilesToRunProvider filesToRun ->
              builder.addTransitiveTools(filesToRun.getFilesToRun());
          case Depset depset ->
              builder.addTransitiveTools(Depset.cast(depset, Artifact.class, "tools"));
          default -> {
            throw Starlark.errorf(
                "Expected a File, FilesToRunProvider or Depset but got %s",
                Starlark.type(toolUnchecked));
          }
        }
      }
    }

    actions.add(builder.buildForStarlarkActionTemplate(actionOwner));
  }

  public void registerAction(AbstractAction action) {
    actions.add(action);
  }

  private SpawnAction.Builder newSpawnActionBuilder() {
    return new SpawnAction.Builder(spawnActionBuilder);
  }

  @Override
  public Artifact declareFile(String filename, FileApi directory) throws EvalException {
    if (!outputDirectories.contains(directory)) {
      throw Starlark.errorf(
          "Cannot declare file `%s` in non-output directory %s", filename, directory);
    }

    return TreeFileArtifact.createTemplateExpansionOutput(
        SpecialArtifact.cast(directory, SpecialArtifactType.TREE, "directory"),
        PathFragment.create(filename),
        artifactOwner);
  }

  @Override
  public Args args(StarlarkThread thread) {
    return Args.newArgs(thread.mutability(), semantics);
  }

  public ImmutableList<AbstractAction> getActions() {
    return actions.build();
  }

  public ImmutableMap<String, String> getExecutionInfo() {
    return executionInfo;
  }

  public ActionOwner getActionOwner() {
    return actionOwner;
  }

  public void close() {
    actions = null;
  }
}
