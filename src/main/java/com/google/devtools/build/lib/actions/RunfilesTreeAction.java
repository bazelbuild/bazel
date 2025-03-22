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
package com.google.devtools.build.lib.actions;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.analysis.platform.PlatformInfo;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.skyframe.TreeArtifactValue;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.vfs.BulkDeleter;
import com.google.devtools.build.lib.vfs.Path;
import java.io.IOException;
import javax.annotation.Nullable;

/**
 * An action that depends on a set of inputs and creates a single output file whenever it runs. This
 * is useful for bundling up a bunch of dependencies that are shared between individual targets in
 * the action graph; for example generated header files.
 */
@Immutable
public final class RunfilesTreeAction extends AbstractAction implements RichDataProducingAction {
  public static final String MNEMONIC = "RunfilesTree";

  /** The runfiles tree created by this action. */
  private final RunfilesTree runfilesTree;

  public RunfilesTreeAction(
      ActionOwner owner,
      RunfilesTree runfilesTree,
      NestedSet<Artifact> inputs,
      ImmutableSet<Artifact> outputs) {
    super(owner, inputs, outputs);

    this.runfilesTree = runfilesTree;
    Preconditions.checkArgument(Iterables.getOnlyElement(outputs).isRunfilesTree(), outputs);
  }

  public RunfilesTree getRunfilesTree() {
    return runfilesTree;
  }

  private RunfilesArtifactValue createRunfilesArtifactValue(
      InputMetadataProvider inputMetadataProvider)
      throws IOException {
    ImmutableList<Artifact> inputs = getInputs().toList();
    ImmutableList.Builder<Artifact> files = ImmutableList.builder();
    ImmutableList.Builder<FileArtifactValue> fileValues = ImmutableList.builder();
    ImmutableList.Builder<Artifact> trees = ImmutableList.builder();
    ImmutableList.Builder<TreeArtifactValue> treeValues = ImmutableList.builder();
    ImmutableList.Builder<Artifact> filesets = ImmutableList.builder();
    ImmutableList.Builder<FilesetOutputTree> filesetValues = ImmutableList.builder();

    // Sort for better equality in RunfilesArtifactValue.
    ImmutableList<Artifact> sortedInputs =
        ImmutableList.sortedCopyOf(Artifact.EXEC_PATH_COMPARATOR, inputs);
    for (Artifact input : sortedInputs) {
      if (input.isFileset()) {
        filesets.add(input);
        filesetValues.add(inputMetadataProvider.getFileset(input));
      } else if (input.isTreeArtifact()) {
        trees.add(input);
        treeValues.add(inputMetadataProvider.getTreeMetadata(input));
      } else {
        files.add(input);
        fileValues.add(inputMetadataProvider.getInputMetadata(input));
      }
    }

    return new RunfilesArtifactValue(
        runfilesTree,
        files.build(),
        fileValues.build(),
        trees.build(),
        treeValues.build(),
        filesets.build(),
        filesetValues.build());
  }

  @Override
  public RichArtifactData reconstructRichDataOnActionCacheHit(
      Path execRoot, InputMetadataProvider inputMetadataProvider) {
    try {
      return createRunfilesArtifactValue(inputMetadataProvider);
    } catch (IOException e) {
      // On action cache hits, all input metadata should already be in RAM
      throw new IllegalStateException(e);
    }
  }

  @Override
  public ActionResult execute(ActionExecutionContext actionExecutionContext) {
    try {
      RunfilesArtifactValue runfilesArtifactValue =
          createRunfilesArtifactValue(
              actionExecutionContext.getInputMetadataProvider());
      actionExecutionContext.setRichArtifactData(runfilesArtifactValue);
    } catch (IOException e) {
      throw new IllegalStateException(e);
    }

    return ActionResult.EMPTY;
  }

  @Override
  public void prepare(
      Path execRoot,
      ArtifactPathResolver pathResolver,
      @Nullable BulkDeleter bulkDeleter,
      boolean cleanupArchivedArtifacts) {
    // Runfiles trees are created as a side effect of building the output manifest, not the runfiles
    // tree artifact. This method is overridden so that depending on the runfiles tree does not
    // delete the runfiles tree that's on the file system that someone decided it must be there.
  }

  @Override
  protected void computeKey(
      ActionKeyContext actionKeyContext,
      @Nullable InputMetadataProvider inputMetadataProvider,
      Fingerprint fp) {
    // Only the set of inputs matters, and the dependency checker is
    // responsible for considering those.
  }

  @Nullable
  @Override
  protected String getRawProgressMessage() {
    return null; // this action doesn't actually do anything so let's not report it
  }

  @Override
  public String prettyPrint() {
    return "runfiles for " + Label.print(getOwner().getLabel());
  }

  @Override
  public String getMnemonic() {
    return MNEMONIC;
  }

  @Override
  public boolean mayInsensitivelyPropagateInputs() {
    return true;
  }

  @Override
  public PlatformInfo getExecutionPlatform() {
    return PlatformInfo.EMPTY_PLATFORM_INFO;
  }

  @Override
  public ImmutableMap<String, String> getExecProperties() {
    // Runfiles tree actions do not execute actual actions, and therefore have no execution
    // platform.
    return ImmutableMap.of();
  }
}
