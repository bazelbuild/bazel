// Copyright 2016 The Bazel Authors. All rights reserved.
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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.AbstractAction;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.ActionExecutionMetadata;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionInputHelper;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Actions;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.actions.ArtifactPrefixConflictException;
import com.google.devtools.build.lib.actions.BaseSpawn;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.Executor;
import com.google.devtools.build.lib.actions.RunfilesSupplier;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.SpawnActionContext;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import java.util.Collection;
import java.util.List;
import java.util.Map;

/**
 * An action that populates a TreeArtifact with the contents of an archive file.
 *
 * <p>Internally, the following happens at execution time:
 * <ol>
 *   <li>The archive entry paths are read from the associated archive manifest file locally.
 *   <li>A spawn is executed to unzip the archive contents under the root of the TreeArtifact.
 *   <li>Child TreeFileArtifacts are created using the archive entry paths and then populated into
 *       the output TreeArtifact.
 * </ol>
 *
 * <p>There are also several requirements regarding the archive and archive manifest file:
 * <ul>
 *   <li>The entry names (paths) of the archive and archive manifest file must be valid ISO-8859-1
 *       strings.
 *   <li>The archive manifest file must not contain absolute, non-normalized
 *       (e.g., containing '..' fragments) or duplicated paths. And no path is allowed to be a
 *       prefix of another path.
 * </ul>
 */
public final class PopulateTreeArtifactAction extends AbstractAction {
  private static final String GUID = "a3d36f29-9f14-42cf-a014-3a51e914e482";

  @VisibleForTesting
  static final String MNEMONIC = "PopulateTreeArtifact";

  private final Artifact archive;
  private final Artifact archiveManifest;
  private final Artifact outputTreeArtifact;
  private final FilesToRunProvider zipper;

  /**
   * Creates a PopulateTreeArtifactAction object.
   *
   * @param owner the owner of the action.
   * @param archive the archive containing files to populate into the TreeArtifact.
   * @param archiveManifest the archive manifest file specifying the entry files to populate into
   *     the TreeArtifact.
   * @param treeArtifactToPopulate the TreeArtifact to be populated with archive member files.
   * @param zipper the zipper executable used to unzip the archive.
   */
  public PopulateTreeArtifactAction(
      ActionOwner owner,
      Artifact archive,
      Artifact archiveManifest,
      Artifact treeArtifactToPopulate,
      FilesToRunProvider zipper) {
    super(
        owner,
        ImmutableList.copyOf(zipper.getFilesToRun()),
        Iterables.concat(
            ImmutableList.of(archive, archiveManifest),
            ImmutableList.copyOf(zipper.getFilesToRun())),
        ImmutableList.of(treeArtifactToPopulate));

    Preconditions.checkArgument(
        treeArtifactToPopulate.isTreeArtifact(),
        "%s is not TreeArtifact",
        treeArtifactToPopulate);

    this.archive = archive;
    this.archiveManifest = archiveManifest;
    this.outputTreeArtifact = treeArtifactToPopulate;
    this.zipper = zipper;
  }

  private static class PopulateTreeArtifactSpawn extends BaseSpawn {
    private final Artifact treeArtifact;
    private final Iterable<PathFragment> entriesToExtract;

    // The output TreeFileArtifacts are created lazily outside of the contructor because potentially
    // we can have a lot of TreeFileArtifacts under a given tree artifact.
    private Collection<TreeFileArtifact> outputTreeFileArtifacts;

    PopulateTreeArtifactSpawn(
        Artifact treeArtifact,
        Iterable<PathFragment> entriesToExtract,
        Iterable<String> commandLine,
        RunfilesSupplier runfilesSupplier,
        ActionExecutionMetadata action) {
      super(
        ImmutableList.copyOf(commandLine),
        ImmutableMap.<String, String>of(),
        ImmutableMap.<String, String>of(),
        runfilesSupplier,
        action,
        AbstractAction.DEFAULT_RESOURCE_SET);
      this.treeArtifact = treeArtifact;
      this.entriesToExtract = entriesToExtract;
    }

    @Override
    public Collection<? extends ActionInput> getOutputFiles() {
      if (outputTreeFileArtifacts == null) {
        outputTreeFileArtifacts = ImmutableList.<TreeFileArtifact>copyOf(
            ActionInputHelper.asTreeFileArtifacts(treeArtifact, entriesToExtract));
      }
      return outputTreeFileArtifacts;
    }
  }

  @Override
  public Artifact getPrimaryOutput() {
    return outputTreeArtifact;
  }

  @Override
  public void execute(ActionExecutionContext actionExecutionContext)
      throws ActionExecutionException, InterruptedException {
    Executor executor = actionExecutionContext.getExecutor();
    Spawn spawn;

    // Create a spawn to unzip the archive file into the output TreeArtifact.
    try {
      spawn = createSpawn();
    } catch (IOException e) {
      throw new ActionExecutionException(e, this, false);
    } catch (IllegalManifestFileException e) {
      throw new ActionExecutionException(e, this, true);
    }

    // If the spawn does not have any output, it means the archive file contains nothing. In this
    // case we just return without generating anything under the output TreeArtifact.
    if (spawn.getOutputFiles().isEmpty()) {
      return;
    }

    // Check spawn output TreeFileArtifact conflicts.
    try {
      checkOutputConflicts(spawn.getOutputFiles());
    } catch (ArtifactPrefixConflictException e) {
      throw new ActionExecutionException(e, this, true);
    }

    // Create parent directories for the output TreeFileArtifacts.
    try {
      for (ActionInput fileEntry : spawn.getOutputFiles()) {
        FileSystemUtils.createDirectoryAndParents(
            ((Artifact) fileEntry).getPath().getParentDirectory());
      }
    } catch (IOException e) {
      throw new ActionExecutionException(e, this, false);
    }

    // Execute the spawn.
    try {
      getContext(executor).exec(spawn, actionExecutionContext);
    } catch (ExecException e) {
      throw e.toActionExecutionException(
          getMnemonic() + " action failed for target: " + getOwner().getLabel(),
          executor.getVerboseFailures(),
          this);
    }

    // Populate the output TreeArtifact with the Spawn output TreeFileArtifacts.
    for (ActionInput fileEntry : spawn.getOutputFiles()) {
      actionExecutionContext.getMetadataHandler().addExpandedTreeOutput(
          (TreeFileArtifact) fileEntry);
    }
  }

  @Override
  protected String computeKey() {
    Fingerprint f = new Fingerprint();
    f.addString(GUID);
    f.addString(getMnemonic());
    f.addStrings(spawnCommandLine());
    f.addPaths(zipper.getRunfilesSupplier().getRunfilesDirs());
    List<Artifact> runfilesManifests = zipper.getRunfilesSupplier().getManifests();
    f.addInt(runfilesManifests.size());
    for (Artifact manifest : runfilesManifests) {
      f.addPath(manifest.getExecPath());
    }
    return f.hexDigestAndReset();
  }

  @Override
  public String getMnemonic() {
    return "PopulateTreeArtifact";
  }

  @Override
  public boolean shouldReportPathPrefixConflict(ActionAnalysisMetadata action) {
    return true;
  }

  private SpawnActionContext getContext(Executor executor) {
    return executor.getSpawnActionContext(getMnemonic());
  }

  /**
   * Creates a spawn to unzip the archive members specified in the archive manifest into the
   * TreeArtifact.
   */
  @VisibleForTesting
  Spawn createSpawn() throws IOException, IllegalManifestFileException {
    Iterable<PathFragment> entries = readAndCheckManifestEntries();
    return new PopulateTreeArtifactSpawn(
        outputTreeArtifact,
        entries,
        spawnCommandLine(),
        zipper.getRunfilesSupplier(),
        this);
  }

  private Iterable<String> spawnCommandLine() {
    return ImmutableList.of(
        zipper.getExecutable().getExecPathString(),
        "x",
        archive.getExecPathString(),
        "-d",
        outputTreeArtifact.getExecPathString(),
        "@" + archiveManifest.getExecPathString());
  }

  private Iterable<PathFragment> readAndCheckManifestEntries()
      throws IOException, IllegalManifestFileException {
    ImmutableList.Builder<PathFragment> manifestEntries = ImmutableList.builder();

    for (String line :
        FileSystemUtils.iterateLinesAsLatin1(archiveManifest.getPath())) {
      if (!line.isEmpty()) {
        PathFragment path = new PathFragment(line);

        if (!path.isNormalized() || path.isAbsolute()) {
          throw new IllegalManifestFileException(
              path + " is not a proper relative path");
        }

        manifestEntries.add(path);
      }
    }

    return manifestEntries.build();
  }

  private void checkOutputConflicts(Collection<? extends ActionInput> outputs)
      throws ArtifactPrefixConflictException {
    ImmutableMap.Builder<Artifact, ActionAnalysisMetadata> generatingActions =
        ImmutableMap.<Artifact, ActionAnalysisMetadata>builder();
    for (ActionInput output : outputs) {
      generatingActions.put((Artifact) output, this);
    }

    Map<ActionAnalysisMetadata, ArtifactPrefixConflictException> artifactPrefixConflictMap =
        Actions.findArtifactPrefixConflicts(generatingActions.build());

    if (!artifactPrefixConflictMap.isEmpty()) {
      throw artifactPrefixConflictMap.values().iterator().next();
    }
  }

  private static class IllegalManifestFileException extends Exception {

    IllegalManifestFileException(String message) {
      super(message);
    }
  }
}
