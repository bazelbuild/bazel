// Copyright 2022 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.sandbox;

import static com.google.common.util.concurrent.Futures.immediateVoidFuture;
import static org.mockito.Mockito.mock;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableSortedSet;
import com.google.common.io.Files;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.actions.ActionContext;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.Artifact.ArtifactExpander;
import com.google.devtools.build.lib.actions.InputMetadataProvider;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.exec.Protos.Digest;
import com.google.devtools.build.lib.exec.SpawnInputExpander;
import com.google.devtools.build.lib.exec.SpawnRunner.ProgressStatus;
import com.google.devtools.build.lib.exec.SpawnRunner.SpawnExecutionContext;
import com.google.devtools.build.lib.testutil.BlazeTestUtils;
import com.google.devtools.build.lib.testutil.TestConstants;
import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.File;
import java.io.IOException;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.SortedMap;
import java.util.TreeMap;
import javax.annotation.Nullable;

// TODO(b/62588075): Use this class for the LocalSpawnRunnerTest as well.
/**
 * Utilities to help test SpawnRunners.
 *
 * <p>For example, to make embedded tools available for tests, or to use a rigged {@link
 * SpawnExecutionContext} for testing purposes.
 */
public final class SpawnRunnerTestUtil {
  private SpawnRunnerTestUtil() {}

  /** A rigged spawn execution policy that can be used for testing purposes. */
  public static final class SpawnExecutionContextForTesting implements SpawnExecutionContext {
    public final List<ProgressStatus> reportedStatus = new ArrayList<>();
    public boolean prefetchCalled;
    public boolean lockOutputFilesCalled;

    private final Spawn spawn;
    private final Duration timeout;
    private final FileOutErr fileOutErr;

    private final ArtifactExpander artifactExpander = treeArtifact -> ImmutableSortedSet.of();

    /**
     * Creates a new spawn execution policy for testing purposes.
     *
     * @param fileOutErr the {@link FileOutErr} object to use. After a {@link Spawn} is executed,
     *     its stdout and stderr can be available here, if the spawn runner uses the fileOutErr
     *     returned by {@link #getFileOutErr()} on the spawn execution policy
     * @param timeout the timeout to use. Spawn runners may request this via {@link #getTimeout()}
     */
    public SpawnExecutionContextForTesting(Spawn spawn, FileOutErr fileOutErr, Duration timeout) {
      this.spawn = spawn;
      this.fileOutErr = fileOutErr;
      this.timeout = timeout;
    }

    @Override
    public int getId() {
      return 0;
    }

    @Override
    public void setDigest(Digest digest) {
      // Intentionally empty.
    }

    @Override
    @Nullable
    public Digest getDigest() {
      return null;
    }

    @Override
    public ListenableFuture<Void> prefetchInputs() {
      prefetchCalled = true;
      return immediateVoidFuture();
    }

    @Override
    public void lockOutputFiles(int exitCode, String errorMessage, FileOutErr outErr)
        throws InterruptedException {
      lockOutputFilesCalled = true;
    }

    @Override
    public boolean speculating() {
      return false;
    }

    @Override
    public InputMetadataProvider getInputMetadataProvider() {
      return mock(InputMetadataProvider.class);
    }

    @Override
    public ArtifactExpander getArtifactExpander() {
      return artifactExpander;
    }

    @Override
    public SpawnInputExpander getSpawnInputExpander() {
      throw new UnsupportedOperationException();
    }

    @Override
    public Duration getTimeout() {
      return timeout;
    }

    @Override
    public FileOutErr getFileOutErr() {
      return fileOutErr;
    }

    @Override
    public SortedMap<PathFragment, ActionInput> getInputMapping(
        PathFragment baseDirectory, boolean willAccessRepeatedly) {
      TreeMap<PathFragment, ActionInput> inputMapping = new TreeMap<>();
      for (ActionInput actionInput : spawn.getInputFiles().toList()) {
        inputMapping.put(baseDirectory.getRelative(actionInput.getExecPath()), actionInput);
      }
      return inputMapping;
    }

    @Override
    public void report(ProgressStatus progress) {
      reportedStatus.add(progress);
    }

    @Override
    public <T extends ActionContext> T getContext(Class<T> identifyingType) {
      throw new UnsupportedOperationException();
    }

    @Override
    public boolean isRewindingEnabled() {
      return false;
    }

    @Override
    public void checkForLostInputs() {}

    @Nullable
    @Override
    public FileSystem getActionFileSystem() {
      return null;
    }
  }

  /**
   * Copies a file into a specific path.
   *
   * @param sourceFile the file to copy
   * @param destinationDirectoryPath the directory to copy the sourceFile into
   */
  public static Path copyFileToPath(File sourceFile, Path destinationDirectoryPath)
      throws IOException {
    Preconditions.checkArgument(sourceFile.exists(), "source file to copy does not exist");
    Preconditions.checkArgument(
        destinationDirectoryPath.exists(), "destination directory to copy to does not exist");

    Path destinationFilePath = destinationDirectoryPath.getRelative(sourceFile.getName());
    File destinationFile = destinationFilePath.getPathFile();

    Preconditions.checkState(!destinationFilePath.exists(), "destination file already exists");
    Files.copy(sourceFile, destinationFile);

    return destinationFilePath;
  }

  private static Path copyToolIntoPath(String sourceToolRelativePath, Path destinationDirectoryPath)
      throws IOException {
    File sourceToolFile =
        new File(
            PathFragment.create(BlazeTestUtils.runfilesDir())
                .getRelative(sourceToolRelativePath)
                .getPathString());
    Preconditions.checkState(sourceToolFile.exists(), "tool not found");

    Path binDirectoryPath = destinationDirectoryPath.getRelative("_bin");
    binDirectoryPath.createDirectory();

    Path destinationToolPath = copyFileToPath(sourceToolFile, binDirectoryPath);

    destinationToolPath.setExecutable(true);

    return destinationToolPath;
  }

  /** Copies the {@code process-wrapper} tool a path where a runner expects to find it. */
  public static Path copyProcessWrapperIntoPath(Path destinationDirectoryPath) throws IOException {
    return copyToolIntoPath(TestConstants.PROCESS_WRAPPER_PATH, destinationDirectoryPath);
  }

  /** Copies the {@code linux-sandbox} tool into a path where a runner expects to find it. */
  public static Path copyLinuxSandboxIntoPath(Path destinationDirectoryPath) throws IOException {
    return copyToolIntoPath(TestConstants.LINUX_SANDBOX_PATH, destinationDirectoryPath);
  }

  /** Copies the {@code spend_cpu_time} test util into a path where a runner expects to find it. */
  public static Path copyCpuTimeSpenderIntoPath(Path destinationDirectoryPath) throws IOException {
    File realCpuTimeSpenderFile =
        new File(
            PathFragment.create(BlazeTestUtils.runfilesDir())
                .getRelative(TestConstants.CPU_TIME_SPENDER_PATH)
                .getPathString());
    Preconditions.checkState(realCpuTimeSpenderFile.exists(), "spend_cpu_time not found");

    Path destinationCpuTimeSpenderPath =
        copyFileToPath(realCpuTimeSpenderFile, destinationDirectoryPath);

    destinationCpuTimeSpenderPath.setExecutable(true);

    return destinationCpuTimeSpenderPath;
  }
}
