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
package com.google.devtools.build.lib.exec;

import static com.google.devtools.build.lib.exec.SpawnLogContext.computeDigest;
import static com.google.devtools.build.lib.exec.SpawnLogContext.getEnvironmentVariables;
import static com.google.devtools.build.lib.exec.SpawnLogContext.getPlatform;
import static com.google.devtools.build.lib.exec.SpawnLogContext.getSpawnMetricsProto;
import static com.google.devtools.build.lib.exec.SpawnLogContext.isInputDirectory;

import com.google.common.collect.ImmutableSet;
import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.InputMetadataProvider;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.SpawnResult;
import com.google.devtools.build.lib.actions.Spawns;
import com.google.devtools.build.lib.actions.cache.VirtualActionInput;
import com.google.devtools.build.lib.exec.Protos.Digest;
import com.google.devtools.build.lib.exec.Protos.File;
import com.google.devtools.build.lib.exec.Protos.Platform;
import com.google.devtools.build.lib.exec.Protos.SpawnExec;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import com.google.devtools.build.lib.remote.options.RemoteOptions;
import com.google.devtools.build.lib.util.io.AsynchronousMessageOutputStream;
import com.google.devtools.build.lib.util.io.MessageOutputStream;
import com.google.devtools.build.lib.util.io.MessageOutputStreamWrapper.BinaryOutputStreamWrapper;
import com.google.devtools.build.lib.util.io.MessageOutputStreamWrapper.JsonOutputStreamWrapper;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.Dirent;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Symlinks;
import com.google.devtools.build.lib.vfs.XattrProvider;
import java.io.IOException;
import java.io.InputStream;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.SortedMap;
import java.util.TreeMap;
import java.util.function.Consumer;
import javax.annotation.Nullable;

/** A {@link SpawnLogContext} implementation that produces a log in expanded format. */
public class ExpandedSpawnLogContext implements SpawnLogContext {

  /** The log encoding. */
  public enum Encoding {
    /** Length-delimited binary protos. */
    BINARY,
    /** Newline-delimited JSON messages. */
    JSON
  }

  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  private final Path tempPath;
  private final boolean sorted;

  private final PathFragment execRoot;
  @Nullable private final RemoteOptions remoteOptions;
  private final DigestHashFunction digestHashFunction;
  private final XattrProvider xattrProvider;

  /** Output stream to write directly into during execution. */
  private final MessageOutputStream<SpawnExec> rawOutputStream;

  /** Output stream to convert the raw output stream into after execution, if required. */
  @Nullable private final MessageOutputStream<SpawnExec> convertedOutputStream;

  public ExpandedSpawnLogContext(
      Path outputPath,
      Path tempPath,
      Encoding encoding,
      boolean sorted,
      PathFragment execRoot,
      @Nullable RemoteOptions remoteOptions,
      DigestHashFunction digestHashFunction,
      XattrProvider xattrProvider)
      throws IOException {
    this.tempPath = tempPath;
    this.sorted = sorted;
    this.execRoot = execRoot;
    this.remoteOptions = remoteOptions;
    this.digestHashFunction = digestHashFunction;
    this.xattrProvider = xattrProvider;

    if (encoding == Encoding.BINARY && !sorted) {
      // The unsorted binary format can be written directly into the output path during execution.
      rawOutputStream = getRawOutputStream(outputPath);
      convertedOutputStream = null;
    } else {
      // Otherwise, write the unsorted binary format into a temporary path first, then convert into
      // the output format after execution.
      rawOutputStream = getRawOutputStream(tempPath);
      convertedOutputStream = getConvertedOutputStream(encoding, outputPath);
    }
  }

  private static MessageOutputStream<SpawnExec> getRawOutputStream(Path path) throws IOException {
    // Use an AsynchronousMessageOutputStream so that writes occur in a separate thread.
    // This ensures concurrent writes don't tear and avoids blocking execution.
    return new AsynchronousMessageOutputStream<>(path);
  }

  private static MessageOutputStream<SpawnExec> getConvertedOutputStream(
      Encoding encoding, Path path) throws IOException {
    switch (encoding) {
      case BINARY:
        return new BinaryOutputStreamWrapper<>(path.getOutputStream());
      case JSON:
        return new JsonOutputStreamWrapper<>(path.getOutputStream());
    }
    throw new IllegalArgumentException(
        String.format("invalid execution log encoding: %s", encoding));
  }

  @Override
  public void logSpawn(
      Spawn spawn,
      InputMetadataProvider inputMetadataProvider,
      SortedMap<PathFragment, ActionInput> inputMap,
      FileSystem fileSystem,
      Duration timeout,
      SpawnResult result)
      throws IOException, ExecException {
    try (SilentCloseable c = Profiler.instance().profile("logSpawn")) {
      SortedMap<Path, ActionInput> existingOutputs = listExistingOutputs(spawn, fileSystem);
      SpawnExec.Builder builder = SpawnExec.newBuilder();
      builder.addAllCommandArgs(spawn.getArguments());
      builder.addAllEnvironmentVariables(getEnvironmentVariables(spawn));

      ImmutableSet<? extends ActionInput> toolFiles = spawn.getToolFiles().toSet();

      try (SilentCloseable c1 = Profiler.instance().profile("logSpawn/inputs")) {
        for (Map.Entry<PathFragment, ActionInput> e : inputMap.entrySet()) {
          PathFragment displayPath = e.getKey();
          ActionInput input = e.getValue();

          if (input instanceof VirtualActionInput.EmptyActionInput) {
            // Do not include a digest, as it's a waste of space.
            builder.addInputsBuilder().setPath(displayPath.getPathString());
            continue;
          }

          boolean isTool =
              toolFiles.contains(input)
                  || (input instanceof TreeFileArtifact
                      && toolFiles.contains(((TreeFileArtifact) input).getParent()));

          Path contentPath = fileSystem.getPath(execRoot.getRelative(input.getExecPathString()));

          if (isInputDirectory(input, inputMetadataProvider)) {
            listDirectoryContents(
                displayPath, contentPath, builder::addInputs, inputMetadataProvider, isTool);
            continue;
          }

          Digest digest =
              computeDigest(
                  input, contentPath, inputMetadataProvider, xattrProvider, digestHashFunction);

          builder
              .addInputsBuilder()
              .setPath(displayPath.getPathString())
              .setDigest(digest)
              .setIsTool(isTool);
        }
      } catch (IOException e) {
        logger.atWarning().withCause(e).log("Error computing spawn input properties");
      }
      try (SilentCloseable c1 = Profiler.instance().profile("logSpawn/outputs")) {
        ArrayList<String> outputPaths = new ArrayList<>();
        for (ActionInput output : spawn.getOutputFiles()) {
          outputPaths.add(output.getExecPathString());
        }
        Collections.sort(outputPaths);
        builder.addAllListedOutputs(outputPaths);
        try {
          for (Map.Entry<Path, ActionInput> e : existingOutputs.entrySet()) {
            Path path = e.getKey();
            ActionInput output = e.getValue();
            if (path.isDirectory()) {
              listDirectoryContents(
                  output.getExecPath(),
                  path,
                  builder::addActualOutputs,
                  inputMetadataProvider,
                  /* isTool= */ false);
              continue;
            }
            builder
                .addActualOutputsBuilder()
                .setPath(output.getExecPathString())
                .setDigest(
                    computeDigest(
                        output, path, inputMetadataProvider, xattrProvider, digestHashFunction));
          }
        } catch (IOException ex) {
          logger.atWarning().withCause(ex).log("Error computing spawn output properties");
        }
      }
      builder.setRemotable(Spawns.mayBeExecutedRemotely(spawn));

      Platform platform = getPlatform(spawn, remoteOptions);
      if (platform != null) {
        builder.setPlatform(platform);
      }
      if (result.status() != SpawnResult.Status.SUCCESS) {
        builder.setStatus(result.status().toString());
      }
      if (!timeout.isZero()) {
        builder.setTimeoutMillis(timeout.toMillis());
      }
      builder.setCacheable(Spawns.mayBeCached(spawn));
      builder.setRemoteCacheable(Spawns.mayBeCachedRemotely(spawn));
      builder.setExitCode(result.exitCode());
      builder.setCacheHit(result.isCacheHit());
      builder.setRunner(result.getRunnerName());

      if (result.getDigest() != null) {
        builder.setDigest(result.getDigest());
      }

      builder.setMnemonic(spawn.getMnemonic());

      if (spawn.getTargetLabel() != null) {
        builder.setTargetLabel(spawn.getTargetLabel().toString());
      }

      builder.setMetrics(getSpawnMetricsProto(result));

      try (SilentCloseable c1 = Profiler.instance().profile("logSpawn/write")) {
        rawOutputStream.write(builder.build());
      }
    }
  }

  @Override
  public void close() throws IOException {
    rawOutputStream.close();

    if (convertedOutputStream == null) {
      // No conversion required.
      return;
    }

    try (InputStream in = tempPath.getInputStream()) {
      if (sorted) {
        StableSort.stableSort(in, convertedOutputStream);
      } else {
        while (in.available() > 0) {
          SpawnExec ex = SpawnExec.parseDelimitedFrom(in);
          convertedOutputStream.write(ex);
        }
      }
    } finally {
      try {
        tempPath.delete();
      } catch (IOException e) {
        // Intentionally ignored.
      }
    }
  }

  private SortedMap<Path, ActionInput> listExistingOutputs(Spawn spawn, FileSystem fileSystem) {
    TreeMap<Path, ActionInput> result = new TreeMap<>();
    for (ActionInput output : spawn.getOutputFiles()) {
      Path outputPath = fileSystem.getPath(execRoot.getRelative(output.getExecPathString()));
      // TODO(olaola): once symlink API proposal is implemented, report symlinks here.
      if (outputPath.exists()) {
        result.put(outputPath, output);
      }
    }
    return result;
  }

  /**
   * Expands a directory into its contents.
   *
   * <p>Note the difference between {@code displayPath} and {@code contentPath}: the first is where
   * the spawn can find the directory, while the second is where Bazel can find it. They're not the
   * same for a directory appearing in a runfiles or fileset tree.
   */
  private void listDirectoryContents(
      PathFragment displayPath,
      Path contentPath,
      Consumer<File> addFile,
      InputMetadataProvider inputMetadataProvider,
      boolean isTool)
      throws IOException {
    // TODO(olaola): once symlink API proposal is implemented, report symlinks here.
    List<Dirent> sortedDirent = new ArrayList<>(contentPath.readdir(Symlinks.NOFOLLOW));
    sortedDirent.sort(Comparator.comparing(Dirent::getName));

    for (Dirent dirent : sortedDirent) {
      String name = dirent.getName();
      PathFragment childDisplayPath = displayPath.getChild(name);
      Path childContentPath = contentPath.getChild(name);

      if (dirent.getType() == Dirent.Type.DIRECTORY) {
        listDirectoryContents(
            childDisplayPath, childContentPath, addFile, inputMetadataProvider, isTool);
        continue;
      }

      addFile.accept(
          File.newBuilder()
              .setPath(childDisplayPath.getPathString())
              .setDigest(
                  computeDigest(
                      null,
                      childContentPath,
                      inputMetadataProvider,
                      xattrProvider,
                      digestHashFunction))
              .setIsTool(isTool)
              .build());
    }
  }
}
