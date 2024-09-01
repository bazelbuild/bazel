// Copyright 2024 The Bazel Authors. All rights reserved.
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

import static com.google.common.collect.ImmutableList.toImmutableList;

import com.github.luben.zstd.ZstdInputStream;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.cmdline.LabelConstants;
import com.google.devtools.build.lib.exec.Protos.ExecLogEntry;
import com.google.devtools.build.lib.exec.Protos.File;
import com.google.devtools.build.lib.exec.Protos.SpawnExec;
import com.google.devtools.build.lib.util.io.MessageInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.SortedMap;
import java.util.SortedSet;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.function.Consumer;
import java.util.regex.Pattern;
import java.util.stream.Stream;
import javax.annotation.Nullable;

/** Reconstructs an execution log in expanded format from the compact format representation. */
public final class SpawnLogReconstructor implements MessageInputStream<SpawnExec> {
  private static final String EXTERNAL_PREFIX =
      LabelConstants.EXTERNAL_PATH_PREFIX.getPathString() + "/";
  // Matches paths of source files and generated files under an external repository.
  private static final Pattern EXTERNAL_PREFIX_PATTERN =
      Pattern.compile(
          "(%1$s|(blaze|bazel)-out/[^/]+/[^/]+/%1$s).*".formatted(Pattern.quote(EXTERNAL_PREFIX)));
  // Matches the prefix of a generated file path.
  private static final Pattern BAZEL_OUT_PREFIX_PATTERN =
      Pattern.compile("^(blaze|bazel)-out/[^/]+/[^/]+/");

  private final ZstdInputStream in;

  private sealed interface Input {
    String path();

    record File(Protos.File file) implements Input {
      @Override
      public String path() {
        return file.getPath();
      }
    }

    record Symlink(Protos.File symlink) implements Input {
      @Override
      public String path() {
        return symlink.getPath();
      }
    }

    record Directory(String path, Collection<Protos.File> files) implements Input {}
  }

  // Stores both Inputs and InputSets. Bazel uses consecutive IDs starting from 1, so we can use
  // an ArrayList to store them together efficiently.
  private final ArrayList<Object> inputMap = new ArrayList<>();
  private String hashFunctionName = "";
  private String workspaceRunfilesDirectory = "";

  public SpawnLogReconstructor(InputStream in) throws IOException {
    this.in = new ZstdInputStream(in);
    // Add a null entry for the 0th index as IDs are 1-based.
    inputMap.add(null);
  }

  @Override
  @Nullable
  public SpawnExec read() throws IOException {
    ExecLogEntry entry;
    while ((entry = ExecLogEntry.parseDelimitedFrom(in)) != null) {
      switch (entry.getTypeCase()) {
        case INVOCATION -> {
          hashFunctionName = entry.getInvocation().getHashFunctionName();
          workspaceRunfilesDirectory = entry.getInvocation().getWorkspaceRunfilesDirectory();
        }
        case FILE -> putInput(entry.getId(), reconstructFile(entry.getFile()));
        case DIRECTORY -> putInput(entry.getId(), reconstructDir(entry.getDirectory()));
        case UNRESOLVED_SYMLINK ->
            putInput(entry.getId(), reconstructSymlink(entry.getUnresolvedSymlink()));
        case RUNFILES_TREE ->
            putInput(entry.getId(), reconstructRunfilesDir(entry.getRunfilesTree()));
        case INPUT_SET -> putInputSet(entry.getId(), entry.getInputSet());
        case SPAWN -> {
          return reconstructSpawnExec(entry.getSpawn());
        }
        case SYMLINK_ACTION -> {
          // Symlink actions are not represented in the expanded format.
        }
        default ->
            throw new IOException(
                String.format("unknown entry type %d", entry.getTypeCase().getNumber()));
      }
    }
    return null;
  }

  private SpawnExec reconstructSpawnExec(ExecLogEntry.Spawn entry) throws IOException {
    SpawnExec.Builder builder =
        SpawnExec.newBuilder()
            .addAllCommandArgs(entry.getArgsList())
            .addAllEnvironmentVariables(entry.getEnvVarsList())
            .setTargetLabel(entry.getTargetLabel())
            .setMnemonic(entry.getMnemonic())
            .setExitCode(entry.getExitCode())
            .setStatus(entry.getStatus())
            .setRunner(entry.getRunner())
            .setCacheHit(entry.getCacheHit())
            .setRemotable(entry.getRemotable())
            .setCacheable(entry.getCacheable())
            .setRemoteCacheable(entry.getRemoteCacheable())
            .setTimeoutMillis(entry.getTimeoutMillis())
            .setMetrics(entry.getMetrics());

    if (entry.hasPlatform()) {
      builder.setPlatform(entry.getPlatform());
    }

    SortedMap<String, File> inputs = new TreeMap<>();
    visitInPostOrder(entry.getInputSetId(), file -> inputs.put(file.getPath(), file), input -> {});
    HashSet<String> toolInputs = new HashSet<>();
    visitInPostOrder(entry.getToolSetId(), file -> toolInputs.add(file.getPath()), input -> {});

    for (Map.Entry<String, File> e : inputs.entrySet()) {
      File file = e.getValue();
      if (toolInputs.contains(e.getKey())) {
        file = file.toBuilder().setIsTool(true).build();
      }
      builder.addInputs(file);
    }

    SortedSet<String> listedOutputs = new TreeSet<>();

    for (ExecLogEntry.Output output : entry.getOutputsList()) {
      switch (output.getTypeCase()) {
        case OUTPUT_ID -> {
          Input input = getInput(output.getOutputId());
          listedOutputs.add(input.path());
          switch (input) {
            case Input.File(File file) -> builder.addActualOutputs(file);
            case Input.Symlink(File symlink) -> builder.addActualOutputs(symlink);
            case Input.Directory(String ignored, Collection<File> files) ->
                builder.addAllActualOutputs(files);
          }
        }
        case INVALID_OUTPUT_PATH -> listedOutputs.add(output.getInvalidOutputPath());
        default ->
            throw new IOException(
                "unknown output type %d".formatted(output.getTypeCase().getNumber()));
      }
    }

    builder.addAllListedOutputs(listedOutputs);

    if (entry.hasDigest()) {
      builder.setDigest(entry.getDigest().toBuilder().setHashFunctionName(hashFunctionName));
    }

    return builder.build();
  }

  private void visitInPostOrder(int setId, Consumer<File> visitFile, Consumer<Input> visitInput)
      throws IOException {
    if (setId == 0) {
      return;
    }
    ArrayDeque<Integer> setsToVisit = new ArrayDeque<>();
    HashMap<Integer, Integer> previousVisitCount = new HashMap<>();
    setsToVisit.push(setId);
    while (!setsToVisit.isEmpty()) {
      int currentSetId = setsToVisit.pop();
      // In case order matters (it does for runfiles, but not for inputs), we visit the set in
      // post-order (corresponds to Order#COMPILE_ORDER). Transitive sets are visited before direct
      // children; both are visited in left-to-right order.
      switch (previousVisitCount.merge(currentSetId, 0, (oldValue, newValue) -> 1)) {
        case 0 -> {
          // First visit, queue transitive sets for visit before revisiting the current set.
          setsToVisit.push(currentSetId);
          for (int transitiveSetId :
              getInputSet(currentSetId).getTransitiveSetIdsList().reversed()) {
            if (!previousVisitCount.containsKey(transitiveSetId)) {
              setsToVisit.push(transitiveSetId);
            }
          }
        }
        case 1 -> {
          // Second visit, visit the direct inputs only.
          for (int inputId : getInputSet(currentSetId).getInputIdsList()) {
            if (previousVisitCount.put(inputId, 1) != null) {
              continue;
            }
            Input input = getInput(inputId);
            visitInput.accept(input);
            switch (input) {
              case Input.File(File file) -> visitFile.accept(file);
              case Input.Symlink(File symlink) -> visitFile.accept(symlink);
              case Input.Directory(String ignored, Collection<File> files) ->
                  files.forEach(visitFile);
            }
          }
        }
      }
    }
  }

  private Input.Directory reconstructDir(ExecLogEntry.Directory dir) {
    ImmutableList.Builder<File> builder =
        ImmutableList.builderWithExpectedSize(dir.getFilesCount());
    for (var dirFile : dir.getFilesList()) {
      builder.add(reconstructFile(dir, dirFile));
    }
    return new Input.Directory(dir.getPath(), builder.build());
  }

  private Input.File reconstructFile(ExecLogEntry.File entry) {
    return new Input.File(reconstructFile(null, entry));
  }

  private File reconstructFile(
      @Nullable ExecLogEntry.Directory parentDir, ExecLogEntry.File entry) {
    File.Builder builder = File.newBuilder();
    builder.setPath(
        parentDir != null ? parentDir.getPath() + "/" + entry.getPath() : entry.getPath());
    if (entry.hasDigest()) {
      builder.setDigest(entry.getDigest().toBuilder().setHashFunctionName(hashFunctionName));
    }
    return builder.build();
  }

  private static Input.Symlink reconstructSymlink(ExecLogEntry.UnresolvedSymlink entry) {
    return new Input.Symlink(
        File.newBuilder()
            .setPath(entry.getPath())
            .setSymlinkTargetPath(entry.getTargetPath())
            .build());
  }

  private Input.Directory reconstructRunfilesDir(ExecLogEntry.RunfilesTree runfilesTree)
      throws IOException {
    // Preserve the order of the symlinks and artifacts to resolve conflicts in the same order as
    // the real Runfiles implementation. See comment in CompactSpawnLogContext#logRunfilesTree
    // for more details.
    LinkedHashMap<String, File> runfiles = new LinkedHashMap<>();
    final boolean[] hasWorkspaceRunfilesDirectory = {false};

    for (var symlink : runfilesTree.getSymlinkTargetIdMap().entrySet()) {
      hasWorkspaceRunfilesDirectory[0] |=
          symlink.getKey().startsWith(workspaceRunfilesDirectory + "/");
      String newPath = runfilesTree.getPath() + "/" + symlink.getKey();
      for (var file : reconstructRunfilesSymlinkTarget(newPath, symlink.getValue())) {
        runfiles.put(newPath, file);
      }
    }

    LinkedHashSet<File> flattenedArtifacts = new LinkedHashSet<>();
    visitInPostOrder(
        runfilesTree.getInputSetId(),
        flattenedArtifacts::add,
        // This is bug-for-bug compatible with the implementation in Runfiles by considering
        // an empty non-external directory as a runfiles entry under the workspace runfiles
        // directory even though it won't be materialized as one.
        input ->
            hasWorkspaceRunfilesDirectory[0] |=
                !EXTERNAL_PREFIX_PATTERN.matcher(input.path()).matches());
    flattenedArtifacts.stream()
        .flatMap(
            file ->
                getRunfilesPaths(file.getPath(), runfilesTree.getLegacyExternalRunfiles())
                    .map(
                        relativePath ->
                            file.toBuilder()
                                .setPath(runfilesTree.getPath() + "/" + relativePath)
                                .build()))
        .forEach(file -> runfiles.put(file.getPath(), file));
    if (!runfilesTree.getLegacyExternalRunfiles() && !hasWorkspaceRunfilesDirectory[0]) {
      String dotRunfilePath =
          "%s/%s/.runfile".formatted(runfilesTree.getPath(), workspaceRunfilesDirectory);
      runfiles.put(dotRunfilePath, File.newBuilder().setPath(dotRunfilePath).build());
    }
    return new Input.Directory(runfilesTree.getPath(), ImmutableList.copyOf(runfiles.values()));
  }

  private Stream<String> getRunfilesPaths(String originalPath, boolean legacyExternalRunfiles) {
    String path = BAZEL_OUT_PREFIX_PATTERN.matcher(originalPath).replaceFirst("");
    if (path.startsWith(EXTERNAL_PREFIX)) {
      Stream.Builder<String> paths = Stream.builder();
      paths.add(path.substring(EXTERNAL_PREFIX.length()));
      if (legacyExternalRunfiles) {
        paths.add(workspaceRunfilesDirectory + "/" + path);
      }
      return paths.build();
    }
    return Stream.of(workspaceRunfilesDirectory + "/" + path);
  }

  private Collection<File> reconstructRunfilesSymlinkTarget(String newPath, int targetId)
      throws IOException {
    if (targetId == 0) {
      return ImmutableList.of(File.newBuilder().setPath(newPath).build());
    }
    return switch (getInput(targetId)) {
      case Input.File(File file) -> ImmutableList.of(file.toBuilder().setPath(newPath).build());
      case Input.Symlink(File symlink) ->
          ImmutableList.of(symlink.toBuilder().setPath(newPath).build());
      case Input.Directory(String path, Collection<File> files) ->
          files.stream()
              .map(
                  file ->
                      file.toBuilder()
                          .setPath(newPath + file.getPath().substring(path.length()))
                          .build())
              .collect(toImmutableList());
    };
  }

  private void putInput(int id, Input input) throws IOException {
    putEntry(id, input);
  }

  private void putInputSet(int id, ExecLogEntry.InputSet inputSet) throws IOException {
    putEntry(id, inputSet);
  }

  private void putEntry(int id, Object entry) throws IOException {
    if (id == 0) {
      // The entry won't be referenced, so we don't need to store it.
      return;
    }
    // Bazel emits consecutive non-zero IDs.
    if (id != inputMap.size()) {
      throw new IOException(
          "ids must be consecutive, got %d after %d".formatted(id, inputMap.size()));
    }
    inputMap.add(
        switch (entry) {
          // Unwrap trivial wrappers to reduce retained memory usage.
          case Input.File file -> file.file;
          case Input.Symlink symlink -> symlink.symlink;
          default -> entry;
        });
  }

  private Input getInput(int id) throws IOException {
    Object value = inputMap.get(id);
    return switch (value) {
      case Input input -> input;
      case Protos.File file ->
          file.getSymlinkTargetPath().isEmpty() ? new Input.File(file) : new Input.Symlink(file);
      case null -> throw new IOException("referenced input %d is missing".formatted(id));
      default -> throw new IOException("entry %d is not an input: %s".formatted(id, value));
    };
  }

  private ExecLogEntry.InputSet getInputSet(int id) throws IOException {
    Object value = inputMap.get(id);
    return switch (value) {
      case ExecLogEntry.InputSet inputSet -> inputSet;
      case null -> throw new IOException("referenced input set %d is missing".formatted(id));
      default -> throw new IOException("entry %d is not an input set: %s".formatted(id, value));
    };
  }

  @Override
  public void close() throws IOException {
    in.close();
  }
}
