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
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.cmdline.LabelConstants;
import com.google.devtools.build.lib.exec.Protos.ExecLogEntry;
import com.google.devtools.build.lib.exec.Protos.File;
import com.google.devtools.build.lib.exec.Protos.SpawnExec;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.util.io.MessageInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayDeque;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.SortedMap;
import java.util.SortedSet;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.function.Supplier;
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

  private final HashMap<Integer, File> fileMap = new HashMap<>();
  private final HashMap<Integer, Pair<String, Collection<File>>> dirMap = new HashMap<>();
  private final HashMap<Integer, File> symlinkMap = new HashMap<>();
  private final HashMap<Integer, ExecLogEntry.InputSet> setMap = new HashMap<>();
  private String hashFunctionName = "";
  private String workspaceRunfilesDirectory = "";

  public SpawnLogReconstructor(InputStream in) throws IOException {
    this.in = new ZstdInputStream(in);
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
        case FILE -> fileMap.put(entry.getId(), reconstructFile(entry.getFile()));
        case DIRECTORY -> dirMap.put(entry.getId(), reconstructDir(entry.getDirectory()));
        case UNRESOLVED_SYMLINK ->
            symlinkMap.put(entry.getId(), reconstructSymlink(entry.getUnresolvedSymlink()));
        case RUNFILES_TREE ->
            dirMap.put(entry.getId(), reconstructRunfilesDir(entry.getRunfilesTree()));
        case INPUT_SET -> setMap.put(entry.getId(), entry.getInputSet());
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

    SortedMap<String, File> inputs = reconstructInputs(entry.getInputSetId(), TreeMap::new).inputs;
    SortedMap<String, File> toolInputs =
        reconstructInputs(entry.getToolSetId(), TreeMap::new).inputs;

    for (Map.Entry<String, File> e : inputs.entrySet()) {
      File file = e.getValue();
      if (toolInputs.containsKey(e.getKey())) {
        file = file.toBuilder().setIsTool(true).build();
      }
      builder.addInputs(file);
    }

    SortedSet<String> listedOutputs = new TreeSet<>();

    for (ExecLogEntry.Output output : entry.getOutputsList()) {
      switch (output.getTypeCase()) {
        case FILE_ID -> {
          File file = getFromMap(fileMap, output.getFileId());
          listedOutputs.add(file.getPath());
          builder.addActualOutputs(file);
        }
        case DIRECTORY_ID -> {
          Pair<String, Collection<File>> dir = getFromMap(dirMap, output.getDirectoryId());
          listedOutputs.add(dir.getFirst());
          for (File dirFile : dir.getSecond()) {
            builder.addActualOutputs(dirFile);
          }
        }
        case UNRESOLVED_SYMLINK_ID -> {
          File symlink = getFromMap(symlinkMap, output.getUnresolvedSymlinkId());
          listedOutputs.add(symlink.getPath());
          builder.addActualOutputs(symlink);
        }
        case INVALID_OUTPUT_PATH -> listedOutputs.add(output.getInvalidOutputPath());
        default ->
            throw new IOException(
                String.format("unknown output type %d", output.getTypeCase().getNumber()));
      }
    }

    builder.addAllListedOutputs(listedOutputs);

    if (entry.hasDigest()) {
      builder.setDigest(entry.getDigest().toBuilder().setHashFunctionName(hashFunctionName));
    }

    return builder.build();
  }

  private record FlattenedInputSet<T extends Map<String, File>>(
      T inputs, boolean hasWorkspaceRunfilesDirectory) {}

  private <T extends Map<String, File>> FlattenedInputSet<T> reconstructInputs(
      int setId, Supplier<T> newMap) throws IOException {
    T inputs = newMap.get();
    ArrayDeque<Integer> setsToVisit = new ArrayDeque<>();
    HashSet<Integer> visited = new HashSet<>();
    boolean hasWorkspaceRunfilesDirectory = false;
    if (setId != 0) {
      setsToVisit.addLast(setId);
      visited.add(setId);
    }
    while (!setsToVisit.isEmpty()) {
      ExecLogEntry.InputSet set = getFromMap(setMap, setsToVisit.removeFirst());
      for (int fileId : set.getFileIdsList()) {
        if (visited.add(fileId)) {
          File file = getFromMap(fileMap, fileId);
          inputs.put(file.getPath(), file);
          if (!hasWorkspaceRunfilesDirectory
              && !EXTERNAL_PREFIX_PATTERN.matcher(file.getPath()).matches()) {
            hasWorkspaceRunfilesDirectory = true;
          }
        }
      }
      for (int dirId : Iterables.concat(set.getDirectoryIdsList(), set.getRunfilesTreeIdsList())) {
        if (visited.add(dirId)) {
          Pair<String, Collection<File>> dir = getFromMap(dirMap, dirId);
          for (File dirFile : dir.getSecond()) {
            inputs.put(dirFile.getPath(), dirFile);
          }
          // This is bug-for-bug compatible with the implementation in Runfiles by considering
          // an empty non-external directory as a runfiles entry under the workspace runfiles
          // directory even though it won't be materialized as one.
          if (!hasWorkspaceRunfilesDirectory
              && !EXTERNAL_PREFIX_PATTERN.matcher(dir.getFirst()).matches()) {
            hasWorkspaceRunfilesDirectory = true;
          }
        }
      }
      for (int symlinkId : set.getUnresolvedSymlinkIdsList()) {
        if (visited.add(symlinkId)) {
          File symlink = getFromMap(symlinkMap, symlinkId);
          inputs.put(symlink.getPath(), symlink);
          if (!hasWorkspaceRunfilesDirectory
              && !EXTERNAL_PREFIX_PATTERN.matcher(symlink.getPath()).matches()) {
            hasWorkspaceRunfilesDirectory = true;
          }
        }
      }
      for (int transitiveSetId : set.getTransitiveSetIdsList()) {
        if (visited.add(transitiveSetId)) {
          setsToVisit.addLast(transitiveSetId);
        }
      }
    }
    return new FlattenedInputSet<>(inputs, hasWorkspaceRunfilesDirectory);
  }

  private Pair<String, Collection<File>> reconstructDir(ExecLogEntry.Directory dir) {
    ImmutableList.Builder<File> builder =
        ImmutableList.builderWithExpectedSize(dir.getFilesCount());
    for (ExecLogEntry.File dirFile : dir.getFilesList()) {
      builder.add(reconstructFile(dir, dirFile));
    }
    return Pair.of(dir.getPath(), builder.build());
  }

  private File reconstructFile(ExecLogEntry.File entry) {
    return reconstructFile(null, entry);
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

  private static File reconstructSymlink(ExecLogEntry.UnresolvedSymlink entry) {
    return File.newBuilder()
        .setPath(entry.getPath())
        .setSymlinkTargetPath(entry.getTargetPath())
        .build();
  }

  private Pair<String, Collection<File>> reconstructRunfilesDir(
      ExecLogEntry.RunfilesTree runfilesTree) throws IOException {
    // Preserve the order of the inputs to resolve conflicts in the same order as the real Runfiles
    // implementation.
    var flattenedInputs = reconstructInputs(runfilesTree.getInputSetId(), LinkedHashMap::new);
    boolean hasWorkspaceRunfilesDirectory = flattenedInputs.hasWorkspaceRunfilesDirectory;
    LinkedHashMap<String, File> builder =
        new LinkedHashMap<>(runfilesTree.getSymlinksCount() + flattenedInputs.inputs.size());
    for (var symlink : runfilesTree.getSymlinksMap().entrySet()) {
      hasWorkspaceRunfilesDirectory |=
          symlink.getKey().startsWith(workspaceRunfilesDirectory + "/");
      String newPath = runfilesTree.getPath() + "/" + symlink.getKey();
      for (var file : reconstructRunfilesSymlinkTarget(newPath, symlink.getValue())) {
        builder.put(newPath, file);
      }
    }
    Collection<File> inputs = flattenedInputs.inputs.values();
    inputs.stream()
        .flatMap(
            file ->
                getRunfilesPaths(file.getPath(), runfilesTree.getLegacyExternalRunfiles())
                    .map(
                        relativePath ->
                            file.toBuilder()
                                .setPath(runfilesTree.getPath() + "/" + relativePath)
                                .build()))
        .forEach(file -> builder.put(file.getPath(), file));
    if (!runfilesTree.getLegacyExternalRunfiles() && !hasWorkspaceRunfilesDirectory) {
      String dotRunfilePath =
          "%s/%s/.runfile".formatted(runfilesTree.getPath(), workspaceRunfilesDirectory);
      builder.put(dotRunfilePath, File.newBuilder().setPath(dotRunfilePath).build());
    }
    return Pair.of(runfilesTree.getPath(), ImmutableList.copyOf(builder.values()));
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

  private Collection<File> reconstructRunfilesSymlinkTarget(
      String newPath, ExecLogEntry.RunfilesTree.SymlinkTarget target) throws IOException {
    return switch (target.getTargetCase()) {
      case EMPTY_FILE -> ImmutableList.of(File.newBuilder().setPath(newPath).build());
      case FILE_ID -> {
        var file = getFromMap(fileMap, target.getFileId());
        yield ImmutableList.of(file.toBuilder().setPath(newPath).build());
      }
      case UNRESOLVED_SYMLINK_ID -> {
        var symlink = getFromMap(symlinkMap, target.getUnresolvedSymlinkId());
        yield ImmutableList.of(symlink.toBuilder().setPath(newPath).build());
      }
      case DIRECTORY_ID -> {
        var dir = getFromMap(dirMap, target.getDirectoryId());
        yield dir.getSecond().stream()
            .map(
                file ->
                    file.toBuilder()
                        .setPath(newPath + file.getPath().substring(dir.getFirst().length()))
                        .build())
            .collect(toImmutableList());
      }
      default ->
          throw new IOException(
              String.format("unknown target type %d", target.getTargetCase().getNumber()));
    };
  }

  private static <T> T getFromMap(Map<Integer, T> map, int id) throws IOException {
    T value = map.get(id);
    if (value == null) {
      throw new IOException(String.format("referenced entry %d is missing or has wrong type", id));
    }
    return value;
  }

  @Override
  public void close() throws IOException {
    in.close();
  }
}
