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

import static com.google.common.base.Preconditions.checkState;
import static com.google.common.collect.ImmutableList.toImmutableList;

import com.github.luben.zstd.ZstdInputStream;
import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
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
import java.util.function.BiConsumer;
import java.util.function.Consumer;
import java.util.regex.MatchResult;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Stream;
import javax.annotation.Nullable;

/** Reconstructs an execution log in expanded format from the compact format representation. */
public final class SpawnLogReconstructor implements MessageInputStream<SpawnExec> {
  // The path of the repo mapping manifest file under the runfiles tree.
  private static final String REPO_MAPPING_MANIFEST = "_repo_mapping";

  // Examples:
  // * bazel-out/k8-fastbuild/bin/pkg/file.txt (repo: null, path: "pkg/file.txt")
  // * bazel-out/k8-fastbuild/bin/external/some_repo/pkg/file.txt (repo: "some_repo", path:
  //   "pkg/file.txt")
  private static final Pattern DEFAULT_GENERATED_FILE_RUNFILES_PATH_PATTERN =
      Pattern.compile("(?:bazel|blaze)-out/[^/]+/[^/]+/(?:external/(?<repo>[^/]+)/)?(?<path>.+)");

  // Examples:
  // * pkg/file.txt (repo: null, path: "pkg/file.txt")
  // * external/some_repo/pkg/file.txt (repo: "some_repo", path: "pkg/file.txt")
  private static final Pattern DEFAULT_SOURCE_FILE_RUNFILES_PATH_PATTERN =
      Pattern.compile("(?:external/(?<repo>[^/]+)/)?(?<path>.+)");

  // Examples:
  // * bazel-out/k8-fastbuild/bin/pkg/file.txt (repo: null, path: "pkg/file.txt")
  // * bazel-out/some_repo/k8-fastbuild/bin/pkg/file.txt (repo: "some_repo", path: "pkg/file.txt")
  // * bazel-out/k8-fastbuild/k8-fastbuild/bin/pkg/file.txt (repo: "k8-fastbuild", path:
  //   "pkg/file.txt")
  //
  // Repo names are distinguished from mnemonics via a positive lookahead on the following segment,
  // which in the case of a repo name is a mnemonic and thus contains a hyphen, whereas a mnemonic
  // is followed by an output directory name, which does not contain a hyphen unless it is
  // "coverage-metadata" (which in turn is not likely to be a mnemonic).
  private static final Pattern SIBLING_LAYOUT_GENERATED_FILE_RUNFILES_PATH_PATTERN =
      Pattern.compile(
          "(?:bazel|blaze)-out/(?:(?<repo>[^/]+(?=/[^/]+-[^/]+/)(?!/coverage-metadata/))/)?[^/]+/[^/]+/(?<path>.+)");

  // Examples:
  // * pkg/file.txt (repo: null, path: "pkg/file.txt")
  // * ../some_repo/pkg/file.txt (repo: "some_repo", path: "pkg/file.txt")
  private static final Pattern SIBLING_LAYOUT_SOURCE_FILE_RUNFILES_PATH_PATTERN =
      Pattern.compile("(?:\\.\\./(?<repo>[^/]+)/)?(?<path>.+)");

  private final ZstdInputStream in;

  /** Represents a reconstructed input file, symlink, or directory. */
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
  private boolean siblingRepositoryLayout = false;

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
          siblingRepositoryLayout = entry.getInvocation().getSiblingRepositoryLayout();
        }
        case FILE -> putInput(entry.getId(), reconstructFile(entry.getFile()));
        case DIRECTORY -> putInput(entry.getId(), reconstructDir(entry.getDirectory()));
        case UNRESOLVED_SYMLINK ->
            putInput(entry.getId(), reconstructSymlink(entry.getUnresolvedSymlink()));
        case RUNFILES_TREE ->
            putInput(entry.getId(), reconstructRunfilesDir(entry.getRunfilesTree()));
        case INPUT_SET -> putInputSet(entry.getId(), entry.getInputSet());
        case SYMLINK_ENTRY_SET -> putSymlinkEntrySet(entry.getId(), entry.getSymlinkEntrySet());
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
            .setTimeoutMillis(entry.getTimeoutMillis());

    if (entry.hasMetrics()) {
      builder.setMetrics(entry.getMetrics());
    }

    if (entry.hasPlatform()) {
      builder.setPlatform(entry.getPlatform());
    }

    SortedMap<String, File> inputs = new TreeMap<>();
    visitInputSet(entry.getInputSetId(), file -> inputs.put(file.getPath(), file), input -> {});
    HashSet<String> toolInputs = new HashSet<>();
    visitInputSet(entry.getToolSetId(), file -> toolInputs.add(file.getPath()), input -> {});

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

  private void visitInputSet(int inputSetId, Consumer<File> visitFile, Consumer<Input> visitInput)
      throws IOException {
    if (inputSetId == 0) {
      return;
    }
    ArrayDeque<Integer> setsToVisit = new ArrayDeque<>();
    HashMap<Integer, Integer> previousVisitCount = new HashMap<>();
    setsToVisit.push(inputSetId);
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
        default ->
            throw new IllegalStateException(
                "expected visit count to be 0 or 1, was " + previousVisitCount.get(currentSetId));
      }
    }
  }

  private void visitSymlinkEntries(
      ExecLogEntry.RunfilesTree runfilesTree,
      boolean rootSymlinks,
      BiConsumer<String, Collection<File>> entryConsumer)
      throws IOException {
    int symlinkEntrySetId =
        rootSymlinks ? runfilesTree.getRootSymlinksId() : runfilesTree.getSymlinksId();
    if (symlinkEntrySetId == 0) {
      return;
    }
    ArrayDeque<Integer> setsToVisit = new ArrayDeque<>();
    HashMap<Integer, Integer> previousVisitCount = new HashMap<>();
    setsToVisit.push(symlinkEntrySetId);
    while (!setsToVisit.isEmpty()) {
      int currentSetId = setsToVisit.pop();
      // As order matters, we visit the set in post-order (corresponds to Order#COMPILE_ORDER).
      // Transitive sets are visited before direct children; both are visited in left-to-right
      // order.
      switch (previousVisitCount.merge(currentSetId, 0, (oldValue, newValue) -> 1)) {
        case 0 -> {
          // First visit, queue transitive sets for visit before revisiting the current set.
          setsToVisit.push(currentSetId);
          for (int transitiveSetId :
              getSymlinkEntrySet(currentSetId).getTransitiveSetIdsList().reversed()) {
            if (!previousVisitCount.containsKey(transitiveSetId)) {
              setsToVisit.push(transitiveSetId);
            }
          }
        }
        case 1 -> {
          // Second visit, visit the direct entries only.
          for (var pathAndInputId :
              getSymlinkEntrySet(currentSetId).getDirectEntriesMap().entrySet()) {
            String runfilesTreeRelativePath;
            if (rootSymlinks) {
              runfilesTreeRelativePath = pathAndInputId.getKey();
            } else if (pathAndInputId.getKey().startsWith("../")) {
              runfilesTreeRelativePath = pathAndInputId.getKey().substring(3);
            } else {
              runfilesTreeRelativePath = workspaceRunfilesDirectory + "/" + pathAndInputId.getKey();
            }
            String path = runfilesTree.getPath() + "/" + runfilesTreeRelativePath;
            entryConsumer.accept(
                runfilesTreeRelativePath,
                reconstructRunfilesSymlinkTarget(path, pathAndInputId.getValue()));
          }
        }
        default ->
            throw new IllegalStateException(
                "expected visit count to be 0 or 1, was " + previousVisitCount.get(currentSetId));
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
    // In case of path collisions, runfiles should be collected in the following order, with
    // later sources overriding earlier ones (see
    // com.google.devtools.build.lib.analysis.Runfiles#getRunfilesInputs):
    //
    // 1. symlinks
    // 2. artifacts at canonical locations
    // 3. empty files
    // 4. root symlinks
    // 5. the _repo_mapping file with the repo mapping manifest
    // 6. the <workspace runfiles directory>/.runfile file (if the workspace runfiles directory
    //    wouldn't exist otherwise)
    //
    // Within each group represented by a nested set, the entries are traversed in postorder (i.e.
    // the transitive sets are visited before the direct children). This is important to resolve
    // conflicts in the same order as the real Runfiles implementation.
    LinkedHashMap<String, File> runfiles = new LinkedHashMap<>();
    final boolean[] hasWorkspaceRunfilesDirectory = {false};

    visitSymlinkEntries(
        runfilesTree,
        /* rootSymlinks= */ false,
        (rootRelativePath, files) -> {
          hasWorkspaceRunfilesDirectory[0] |=
              rootRelativePath.startsWith(workspaceRunfilesDirectory + "/");
          for (var file : files) {
            runfiles.put(file.getPath(), file);
          }
        });

    LinkedHashSet<File> flattenedArtifacts = new LinkedHashSet<>();
    visitInputSet(
        runfilesTree.getInputSetId(),
        flattenedArtifacts::add,
        // This is bug-for-bug compatible with the implementation in Runfiles by considering
        // an empty non-external directory as a runfiles entry under the workspace runfiles
        // directory even though it won't be materialized as one.
        input -> hasWorkspaceRunfilesDirectory[0] |= hasWorkspaceRunfilesDirectory(input.path()));
    flattenedArtifacts.stream()
        .flatMap(
            file ->
                getRunfilesPaths(file.getPath())
                    .map(
                        relativePath ->
                            file.toBuilder()
                                .setPath(runfilesTree.getPath() + "/" + relativePath)
                                .build()))
        .forEach(file -> runfiles.put(file.getPath(), file));

    for (String emptyFile : runfilesTree.getEmptyFilesList()) {
      // Empty files are only created as siblings or parents of existing files, so they can't
      // by themselves create a workspace runfiles directory if it wouldn't exist otherwise.
      String newPath;
      if (emptyFile.startsWith("../")) {
        newPath = runfilesTree.getPath() + "/" + emptyFile.substring(3);
      } else {
        newPath = runfilesTree.getPath() + "/" + workspaceRunfilesDirectory + "/" + emptyFile;
      }
      runfiles.put(newPath, File.newBuilder().setPath(newPath).build());
    }

    visitSymlinkEntries(
        runfilesTree,
        /* rootSymlinks= */ true,
        (rootRelativePath, files) -> {
          hasWorkspaceRunfilesDirectory[0] |=
              rootRelativePath.startsWith(workspaceRunfilesDirectory + "/");
          for (var file : files) {
            runfiles.put(file.getPath(), file);
          }
        });

    if (runfilesTree.hasRepoMappingManifest()) {
      runfiles.put(
          REPO_MAPPING_MANIFEST,
          File.newBuilder()
              .setPath(runfilesTree.getPath() + "/" + REPO_MAPPING_MANIFEST)
              .setDigest(runfilesTree.getRepoMappingManifest().getDigest())
              .build());
    }

    if (!hasWorkspaceRunfilesDirectory[0]) {
      String dotRunfilePath =
          "%s/%s/.runfile".formatted(runfilesTree.getPath(), workspaceRunfilesDirectory);
      runfiles.put(dotRunfilePath, File.newBuilder().setPath(dotRunfilePath).build());
    }
    // Copy to avoid retaining the entire runfiles map.
    return new Input.Directory(runfilesTree.getPath(), ImmutableList.copyOf(runfiles.values()));
  }

  @VisibleForTesting
  static MatchResult extractRunfilesPath(String execPath, boolean siblingRepositoryLayout) {
    Matcher matcher =
        (siblingRepositoryLayout
                ? SIBLING_LAYOUT_GENERATED_FILE_RUNFILES_PATH_PATTERN
                : DEFAULT_GENERATED_FILE_RUNFILES_PATH_PATTERN)
            .matcher(execPath);
    if (matcher.matches()) {
      return matcher;
    }
    matcher =
        (siblingRepositoryLayout
                ? SIBLING_LAYOUT_SOURCE_FILE_RUNFILES_PATH_PATTERN
                : DEFAULT_SOURCE_FILE_RUNFILES_PATH_PATTERN)
            .matcher(execPath);
    checkState(matcher.matches());
    return matcher;
  }

  private boolean hasWorkspaceRunfilesDirectory(String path) {
    return extractRunfilesPath(path, siblingRepositoryLayout).group("repo") == null;
  }

  private Stream<String> getRunfilesPaths(String execPath) {
    MatchResult matchResult = extractRunfilesPath(execPath, siblingRepositoryLayout);
    String repo = matchResult.group("repo");
    String repoRelativePath = matchResult.group("path");
    if (repo == null) {
      return Stream.of(workspaceRunfilesDirectory + "/" + repoRelativePath);
    } else {
      Stream.Builder<String> paths = Stream.builder();
      paths.add(repo + "/" + repoRelativePath);
      return paths.build();
    }
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

  private void putSymlinkEntrySet(int id, ExecLogEntry.SymlinkEntrySet symlinkEntries)
      throws IOException {
    putEntry(id, symlinkEntries);
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

  private ExecLogEntry.SymlinkEntrySet getSymlinkEntrySet(int id) throws IOException {
    Object value = inputMap.get(id);
    return switch (value) {
      case ExecLogEntry.SymlinkEntrySet symlinkEntries -> symlinkEntries;
      case null ->
          throw new IOException("referenced set of symlink entries %d is missing".formatted(id));
      default ->
          throw new IOException(
              "entry %d is not a set of symlink entries: %s".formatted(id, value));
    };
  }

  @Override
  public void close() throws IOException {
    in.close();
  }
}
