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

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.base.Preconditions.checkState;
import static com.google.devtools.build.lib.profiler.ProfilerTask.SPAWN_LOG;
import static com.google.devtools.build.lib.util.StringEncoding.internalToUnicode;

import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.lib.actions.AbstractAction;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.InputMetadataProvider;
import com.google.devtools.build.lib.actions.ParamFileActionInput;
import com.google.devtools.build.lib.actions.RunfilesTree;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.SpawnResult;
import com.google.devtools.build.lib.actions.Spawns;
import com.google.devtools.build.lib.actions.VirtualActionInput;
import com.google.devtools.build.lib.analysis.SymlinkEntry;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelConstants;
import com.google.devtools.build.lib.collect.ObjectToIntMap;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.compress.CompressionService;
import com.google.devtools.build.lib.concurrent.AbstractQueueVisitor;
import com.google.devtools.build.lib.concurrent.ErrorClassifier;
import com.google.devtools.build.lib.concurrent.NamedForkJoinPool;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.exec.Protos.Digest;
import com.google.devtools.build.lib.exec.Protos.ExecLogEntry;
import com.google.devtools.build.lib.exec.Protos.Platform;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import com.google.devtools.build.lib.remote.options.RemoteOptions;
import com.google.devtools.build.lib.skyframe.TreeArtifactValue;
import com.google.devtools.build.lib.util.StringEncoding;
import com.google.devtools.build.lib.util.io.AsynchronousMessageOutputStream;
import com.google.devtools.build.lib.util.io.MessageOutputStream;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.Dirent;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Symlinks;
import com.google.devtools.build.lib.vfs.XattrProvider;
import com.google.errorprone.annotations.CheckReturnValue;
import java.io.BufferedOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.UncheckedIOException;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.SortedMap;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Predicate;
import java.util.function.Supplier;
import javax.annotation.Nullable;
import javax.annotation.concurrent.GuardedBy;

/** A {@link SpawnLogContext} implementation that produces a log in compact format. */
public class CompactSpawnLogContext extends SpawnLogContext {

  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  private static final Comparator<ExecLogEntry.File> EXEC_LOG_ENTRY_FILE_COMPARATOR =
      Comparator.comparing(ExecLogEntry.File::getPath);

  private static final ForkJoinPool VISITOR_POOL =
      NamedForkJoinPool.newNamedPool(
          "execlog-directory-visitor", Runtime.getRuntime().availableProcessors());

  /** Visitor for use in {@link #visitDirectory}. */
  protected interface DirectoryChildVisitor {
    void visit(Path path) throws IOException;
  }

  private static class DirectoryVisitor extends AbstractQueueVisitor {
    private final Path rootDir;
    private final DirectoryChildVisitor childVisitor;

    private DirectoryVisitor(Path rootDir, DirectoryChildVisitor childVisitor) {
      super(
          VISITOR_POOL,
          ExecutorOwnership.SHARED,
          ExceptionHandlingMode.FAIL_FAST,
          ErrorClassifier.DEFAULT);
      this.rootDir = checkNotNull(rootDir);
      this.childVisitor = checkNotNull(childVisitor);
    }

    private void run() throws IOException, InterruptedException {
      execute(() -> visitSubdirectory(rootDir));
      try {
        awaitQuiescence(true);
      } catch (UncheckedIOException e) {
        throw e.getCause();
      }
    }

    private void visitSubdirectory(Path dir) {
      try {
        for (Dirent dirent : dir.readdir(Symlinks.FOLLOW)) {
          Path child = dir.getChild(dirent.getName());
          if (dirent.getType() == Dirent.Type.DIRECTORY) {
            execute(() -> visitSubdirectory(child));
            continue;
          }
          childVisitor.visit(child);
        }
      } catch (IOException e) {
        throw new UncheckedIOException(e);
      }
    }
  }

  /**
   * Visits a directory hierarchy in parallel.
   *
   * <p>Calls {@code childVisitor} for every descendant path of {@code rootDir} that isn't itself a
   * directory, following symlinks. The visitor may be concurrently called by multiple threads, and
   * must synchronize accesses to shared data.
   */
  private void visitDirectory(Path rootDir, DirectoryChildVisitor childVisitor)
      throws IOException, InterruptedException {
    new DirectoryVisitor(rootDir, childVisitor).run();
  }

  private interface ExecLogEntrySupplier {
    ExecLogEntry.Builder get() throws IOException, InterruptedException;
  }

  private final CompressionService compressionService;
  private final PathFragment execRoot;
  @Nullable private final RemoteOptions remoteOptions;
  private final DigestHashFunction digestHashFunction;
  private final XattrProvider xattrProvider;
  private final UUID invocationId;
  private final ExtendedEventHandler reporter;
  private final AtomicBoolean outputLoggingFailed = new AtomicBoolean(false);

  /**
   * Stands in for an entry in {@link #pendingEntries} while the entry is being computed.
   *
   * <p>Counted down once the entry has been written to the log or has failed to be computed.
   *
   * <p>We extend {@link CountDownLatch} directly with a primitive {@code int} ID instead of using a
   * {@code Future<Integer>} to avoid extra allocations per attempt, and because waiting threads
   * retry computing the entry when the computing thread fails or is interrupted, rather than
   * rethrowing its exception.
   */
  private static final class PendingEntry extends CountDownLatch {
    // The entry ID, or 0 if the entry failed to be computed. Must only be written before counting
    // down and only be read after awaiting.
    private int id;

    PendingEntry() {
      super(1);
    }
  }

  // Maps a key identifying an entry that has been written to the log into its ID.
  // Each key is either a NestedSet.Node or the String path of a file, directory, symlink or
  // runfiles tree.
  // Only entries that are likely to be referenced by future entries are stored.
  // The ID is only stored after the entry has been handed to the output stream, so that an ID
  // obtained from this map always identifies an entry that precedes every entry written afterwards.
  // The map holds an entry for every file, directory, symlink, runfiles tree and shared nested set
  // mentioned in the build, which is millions of entries for a large build, and is looked up by
  // every thread that logs a spawn. Use a specialized map for minimal memory footprint, whose
  // lookups don't all contend for the same lock.
  private final ObjectToIntMap<Object> entryIds = new ObjectToIntMap<>();

  // Maps the key of an entry that is being computed into its PendingEntry, which lets other threads
  // that need the same entry wait for it instead of computing it again.
  // A key is only removed after the ID of its entry has been stored in entryIds, unless the entry
  // failed to be computed. A thread that has added a key can thus tell from entryIds whether the
  // entry was written in the meantime.
  // Only holds the entries that are being computed at a given time, so its memory footprint
  // doesn't matter.
  private final ConcurrentHashMap<Object, PendingEntry> pendingEntries = new ConcurrentHashMap<>();

  // Returned by entryIds for a key that it doesn't contain.
  private static final int ABSENT = -1;

  // The next available entry ID.
  // Guarded by the lock that is held while an entry with an ID is handed to the output stream, so
  // that IDs are consecutive in the order in which entries appear in the log.
  @GuardedBy("this")
  private int nextEntryId = 1;

  // Output stream to write to.
  private final MessageOutputStream<ExecLogEntry> outputStream;

  public CompactSpawnLogContext(
      BufferedOutputStream out,
      String displayName,
      PathFragment execRoot,
      @Nullable RemoteOptions remoteOptions,
      DigestHashFunction digestHashFunction,
      XattrProvider xattrProvider,
      CompressionService compressionService,
      UUID invocationId,
      ExtendedEventHandler reporter,
      Predicate<Spawn> logSpawnPredicate,
      boolean expandParamFiles)
      throws IOException, InterruptedException {
    super(logSpawnPredicate, expandParamFiles);
    this.execRoot = execRoot;
    this.remoteOptions = remoteOptions;
    this.digestHashFunction = digestHashFunction;
    this.xattrProvider = xattrProvider;
    this.compressionService = compressionService;
    this.invocationId = invocationId;
    this.reporter = reporter;
    this.outputStream = getOutputStream(out, displayName);

    logInvocation();
  }

  private MessageOutputStream<ExecLogEntry> getOutputStream(OutputStream out, String name)
      throws IOException {
    // Use an AsynchronousMessageOutputStream so that compression and I/O occur in a separate
    // thread. This ensures concurrent writes don't tear and avoids blocking execution.
    return new AsynchronousMessageOutputStream<>(name, compressionService.newZstdOutputStream(out));
  }

  private void logInvocation() throws IOException, InterruptedException {
    logEntryWithoutId(() ->
        ExecLogEntry.newBuilder()
            .setInvocation(
                ExecLogEntry.Invocation.newBuilder()
                    .setHashFunctionName(internalToUnicode(digestHashFunction.toString()))
                    .setWorkspaceRunfilesDirectory(
                        internalToUnicode(LabelConstants.MAIN_REPO_DIR_NAME))
                    .setId(internalToUnicode(invocationId.toString()))));
  }

  @Override
  public boolean shouldPublish() {
    // The compact log is small enough to be uploaded to a remote store.
    return true;
  }

  @Override
  public void logSpawn(
      Spawn spawn,
      InputMetadataProvider inputMetadataProvider,
      Supplier<SortedMap<PathFragment, ActionInput>> inputMap,
      FileSystem fileSystem,
      Duration timeout,
      SpawnResult result)
      throws IOException, InterruptedException, ExecException {
    if (!shouldLog(spawn)) {
      return;
    }
    try (SilentCloseable c = Profiler.instance().profile(SPAWN_LOG, "logSpawn")) {
      ExecLogEntry.Spawn.Builder builder = ExecLogEntry.Spawn.newBuilder();

      builder.addAllArgs(
          Lists.transform(getLoggedArguments(spawn), StringEncoding::internalToUnicode));
      builder.addAllEnvVars(getEnvironmentVariables(spawn));
      Platform platform = getPlatform(spawn, remoteOptions);
      if (platform != null) {
        builder.setPlatform(platform);
      }

      builder.setInputSetId(logInputs(spawn, inputMetadataProvider, fileSystem));
      builder.setToolSetId(logTools(spawn, inputMetadataProvider, fileSystem));

      if (spawn.getTargetLabel() != null) {
        builder.setTargetLabel(internalToUnicode(spawn.getTargetLabel().getCanonicalForm()));
      }
      builder.setMnemonic(internalToUnicode(spawn.getMnemonic()));

      boolean warned = false;
      for (ActionInput output : spawn.getOutputFiles()) {
        var path = fileSystem.getPath(execRoot.getRelative(output.getExecPath()));
        var outputBuilder = ExecLogEntry.Output.newBuilder();
        try {
          if (!output.isDirectory() && !output.isSymlink() && path.isFile()) {
            outputBuilder.setOutputId(logFile(output, path, /* inputMetadataProvider= */ null));
          } else if (output.isDirectory() && path.isDirectory()) {
            outputBuilder.setOutputId(
                logDirectory(output, path, /* inputMetadataProvider= */ null));
          } else if (output.isSymlink() && path.isSymbolicLink()) {
            outputBuilder.setOutputId(
                logUnresolvedSymlink(output, path, /* inputMetadataProvider= */ null));
          } else {
            outputBuilder.setInvalidOutputPath(internalToUnicode(output.getExecPathString()));
          }
        } catch (IOException e) {
          if (!warned) {
            outputLoggingFailed.set(true);
            warned = true;
            logger.atInfo().withCause(e).log(
                "Failed to log outputs of spawn with mnemonic %s and primary output %s",
                spawn.getMnemonic(),
                Iterables.getFirst(spawn.getOutputFiles(), /* not reached */ null));
          }
          outputBuilder.setInvalidOutputPath(internalToUnicode(output.getExecPathString()));
        }
        builder.addOutputs(outputBuilder);
      }

      builder.setExitCode(result.exitCode());
      if (result.status() != SpawnResult.Status.SUCCESS) {
        builder.setStatus(internalToUnicode(result.status().toString()));
      }
      builder.setRunner(internalToUnicode(result.getRunnerName()));
      builder.setCacheHit(result.isCacheHit());
      builder.setRemotable(Spawns.mayBeExecutedRemotely(spawn));
      builder.setCacheable(Spawns.mayBeCached(spawn));
      builder.setRemoteCacheable(Spawns.mayBeCachedRemotely(spawn));

      if (result.getDigest() != null) {
        builder.setDigest(result.getDigest().toBuilder().clearHashFunctionName().build());
      }

      builder.setTimeoutMillis(timeout.toMillis());
      builder.setMetrics(getSpawnMetricsProto(result));

      logEntryWithoutId(() -> ExecLogEntry.newBuilder().setSpawn(builder));
    }
  }

  @Override
  public void logSymlinkAction(AbstractAction action) throws IOException, InterruptedException {
    try (SilentCloseable c = Profiler.instance().profile(SPAWN_LOG, "logSymlinkAction")) {
      ExecLogEntry.SymlinkAction.Builder builder = ExecLogEntry.SymlinkAction.newBuilder();

      Artifact input = action.getPrimaryInput();
      if (input == null) {
        // Symlinks to absolute paths are only used by FDO and not worth logging as they can be
        // treated just like source files.
        return;
      }
      builder.setInputPath(internalToUnicode(input.getExecPathString()));
      builder.setOutputPath(internalToUnicode(action.getPrimaryOutput().getExecPathString()));

      Label label = action.getOwner().getLabel();
      if (label != null) {
        builder.setTargetLabel(internalToUnicode(label.getCanonicalForm()));
      }
      builder.setMnemonic(internalToUnicode(action.getMnemonic()));

      logEntryWithoutId(() -> ExecLogEntry.newBuilder().setSymlinkAction(builder));
    }
  }

  /**
   * Logs the inputs.
   *
   * @return the entry ID of the {@link ExecLogEntry.InputSet} describing the inputs, or 0 if there
   *     are no inputs.
   */
  private int logInputs(
      Spawn spawn, InputMetadataProvider inputMetadataProvider, FileSystem fileSystem)
      throws IOException, InterruptedException {

    return logInputSet(
        spawn.getInputFiles().asNestedSet(),
        inputMetadataProvider,
        fileSystem,
        /* shared= */ false);
  }

  /**
   * Logs the tool inputs.
   *
   * @return the entry ID of the {@link ExecLogEntry.InputSet} describing the tool inputs, or 0 if
   *     there are no tool inputs.
   */
  private int logTools(
      Spawn spawn, InputMetadataProvider inputMetadataProvider, FileSystem fileSystem)
      throws IOException, InterruptedException {
    return logInputSet(spawn.getToolFiles(), inputMetadataProvider, fileSystem, /* shared= */ true);
  }

  /**
   * Logs a nested set.
   *
   * @param set the nested set
   * @param shared whether this nested set is likely to be a transitive member of other sets
   * @return the entry ID of the {@link ExecLogEntry.InputSet} describing the nested set, or 0 if
   *     the nested set is empty.
   */
  private int logInputSet(
      NestedSet<? extends ActionInput> set,
      InputMetadataProvider inputMetadataProvider,
      FileSystem fileSystem,
      boolean shared)
      throws IOException, InterruptedException {
    if (set.isEmpty()) {
      return 0;
    }

    return logEntry(
        shared ? set.toNode() : null,
        () -> {
          ExecLogEntry.InputSet.Builder builder = ExecLogEntry.InputSet.newBuilder();

          for (NestedSet<? extends ActionInput> transitive : set.getNonLeaves()) {
            checkState(!transitive.isEmpty());
            builder.addTransitiveSetIds(
                logInputSet(transitive, inputMetadataProvider, fileSystem, /* shared= */ true));
          }

          for (ActionInput input : set.getLeaves()) {
            if (input instanceof Artifact artifact && artifact.isRunfilesTree()) {
              RunfilesTree runfilesTree =
                  inputMetadataProvider.getRunfilesMetadata(input).getRunfilesTree();
              builder.addInputIds(
                  logRunfilesTree(
                      runfilesTree,
                      inputMetadataProvider,
                      fileSystem,
                      // Only share runfiles trees when their mapping is cached; otherwise the
                      // mapping can vary per spawn and should be treated as unique.
                      runfilesTree.isMappingCached()));
              continue;
            }

            if (input instanceof Artifact artifact && artifact.isFileset()) {
              // The fileset symlink tree is always materialized on disk.
              builder.addInputIds(
                  logDirectory(
                      input,
                      fileSystem.getPath(execRoot.getRelative(input.getExecPath())),
                      inputMetadataProvider));
            }

            builder.addInputIds(logInput(input, inputMetadataProvider, fileSystem));
          }

          return ExecLogEntry.newBuilder().setInputSet(builder);
        });
  }

  /**
   * Logs a nested set of {@link SymlinkEntry}.
   *
   * @return the entry ID of the {@link ExecLogEntry.SymlinkEntrySet} describing the nested set, or
   *     0 if the nested set is empty.
   */
  private int logSymlinkEntries(
      NestedSet<SymlinkEntry> symlinks,
      InputMetadataProvider inputMetadataProvider,
      FileSystem fileSystem)
      throws IOException, InterruptedException {
    if (symlinks.isEmpty()) {
      return 0;
    }

    return logEntry(
        symlinks.toNode(),
        () -> {
          ExecLogEntry.SymlinkEntrySet.Builder builder = ExecLogEntry.SymlinkEntrySet.newBuilder();

          for (NestedSet<SymlinkEntry> transitive : symlinks.getNonLeaves()) {
            checkState(!transitive.isEmpty());
            builder.addTransitiveSetIds(
                logSymlinkEntries(transitive, inputMetadataProvider, fileSystem));
          }

          for (SymlinkEntry input : symlinks.getLeaves()) {
            builder.putDirectEntries(
                internalToUnicode(input.getPathString()),
                logInput(input.getArtifact(), inputMetadataProvider, fileSystem));
          }

          return ExecLogEntry.newBuilder().setSymlinkEntrySet(builder);
        });
  }

  /**
   * Logs a single input that is either a file, a directory or a symlink.
   *
   * @return the entry ID of the {@link ExecLogEntry} describing the input.
   */
  private int logInput(
      ActionInput input, InputMetadataProvider inputMetadataProvider, FileSystem fileSystem)
      throws IOException, InterruptedException {
    Path path = fileSystem.getPath(execRoot.getRelative(input.getExecPath()));
    if (isInputDirectory(input, inputMetadataProvider)) {
      return logDirectory(input, path, inputMetadataProvider);
    } else if (input.isSymlink()) {
      return logUnresolvedSymlink(input, path, inputMetadataProvider);
    } else {
      return logFile(input, path, inputMetadataProvider);
    }
  }

  /**
   * Logs a file.
   *
   * @param input the input representing the file.
   * @param path the path to the file, which must have already been verified to be of the correct
   *     type.
   * @param inputMetadataProvider provides metadata for inputs; null if logging an output
   * @return the entry ID of the {@link ExecLogEntry.File} describing the file.
   */
  private int logFile(
      ActionInput input, Path path, @Nullable InputMetadataProvider inputMetadataProvider)
      throws IOException, InterruptedException {
    checkState(!(input instanceof VirtualActionInput.EmptyActionInput));

    return logEntry(
        // A ParamFileActionInput is never shared between spawns.
        input instanceof ParamFileActionInput ? null : input.getExecPathString(),
        () -> {
          ExecLogEntry.File.Builder builder = ExecLogEntry.File.newBuilder();

          builder.setPath(internalToUnicode(input.getExecPathString()));

          Digest digest =
              computeDigest(
                  input,
                  path,
                  inputMetadataProvider,
                  xattrProvider,
                  digestHashFunction,
                  /* includeHashFunctionName= */ false);

          builder.setDigest(digest);

          return ExecLogEntry.newBuilder().setFile(builder);
        });
  }

  /**
   * Logs a directory.
   *
   * <p>This may be either a source directory, a fileset or an output directory. For runfiles,
   * {@link #logRunfilesTree} must be used instead.
   *
   * @param input the input representing the directory.
   * @param root the path to the directory, which must have already been verified to be of the
   *     correct type.
   * @param inputMetadataProvider provides metadata for inputs; null if logging an output
   * @return the entry ID of the {@link ExecLogEntry.Directory} describing the directory.
   */
  private int logDirectory(
      ActionInput input, Path root, @Nullable InputMetadataProvider inputMetadataProvider)
      throws IOException, InterruptedException {
    return logEntry(
        input.getExecPathString(),
        () ->
            ExecLogEntry.newBuilder()
                .setDirectory(
                    ExecLogEntry.Directory.newBuilder()
                        .setPath(internalToUnicode(input.getExecPathString()))
                        .addAllFiles(expandDirectory(input, root, inputMetadataProvider))));
  }

  /**
   * Logs a runfiles directory by storing the information in its {@link RunfilesTree}.
   *
   * <p>Since runfiles trees can be very large and, for tests, are only used by a single spawn, we
   * store them in the log as a special entry that references the nested set of artifacts instead of
   * as a flat directory.
   *
   * @param shared whether this runfiles tree is likely to be contained in more than one Spawn's
   *     inputs
   * @param inputMetadataProvider provides metadata for inputs
   * @return the entry ID of the {@link ExecLogEntry.RunfilesTree} describing the directory.
   */
  private int logRunfilesTree(
      RunfilesTree runfilesTree,
      InputMetadataProvider inputMetadataProvider,
      FileSystem fileSystem,
      boolean shared)
      throws IOException, InterruptedException {
    return logEntry(
        shared ? runfilesTree.getExecPath().getPathString() : null,
        () -> {
          ExecLogEntry.RunfilesTree.Builder builder =
              ExecLogEntry.RunfilesTree.newBuilder()
                  .setPath(internalToUnicode(runfilesTree.getExecPath().getPathString()));

          builder.setInputSetId(
              logInputSet(
                  runfilesTree.getArtifactsAtCanonicalLocationsForLogging(),
                  inputMetadataProvider,
                  fileSystem,
                  // The runfiles tree itself is shared, but the nested set is unique to the tree as
                  // it contains the executable.
                  /* shared= */ false));
          builder.setSymlinksId(
              logSymlinkEntries(
                  runfilesTree.getSymlinksForLogging(), inputMetadataProvider, fileSystem));
          builder.setRootSymlinksId(
              logSymlinkEntries(
                  runfilesTree.getRootSymlinksForLogging(), inputMetadataProvider, fileSystem));
          builder.addAllEmptyFiles(
              Iterables.transform(
                  runfilesTree.getEmptyFilenamesForLogging(),
                  emptyFile -> internalToUnicode(emptyFile.getPathString())));
          Artifact repoMappingManifest = runfilesTree.getRepoMappingManifestForLogging();
          if (repoMappingManifest != null) {
            builder.setRepoMappingManifest(
                ExecLogEntry.File.newBuilder()
                    .setDigest(
                        computeDigest(
                            repoMappingManifest,
                            repoMappingManifest.getPath(),
                            inputMetadataProvider,
                            xattrProvider,
                            digestHashFunction,
                            /* includeHashFunctionName= */ false)));
          }

          return ExecLogEntry.newBuilder().setRunfilesTree(builder);
        });
  }

  /**
   * Expands a directory.
   *
   * @param input the input representing the directory
   * @param root the path to the directory
   * @param inputMetadataProvider provides metadata for inputs; null if logging an output
   * @return the list of files transitively contained in the directory
   */
  private List<ExecLogEntry.File> expandDirectory(
      ActionInput input, Path root, @Nullable InputMetadataProvider inputMetadataProvider)
      throws IOException, InterruptedException {
    if (inputMetadataProvider != null
        && input instanceof Artifact artifact
        && artifact.isTreeArtifact()) {
      TreeArtifactValue treeMetadata = inputMetadataProvider.getTreeMetadata(artifact);
      if (treeMetadata != null) {
        // Using the metadata over a filesystem traversal is not just an optimization: an empty tree
        // artifact may not be materialized on disk.
        return expandTreeArtifact(treeMetadata, root, inputMetadataProvider);
      }
    }
    return expandDirectoryFromFileSystem(root, inputMetadataProvider);
  }

  /** Expands a tree artifact into its contents as recorded in its metadata. */
  private List<ExecLogEntry.File> expandTreeArtifact(
      TreeArtifactValue treeMetadata, Path root, InputMetadataProvider inputMetadataProvider)
      throws IOException {
    var files = new ArrayList<ExecLogEntry.File>(treeMetadata.getChildren().size());
    for (var child : treeMetadata.getChildren()) {
      PathFragment parentRelativePath = child.getParentRelativePath();
      Digest digest =
          computeDigest(
              child,
              root.getRelative(parentRelativePath),
              inputMetadataProvider,
              xattrProvider,
              digestHashFunction,
              /* includeHashFunctionName= */ false);
      files.add(
          ExecLogEntry.File.newBuilder()
              .setPath(internalToUnicode(parentRelativePath.getPathString()))
              .setDigest(digest)
              .build());
    }
    files.sort(EXEC_LOG_ENTRY_FILE_COMPARATOR);
    return files;
  }

  /** Expands a directory by traversing it on the filesystem. */
  private List<ExecLogEntry.File> expandDirectoryFromFileSystem(
      Path root, @Nullable InputMetadataProvider inputMetadataProvider)
      throws IOException, InterruptedException {
    ArrayList<ExecLogEntry.File> files = new ArrayList<>();
    visitDirectory(
        root,
        (child) -> {
          Digest digest =
              computeDigest(
                  /* input= */ null,
                  child,
                  inputMetadataProvider,
                  xattrProvider,
                  digestHashFunction,
                  /* includeHashFunctionName= */ false);

          ExecLogEntry.File file =
              ExecLogEntry.File.newBuilder()
                  .setPath(internalToUnicode(child.relativeTo(root).getPathString()))
                  .setDigest(digest)
                  .build();

          synchronized (files) {
            files.add(file);
          }
        });

    files.sort(EXEC_LOG_ENTRY_FILE_COMPARATOR);

    return files;
  }

  /**
   * Logs an unresolved symlink.
   *
   * @param input the input representing the unresolved symlink.
   * @param path the path to the unresolved symlink, which must have already been verified to be of
   *     the correct type.
   * @param inputMetadataProvider provides metadata for inputs; null if logging an output
   * @return the entry ID of the {@link ExecLogEntry.UnresolvedSymlink} describing the unresolved
   *     symlink.
   */
  private int logUnresolvedSymlink(
      ActionInput input, Path path, @Nullable InputMetadataProvider inputMetadataProvider)
      throws IOException, InterruptedException {
    return logEntry(
        input.getExecPathString(),
        () -> {
          FileArtifactValue metadata = null;
          if (inputMetadataProvider != null) {
            metadata = inputMetadataProvider.getInputMetadata(input);
          }
          String targetPath;
          if (metadata != null) {
            checkState(metadata.getType().isSymlink(), metadata);
            targetPath = metadata.getUnresolvedSymlinkTarget();
          } else {
            targetPath = path.readSymbolicLink().getPathString();
          }

          return ExecLogEntry.newBuilder()
              .setUnresolvedSymlink(
                  ExecLogEntry.UnresolvedSymlink.newBuilder()
                      .setPath(internalToUnicode(input.getExecPathString()))
                      .setTargetPath(internalToUnicode(targetPath)));
        });
  }

  /**
   * Ensures an entry is written to the log without an ID.
   *
   * <p>No lock is required: every entry referenced by the entry was handed to the output stream
   * before its ID was returned to the supplier, and the output stream preserves the order of
   * writes.
   *
   * @param supplier called to compute the entry; may cause other entries to be logged
   */
  private void logEntryWithoutId(ExecLogEntrySupplier supplier)
      throws IOException, InterruptedException {
    try (SilentCloseable c = Profiler.instance().profile(SPAWN_LOG, "logEntryWithoutId")) {
      outputStream.write(supplier.get().build());
    }
  }

  /**
   * Ensures an entry is written to the log and returns its assigned ID.
   *
   * <p>If an entry with the same non-null key was previously added to the log, its recorded ID is
   * returned. Otherwise, the entry is computed, assigned an ID, and written to the log.
   *
   * <p>May be called concurrently. An entry is computed without holding a lock, so that threads
   * logging unrelated entries don't wait for each other. If a thread requests an entry with the
   * same non-null key as an entry that another thread is computing, it waits for that entry to be
   * written instead of computing it again.
   *
   * @param key the deduplication key in {@link #entryIds}, or {@code null} for unshared entries
   *     that still need an assigned ID so their parent entry can reference them, but should not be
   *     deduplicated across spawns
   * @param supplier called to compute the entry; may cause other entries to be logged
   * @return the entry ID
   */
  @CheckReturnValue
  private int logEntry(@Nullable Object key, ExecLogEntrySupplier supplier)
      throws IOException, InterruptedException {
    try (SilentCloseable c = Profiler.instance().profile(SPAWN_LOG, "logEntry")) {
      if (key == null) {
        // Unshared entry (e.g. a spawn's top-level input set or param file): still needs an ID so
        // its parent entry can reference it, but skip deduplication in entryIds/pendingEntries.
        return writeEntryWithNewId(supplier.get());
      }

      checkState(key instanceof NestedSet.Node || key instanceof String);

      while (true) {
        // Check for a previously added entry.
        int id = entryIds.get(key);
        if (id != ABSENT) {
          return id;
        }
        PendingEntry mine = new PendingEntry();
        PendingEntry pending = pendingEntries.putIfAbsent(key, mine);
        if (pending == null) {
          // No other thread is computing the entry, so it is up to this thread to log it.
          return computeAndWriteEntry(key, mine, supplier);
        }

        // Another thread is computing the entry: wait until it has been written.
        // This can't deadlock: a thread only waits for entries referenced by the entry it is
        // computing, and an entry never transitively references itself.
        try (SilentCloseable c2 =
            Profiler.instance().profile(SPAWN_LOG, "logEntry/waitForPendingEntry")) {
          pending.await();
        }
        if (pending.id != 0) {
          return pending.id;
        }
        // The other thread aborted before writing the entry (e.g. it was part of a cancelled
        // branch under dynamic execution, or its spawn hit an IOException under --keep_going)
        // so retry computing it here instead of failing this spawn with the other thread's error.
      }
    }
  }

  /**
   * Computes a fresh entry, writes it to the log and records its ID, on behalf of the thread that
   * has added the given {@link PendingEntry} to {@link #pendingEntries}.
   */
  private int computeAndWriteEntry(Object key, PendingEntry pending, ExecLogEntrySupplier supplier)
      throws IOException, InterruptedException {
    int id = 0;
    try {
      // Another thread may have written the entry after this thread looked for its ID and before
      // this thread added the PendingEntry. Since that thread stored the ID before it removed its
      // own PendingEntry, the ID is found now.
      int writtenId = entryIds.get(key);
      if (writtenId != ABSENT) {
        id = writtenId;
        return id;
      }

      // The following order of operations is crucial to ensure that this entry is preceded by any
      // entries it references, which in turn ensures the log can be parsed in a single pass: the
      // entry is written after the supplier has obtained an ID for every entry it references, and
      // its own ID is made available to other threads after it has been written.
      id = writeEntryWithNewId(supplier.get());
      entryIds.put(key, id);
      return id;
    } finally {
      // Must come after the ID has been stored, so that a thread that doesn't find the PendingEntry
      // finds the ID, and before the threads waiting for the entry are woken up, so that they
      // don't find the PendingEntry again if the entry failed to be computed and they are to
      // attempt to compute it themselves.
      pendingEntries.remove(key, pending);
      pending.id = id;
      pending.countDown();
    }
  }

  /**
   * Assigns the next ID to a fully computed entry and writes it to the log.
   *
   * <p>This is the only part of logging an entry that is mutually exclusive between threads, and it
   * must not do more than is required to keep IDs consecutive in the order in which entries appear
   * in the log.
   */
  private synchronized int writeEntryWithNewId(ExecLogEntry.Builder entry) throws IOException {
    int id = nextEntryId++;
    outputStream.write(entry.setId(id).build());
    return id;
  }

  @Override
  public void close() throws IOException {
    if (outputLoggingFailed.get()) {
      reporter.handle(
          Event.warn(
              "The compact execution log is incomplete because some outputs could not be read."
                  + " Refer to the server log file for details."));
    }
    outputStream.close();
  }
}
