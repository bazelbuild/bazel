package com.google.devtools.build.lib.worker;

import com.google.common.base.MoreObjects;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.hash.HashCode;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.Spawns;
import com.google.devtools.build.lib.actions.UserExecException;
import com.google.devtools.build.lib.exec.BinTools;
import com.google.devtools.build.lib.exec.SpawnRunner;
import com.google.devtools.build.lib.exec.local.LocalEnvProvider;
import com.google.devtools.build.lib.server.FailureDetails;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.SortedMap;
import java.util.regex.Pattern;

class WorkerParser {
  public static final String ERROR_MESSAGE_PREFIX =
      "Worker strategy cannot execute this %s action, ";
  public static final String REASON_NO_FLAGFILE =
      "because the command-line arguments do not contain at least one @flagfile or --flagfile=";

  /** Pattern for @flagfile.txt and --flagfile=flagfile.txt */
  private static final Pattern FLAG_FILE_PATTERN = Pattern.compile("(?:@|--?flagfile=)(.+)");

  private final Path execRoot;
  private final boolean multiplex;
  private final WorkerOptions workerOptions;
  private final LocalEnvProvider localEnvProvider;
  private final BinTools binTools;

  public WorkerParser(
      Path execRoot,
      boolean multiplex,
      WorkerOptions workerOptions,
      LocalEnvProvider localEnvProvider,
      BinTools binTools) {
    this.execRoot = execRoot;
    this.multiplex = multiplex;
    this.workerOptions = workerOptions;
    this.localEnvProvider = localEnvProvider;
    this.binTools = binTools;
  }

  public WorkerConfig compute(Spawn spawn, SpawnRunner.SpawnExecutionContext context)
      throws ExecException, IOException, InterruptedException {
    // We assume that the spawn to be executed always gets at least one @flagfile.txt or
    // --flagfile=flagfile.txt argument, which contains the flags related to the work itself (as
    // opposed to start-up options for the executed tool). Thus, we can extract those elements from
    // its args and put them into the WorkRequest instead.
    List<String> flagFiles = new ArrayList<>();
    ImmutableList<String> workerArgs = splitSpawnArgsIntoWorkerArgsAndFlagFiles(spawn, flagFiles);
    ImmutableMap<String, String> env =
        localEnvProvider.rewriteLocalEnv(spawn.getEnvironment(), binTools, "/tmp");

    SortedMap<PathFragment, HashCode> workerFiles =
        WorkerFilesHash.getWorkerFilesWithHashes(
            spawn, context.getArtifactExpander(), context.getMetadataProvider());

    HashCode workerFilesCombinedHash = WorkerFilesHash.getCombinedHash(workerFiles);

    WorkerKey key =
        new WorkerKey(
            workerArgs,
            env,
            execRoot,
            Spawns.getWorkerKeyMnemonic(spawn),
            workerFilesCombinedHash,
            workerFiles,
            context.speculating(),
            multiplex && Spawns.supportsMultiplexWorkers(spawn),
            Spawns.supportsWorkerCancellation(spawn),
            Spawns.getWorkerProtocolFormat(spawn));
    return new WorkerConfig(key, flagFiles);
  }

  /**
   * Splits the command-line arguments of the {@code Spawn} into the part that is used to start the
   * persistent worker ({@code workerArgs}) and the part that goes into the {@code WorkRequest}
   * protobuf ({@code flagFiles}).
   */
  private ImmutableList<String> splitSpawnArgsIntoWorkerArgsAndFlagFiles(
      Spawn spawn, List<String> flagFiles) throws UserExecException {
    ImmutableList.Builder<String> workerArgs = ImmutableList.builder();
    for (String arg : spawn.getArguments()) {
      if (FLAG_FILE_PATTERN.matcher(arg).matches()) {
        flagFiles.add(arg);
      } else {
        workerArgs.add(arg);
      }
    }

    if (flagFiles.isEmpty()) {
      throw createUserExecException(
          String.format(ERROR_MESSAGE_PREFIX + REASON_NO_FLAGFILE, spawn.getMnemonic()),
          FailureDetails.Worker.Code.NO_FLAGFILE);
    }

    ImmutableList.Builder<String> mnemonicFlags = ImmutableList.builder();

    workerOptions.workerExtraFlags.stream()
        .filter(entry -> entry.getKey().equals(spawn.getMnemonic()))
        .forEach(entry -> mnemonicFlags.add(entry.getValue()));

    return workerArgs
        .add("--persistent_worker")
        .addAll(MoreObjects.firstNonNull(mnemonicFlags.build(), ImmutableList.of()))
        .build();
  }

  private static UserExecException createUserExecException(
      String message, FailureDetails.Worker.Code detailedCode) {
    return new UserExecException(
        FailureDetails.FailureDetail.newBuilder()
            .setMessage(message)
            .setWorker(FailureDetails.Worker.newBuilder().setCode(detailedCode))
            .build());
  }

  public static class WorkerConfig {
    private final WorkerKey workerKey;
    private final List<String> flagFiles;

    public WorkerConfig(WorkerKey workerKey, List<String> flagFiles) {
      this.workerKey = workerKey;
      this.flagFiles = ImmutableList.copyOf(flagFiles);
    }

    public WorkerKey getWorkerKey() {
      return workerKey;
    }

    public List<String> getFlagFiles() {
      return flagFiles;
    }
  }
}
