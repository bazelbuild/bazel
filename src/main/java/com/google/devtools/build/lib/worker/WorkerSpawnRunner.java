// Copyright 2017 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.lib.worker;

import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Joiner;
import com.google.common.base.Preconditions;
import com.google.common.base.Splitter;
import com.google.common.base.Stopwatch;
import com.google.common.eventbus.EventBus;
import com.google.common.eventbus.Subscribe;
import com.google.common.flogger.GoogleLogger;
import com.google.common.hash.HashCode;
import com.google.devtools.build.lib.actions.ActionExecutionMetadata;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionInputHelper;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.ForbiddenActionInputException;
import com.google.devtools.build.lib.actions.MetadataProvider;
import com.google.devtools.build.lib.actions.ResourceManager;
import com.google.devtools.build.lib.actions.ResourceManager.ResourceHandle;
import com.google.devtools.build.lib.actions.ResourceManager.ResourceHandleWithWorker;
import com.google.devtools.build.lib.actions.ResourceManager.ResourcePriority;
import com.google.devtools.build.lib.actions.Spawn;
import com.google.devtools.build.lib.actions.SpawnExecutedEvent;
import com.google.devtools.build.lib.actions.SpawnMetrics;
import com.google.devtools.build.lib.actions.SpawnResult;
import com.google.devtools.build.lib.actions.SpawnResult.Status;
import com.google.devtools.build.lib.actions.Spawns;
import com.google.devtools.build.lib.actions.UserExecException;
import com.google.devtools.build.lib.buildtool.CollectMetricsEvent;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.exec.BinTools;
import com.google.devtools.build.lib.exec.RunfilesTreeUpdater;
import com.google.devtools.build.lib.exec.SpawnExecutingEvent;
import com.google.devtools.build.lib.exec.SpawnRunner;
import com.google.devtools.build.lib.exec.SpawnSchedulingEvent;
import com.google.devtools.build.lib.exec.local.LocalEnvProvider;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.ProfilerTask;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import com.google.devtools.build.lib.sandbox.SandboxHelpers;
import com.google.devtools.build.lib.sandbox.SandboxHelpers.SandboxInputs;
import com.google.devtools.build.lib.sandbox.SandboxHelpers.SandboxOutputs;
import com.google.devtools.build.lib.server.FailureDetails;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.server.FailureDetails.Worker.Code;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.util.io.FileOutErr;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.XattrProvider;
import com.google.devtools.build.lib.worker.WorkerProtocol.WorkRequest;
import com.google.devtools.build.lib.worker.WorkerProtocol.WorkResponse;
import com.google.protobuf.ByteString;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.InterruptedIOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;

/**
 * A spawn runner that launches Spawns the first time they are used in a persistent mode and then
 * shards work over all the processes.
 */
final class WorkerSpawnRunner implements SpawnRunner {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  public static final String ERROR_MESSAGE_PREFIX =
      "Worker strategy cannot execute this %s action, ";
  public static final String REASON_NO_TOOLS = "because the action has no tools";
  /**
   * The verbosity level implied by `--worker_verbose`. This value allows for manually setting some
   * only-slightly-verbose levels.
   */
  private static final int VERBOSE_LEVEL = 10;

  private final SandboxHelpers helpers;
  private final Path execRoot;
  private final WorkerPool workers;
  private final ExtendedEventHandler reporter;
  private final BinTools binTools;
  private final ResourceManager resourceManager;
  private final RunfilesTreeUpdater runfilesTreeUpdater;
  private final WorkerOptions workerOptions;
  private final WorkerParser workerParser;
  private final AtomicInteger requestIdCounter = new AtomicInteger(1);
  private final Runtime runtime;
  private final XattrProvider xattrProvider;

  /** Mapping of worker ids to their metrics. */
  private Map<Integer, WorkerMetric> workerIdToWorkerMetric = new ConcurrentHashMap<>();

  public WorkerSpawnRunner(
      SandboxHelpers helpers,
      Path execRoot,
      WorkerPool workers,
      ExtendedEventHandler reporter,
      LocalEnvProvider localEnvProvider,
      BinTools binTools,
      ResourceManager resourceManager,
      RunfilesTreeUpdater runfilesTreeUpdater,
      WorkerOptions workerOptions,
      EventBus eventBus,
      Runtime runtime,
      XattrProvider xattrProvider) {
    this.helpers = helpers;
    this.execRoot = execRoot;
    this.workers = Preconditions.checkNotNull(workers);
    this.reporter = reporter;
    this.binTools = binTools;
    this.resourceManager = resourceManager;
    this.runfilesTreeUpdater = runfilesTreeUpdater;
    this.xattrProvider = xattrProvider;
    this.workerParser = new WorkerParser(execRoot, workerOptions, localEnvProvider, binTools);
    this.workerOptions = workerOptions;
    this.runtime = runtime;
    this.resourceManager.setWorkerPool(workers);
    eventBus.register(this);
  }

  @Override
  public String getName() {
    return "worker";
  }

  @Override
  public boolean canExec(Spawn spawn) {
    if (!Spawns.supportsWorkers(spawn) && !Spawns.supportsMultiplexWorkers(spawn)) {
      return false;
    }
    if (spawn.getToolFiles().isEmpty()) {
      return false;
    }
    return true;
  }

  @Override
  public boolean handlesCaching() {
    return false;
  }

  @Override
  public SpawnResult exec(Spawn spawn, SpawnExecutionContext context)
      throws ExecException, IOException, InterruptedException, ForbiddenActionInputException {
    context.report(
        SpawnSchedulingEvent.create(
            WorkerKey.makeWorkerTypeName(
                Spawns.supportsMultiplexWorkers(spawn), context.speculating())));
    if (spawn.getToolFiles().isEmpty()) {
      throw createUserExecException(
          String.format(ERROR_MESSAGE_PREFIX + REASON_NO_TOOLS, spawn.getMnemonic()),
          Code.NO_TOOLS);
    }

    Instant startTime = Instant.now();
    SpawnMetrics.Builder spawnMetrics;
    WorkResponse response;

    try (SilentCloseable c =
        Profiler.instance()
            .profile(
                String.format(
                    "%s worker %s", spawn.getMnemonic(), spawn.getResourceOwner().describe()))) {

      runfilesTreeUpdater.updateRunfilesDirectory(
          execRoot,
          spawn.getRunfilesSupplier(),
          binTools,
          spawn.getEnvironment(),
          context.getFileOutErr(),
          xattrProvider);

      MetadataProvider inputFileCache = context.getMetadataProvider();

      SandboxInputs inputFiles;
      try (SilentCloseable c1 =
          Profiler.instance().profile(ProfilerTask.WORKER_SETUP, "Setting up inputs")) {
        inputFiles =
            helpers.processInputFiles(
                context.getInputMapping(PathFragment.EMPTY_FRAGMENT),
                spawn,
                context.getArtifactExpander(),
                execRoot);
      }
      SandboxOutputs outputs = helpers.getOutputs(spawn);

      WorkerParser.WorkerConfig workerConfig = workerParser.compute(spawn, context);
      WorkerKey key = workerConfig.getWorkerKey();
      List<String> flagFiles = workerConfig.getFlagFiles();

      spawnMetrics =
          SpawnMetrics.Builder.forWorkerExec()
              .setInputFiles(inputFiles.getFiles().size() + inputFiles.getSymlinks().size());
      response =
          execInWorker(
              spawn, key, context, inputFiles, outputs, flagFiles, inputFileCache, spawnMetrics);

      FileOutErr outErr = context.getFileOutErr();
      response.getOutputBytes().writeTo(outErr.getErrorStream());
    }
    Duration wallTime = Duration.between(startTime, Instant.now());

    int exitCode = response.getExitCode();
    SpawnResult.Builder builder =
        new SpawnResult.Builder()
            .setRunnerName(getName())
            .setExitCode(exitCode)
            .setStatus(exitCode == 0 ? Status.SUCCESS : Status.NON_ZERO_EXIT)
            .setWallTime(wallTime)
            .setSpawnMetrics(spawnMetrics.setTotalTime(wallTime).build());
    if (exitCode != 0) {
      builder.setFailureDetail(
          FailureDetail.newBuilder()
              .setMessage("worker spawn failed for " + spawn.getMnemonic())
              .setSpawn(
                  FailureDetails.Spawn.newBuilder()
                      .setCode(FailureDetails.Spawn.Code.NON_ZERO_EXIT)
                      .setSpawnExitCode(exitCode))
              .build());
    }
    SpawnResult result = builder.build();
    reporter.post(new SpawnExecutedEvent(spawn, result, startTime));
    return result;
  }

  private WorkRequest createWorkRequest(
      Spawn spawn,
      SpawnExecutionContext context,
      List<String> flagfiles,
      MetadataProvider inputFileCache,
      WorkerKey key)
      throws IOException {
    WorkRequest.Builder requestBuilder = WorkRequest.newBuilder();
    for (String flagfile : flagfiles) {
      expandArgument(execRoot, flagfile, requestBuilder);
    }

    List<ActionInput> inputs =
        ActionInputHelper.expandArtifacts(spawn.getInputFiles(), context.getArtifactExpander());

    for (ActionInput input : inputs) {
      byte[] digestBytes = inputFileCache.getMetadata(input).getDigest();
      ByteString digest;
      if (digestBytes == null) {
        digest = ByteString.EMPTY;
      } else {
        digest = ByteString.copyFromUtf8(HashCode.fromBytes(digestBytes).toString());
      }

      requestBuilder.addInputsBuilder().setPath(input.getExecPathString()).setDigest(digest);
    }
    if (workerOptions.workerVerbose) {
      requestBuilder.setVerbosity(VERBOSE_LEVEL);
    }
    if (key.isMultiplex()) {
      requestBuilder.setRequestId(requestIdCounter.getAndIncrement());
    }
    return requestBuilder.build();
  }

  /**
   * Recursively expands arguments by replacing @filename args with the contents of the referenced
   * files. The @ itself can be escaped with @@. This deliberately does not expand --flagfile= style
   * arguments, because we want to get rid of the expansion entirely at some point in time.
   *
   * <p>Also check that the argument is not an external repository label, because they start with
   * `@` and are not flagfile locations.
   *
   * @param execRoot the current execroot of the build (relative paths will be assumed to be
   *     relative to this directory).
   * @param arg the argument to expand.
   * @param requestBuilder the WorkRequest to whose arguments the expanded arguments will be added.
   * @throws java.io.IOException if one of the files containing options cannot be read.
   */
  static void expandArgument(Path execRoot, String arg, WorkRequest.Builder requestBuilder)
      throws IOException {
    if (arg.startsWith("@") && !arg.startsWith("@@") && !isExternalRepositoryLabel(arg)) {
      for (String line :
          Files.readAllLines(
              Paths.get(execRoot.getRelative(arg.substring(1)).getPathString()), UTF_8)) {
        expandArgument(execRoot, line, requestBuilder);
      }
    } else {
      requestBuilder.addArguments(arg);
    }
  }

  private static boolean isExternalRepositoryLabel(String arg) {
    return arg.matches("^@.*//.*");
  }

  private static UserExecException createEmptyResponseException(Path logfile) {
    String message =
        ErrorMessage.builder()
            .message("Worker process did not return a WorkResponse:")
            .logFile(logfile)
            .logSizeLimit(4096)
            .build()
            .toString();
    return createUserExecException(message, Code.NO_RESPONSE);
  }

  private static UserExecException createUnparsableResponseException(
      String recordingStreamMessage, Path logfile, Exception e) {
    String message =
        ErrorMessage.builder()
            .message(
                "Worker process returned an unparseable WorkResponse!\n\n"
                    + "Did you try to print something to stdout? Workers aren't allowed to "
                    + "do this, as it breaks the protocol between Bazel and the worker "
                    + "process.\n\n"
                    + "---8<---8<--- Start of response ---8<---8<---\n"
                    + recordingStreamMessage
                    + "---8<---8<--- End of response ---8<---8<---\n\n")
            .logFile(logfile)
            .logSizeLimit(8192)
            .exception(e)
            .build()
            .toString();
    return createUserExecException(message, Code.PARSE_RESPONSE_FAILURE);
  }

  @VisibleForTesting
  WorkResponse execInWorker(
      Spawn spawn,
      WorkerKey key,
      SpawnExecutionContext context,
      SandboxInputs inputFiles,
      SandboxOutputs outputs,
      List<String> flagFiles,
      MetadataProvider inputFileCache,
      SpawnMetrics.Builder spawnMetrics)
      throws InterruptedException, ExecException {
    WorkerOwner workerOwner = new WorkerOwner();
    WorkResponse response;
    WorkRequest request;
    ActionExecutionMetadata owner = spawn.getResourceOwner();
    try {
      Stopwatch setupInputsStopwatch = Stopwatch.createStarted();
      try (SilentCloseable c =
          Profiler.instance().profile(ProfilerTask.WORKER_SETUP, "Preparing inputs")) {
        try {
          inputFiles.materializeVirtualInputs(execRoot);
        } catch (IOException e) {
          restoreInterrupt(e);
          String message = "IOException while materializing virtual inputs:";
          throw createUserExecException(e, message, Code.VIRTUAL_INPUT_MATERIALIZATION_FAILURE);
        }

        try {
          context.prefetchInputsAndWait();
        } catch (IOException e) {
          restoreInterrupt(e);
          String message = "IOException while prefetching for worker:";
          throw createUserExecException(e, message, Code.PREFETCH_FAILURE);
        } catch (ForbiddenActionInputException e) {
          throw createUserExecException(
              e, "Forbidden input found while prefetching for worker:", Code.FORBIDDEN_INPUT);
        }
      }
      Duration setupInputsTime = setupInputsStopwatch.elapsed();
      spawnMetrics.setSetupTime(setupInputsTime);

      Stopwatch queueStopwatch = Stopwatch.createStarted();
      if (workerOptions.workerAsResource) {
        // Worker doesn't automatically return to pool after closing of the handle.
        try (ResourceHandleWithWorker handle =
            resourceManager.acquireWorkerResources(
                owner,
                spawn.getLocalResources(),
                key,
                context.speculating() ? ResourcePriority.DYNAMIC_WORKER : ResourcePriority.LOCAL)) {
          workerOwner.setWorker(handle.getWorker());
          workerOwner.getWorker().setReporter(workerOptions.workerVerbose ? reporter : null);
          request = createWorkRequest(spawn, context, flagFiles, inputFileCache, key);

          // We acquired a worker and resources -- mark that as queuing time.
          spawnMetrics.setQueueTime(queueStopwatch.elapsed());
          response =
              executeRequest(
                  spawn, context, inputFiles, outputs, workerOwner, key, request, spawnMetrics);
        } catch (IOException e) {
          restoreInterrupt(e);
          String message = "IOException while borrowing a worker from the pool:";
          throw createUserExecException(e, message, Code.BORROW_FAILURE);
        }
      } else {
        try (SilentCloseable c =
            Profiler.instance().profile(ProfilerTask.WORKER_BORROW, "Waiting to borrow worker")) {
          workerOwner.setWorker(workers.borrowObject(key));
          workerOwner.getWorker().setReporter(workerOptions.workerVerbose ? reporter : null);
          request = createWorkRequest(spawn, context, flagFiles, inputFileCache, key);
        } catch (IOException e) {
          restoreInterrupt(e);
          String message = "IOException while borrowing a worker from the pool:";
          throw createUserExecException(e, message, Code.BORROW_FAILURE);
        }

        try (ResourceHandle handle =
            resourceManager.acquireResources(
                owner,
                spawn.getLocalResources(),
                context.speculating() ? ResourcePriority.DYNAMIC_WORKER : ResourcePriority.LOCAL)) {
          // We acquired a worker and resources -- mark that as queuing time.
          spawnMetrics.setQueueTime(queueStopwatch.elapsed());
          response =
              executeRequest(
                  spawn, context, inputFiles, outputs, workerOwner, key, request, spawnMetrics);
        }
      }

      if (response == null) {
        throw createEmptyResponseException(workerOwner.getWorker().getLogFile());
      }

      if (response.getWasCancelled()) {
        throw createUserExecException(
            "Received cancel response for " + response.getRequestId() + " without having cancelled",
            Code.FINISH_FAILURE);
      }

      try (SilentCloseable c =
          Profiler.instance()
              .profile(
                  ProfilerTask.WORKER_COPYING_OUTPUTS,
                  String.format(
                      "Worker #%d copying output files", workerOwner.getWorker().getWorkerId()))) {
        Stopwatch processOutputsStopwatch = Stopwatch.createStarted();
        context.lockOutputFiles(response.getExitCode(), response.getOutput(), null);
        workerOwner.getWorker().finishExecution(execRoot, outputs);
        spawnMetrics.setProcessOutputsTime(processOutputsStopwatch.elapsed());
      } catch (IOException e) {
        restoreInterrupt(e);
        String message =
            ErrorMessage.builder()
                .message("IOException while finishing worker execution:")
                .logFile(workerOwner.getWorker().getLogFile())
                .exception(e)
                .build()
                .toString();
        throw createUserExecException(message, Code.FINISH_FAILURE);
      }
    } catch (UserExecException e) {
      if (workerOwner.getWorker() != null) {
        try {
          workers.invalidateObject(key, workerOwner.getWorker());
        } catch (IOException e1) {
          // The original exception is more important / helpful, so we'll just ignore this one.
          restoreInterrupt(e1);
        } finally {
          workerOwner.setWorker(null);
        }
      }

      throw e;
    } finally {
      if (workerOwner.getWorker() != null) {
        workers.returnObject(key, workerOwner.getWorker());
      }
    }

    return response;
  }

  /**
   * Executes worker request in worker, waits until the response is ready. Worker and resources
   * should be allocated before call.
   */
  private WorkResponse executeRequest(
      Spawn spawn,
      SpawnExecutionContext context,
      SandboxInputs inputFiles,
      SandboxOutputs outputs,
      WorkerOwner workerOwner,
      WorkerKey key,
      WorkRequest request,
      SpawnMetrics.Builder spawnMetrics)
      throws ExecException, InterruptedException {
    WorkResponse response;
    context.report(SpawnExecutingEvent.create(key.getWorkerTypeName()));
    Worker worker = workerOwner.getWorker();

    try (SilentCloseable c =
        Profiler.instance()
            .profile(
                ProfilerTask.WORKER_SETUP,
                String.format("Worker #%d preparing execution", worker.getWorkerId()))) {
      // We consider `prepareExecution` to be also part of setup.
      Stopwatch prepareExecutionStopwatch = Stopwatch.createStarted();
      worker.prepareExecution(inputFiles, outputs, key.getWorkerFilesWithDigests().keySet());
      initializeMetricsSet(key, worker);
      spawnMetrics.addSetupTime(prepareExecutionStopwatch.elapsed());
    } catch (IOException e) {
      restoreInterrupt(e);
      String message =
          ErrorMessage.builder()
              .message("IOException while preparing the execution environment of a worker:")
              .logFile(worker.getLogFile())
              .exception(e)
              .build()
              .toString();
      throw createUserExecException(message, Code.PREPARE_FAILURE);
    }

    Stopwatch executionStopwatch = Stopwatch.createStarted();
    try {
      worker.putRequest(request);
    } catch (IOException e) {
      restoreInterrupt(e);
      String message =
          ErrorMessage.builder()
              .message(
                  "Worker process quit or closed its stdin stream when we tried to send a"
                      + " WorkRequest:")
              .logFile(worker.getLogFile())
              .exception(e)
              .build()
              .toString();
      throw createUserExecException(message, Code.REQUEST_FAILURE);
    }

    try (SilentCloseable c =
        Profiler.instance()
            .profile(
                ProfilerTask.WORKER_WORKING,
                String.format("Worker #%d working", worker.getWorkerId()))) {
      response = worker.getResponse(request.getRequestId());
    } catch (InterruptedException e) {
      if (worker.isSandboxed()) {
        // Sandboxed workers can safely finish their work async.
        finishWorkAsync(
            key,
            worker,
            request,
            workerOptions.workerCancellation && Spawns.supportsWorkerCancellation(spawn));
        workerOwner.setWorker(null);
      } else if (!context.speculating()) {
        // Non-sandboxed workers interrupted outside of dynamic execution can only mean that
        // the user interrupted the build, and we don't want to delay finishing. Instead we
        // kill the worker.
        // Technically, workers are always sandboxed under dynamic execution, at least for now.
        try {
          workers.invalidateObject(key, workerOwner.getWorker());
        } catch (IOException e1) {
          // Nothing useful we can do here, in fact it may not be possible to get here.
        } finally {
          workerOwner.setWorker(null);
        }
      }
      throw e;
    } catch (IOException e) {
      restoreInterrupt(e);
      // If protobuf or json reader couldn't parse the response, try to print whatever the
      // failing worker wrote to stdout - it's probably a stack trace or some kind of error
      // message that will help the user figure out why the compiler is failing.
      String recordingStreamMessage = worker.getRecordingStreamMessage();
      if (recordingStreamMessage.isEmpty()) {
        throw createEmptyResponseException(worker.getLogFile());
      } else {
        throw createUnparsableResponseException(recordingStreamMessage, worker.getLogFile(), e);
      }
    }

    spawnMetrics.setExecutionWallTime(executionStopwatch.elapsed());

    return response;
  }

  /**
   * Initializes metricsSet for workers. If worker metrics already exists for this worker, does
   * nothing
   */
  private void initializeMetricsSet(WorkerKey workerKey, Worker worker) {

    if (workerIdToWorkerMetric.containsKey(worker.getWorkerId())) {
      return;
    }
    long processId = worker.getProcessId();

    WorkerMetric workerMetric =
        new WorkerMetric(
            worker.getWorkerId(),
            processId,
            workerKey.getMnemonic(),
            workerKey.isMultiplex(),
            workerKey.isSandboxed());

    workerIdToWorkerMetric.put(worker.getWorkerId(), workerMetric);
  }

  // Collects process stats for each worker
  @VisibleForTesting
  public Map<Long, WorkerMetric.WorkerStat> collectStats(OS os, List<Long> processIds) {
    Map<Long, WorkerMetric.WorkerStat> pidResults = new HashMap<>();

    if (os != OS.LINUX && os != OS.DARWIN) {
      return pidResults;
    }

    List<Long> filteredProcessIds =
        processIds.stream().filter(p -> p > 0).collect(Collectors.toList());
    String pids = Joiner.on(",").join(filteredProcessIds);
    BufferedReader psOutput;

    try {
      String command = "ps -o pid,rss -p " + pids;
      psOutput =
          new BufferedReader(
              new InputStreamReader(
                  runtime.exec(new String[] {"bash", "-c", command}).getInputStream(), "UTF-8"));
    } catch (IOException e) {
      logger.atWarning().withCause(e).log("Error while executing command for pids: %s", pids);
      return pidResults;
    }

    try {
      // The output of the above ps command looks similar to this:
      // PID RSS
      // 211706 222972
      // 2612333 6180
      // We skip over the first line (the header) and then parse the PID and the resident memory
      // size in kilobytes.
      Instant now = Instant.now();
      String output = null;
      boolean isFirst = true;
      while ((output = psOutput.readLine()) != null) {
        if (isFirst) {
          isFirst = false;
          continue;
        }

        List<String> line = Splitter.on(" ").trimResults().omitEmptyStrings().splitToList(output);
        if (line.size() != 2) {
          logger.atWarning().log("Unexpected length of split line %s %d", output, line.size());
          continue;
        }

        long pid = Long.parseLong(line.get(0));
        int memoryInKb = Integer.parseInt(line.get(1)) / 1000;

        pidResults.put(pid, new WorkerMetric.WorkerStat(memoryInKb, now));
      }
    } catch (IllegalArgumentException | IOException e) {
      logger.atWarning().withCause(e).log("Error while parsing psOutput: %s", psOutput);
    }
    return pidResults;
  }

  /**
   * Starts a thread to collect the response from a worker when it's no longer of interest.
   *
   * <p>This can happen either when we lost the race in dynamic execution or the build got
   * interrupted. This takes ownership of the worker for purposes of returning it to the worker
   * pool.
   */
  private void finishWorkAsync(
      WorkerKey key, Worker worker, WorkRequest request, boolean canCancel) {
    Thread reaper =
        new Thread(
            () -> {
              Worker w = worker;
              try {
                if (canCancel) {
                  WorkRequest cancelRequest =
                      WorkRequest.newBuilder()
                          .setRequestId(request.getRequestId())
                          .setCancel(true)
                          .build();
                  w.putRequest(cancelRequest);
                }
                w.getResponse(request.getRequestId());
              } catch (IOException | InterruptedException e1) {
                // If this happens, we either can't trust the output of the worker, or we got
                // interrupted while handling being interrupted. In the latter case, let's stop
                // trying and just destroy the worker. If it's a singleplex worker, there will
                // be a dangling response that we don't want to keep trying to read, so we destroy
                // the worker.
                try {
                  workers.invalidateObject(key, w);
                  w = null;
                } catch (IOException | InterruptedException e2) {
                  // The reaper thread can't do anything useful about this.
                }
              } finally {
                if (w != null) {
                  try {
                    workers.returnObject(key, w);
                  } catch (IllegalStateException e3) {
                    // The worker already not part of the pool
                  }
                }
              }
            },
            "AsyncFinish-Worker-" + worker.workerId);
    reaper.start();
  }

  /**
   * The structure helps to pass the worker's ownership from one function to another. If worker is
   * set to null, then the ownership is taken by another function. E.g. used in finishWorkAsync.
   */
  private static class WorkerOwner {
    Worker worker;

    public void setWorker(Worker worker) {
      this.worker = worker;
    }

    public Worker getWorker() {
      return worker;
    }
  }

  private static void restoreInterrupt(IOException e) {
    if (e instanceof InterruptedIOException) {
      Thread.currentThread().interrupt();
    }
  }

  private static UserExecException createUserExecException(
      IOException e, String message, Code detailedCode) {
    return createUserExecException(
        ErrorMessage.builder().message(message).exception(e).build().toString(), detailedCode);
  }

  private static UserExecException createUserExecException(
      ForbiddenActionInputException e, String message, Code detailedCode) {
    return createUserExecException(
        ErrorMessage.builder().message(message).exception(e).build().toString(), detailedCode);
  }

  private static UserExecException createUserExecException(String message, Code detailedCode) {
    return new UserExecException(
        FailureDetail.newBuilder()
            .setMessage(message)
            .setWorker(FailureDetails.Worker.newBuilder().setCode(detailedCode))
            .build());
  }

  @SuppressWarnings("unused")
  @Subscribe
  public void onCollectMetricsEvent(CollectMetricsEvent event) {
    Map<Long, WorkerMetric.WorkerStat> workerStats =
        collectStats(
            OS.getCurrent(),
            this.workerIdToWorkerMetric.values().stream()
                .map(WorkerMetric::getProcessId)
                .collect(Collectors.toList()));

    for (WorkerMetric workerMetric : this.workerIdToWorkerMetric.values()) {
      WorkerMetric.WorkerStat workerStat = workerStats.get(workerMetric.getProcessId());
      if (workerStat == null) {
        workerMetric.setIsMeasurable(false);
        continue;
      }
      workerMetric.addWorkerStat(workerStat);
    }

    this.reporter.post(
        new WorkerMetricsEvent(new ArrayList<>(this.workerIdToWorkerMetric.values())));
    this.workerIdToWorkerMetric.clear();

    // remove dead workers from metrics list
    Map<Integer, WorkerMetric> measurableWorkerMetrics = new HashMap<>();
    for (WorkerMetric workerMetric : workerIdToWorkerMetric.values()) {
      if (workerMetric.getIsMeasurable()) {
        measurableWorkerMetrics.put(workerMetric.getWorkerId(), workerMetric);
      }
    }

    this.workerIdToWorkerMetric = measurableWorkerMetrics;
  }
}
