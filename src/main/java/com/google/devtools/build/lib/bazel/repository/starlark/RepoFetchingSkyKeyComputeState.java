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

package com.google.devtools.build.lib.bazel.repository.starlark;

import com.google.common.util.concurrent.Uninterruptibles;
import com.google.devtools.build.lib.bazel.repository.starlark.RepoFetchingSkyKeyComputeState.HostState.EnvironmentSent;
import com.google.devtools.build.lib.bazel.repository.starlark.RepoFetchingSkyKeyComputeState.WorkerState.EnvironmentReceived;
import com.google.devtools.build.lib.bazel.repository.starlark.RepoFetchingSkyKeyComputeState.HostState.TerminationRequested;
import com.google.devtools.build.lib.rules.repository.RepoRecordedInput;
import com.google.devtools.build.lib.rules.repository.RepositoryDirectoryValue;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunction.Environment.SkyKeyComputeState;
import java.util.Map;
import java.util.TreeMap;
import java.util.concurrent.Exchanger;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Function;

/**
 * Captures state that persists across different invocations of {@link
 * com.google.devtools.build.lib.rules.repository.RepositoryDelegatorFunction}, specifically {@link
 * StarlarkRepositoryFunction}.
 *
 * <p>This class is used to hold on to a worker thread (in reality just a {@link Future} object)
 * when fetching repos using a worker thread is enabled. The worker thread uses a {@link
 * SkyFunction.Environment} object acquired from the host thread, and can signal the host thread to
 * restart to get a fresh environment object.
 */
class RepoFetchingSkyKeyComputeState implements SkyKeyComputeState {
  /** The state the worker thread is stopped in. */
  sealed interface WorkerState {
    /**
     * Indicates that the worker thread added a Skyframe dependency that's not evaluated yet.
     *
     * <p>The host thread should return {@code null}, causing a Skyframe restart, then send a new
     * environment to the worker to unblock it.
     */
    record EnvironmentRequested() implements WorkerState {}

    /** Indicates that the worker received an environment it requested.
     *
     * <P>The next state it will be in is {@code EnvironmentRequested}, {@code Success} or
     * {@code Failure}.
     */
    record EnvironmentReceived() implements WorkerState {}

    /** Indicates that the worker thread has finished running with a failure.
     *
     * <p>It will terminate after successfully sending this signal.
     */
    record Failure(Throwable e) implements WorkerState {}

    /** Indicates that the worker thread has finished fetching the repository successfully.
     *
     * <p>It is now waiting for the host thread to decide whether it should terminate or fetch the
     * repository again (happens due to a nuance in environment handling, see {@code
     * StarlarkRepositoryFunction.fetch()}.
     */
    record Success(RepositoryDirectoryValue.Builder result) implements WorkerState {}

    /** The worker thread has finished fetching the repository successfully.
     *
     * <p>It is now waiting for the host thread to decide whether it should terminate or fetch the
     * repository again (happens due to a nuance in environment handling, see {@code
     * StarlarkRepositoryFunction.fetch()}. It should respond with {@code FullRestartRequested}
     * or {@code TerminationRequested} as appropriate.
     */
    record WaitingForRestartDecision() implements WorkerState {}
  }

  /** The state the host thread is stopped in. */
  sealed interface HostState {

    /** A new environment the worker thread requested.
     *
     * <p>The worker thread needs to reply with {@code EnvironmentReceived}.
     */
    record EnvironmentSent(SkyFunction.Environment environment) implements HostState {}

    /** The host thread has sent an environment to the worker thread.
     *
     * <p>It is now waiting for the worker to decide whether it needs a new environment or it can
     * do its job without one. The worker should respond with {@code EnvironmentSent},
     * {@code Success} or {@code Failure} as a appropriate.
     */
    record WaitingForWorker() implements HostState {}

    /** Indicates that the worker thread needs to be restarted, despite finishing with success.
     *
     * <p>After this, the host thread will send a new environment (with {@code EnvironmentSent}).
     */
    record FullRestartRequested() implements HostState {}

    /** Indicates that the worker thread should terminate after finishing with success.
     *
     * <p>After this, the host thread will join the worker one.
     */
    record TerminationRequested() implements HostState {}
  }

  /** The name of the repository this state is for. Only there to ease debugging. */
  private final String repositoryName;

  /** Used to ensure that the worker and Skyframe threads both are at a known place. */
  private final Exchanger<Object> rendezvous = new Exchanger();

  /** The working thread that actually performs the fetching logic. */
  private final Thread workerThread;

  /** Set to indicate that close() has been called at least once. */
  private final AtomicBoolean closeInProgress = new AtomicBoolean(false);

  /**
   * This is where the recorded inputs & values for the whole invocation is collected.
   *
   * <p>{@link com.google.devtools.build.lib.rules.repository.RepositoryDelegatorFunction} creates a
   * new map on each restart, so we can't simply plumb that in.
   */
  final Map<RepoRecordedInput, String> recordedInputValues = new TreeMap<>();

  RepoFetchingSkyKeyComputeState(String repositoryName,
      Function<RepoFetchingSkyKeyComputeState, Runnable> threadFunc) {
    this.repositoryName = repositoryName;
    workerThread =
        Thread.ofVirtual()
            .name("starlark-repository-" + repositoryName)
            .start(threadFunc.apply(this));
  }

  SkyFunction.Environment signalForFreshEnv() throws InterruptedException {
    // First unblock the host thread, then wait until it has something else to say. The only thing
    // the host thread can do to this one is an interrupt so there is no need to do anything special
    // in between.
    HostState hostState = rendezvousFromWorker(new WorkerState.EnvironmentRequested());
    if (!(hostState instanceof HostState.WaitingForWorker)) {
      throw new IllegalStateException("Host thread is in unexpected state %s" + hostState);
    }
    return getEnvironmentFromHost();
  }

  /** Get a new Environment from the host.
   *
   * <p>This should either be called right after the worker thread starts or after sending a
   * {@code EnvironmentSent} to the host.
   */
  SkyFunction.Environment getEnvironmentFromHost() throws InterruptedException {
    HostState hostState = rendezvousFromWorker(new EnvironmentReceived());
    return switch (hostState) {
      case EnvironmentSent(SkyFunction.Environment env) -> env;
      default -> throw new IllegalStateException(
          "Host thread is in unexpected state %s" + hostState);
    };
  }

  private void logMaybe(String msg) {
  }

  HostState rendezvousFromWorker(WorkerState workerState) throws InterruptedException {
    logMaybe(String.format("LOG %s/worker: sending %s", repositoryName, workerState));
    HostState result = (HostState) rendezvous.exchange(workerState);
    logMaybe(String.format("LOG %s/worker: received %s", repositoryName, result));
    return result;
  }

  WorkerState rendezvousFromHost(HostState hostState) throws InterruptedException {
    logMaybe(String.format("LOG %s/host: sending %s", repositoryName, hostState));
    WorkerState result = (WorkerState) rendezvous.exchange(hostState);
    logMaybe(String.format("LOG %s/host: received %s", repositoryName, result));
    return result;
  }

  HostState rendezvousUninterruptiblyFromWorker(WorkerState signal) {
    logMaybe(String.format("LOG %s/worker: sending %s uninterruptibly", repositoryName, signal));
    while(true) {
      try {
        return rendezvousFromWorker(signal);
      } catch (InterruptedException e) {
        logMaybe(String.format("LOG %s/worker: retrying on interrupt", repositoryName));
      }
    }
  }

  WorkerState rendezvousUninterruptiblyFromHost(HostState signal) {
    logMaybe(String.format("LOG %s/host: sending %s uninterruptibly", repositoryName, signal));
    while(true) {
      try {
        return rendezvousFromHost(signal);
      } catch (InterruptedException e) {
        logMaybe(String.format("LOG %s/host: retrying on interrupt", repositoryName));
      }
    }
  }

  /** Join the worker thread.
   *
   * This method should only bee called if it either sent a {@code Failure} message or it received
   * a {@code Finish} one.
   */
  public void join() {
    logMaybe(String.format("LOG %s/host: joining", repositoryName));
    closeInProgress.set(true);
    Uninterruptibles.joinUninterruptibly(workerThread);
    logMaybe(String.format("LOG %s/host: joined", repositoryName));
  }

  /** Closes the worker thread forcefully by interrupting it.
   *
   * Waits for the worker thread to terminate before returning.
   */
  @Override
  public void close() {
    if (closeInProgress.getAndSet(true)) {
      logMaybe(String.format("LOG %s/host: not closing", repositoryName));
      // Someone else already called close(). Just wait until they are done.
      Uninterruptibles.joinUninterruptibly(workerThread);
      logMaybe(String.format("LOG %s/host: not closing done", repositoryName));
      return;
    }

    logMaybe(String.format("LOG %s/host: closing with interrupt", repositoryName));
    workerThread.interrupt();

    // Wait until the worker thread actually gets interrupted. Be resilient to cases where despite
    // the interrupt above, the worker thread was already trying to post a restart. I'm not sure if
    // that can happen; theoretically, it looks like it shouldn't be but I'm not intimately familiar
    // with the exact semantics of thread interruption and it's cheap to be resilient. The important
    // part is that in case an interrupt happens, a Success or Failure is eventually posted by the
    // worker thread.
SignalLoop:
    while (true) {
      WorkerState signal = rendezvousUninterruptiblyFromHost(null);
      logMaybe(String.format("LOG %s/host: close signal is %s", repositoryName, signal));
      switch (signal) {
        case WorkerState.Failure unused: break SignalLoop;
        case WorkerState.Success unused: {
          // In this case, the worker thread expects an answer. */
          rendezvousUninterruptiblyFromHost(new TerminationRequested());
          break SignalLoop;
        }
        default: continue SignalLoop;
      }
    }

    Uninterruptibles.joinUninterruptibly(workerThread);
    logMaybe(String.format("LOG %s/host: closing with interrupt done", repositoryName));
  }
}
