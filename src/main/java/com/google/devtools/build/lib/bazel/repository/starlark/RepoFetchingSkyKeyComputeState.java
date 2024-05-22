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

import static com.google.common.util.concurrent.MoreExecutors.directExecutor;

import com.google.common.base.Preconditions;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.ListeningExecutorService;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.common.util.concurrent.Uninterruptibles;
import com.google.devtools.build.lib.rules.repository.RepoRecordedInput;
import com.google.devtools.build.lib.rules.repository.RepositoryDirectoryValue;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunction.Environment.SkyKeyComputeState;
import java.util.Map;
import java.util.TreeMap;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.Callable;
import java.util.concurrent.Executors;
import java.util.concurrent.Semaphore;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Function;
import javax.annotation.Nullable;

/**
 * Captures state that persists across different invocations of {@link
 * com.google.devtools.build.lib.rules.repository.RepositoryDelegatorFunction}, specifically {@link
 * StarlarkRepositoryFunction}.
 *
 * <p>This class is used to hold on to a worker thread when fetching repos using a worker thread is
 * enabled. The worker thread uses a {@link SkyFunction.Environment} object acquired from the host
 * thread, and can signal the host thread to restart to get a fresh environment object.
 */
class RepoFetchingSkyKeyComputeState implements SkyKeyComputeState {
  sealed interface Message {}
  record Packet<M extends Message>(int sequenceNo, M message) {}

  /** Requests the worker thread can make to the host thread. */
  sealed interface Request extends Message {

    /** Ask for the current Skyframe environment. This does *not* cause a restart, just causes
     * the host thread to send over the current one, whatever it may be.
     *
     * <p>The host thread will answer with {@code Environment}.
     */
    record GetEnvironment() implements Request {}

    /**
     * Signals that the worker thread discovered new dependencies and that the host thread must
     * restart.
     *
     * <p>The host thread will respond with {@code Restarting}.
     */
    record NewDependencies() implements Request {}

    /** Signals that the repository was successfully fetched.
     *
     * <p>The host thread will respnond with {@cod RestartDecision}, which can cause the fetching to
     * be done again. If the response is that no restart is required, the worker thread will
     * terminate.
     */
    record Success(RepositoryDirectoryValue.Builder result) implements Request {}

    /**
     * Fetching failed. The host will answer with {@code FailureAcknowledged}. The worker thread
     * will terminate.
     */
    record Failure(Throwable e) implements Request {}
  }

  /** Runtime exception thrown on the worker thread when a {@link Response.Closing} is received.
   */
  class ClosingException extends RuntimeException {
  }

  /** Responses to the worker thread. */
  sealed interface Response extends Message {
    /** A new Skyframe environment for the worker thread. */
    record Environment(SkyFunction.Environment environment) implements Response {}

    /** Acknowledgement of a restart request. */
    record Restarting() implements Response {}

    /** Response for {@code Success}. If {@code restart} is true, the fetching needs to be repeated.
     */
    record RestartDecision(boolean restart) implements Response {}

    /** Response for {@code Failure}. */
    record FailureAcknowledged() implements Response {}

    /** Special response for when the close() interrupted the worker thread and it's just waiting
     * for it to die.
     */
    record Closing() implements Response {}
  }

  /** The name of the repository this state is for. Only there to ease debugging. */
  private final String repositoryName;

  /** The sequence number of the next request the worker thread will make. */
  private int nextRequestSequenceNo = 0;

  private final BlockingQueue<Packet<Request>> requestQueue = new ArrayBlockingQueue<>(1);
  private final BlockingQueue<Packet<Response>> responseQueue = new ArrayBlockingQueue<>(1);

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

  /**
   * Releases a permit on the {@code signalSemaphore} and immediately expect a fresh Environment
   * back. This may only be called from the worker thread.
   */
  SkyFunction.Environment signalForFreshEnv() throws InterruptedException {
    Response restartResponse = sendRequest(new Request.NewDependencies());
    Preconditions.checkState(restartResponse instanceof Response.Restarting);
    Response.Environment envResponse = (Response.Environment) sendRequest(new Request.GetEnvironment());
    return envResponse.environment;
  }

  private void logMaybe(String msg) {
    // System.err.println(msg);
  }

  Response sendRequest(Request request) throws InterruptedException {
    int seq = nextRequestSequenceNo++;
    Packet<Request> packet = new Packet(seq, request);
    logMaybe(String.format("LOG %s/worker: sending request %s", repositoryName, packet));
    requestQueue.put(packet);
    // This loop is necessary because the worker and host threads can get out of sync:
    // 1. The worker thread sends a request (say, NewDependencies)
    // 2. The host thread gets busy processing it
    // 3. The worker thread gets interrupted, sends a Failure request to signal its demise
    // 4. The host thread finishes processing the request in (1)
    // 5. Now the worker thread receives an answer to (1), even though it expects an answer to (3)
    while (true) {
      Packet<Response> response = responseQueue.take();
      logMaybe(String.format("LOG %s/worker: got response packet %s", repositoryName, response));
      Preconditions.checkState(response.sequenceNo <= seq);
      if (response.message instanceof Response.Closing) {
        throw new ClosingException();
      }

      if (response.sequenceNo == seq) {
        return response.message;
      }
    }
  }

  Response sendRequestUninterruptibly(Request request) {
    int seq = nextRequestSequenceNo++;
    Packet<Request> packet = new Packet(seq, request);
    logMaybe(String.format("LOG %s/worker: sending request uninterruptibly %s", repositoryName, packet));
    Uninterruptibles.putUninterruptibly(requestQueue, packet);
    while (true) {  // See sendRequest() to see why this loop is necessary
      Packet<Response> response = Uninterruptibles.takeUninterruptibly(responseQueue);
      logMaybe(String.format("LOG %s/worker: got response packet %s", repositoryName, response));
      Preconditions.checkState(response.sequenceNo <= seq);
      if (response.sequenceNo == seq) {
        return response.message;
      }
    }
  }

  Packet<Request> getRequest() throws InterruptedException {
    logMaybe(String.format("LOG %s/host: taking request", repositoryName));
    Packet<Request> result = requestQueue.take();
    logMaybe(String.format("LOG %s/host: Got request %s", repositoryName, result));
    return result;
  }

  Packet<Request> getRequestUninterruptibly() {
    logMaybe(String.format("LOG %s/host: taking request uninterruptibly", repositoryName));
    Packet<Request> result = Uninterruptibles.takeUninterruptibly(requestQueue);
    logMaybe(String.format("LOG %s/host: request received %s", repositoryName, result));
    return result;
  }

  void sendResponse(Packet<Request> request, Response response) throws InterruptedException {
    Packet<Response> packet = new Packet<>(request.sequenceNo, response);
    logMaybe(String.format("LOG %s/host: sending response %s", repositoryName, packet));
    responseQueue.put(packet);
  }

  void sendResponseUninterruptibly(Packet<Request> request, Response response) {
    Packet<Response> packet = new Packet<>(request.sequenceNo, response);
    logMaybe(String.format("LOG %s/host: sending response uninterruptibly %s", repositoryName, packet));
    Uninterruptibles.putUninterruptibly(responseQueue, packet);
  }

  /** Join the worker thread.
   *
   * This method should only be called if it either sent a {@code Failure} message or it received
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

  // This may be called from any thread, including the host Skyframe thread and the
  // high-memory-pressure listener thread.
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

    // Drain the request queue until we get a message that signals that the worker thread will
    // quit. Don't let an InterruptedException deter us from joining the worker thread because
    // otherwise, it'll be in an unknown state.
    MessageLoop:
    while (true) {
      Packet<Request> packet = getRequestUninterruptibly();
      switch (packet.message) {
        case Request.Failure unused: {
          sendResponseUninterruptibly(packet, new Response.FailureAcknowledged());
          break MessageLoop;
        }
        case Request.Success unused: {
          sendResponseUninterruptibly(packet, new Response.RestartDecision(false));
          break MessageLoop;
        }

        default: {
          // This may crash the worker thread, but we don't care -- it has already been interrupted
          // so we don't expect useful work from it anyway.
          sendResponseUninterruptibly(packet, new Response.Closing());
          continue MessageLoop;
        }
      }
    }

    Uninterruptibles.joinUninterruptibly(workerThread);
    logMaybe(String.format("LOG %s/host: closing with interrupt done", repositoryName));
  }
}
