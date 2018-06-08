// Copyright 2018 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.skylarkdebug.server;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.events.EventKind;
import com.google.devtools.build.lib.events.util.EventCollectionApparatus;
import com.google.devtools.build.lib.skylarkdebugging.SkylarkDebuggingProtos;
import com.google.devtools.build.lib.skylarkdebugging.SkylarkDebuggingProtos.Breakpoint;
import com.google.devtools.build.lib.skylarkdebugging.SkylarkDebuggingProtos.ContinueExecutionRequest;
import com.google.devtools.build.lib.skylarkdebugging.SkylarkDebuggingProtos.DebugEvent;
import com.google.devtools.build.lib.skylarkdebugging.SkylarkDebuggingProtos.DebugRequest;
import com.google.devtools.build.lib.skylarkdebugging.SkylarkDebuggingProtos.EvaluateRequest;
import com.google.devtools.build.lib.skylarkdebugging.SkylarkDebuggingProtos.Frame;
import com.google.devtools.build.lib.skylarkdebugging.SkylarkDebuggingProtos.ListFramesRequest;
import com.google.devtools.build.lib.skylarkdebugging.SkylarkDebuggingProtos.ListFramesResponse;
import com.google.devtools.build.lib.skylarkdebugging.SkylarkDebuggingProtos.ListThreadsRequest;
import com.google.devtools.build.lib.skylarkdebugging.SkylarkDebuggingProtos.ListThreadsResponse;
import com.google.devtools.build.lib.skylarkdebugging.SkylarkDebuggingProtos.Location;
import com.google.devtools.build.lib.skylarkdebugging.SkylarkDebuggingProtos.Scope;
import com.google.devtools.build.lib.skylarkdebugging.SkylarkDebuggingProtos.SetBreakpointsRequest;
import com.google.devtools.build.lib.skylarkdebugging.SkylarkDebuggingProtos.StartDebuggingRequest;
import com.google.devtools.build.lib.skylarkdebugging.SkylarkDebuggingProtos.StartDebuggingResponse;
import com.google.devtools.build.lib.skylarkdebugging.SkylarkDebuggingProtos.Stepping;
import com.google.devtools.build.lib.syntax.BuildFileAST;
import com.google.devtools.build.lib.syntax.DebugServerUtils;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.Mutability;
import com.google.devtools.build.lib.syntax.ParserInputSource;
import com.google.devtools.build.lib.syntax.SkylarkList;
import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import java.io.IOException;
import java.net.InetAddress;
import java.net.ServerSocket;
import java.time.Duration;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Integration tests for {@link SkylarkDebugServer}. */
@RunWith(JUnit4.class)
public class SkylarkDebugServerTest {

  private final ExecutorService executor = Executors.newFixedThreadPool(2);
  private final Scratch scratch = new Scratch();
  private final EventCollectionApparatus events =
      new EventCollectionApparatus(EventKind.ALL_EVENTS);

  private MockDebugClient client;
  private SkylarkDebugServer server;

  @Before
  public void setUpServerAndClient() throws Exception {
    ServerSocket serverSocket = new ServerSocket(0, 1, InetAddress.getByName(null));
    Future<SkylarkDebugServer> future =
        executor.submit(
            () -> SkylarkDebugServer.createAndWaitForConnection(events.reporter(), serverSocket));
    client = new MockDebugClient();
    client.connect(serverSocket, Duration.ofSeconds(10));

    server = future.get(10, TimeUnit.SECONDS);
    assertThat(server).isNotNull();
    DebugServerUtils.initializeDebugServer(server);
  }

  @After
  public void shutDown() throws Exception {
    if (client != null) {
      client.close();
    }
    if (server != null) {
      server.close();
    }
  }

  @Test
  public void testStartDebuggingResponseReceived() throws Exception {
    DebugEvent response =
        client.sendRequestAndWaitForResponse(
            DebugRequest.newBuilder()
                .setSequenceNumber(1)
                .setStartDebugging(StartDebuggingRequest.newBuilder())
                .build());
    assertThat(response)
        .isEqualTo(
            DebugEvent.newBuilder()
                .setSequenceNumber(1)
                .setStartDebugging(StartDebuggingResponse.newBuilder().build())
                .build());
  }

  @Test
  public void testThreadRegisteredEvents() throws Exception {
    sendStartDebuggingRequest();
    String threadName = Thread.currentThread().getName();
    long threadId = Thread.currentThread().getId();
    DebugServerUtils.runWithDebuggingIfEnabled(newEnvironment(), () -> threadName, () -> true);

    client.waitForEvent(DebugEvent::hasThreadEnded, Duration.ofSeconds(5));

    assertThat(client.unnumberedEvents)
        .containsExactly(
            DebugEventHelper.threadStartedEvent(threadId, threadName),
            DebugEventHelper.threadEndedEvent(threadId, threadName));
  }

  @Test
  public void testPausedUntilStartDebuggingRequestReceived() throws Exception {
    BuildFileAST buildFile = parseBuildFile("/a/build/file/BUILD", "x = [1,2,3]");
    Environment env = newEnvironment();

    Thread evaluationThread = execInWorkerThread(buildFile, env);
    String threadName = evaluationThread.getName();
    long threadId = evaluationThread.getId();

    // wait for BUILD evaluation to start
    client.waitForEvent(DebugEvent::hasThreadPaused, Duration.ofSeconds(5));

    assertThat(listThreads().getThreadList())
        .containsExactly(
            SkylarkDebuggingProtos.Thread.newBuilder()
                .setId(threadId)
                .setName(threadName)
                .setLocation(
                    DebugEventHelper.getLocationProto(
                        buildFile.getStatements().get(0).getLocation()))
                .setIsPaused(true)
                .build());

    sendStartDebuggingRequest();
    client.waitForEvent(DebugEvent::hasThreadEnded, Duration.ofSeconds(5));
    assertThat(listThreads().getThreadList()).isEmpty();
    assertThat(client.unnumberedEvents)
        .containsAllOf(
            DebugEventHelper.threadContinuedEvent(
                SkylarkDebuggingProtos.Thread.newBuilder()
                    .setName(threadName)
                    .setId(threadId)
                    .build()),
            DebugEventHelper.threadEndedEvent(threadId, threadName));
  }

  @Test
  public void testPauseAtBreakpoint() throws Exception {
    sendStartDebuggingRequest();
    BuildFileAST buildFile = parseBuildFile("/a/build/file/BUILD", "x = [1,2,3]", "y = [2,3,4]");
    Environment env = newEnvironment();

    Location breakpoint =
        Location.newBuilder().setLineNumber(1).setPath("/a/build/file/BUILD").build();
    setBreakpoints(ImmutableList.of(breakpoint));

    Thread evaluationThread = execInWorkerThread(buildFile, env);
    String threadName = evaluationThread.getName();
    long threadId = evaluationThread.getId();

    // wait for breakpoint to be hit
    client.waitForEvent(DebugEvent::hasThreadPaused, Duration.ofSeconds(5));

    assertThat(client.unnumberedEvents)
        .contains(
            DebugEventHelper.threadPausedEvent(
                SkylarkDebuggingProtos.Thread.newBuilder()
                    .setName(threadName)
                    .setId(threadId)
                    .setIsPaused(true)
                    .setLocation(breakpoint.toBuilder().setColumnNumber(1))
                    .build(),
                ImmutableList.of(
                    Frame.newBuilder()
                        .setFunctionName("<top level>")
                        .setLocation(breakpoint.toBuilder().setColumnNumber(1))
                        .addScope(Scope.newBuilder().setName("global"))
                        .build())));

    assertThat(listThreads().getThreadList())
        .containsExactly(
            SkylarkDebuggingProtos.Thread.newBuilder()
                .setId(threadId)
                .setName(threadName)
                .setLocation(breakpoint.toBuilder().setColumnNumber(1))
                .setIsPaused(true)
                .build());
  }

  @Test
  public void testListFramesForInvalidThread() throws Exception {
    sendStartDebuggingRequest();
    DebugEvent event =
        client.sendRequestAndWaitForResponse(
            DebugRequest.newBuilder()
                .setSequenceNumber(1)
                .setListFrames(ListFramesRequest.newBuilder().setThreadId(20).build())
                .build());
    assertThat(event.hasError()).isTrue();
    assertThat(event.getError().getMessage()).contains("Thread 20 is not running");
  }

  @Test
  public void testSimpleListFramesRequest() throws Exception {
    sendStartDebuggingRequest();
    BuildFileAST buildFile = parseBuildFile("/a/build/file/BUILD", "x = [1,2,3]", "y = [2,3,4]");
    Environment env = newEnvironment();

    Location breakpoint =
        Location.newBuilder().setLineNumber(2).setPath("/a/build/file/BUILD").build();
    setBreakpoints(ImmutableList.of(breakpoint));

    Thread evaluationThread = execInWorkerThread(buildFile, env);
    long threadId = evaluationThread.getId();

    // wait for breakpoint to be hit
    client.waitForEvent(DebugEvent::hasThreadPaused, Duration.ofSeconds(5));

    ListFramesResponse frames = listFrames(threadId);
    assertThat(frames.getFrameCount()).isEqualTo(1);
    assertThat(frames.getFrame(0))
        .isEqualTo(
            Frame.newBuilder()
                .setFunctionName("<top level>")
                .setLocation(breakpoint.toBuilder().setColumnNumber(1))
                .addScope(
                    Scope.newBuilder()
                        .setName("global")
                        .addBinding(
                            DebuggerSerialization.getValueProto(
                                "x", SkylarkList.createImmutable(ImmutableList.of(1, 2, 3)))))
                .build());
  }

  @Test
  public void testListFramesShadowedBinding() throws Exception {
    sendStartDebuggingRequest();
    BuildFileAST bzlFile =
        parseSkylarkFile(
            "/a/build/file/test.bzl",
            "a = 1",
            "c = 3",
            "def fn():",
            "  a = 2",
            "  b = 1",
            "  b + 1",
            "fn()");
    Environment env = newEnvironment();

    Location breakpoint =
        Location.newBuilder().setPath("/a/build/file/test.bzl").setLineNumber(6).build();
    setBreakpoints(ImmutableList.of(breakpoint));

    Thread evaluationThread = execInWorkerThread(bzlFile, env);
    long threadId = evaluationThread.getId();

    // wait for breakpoint to be hit
    client.waitForEvent(DebugEvent::hasThreadPaused, Duration.ofSeconds(5));

    ListFramesResponse frames = listFrames(threadId);
    assertThat(frames.getFrameCount()).isEqualTo(2);

    assertThat(frames.getFrame(0))
        .isEqualTo(
            Frame.newBuilder()
                .setFunctionName("fn")
                .setLocation(breakpoint.toBuilder().setColumnNumber(3))
                .addScope(
                    Scope.newBuilder()
                        .setName("local")
                        .addBinding(DebuggerSerialization.getValueProto("a", 2))
                        .addBinding(DebuggerSerialization.getValueProto("b", 1)))
                .addScope(
                    Scope.newBuilder()
                        .setName("global")
                        .addBinding(DebuggerSerialization.getValueProto("c", 3))
                        .addBinding(DebuggerSerialization.getValueProto("fn", env.lookup("fn"))))
                .build());

    assertThat(frames.getFrame(1))
        .isEqualTo(
            Frame.newBuilder()
                .setFunctionName("<top level>")
                .setLocation(
                    Location.newBuilder()
                        .setPath("/a/build/file/test.bzl")
                        .setLineNumber(7)
                        .setColumnNumber(1))
                .addScope(
                    Scope.newBuilder()
                        .setName("global")
                        .addBinding(DebuggerSerialization.getValueProto("a", 1))
                        .addBinding(DebuggerSerialization.getValueProto("c", 3))
                        .addBinding(DebuggerSerialization.getValueProto("fn", env.lookup("fn"))))
                .build());
  }

  @Test
  public void testEvaluateRequest() throws Exception {
    sendStartDebuggingRequest();
    BuildFileAST buildFile = parseBuildFile("/a/build/file/BUILD", "x = [1,2,3]", "y = [2,3,4]");
    Environment env = newEnvironment();

    Location breakpoint =
        Location.newBuilder().setLineNumber(2).setPath("/a/build/file/BUILD").build();
    setBreakpoints(ImmutableList.of(breakpoint));

    Thread evaluationThread = execInWorkerThread(buildFile, env);
    long threadId = evaluationThread.getId();

    // wait for breakpoint to be hit
    client.waitForEvent(DebugEvent::hasThreadPaused, Duration.ofSeconds(5));

    DebugEvent response =
        client.sendRequestAndWaitForResponse(
            DebugRequest.newBuilder()
                .setSequenceNumber(123)
                .setEvaluate(
                    EvaluateRequest.newBuilder()
                        .setThreadId(threadId)
                        .setExpression("x[1]")
                        .build())
                .build());
    assertThat(response.hasEvaluate()).isTrue();
    assertThat(response.getEvaluate().getResult())
        .isEqualTo(DebuggerSerialization.getValueProto("Evaluation result", 2));
  }

  @Test
  public void testEvaluateRequestThrowingException() throws Exception {
    sendStartDebuggingRequest();
    BuildFileAST buildFile = parseBuildFile("/a/build/file/BUILD", "x = [1,2,3]", "y = [2,3,4]");
    Environment env = newEnvironment();

    Location breakpoint =
        Location.newBuilder().setLineNumber(2).setPath("/a/build/file/BUILD").build();
    setBreakpoints(ImmutableList.of(breakpoint));

    Thread evaluationThread = execInWorkerThread(buildFile, env);
    long threadId = evaluationThread.getId();

    // wait for breakpoint to be hit
    client.waitForEvent(DebugEvent::hasThreadPaused, Duration.ofSeconds(5));

    DebugEvent response =
        client.sendRequestAndWaitForResponse(
            DebugRequest.newBuilder()
                .setSequenceNumber(123)
                .setEvaluate(
                    EvaluateRequest.newBuilder()
                        .setThreadId(threadId)
                        .setExpression("z[0]")
                        .build())
                .build());
    assertThat(response.hasError()).isTrue();
    assertThat(response.getError().getMessage()).isEqualTo("name 'z' is not defined");
  }

  @Test
  public void testStepIntoFunction() throws Exception {
    sendStartDebuggingRequest();
    BuildFileAST bzlFile =
        parseSkylarkFile(
            "/a/build/file/test.bzl",
            "def fn():",
            "  a = 2",
            "  return a",
            "x = fn()",
            "y = [2,3,4]");
    Environment env = newEnvironment();

    Location breakpoint =
        Location.newBuilder().setLineNumber(4).setPath("/a/build/file/test.bzl").build();
    setBreakpoints(ImmutableList.of(breakpoint));

    Thread evaluationThread = execInWorkerThread(bzlFile, env);
    long threadId = evaluationThread.getId();

    // wait for breakpoint to be hit
    client.waitForEvent(DebugEvent::hasThreadPaused, Duration.ofSeconds(5));

    assertThat(listThreads().getThread(0).getLocation().getLineNumber()).isEqualTo(4);

    client.unnumberedEvents.clear();
    client.sendRequestAndWaitForResponse(
        DebugRequest.newBuilder()
            .setSequenceNumber(2)
            .setContinueExecution(
                ContinueExecutionRequest.newBuilder()
                    .setThreadId(threadId)
                    .setStepping(Stepping.INTO)
                    .build())
            .build());
    client.waitForEvent(DebugEvent::hasThreadPaused, Duration.ofSeconds(5));

    // check we're paused inside the function
    assertThat(listFrames(threadId).getFrameCount()).isEqualTo(2);

    // and verify the exact line index as well
    ListThreadsResponse threads = listThreads();
    assertThat(threads.getThreadList()).hasSize(1);
    assertThat(threads.getThread(0).getIsPaused()).isTrue();
    assertThat(threads.getThread(0).getLocation().getLineNumber()).isEqualTo(2);
  }

  @Test
  public void testStepOverFunction() throws Exception {
    sendStartDebuggingRequest();
    BuildFileAST bzlFile =
        parseSkylarkFile(
            "/a/build/file/test.bzl",
            "def fn():",
            "  a = 2",
            "  return a",
            "x = fn()",
            "y = [2,3,4]");
    Environment env = newEnvironment();

    Location breakpoint =
        Location.newBuilder().setLineNumber(4).setPath("/a/build/file/test.bzl").build();
    setBreakpoints(ImmutableList.of(breakpoint));

    Thread evaluationThread = execInWorkerThread(bzlFile, env);
    long threadId = evaluationThread.getId();

    // wait for breakpoint to be hit
    client.waitForEvent(DebugEvent::hasThreadPaused, Duration.ofSeconds(5));

    assertThat(listThreads().getThread(0).getLocation().getLineNumber()).isEqualTo(4);

    client.unnumberedEvents.clear();
    client.sendRequestAndWaitForResponse(
        DebugRequest.newBuilder()
            .setSequenceNumber(2)
            .setContinueExecution(
                ContinueExecutionRequest.newBuilder()
                    .setThreadId(threadId)
                    .setStepping(Stepping.OVER)
                    .build())
            .build());
    client.waitForEvent(DebugEvent::hasThreadPaused, Duration.ofSeconds(5));

    ListThreadsResponse threads = listThreads();
    assertThat(threads.getThreadList()).hasSize(1);
    assertThat(threads.getThread(0).getIsPaused()).isTrue();
    assertThat(threads.getThread(0).getLocation().getLineNumber()).isEqualTo(5);
  }

  @Test
  public void testStepOutOfFunction() throws Exception {
    sendStartDebuggingRequest();
    BuildFileAST bzlFile =
        parseSkylarkFile(
            "/a/build/file/test.bzl",
            "def fn():",
            "  a = 2",
            "  return a",
            "x = fn()",
            "y = [2,3,4]");
    Environment env = newEnvironment();

    Location breakpoint =
        Location.newBuilder().setLineNumber(2).setPath("/a/build/file/test.bzl").build();
    setBreakpoints(ImmutableList.of(breakpoint));

    Thread evaluationThread = execInWorkerThread(bzlFile, env);
    long threadId = evaluationThread.getId();

    // wait for breakpoint to be hit
    client.waitForEvent(DebugEvent::hasThreadPaused, Duration.ofSeconds(5));

    assertThat(listFrames(threadId).getFrameCount()).isEqualTo(2);

    client.unnumberedEvents.clear();
    client.sendRequestAndWaitForResponse(
        DebugRequest.newBuilder()
            .setSequenceNumber(2)
            .setContinueExecution(
                ContinueExecutionRequest.newBuilder()
                    .setThreadId(threadId)
                    .setStepping(Stepping.OUT)
                    .build())
            .build());
    client.waitForEvent(DebugEvent::hasThreadPaused, Duration.ofSeconds(5));

    ListThreadsResponse threads = listThreads();
    assertThat(threads.getThreadList()).hasSize(1);
    assertThat(threads.getThread(0).getIsPaused()).isTrue();
    assertThat(threads.getThread(0).getLocation().getLineNumber()).isEqualTo(5);
  }

  private void setBreakpoints(Iterable<Location> locations) throws Exception {
    SetBreakpointsRequest.Builder request = SetBreakpointsRequest.newBuilder();
    locations.forEach(l -> request.addBreakpoint(Breakpoint.newBuilder().setLocation(l)));
    DebugEvent response =
        client.sendRequestAndWaitForResponse(
            DebugRequest.newBuilder().setSequenceNumber(10).setSetBreakpoints(request).build());
    assertThat(response.hasSetBreakpoints()).isTrue();
    assertThat(response.getSequenceNumber()).isEqualTo(10);
  }

  private void sendStartDebuggingRequest() throws Exception {
    client.sendRequestAndWaitForResponse(
        DebugRequest.newBuilder()
            .setSequenceNumber(1)
            .setStartDebugging(StartDebuggingRequest.newBuilder())
            .build());
  }

  private ListThreadsResponse listThreads() throws Exception {
    DebugEvent event =
        client.sendRequestAndWaitForResponse(
            DebugRequest.newBuilder()
                .setSequenceNumber(1)
                .setListThreads(ListThreadsRequest.newBuilder())
                .build());
    assertThat(event.hasListThreads()).isTrue();
    assertThat(event.getSequenceNumber()).isEqualTo(1);
    return event.getListThreads();
  }

  private ListFramesResponse listFrames(long threadId) throws Exception {
    DebugEvent event =
        client.sendRequestAndWaitForResponse(
            DebugRequest.newBuilder()
                .setSequenceNumber(1)
                .setListFrames(ListFramesRequest.newBuilder().setThreadId(threadId).build())
                .build());
    assertThat(event.hasListFrames()).isTrue();
    assertThat(event.getSequenceNumber()).isEqualTo(1);
    return event.getListFrames();
  }

  private static Environment newEnvironment() {
    Mutability mutability = Mutability.create("test");
    return Environment.builder(mutability).useDefaultSemantics().build();
  }

  private BuildFileAST parseBuildFile(String path, String... lines) throws IOException {
    Path file = scratch.file(path, lines);
    byte[] bytes = FileSystemUtils.readWithKnownFileSize(file, file.getFileSize());
    ParserInputSource inputSource = ParserInputSource.create(bytes, file.asFragment());
    return BuildFileAST.parseBuildFile(inputSource, events.reporter());
  }

  private BuildFileAST parseSkylarkFile(String path, String... lines) throws IOException {
    Path file = scratch.file(path, lines);
    byte[] bytes = FileSystemUtils.readWithKnownFileSize(file, file.getFileSize());
    ParserInputSource inputSource = ParserInputSource.create(bytes, file.asFragment());
    return BuildFileAST.parseSkylarkFile(inputSource, events.reporter());
  }

  /**
   * Creates and starts a worker thread executing the given {@link BuildFileAST} in the given
   * environment.
   */
  private Thread execInWorkerThread(BuildFileAST ast, Environment env) {
    Thread thread =
        new Thread(
            () -> {
              try {
                ast.exec(env, events.collector());
              } catch (Throwable e) {
                throw new AssertionError(e);
              }
            });
    thread.start();
    return thread;
  }
}
