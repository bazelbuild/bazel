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

package com.google.devtools.build.lib.starlarkdebug.server;

import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.common.truth.Truth.assertThat;
import static java.util.Arrays.stream;
import static java.util.concurrent.TimeUnit.SECONDS;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.buildtool.BuildResult;
import com.google.devtools.build.lib.buildtool.util.BuildIntegrationTestCase;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventCollector;
import com.google.devtools.build.lib.events.EventKind;
import com.google.devtools.build.lib.runtime.BlazeRuntime;
import com.google.devtools.build.lib.skyframe.SkyFunctions;
import com.google.devtools.build.lib.starlarkdebug.module.StarlarkDebuggerModule;
import com.google.devtools.build.lib.starlarkdebugging.StarlarkDebuggingProtos.Breakpoint;
import com.google.devtools.build.lib.starlarkdebugging.StarlarkDebuggingProtos.DebugRequest;
import com.google.devtools.build.lib.starlarkdebugging.StarlarkDebuggingProtos.Location;
import com.google.devtools.build.lib.starlarkdebugging.StarlarkDebuggingProtos.SetBreakpointsRequest;
import com.google.devtools.build.lib.starlarkdebugging.StarlarkDebuggingProtos.StartDebuggingRequest;
import com.google.devtools.build.lib.testutil.MoreAsserts;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.QueryableGraph.Reason;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.io.IOException;
import java.net.InetAddress;
import java.time.Duration;
import java.util.Collection;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Consumer;
import java.util.regex.Pattern;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class StarlarkDebugIntegrationTest extends BuildIntegrationTestCase {
  private static final AtomicInteger sequenceIds = new AtomicInteger(1);

  private final ExecutorService executor = Executors.newFixedThreadPool(1);
  private final Collection<Event> eventCollector = new ConcurrentLinkedQueue<>();

  @Override
  protected BlazeRuntime.Builder getRuntimeBuilder() throws Exception {
    return super.getRuntimeBuilder().addBlazeModule(new StarlarkDebuggerModule());
  }

  @Before
  public void setup() throws Exception {
    addOptions("--experimental_skylark_debug", "--experimental_skylark_debug_server_port=0");
    eventCollector.clear();
    events.addHandler(new EventCollector(EventKind.ALL_EVENTS, eventCollector));
  }

  @Test
  public void testAnalysisResetBlocksOnDebuggingStart() throws Exception {
    addOptions("--experimental_skylark_debug_reset_analysis");
    write("foo/BUILD", "genrule(name = 'foo', outs = ['foo.out'], cmd = 'touch $@')");

    // run async, otherwise this will just block on the result indefinitely
    CompletableFuture<BuildResult> resultCf =
        CompletableFuture.supplyAsync(
            () -> {
              try {
                return buildTarget(StarlarkDebugIntegrationTest::createClient, "//foo");
              } catch (Exception e) {
                throw new RuntimeException(e);
              }
            },
            Executors.newSingleThreadExecutor());

    TimeoutException unusedError =
        assertThrows(TimeoutException.class, () -> resultCf.get(10, SECONDS));
  }

  @Test
  public void testAnalysisResetWithNoBreakpoints() throws Exception {
    addOptions("--experimental_skylark_debug_reset_analysis");
    write("foo/BUILD", "genrule(name = 'foo', outs = ['foo.out'], cmd = 'touch $@')");

    BuildResult result = buildTarget(this::createClientAndSetBreakpoints, "//foo");

    assertThat(result).isNotNull();
    assertThat(result.getSuccessfulTargets()).hasSize(1);
    MoreAsserts.assertDoesNotContainEvent(eventCollector, "did not receive breakpoints");
    MoreAsserts.assertContainsEvent(eventCollector, "resetting analysis for: []");
  }

  @Test
  public void testAnalysisResetWithBreakpoint() throws Exception {
    addOptions("--experimental_skylark_debug_reset_analysis");
    write("foo/BUILD", "genrule(name = 'foo', outs = ['foo.out'], cmd = 'touch $@')");

    BuildResult result =
        buildTarget(debugPort -> createClientAndSetBreakpoints(debugPort, "foo/BUILD"), "//foo");

    MoreAsserts.assertContainsEvent(
        eventCollector, Pattern.compile("resetting analysis for: .*/foo/BUILD"));
    assertThat(result).isNotNull();
    assertThat(result.getSuccessfulTargets()).hasSize(1);
  }

  @Test
  public void testAnalysisResetWithBreakpointDeletesSkyframeFileNode() throws Exception {
    write("foo/BUILD", "genrule(name = 'foo', outs = ['foo.out'], cmd = 'touch $@')");

    // first build to populate skyframe
    BuildResult result =
        buildTarget(StarlarkDebugIntegrationTest::createClientAndStartDebugging, "//foo");
    assertThat(result).isNotNull();

    Set<String> deletedFiles = ConcurrentHashMap.newKeySet();
    injectListenerAtStartOfNextBuild(
        (key, type, order, context) -> {
          if (Objects.equals(key.functionName(), SkyFunctions.FILE)
              && Objects.equals(context, Reason.INVALIDATION)) {
            deletedFiles.add(((RootedPath) key.argument()).getRootRelativePath().getPathString());
          }
        });
    addOptions("--experimental_skylark_debug_reset_analysis");

    // rebuild with non-existent breakpoint
    result =
        buildTarget(debugPort -> createClientAndSetBreakpoints(debugPort, "bar/BUILD"), "//foo");
    assertThat(result).isNotNull();
    assertThat(deletedFiles).isEmpty();

    // rebuild with breakpoint on build file
    result =
        buildTarget(debugPort -> createClientAndSetBreakpoints(debugPort, "foo/BUILD"), "//foo");
    assertThat(result).isNotNull();
    assertThat(deletedFiles).contains("foo/BUILD");
  }

  private BuildResult buildTarget(Consumer<Integer> clientSetup, String target) throws Exception {
    DebugServerTransport.onListenPortCallbackForTests =
        port -> {
          var unused = executor.submit(() -> clientSetup.accept(port));
        };
    return super.buildTarget(target);
  }

  @CanIgnoreReturnValue
  private static MockDebugClient createClient(int debugPort) {
    MockDebugClient client = new MockDebugClient();
    client.connect(InetAddress.getLoopbackAddress(), debugPort, Duration.ofSeconds(5));
    return client;
  }

  private static void startDebugging(MockDebugClient client) {
    try {
      client.sendRequestAndWaitForResponse(
          DebugRequest.newBuilder()
              .setSequenceNumber(sequenceIds.getAndIncrement())
              .setStartDebugging(StartDebuggingRequest.getDefaultInstance())
              .build());
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  private static void createClientAndStartDebugging(int debugPort) {
    MockDebugClient client = createClient(debugPort);
    startDebugging(client);
  }

  private void createClientAndSetBreakpoints(int debugPort, String... paths) {
    MockDebugClient client = createClient(debugPort);
    setBreakpoints(client, paths);
    startDebugging(client);
  }

  private void setBreakpoints(MockDebugClient client, String... paths) {
    ImmutableList<Breakpoint> breakpoints =
        stream(paths)
            .map(path -> getWorkspace().getRelative(path).getPathString())
            .map(
                path ->
                    Breakpoint.newBuilder()
                        .setLocation(Location.newBuilder().setPath(path).build())
                        .build())
            .collect(toImmutableList());
    try {
      client.sendRequestAndWaitForResponse(
          DebugRequest.newBuilder()
              .setSequenceNumber(sequenceIds.getAndIncrement())
              .setSetBreakpoints(
                  SetBreakpointsRequest.newBuilder().addAllBreakpoint(breakpoints).build())
              .build());
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }
}
