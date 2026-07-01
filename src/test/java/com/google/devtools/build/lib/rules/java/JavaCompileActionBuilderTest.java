// Copyright 2022 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.java;

import static com.google.common.collect.MoreCollectors.onlyElement;
import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.rules.java.JavaCompileActionTestHelper.getDirectJars;
import static com.google.devtools.build.lib.rules.java.JavaCompileActionTestHelper.getJavacArguments;
import static org.junit.Assert.assertThrows;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import build.bazel.remote.execution.v2.Digest;
import com.google.common.collect.ImmutableList;
import com.google.common.util.concurrent.Futures;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionInputMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.ArtifactRoot.RootType;
import com.google.devtools.build.lib.actions.CommandAction;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.LostInputsActionExecutionException;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.remote.RemoteActionFileSystem;
import com.google.devtools.build.lib.remote.RemoteActionInputFetcher;
import com.google.devtools.build.lib.remote.common.BulkTransferException;
import com.google.devtools.build.lib.remote.common.CacheNotFoundException;
import com.google.devtools.build.lib.testutil.MoreAsserts;
import com.google.devtools.build.lib.testutil.TestConstants;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import com.google.devtools.build.lib.view.proto.Deps;
import com.google.devtools.build.lib.view.proto.Deps.Dependency.Kind;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Supplier;
import java.util.stream.Stream;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link JavaCompileActionBuilder}. */
@RunWith(JUnit4.class)
public final class JavaCompileActionBuilderTest extends BuildViewTestCase {

  @Test
  public void testClassdirIsInBlazeOut() throws Exception {
    scratch.file(
        "java/com/google/test/BUILD",
        """
        load("@rules_java//java:defs.bzl", "java_binary")
        java_binary(
            name = "a",
            srcs = ["a.java"],
        )
        """);
    JavaCompileAction action =
        (JavaCompileAction) getGeneratingActionForLabel("//java/com/google/test:a.jar");
    List<String> command = new ArrayList<>();
    command.addAll(getJavacArguments(action));
    MoreAsserts.assertContainsSublist(
        command,
        "--output",
        targetConfig
            .getBinFragment(RepositoryName.MAIN)
            .getRelative("java/com/google/test/a.jar")
            .getPathString());
  }

  @Test
  public void progressMessage() throws Exception {
    scratch.file(
        "java/com/google/test/BUILD",
        """
        load("@rules_java//java:defs.bzl", "java_library")
        java_library(
            name = "a",
            srcs = [
                "a.java",
                "b.java",
            ],
        )
        """);
    JavaCompileAction action =
        (JavaCompileAction) getGeneratingActionForLabel("//java/com/google/test:liba.jar");
    assertThat(action.getProgressMessage())
        .isEqualTo("Building java/com/google/test/liba.jar (2 source files)");
  }

  @Test
  public void progressMessageWithSourceJars() throws Exception {
    scratch.file(
        "java/com/google/test/BUILD",
        """
        load("@rules_java//java:defs.bzl", "java_library")
        java_library(
            name = "a",
            srcs = [
                "a.java",
                "archive.srcjar",
                "b.java",
            ],
        )
        """);
    JavaCompileAction action =
        (JavaCompileAction) getGeneratingActionForLabel("//java/com/google/test:liba.jar");
    assertThat(action.getProgressMessage())
        .isEqualTo("Building java/com/google/test/liba.jar (2 source files, 1 source jar)");
  }

  @Test
  public void progressMessageAnnotationProcessors() throws Exception {
    scratch.file(
        "java/com/google/test/BUILD",
        """
        load("@rules_java//java:defs.bzl", "java_library", "java_plugin")
        java_plugin(
            name = "foo",
            srcs = ["Foo.java"],
            processor_class = "Foo",
        )

        java_plugin(
            name = "bar",
            srcs = ["Bar.java"],
            processor_class = "com.google.Bar",
        )

        java_library(
            name = "a",
            srcs = [
                "a.java",
                "archive.srcjar",
                "b.java",
            ],
            plugins = [
                ":foo",
                ":bar",
            ],
        )
        """);
    JavaCompileAction action =
        (JavaCompileAction) getGeneratingActionForLabel("//java/com/google/test:liba.jar");
    assertThat(action.getProgressMessage())
        .isEqualTo(
            "Building java/com/google/test/liba.jar (2 source files, 1 source jar)"
                + " and running annotation processors (Foo, Bar)");
  }

  @Test
  public void testLocale() throws Exception {
    scratch.file(
        "java/com/google/test/BUILD",
        """
        load("@rules_java//java:defs.bzl", "java_library")
        java_library(
            name = "a",
            srcs = ["A.java"],
        )
        """);
    JavaCompileAction action =
        (JavaCompileAction) getGeneratingActionForLabel("//java/com/google/test:liba.jar");
    assertThat(action.getIncompleteEnvironmentForTesting())
        .containsEntry("LC_CTYPE", analysisMock.isThisBazel() ? "C.UTF-8" : "en_US.UTF-8");
  }

  @Test
  public void testClasspathReduction() throws Exception {
    scratch.file(
        "java/com/google/test/BUILD",
        """
        load("@rules_java//java:defs.bzl", "java_library")
        java_library(
            name = "a",
            srcs = ["A.java"],
            deps = [":b"],
        )

        java_library(
            name = "b",
            srcs = ["B.java"],
            deps = [
                ":c",
                ":d",
            ],
        )

        java_library(
            name = "c",
            srcs = ["C.java"],
        )

        java_library(
            name = "d",
            srcs = ["D.java"],
        )
        """);
    Artifact bJdeps =
        getBinArtifact("libb-hjar.jdeps", getConfiguredTarget("//java/com/google/test:b"));
    Artifact cHjar =
        getBinArtifact("libc-hjar.jar", getConfiguredTarget("//java/com/google/test:libc.jar"));
    JavaCompileAction action =
        (JavaCompileAction) getGeneratingActionForLabel("//java/com/google/test:liba.jar");
    JavaCompileActionContext context = new JavaCompileActionContext();
    Deps.Dependency dep =
        Deps.Dependency.newBuilder()
            .setKind(Kind.EXPLICIT)
            .setPath(cHjar.getExecPathString())
            .build();
    context.insertDependencies(bJdeps, Deps.Dependencies.newBuilder().addDependency(dep).build());
    assertThat(
            artifactsToStrings(
                action.getReducedClasspath(new ActionExecutionContextBuilder().build(), context)))
        .containsExactly(
            "bin java/com/google/test/libb-hjar.jar", "bin java/com/google/test/libc-hjar.jar");
  }

  @Test
  public void testTurbineCpuReservation() throws Exception {
    useConfiguration("--java_header_compilation=true", "--experimental_turbine_cpu_reservation=2");
    scratch.file(
        "java/com/google/test/BUILD",
        """
        load("@rules_java//java:defs.bzl", "java_library")
        java_library(
            name = "a",
            srcs = ["A.java"],
            deps = [":b"],
        )
        java_library(
            name = "b",
            srcs = ["b.java"],
        )
        """);
    JavaCompileAction compileAction =
        (JavaCompileAction) getGeneratingActionForLabel("//java/com/google/test:liba.jar");
    Action action = getTurbineAction(compileAction);

    if (TestConstants.PRODUCT_NAME.equals("bazel")) {
      assertThat(paramFileArgsForAction(action)).contains("-XDnoParallel");
    } else {
      assertThat(paramFileArgsForAction(action)).doesNotContain("-XDnoParallel");
    }
    assertThat(action.getExecutionInfo().keySet().stream().filter(k -> k.startsWith("cpu:")))
        .containsExactly("cpu:2");
  }

  @Test
  public void testNoTurbineCpuReservation() throws Exception {
    useConfiguration("--java_header_compilation=true");
    scratch.file(
        "java/com/google/test/BUILD",
        """
        load("@rules_java//java:defs.bzl", "java_library")
        java_library(
            name = "a",
            srcs = ["A.java"],
            deps = [":b"],
        )
        java_library(
            name = "b",
            srcs = ["b.java"],
        )
        """);
    JavaCompileAction compileAction =
        (JavaCompileAction) getGeneratingActionForLabel("//java/com/google/test:liba.jar");
    Action action = getTurbineAction(compileAction);

    if (TestConstants.PRODUCT_NAME.equals("bazel")) {
      assertThat(paramFileArgsForAction(action)).contains("-XDnoParallel");
    } else {
      assertThat(paramFileArgsForAction(action)).doesNotContain("-XDnoParallel");
    }
    assertThat(action.getExecutionInfo().keySet().stream().filter(k -> k.startsWith("cpu:")))
        .isEmpty();
  }

  private CommandAction getTurbineAction(JavaCompileAction compileAction) throws Exception {
    return (CommandAction)
        getGeneratingAction(getBinArtifacts(compileAction).collect(onlyElement()));
  }

  private static Stream<Artifact> getBinArtifacts(JavaCompileAction compileAction)
      throws Exception {
    return getInputs(compileAction, getDirectJars(compileAction)).stream()
        .filter(a -> a.getFilename().endsWith("-hjar.jar"));
  }

  /**
   * Regression test for the {@link java.util.ConcurrentModificationException} crash in
   * https://github.com/bazelbuild/bazel/pull/30001.
   *
   * <p>{@link JavaCompileActionContext#addDependencies} reads the dependencies' {@code .jdeps}
   * files on an internal thread pool. Each read goes through {@link
   * RemoteActionFileSystem#getInputStream}, which records a lost input when a remote input has been
   * evicted from the cache (a {@link BulkTransferException}). If one read fails, {@code
   * addDependencies} must still wait for the other reads to finish before propagating the failure.
   * Otherwise an abandoned read keeps running and records a lost input into the per-action {@link
   * RemoteActionFileSystem} concurrently with the {@link RemoteActionFileSystem#checkForLostInputs}
   * call that the action machinery makes right after {@code addDependencies} returns -- a data race
   * that threw {@code ConcurrentModificationException} while {@code checkForLostInputs} iterated the
   * lost-input collection. This test makes the interleaving deterministic by blocking one read on a
   * latch.
   */
  @Test
  public void addDependencies_doesNotReturnWhileAJdepsReadIsStillInFlight() throws Exception {
    FileSystem fs = new InMemoryFileSystem(DigestHashFunction.SHA256);
    Path execRoot = fs.getPath("/exec");
    ArtifactRoot outputRoot = ArtifactRoot.asDerivedRoot(execRoot, RootType.OUTPUT, "out");
    execRoot.createDirectoryAndParents();
    outputRoot.getRoot().asPath().createDirectoryAndParents();

    // Two .jdeps inputs that are both remote and both evicted from the cache, i.e. both "lost".
    RemoteActionInputFetcher inputFetcher = mock(RemoteActionInputFetcher.class);
    ActionInputMap inputs = new ActionInputMap(2);
    Artifact jdepsA = remoteJdepsArtifact(outputRoot, "liba.jdeps", inputs);
    Artifact jdepsB = remoteJdepsArtifact(outputRoot, "libb.jdeps", inputs);

    RemoteActionFileSystem actionFs =
        new RemoteActionFileSystem(fs, execRoot.asFragment(), "out", inputs, inputFetcher);
    Action action = mock(Action.class);
    actionFs.updateContext(action);

    // Reading jdepsA fails fast with a lost-input error, while reading jdepsB blocks until we
    // release it and only then fails. This keeps the jdepsB read in flight after jdepsA has failed.
    CountDownLatch bStarted = new CountDownLatch(1);
    CountDownLatch bRelease = new CountDownLatch(1);
    when(inputFetcher.prefetchFiles(any(), any(), any(), any(), any(), any()))
        .thenAnswer(
            invocation -> {
              Object requested = invocation.<Supplier<?>>getArgument(2).get();
              if (ImmutableList.of(jdepsA).equals(requested)) {
                return Futures.immediateFailedFuture(lostInputException(jdepsA, "a".repeat(64)));
              }
              bStarted.countDown();
              bRelease.await();
              return Futures.immediateFailedFuture(lostInputException(jdepsB, "b".repeat(64)));
            });

    ActionExecutionContext actionExecutionContext = mock(ActionExecutionContext.class);
    when(actionExecutionContext.getInputPath(jdepsA))
        .thenReturn(actionFs.getPath(jdepsA.getPath().asFragment()));
    when(actionExecutionContext.getInputPath(jdepsB))
        .thenReturn(actionFs.getPath(jdepsB.getPath().asFragment()));

    JavaCompileActionContext context = new JavaCompileActionContext();
    AtomicReference<Throwable> thrown = new AtomicReference<>();
    Thread worker =
        new Thread(
            () -> {
              try {
                context.addDependencies(
                    ImmutableList.of(jdepsA, jdepsB), actionExecutionContext, new HashSet<>());
                thrown.set(new AssertionError("addDependencies should have thrown"));
              } catch (Throwable t) {
                thrown.set(t);
              }
            });
    worker.start();

    // jdepsA has failed, but addDependencies must keep waiting for the in-flight jdepsB read.
    assertThat(bStarted.await(30, TimeUnit.SECONDS)).isTrue();
    worker.join(TimeUnit.SECONDS.toMillis(1));
    assertThat(worker.isAlive()).isTrue();

    // Let jdepsB finish. addDependencies now propagates the failure, with both lost inputs already
    // recorded -- there is no longer a reader thread mutating the file system after this point.
    bRelease.countDown();
    worker.join(TimeUnit.SECONDS.toMillis(30));
    assertThat(worker.isAlive()).isFalse();
    assertThat(thrown.get()).isInstanceOf(IOException.class);

    LostInputsActionExecutionException lostInputs =
        assertThrows(
            LostInputsActionExecutionException.class, () -> actionFs.checkForLostInputs(action));
    assertThat(lostInputs.getLostInputs().values()).containsExactly(jdepsA, jdepsB);
  }

  /** Adds a remote artifact to {@code inputs} and returns it. */
  private static Artifact remoteJdepsArtifact(
      ArtifactRoot outputRoot, String name, ActionInputMap inputs) {
    Artifact a = ActionsTestUtil.createArtifact(outputRoot, name);
    inputs.put(
        a,
        FileArtifactValue.createForRemoteFileWithMaterializationData(
            new byte[32],
            /* size= */ 1,
            /* locationIndex= */ 1,
            /* expirationTime= */ null,
            /* inMemoryOutput= */ false));
    return a;
  }

  /**
   * Builds a {@link BulkTransferException} representing the loss of {@code artifact} due to a cache
   * miss, matching what {@link RemoteActionFileSystem#getInputStream} turns into a recorded lost
   * input.
   */
  private static BulkTransferException lostInputException(Artifact artifact, String digestHash) {
    Digest digest = Digest.newBuilder().setHash(digestHash).setSizeBytes(1).build();
    CacheNotFoundException cacheMiss = new CacheNotFoundException(digest, artifact.getExecPath());
    cacheMiss.setFilename(artifact.getExecPathString());
    return new BulkTransferException(cacheMiss);
  }
}
