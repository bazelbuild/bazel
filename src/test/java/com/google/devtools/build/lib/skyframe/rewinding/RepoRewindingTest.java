// Copyright 2026 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe.rewinding;

import static com.google.common.base.Preconditions.checkState;
import static com.google.common.truth.Truth.assertThat;
import static java.util.concurrent.TimeUnit.SECONDS;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.util.concurrent.Uninterruptibles;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.FileStateValue;
import com.google.devtools.build.lib.actions.FileValue;
import com.google.devtools.build.lib.buildtool.util.BuildIntegrationTestCase;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.rules.repository.RepositoryDirectoryValue;
import com.google.devtools.build.lib.runtime.BlazeRuntime;
import com.google.devtools.build.lib.runtime.BlockWaitingModule;
import com.google.devtools.build.lib.testutil.ActionEventRecorder;
import com.google.devtools.build.lib.testutil.SpawnController.ExecResult;
import com.google.devtools.build.lib.testutil.SpawnController.SpawnShim;
import com.google.devtools.build.lib.testutil.SpawnInputUtils;
import com.google.devtools.build.lib.vfs.DelegateFileSystem;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RewindableRepoFileSystem;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyKey;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.BooleanSupplier;
import javax.annotation.Nullable;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Integration tests for rewinding of external repository fetches to recover lost source files.
 *
 * <p>This mirrors the situation in which the contents of a source file in an external repository
 * are served from the remote repo contents cache and lost from the remote cache: the consuming
 * action fails with a lost input that has no generating action and is instead recovered by
 * rewinding the file's metadata nodes together with the repository fetch.
 *
 * <p>These tests are kept separate from {@link RewindingTest}, which disables external
 * repositories to preserve its action graph structure between blaze and bazel.
 */
@RunWith(JUnit4.class)
public final class RepoRewindingTest extends BuildIntegrationTestCase {

  private final ActionEventRecorder actionEventRecorder = new ActionEventRecorder();
  private final RewindingTestsHelper helper = new RewindingTestsHelper(this, actionEventRecorder);

  private RewindableRepoFileSystemForTesting rewindableFs;

  @Override
  protected FileSystem createFileSystemForBuildArtifacts(FileSystem fileSystem) {
    rewindableFs = new RewindableRepoFileSystemForTesting(fileSystem, outputBaseName);
    return rewindableFs;
  }

  @Override
  protected BlazeRuntime.Builder getRuntimeBuilder() throws Exception {
    return super.getRuntimeBuilder()
        .addBlazeModule(new BlockWaitingModule())
        .addBlazeModule(helper.makeControllableActionStrategyModule("standalone"))
        .addBlazeModule(helper.getLostOutputsModule());
  }

  @Override
  protected void setupOptions() throws Exception {
    super.setupOptions();
    addOptions("--spawn_strategy=standalone", "--rewind_lost_inputs", "--jobs=8");
    runtimeWrapper.registerSubscriber(actionEventRecorder);
  }

  /**
   * Writes a repo rule whose repos contain a file {@code src.txt} with the contents of the given
   * workspace file, which is intentionally not watched so that it can be modified mid-build to
   * observe refetches, as well as a file {@code other.txt} with fixed contents.
   *
   * <p>Also writes a {@code dep_repo} rule whose repos contain a file {@code own.txt} with the
   * contents of {@code @repo_a//:src.txt}. Reading another repo's file by label is the analog of
   * (and in production triggers) the materialization of that repo from the remote repo contents
   * cache.
   */
  private void writeRepoRule() throws Exception {
    write("repo/BUILD");
    write(
        "repo/repo.bzl",
        """
        def _my_repo_impl(rctx):
            rctx.file("BUILD", "exports_files(['src.txt', 'other.txt'])")
            content_path = rctx.workspace_root.get_child("repo", rctx.attr.content_file)
            rctx.file("src.txt", rctx.read(content_path, watch = "no"))
            rctx.file("other.txt", "other")

        my_repo = repository_rule(
            implementation = _my_repo_impl,
            attrs = {"content_file": attr.string()},
        )

        def _dep_repo_impl(rctx):
            rctx.file("BUILD", "exports_files(['own.txt'])")
            rctx.file("own.txt", rctx.read(Label("@repo_a//:src.txt")))

        dep_repo = repository_rule(implementation = _dep_repo_impl)
        """);
  }

  private void appendToModuleFile(String... lines) throws Exception {
    FileSystemUtils.appendIsoLatin1(getWorkspace().getRelative("MODULE.bazel"), lines);
  }

  /**
   * Returns a spawn shim that simulates the loss of the given source input from the remote repo
   * contents cache: it changes the contents the repo rule would produce on a refetch, deletes the
   * repo's marker file so that a rewound repository fetch actually re-executes the repo rule (as a
   * cache lookup miss would in production), and fails with a lost input.
   *
   * <p>The shim waits on {@code allSpawnsObservedLostInputs} before failing so that all lost
   * inputs are reported concurrently and the resulting rewinds race with each other.
   */
  private SpawnShim lostRepoFileShim(
      String inputName,
      String contentFile,
      String newContent,
      CountDownLatch allSpawnsObservedLostInputs,
      AtomicReference<Artifact> lostInput)
      throws Exception {
    return (spawn, context) -> {
      Artifact input = (Artifact) SpawnInputUtils.getInputWithName(spawn, inputName);
      lostInput.set(input);
      write("repo/" + contentFile, newContent);
      // The marker file may have already been deleted by another shim losing a file from the same
      // repo.
      var unused = markerFileForRepoOf(input).delete();
      allSpawnsObservedLostInputs.countDown();
      checkState(
          Uninterruptibles.awaitUninterruptibly(allSpawnsObservedLostInputs, 60, SECONDS),
          "timed out waiting for all spawns to observe lost inputs");
      return helper.createLostInputsExecException(context, ImmutableList.of(input));
    };
  }

  /** The chain of Skyframe nodes expected to be rewound for a lost repo source file. */
  private static ImmutableList<SkyKey> expectedRewoundChain(Artifact lostInput) {
    RootedPath rootedPath =
        RootedPath.toRootedPath(lostInput.getRoot().getRoot(), lostInput.getPath());
    return ImmutableList.of(
        RepositoryDirectoryValue.key(
            RepositoryName.createUnvalidated(
                lostInput.getPath().getParentDirectory().getBaseName())),
        FileStateValue.key(rootedPath),
        FileValue.key(rootedPath),
        lostInput);
  }

  private static void assertRewoundInOrder(List<SkyKey> rewoundKeys, List<SkyKey> chain) {
    int lastIndex = -1;
    for (SkyKey key : chain) {
      int index = rewoundKeys.indexOf(key);
      assertThat(index).isGreaterThan(lastIndex);
      lastIndex = index;
    }
  }

  @Test
  public void lostFilesFromMultipleRepos_reposRewoundConcurrently() throws Exception {
    writeRepoRule();
    write("repo/content_a.txt", "old_a");
    write("repo/content_b.txt", "old_b");
    appendToModuleFile(
        "my_repo = use_repo_rule('//repo:repo.bzl', 'my_repo')",
        "my_repo(name = 'repo_a', content_file = 'content_a.txt')",
        "my_repo(name = 'repo_b', content_file = 'content_b.txt')");
    write(
        "test/BUILD",
        """
        genrule(
            name = "consume_a",
            srcs = ["@repo_a//:src.txt"],
            outs = ["out_a.txt"],
            cmd = "cp $< $@",
        )

        genrule(
            name = "consume_b",
            srcs = ["@repo_b//:src.txt"],
            outs = ["out_b.txt"],
            cmd = "cp $< $@",
        )
        """);

    CountDownLatch allSpawnsObservedLostInputs = new CountDownLatch(2);
    AtomicReference<Artifact> lostInputA = new AtomicReference<>();
    AtomicReference<Artifact> lostInputB = new AtomicReference<>();
    helper.addSpawnShim(
        "Executing genrule //test:consume_a",
        lostRepoFileShim(
            "src.txt", "content_a.txt", "new_a", allSpawnsObservedLostInputs, lostInputA));
    helper.addSpawnShim(
        "Executing genrule //test:consume_b",
        lostRepoFileShim(
            "src.txt", "content_b.txt", "new_b", allSpawnsObservedLostInputs, lostInputB));

    List<SkyKey> rewoundKeys = helper.collectOrderedRewoundKeys();
    buildTarget("//test:consume_a", "//test:consume_b");

    helper.verifyAllSpawnShimsConsumed();
    // The outputs contain the new contents, which shows that the repos were refetched before the
    // consuming actions were retried.
    assertContents("new_a", "//test:consume_a");
    assertContents("new_b", "//test:consume_b");
    // Each consuming action ran twice: once failing with a lost input and once after rewinding.
    assertThat(helper.getExecutedSpawnDescriptions())
        .containsExactly(
            "Executing genrule //test:consume_a",
            "Executing genrule //test:consume_b",
            "Executing genrule //test:consume_a",
            "Executing genrule //test:consume_b");
    // Both lost files were marked lost in the file system.
    assertThat(rewindableFs.lostRepoFiles)
        .containsExactly(
            lostInputA.get().getPath().asFragment(), lostInputB.get().getPath().asFragment());
    // Both rewinds dirtied the full chain from the source artifact to the repository fetch, with
    // each chain dirtied in reverse dependency order.
    ImmutableList<SkyKey> chainA = expectedRewoundChain(lostInputA.get());
    ImmutableList<SkyKey> chainB = expectedRewoundChain(lostInputB.get());
    assertThat(rewoundKeys)
        .containsExactlyElementsIn(
            ImmutableList.builder().addAll(chainA).addAll(chainB).build());
    assertRewoundInOrder(rewoundKeys, chainA);
    assertRewoundInOrder(rewoundKeys, chainB);
    actionEventRecorder.assertTotalLostInputCountsFromStats(ImmutableList.of(2));
  }

  @Test
  public void lostFilesFromSameRepo_repoRewoundConcurrently() throws Exception {
    writeRepoRule();
    write("repo/content_a.txt", "old");
    appendToModuleFile(
        "my_repo = use_repo_rule('//repo:repo.bzl', 'my_repo')",
        "my_repo(name = 'repo_a', content_file = 'content_a.txt')");
    write(
        "test/BUILD",
        """
        genrule(
            name = "consume_1",
            srcs = ["@repo_a//:src.txt"],
            outs = ["out_1.txt"],
            cmd = "cp $< $@",
        )

        genrule(
            name = "consume_2",
            srcs = ["@repo_a//:src.txt"],
            outs = ["out_2.txt"],
            cmd = "cp $< $@",
        )
        """);

    CountDownLatch allSpawnsObservedLostInputs = new CountDownLatch(2);
    AtomicReference<Artifact> lostInput1 = new AtomicReference<>();
    AtomicReference<Artifact> lostInput2 = new AtomicReference<>();
    helper.addSpawnShim(
        "Executing genrule //test:consume_1",
        lostRepoFileShim(
            "src.txt", "content_a.txt", "new", allSpawnsObservedLostInputs, lostInput1));
    helper.addSpawnShim(
        "Executing genrule //test:consume_2",
        lostRepoFileShim(
            "src.txt", "content_a.txt", "new", allSpawnsObservedLostInputs, lostInput2));

    List<SkyKey> rewoundKeys = helper.collectOrderedRewoundKeys();
    buildTarget("//test:consume_1", "//test:consume_2");

    helper.verifyAllSpawnShimsConsumed();
    assertContents("new", "//test:consume_1");
    assertContents("new", "//test:consume_2");
    assertThat(helper.getExecutedSpawnDescriptions())
        .containsExactly(
            "Executing genrule //test:consume_1",
            "Executing genrule //test:consume_2",
            "Executing genrule //test:consume_1",
            "Executing genrule //test:consume_2");
    // Both actions lost the same source artifact, whose chain of rewound nodes is shared between
    // the two concurrent rewinds. Depending on timing, nodes rewound by the first reset may have
    // been re-evaluated by the time the second reset rewinds them again, so keys may be reported
    // more than once.
    assertThat(lostInput2.get()).isSameInstanceAs(lostInput1.get());
    ImmutableList<SkyKey> chain = expectedRewoundChain(lostInput1.get());
    assertThat(ImmutableSet.copyOf(rewoundKeys)).containsExactlyElementsIn(chain);
    actionEventRecorder.assertTotalLostInputCountsFromStats(ImmutableList.of(2));
  }

  @Test
  public void unrelatedActionExecutingWhileRepoRefetches() throws Exception {
    writeRepoRule();
    write("repo/content_a.txt", "old");
    appendToModuleFile(
        "my_repo = use_repo_rule('//repo:repo.bzl', 'my_repo')",
        "my_repo(name = 'repo_a', content_file = 'content_a.txt')");
    write(
        "test/BUILD",
        """
        genrule(
            name = "consume_lost",
            srcs = ["@repo_a//:src.txt"],
            outs = ["out_lost.txt"],
            cmd = "cp $< $@",
        )

        genrule(
            name = "reader",
            srcs = ["@repo_a//:other.txt"],
            outs = ["out_reader.txt"],
            cmd = "cp $< $@",
        )

        genrule(
            name = "all",
            srcs = [
                "out_lost.txt",
                "out_reader.txt",
            ],
            outs = ["out_all.txt"],
            cmd = "cat $(SRCS) > $@",
        )
        """);

    CountDownLatch readerStarted = new CountDownLatch(1);
    AtomicReference<Artifact> lostInput = new AtomicReference<>();
    helper.addSpawnShim(
        "Executing genrule //test:consume_lost",
        (spawn, context) -> {
          Artifact input = (Artifact) SpawnInputUtils.getInputWithName(spawn, "src.txt");
          lostInput.set(input);
          // Make sure that the reader's spawn is executing before the lost input is reported, so
          // that it remains in flight while the repo is refetched.
          checkState(
              Uninterruptibles.awaitUninterruptibly(readerStarted, 60, SECONDS),
              "timed out waiting for the reader to start executing");
          write("repo/content_a.txt", "new");
          Path markerFile = markerFileForRepoOf(input);
          checkState(markerFile.delete(), "marker file %s did not exist", markerFile);
          return helper.createLostInputsExecException(context, ImmutableList.of(input));
        });
    helper.addSpawnShim(
        "Executing genrule //test:reader",
        (spawn, context) -> {
          readerStarted.countDown();
          // Hold off the actual execution until the repo has been refetched (which rewrites the
          // marker file deleted by the shim above) to ensure that an action that was already
          // executing when the repo's contents were replaced still completes successfully.
          Artifact input = (Artifact) SpawnInputUtils.getInputWithName(spawn, "other.txt");
          Path markerFile = markerFileForRepoOf(input);
          Path srcFile = input.getPath().getParentDirectory().getChild("src.txt");
          waitFor(
              () -> {
                try {
                  return markerFile.exists()
                      && new String(FileSystemUtils.readContentAsLatin1(srcFile)).equals("new\n");
                } catch (IOException e) {
                  return false;
                }
              },
              "the repo to be refetched");
          return ExecResult.delegate();
        });

    List<SkyKey> rewoundKeys = helper.collectOrderedRewoundKeys();
    buildTarget("//test:all");

    helper.verifyAllSpawnShimsConsumed();
    assertContents("new\nother", "//test:all");
    // The reader executed only once: it was unaffected by the concurrent refetch of its repo.
    assertThat(helper.getExecutedSpawnDescriptions())
        .containsExactly(
            "Executing genrule //test:consume_lost",
            "Executing genrule //test:reader",
            "Executing genrule //test:consume_lost",
            "Executing genrule //test:all");
    assertThat(rewoundKeys).containsExactlyElementsIn(expectedRewoundChain(lostInput.get()));
    actionEventRecorder.assertTotalLostInputCountsFromStats(ImmutableList.of(1));
  }

  @Test
  public void repoMaterializedByOtherRepoRefetched() throws Exception {
    writeRepoRule();
    write("repo/content_a.txt", "old");
    appendToModuleFile(
        "my_repo = use_repo_rule('//repo:repo.bzl', 'my_repo')",
        "my_repo(name = 'repo_a', content_file = 'content_a.txt')",
        "dep_repo = use_repo_rule('//repo:repo.bzl', 'dep_repo')",
        "dep_repo(name = 'repo_b')");
    write(
        "test/BUILD",
        """
        genrule(
            name = "consume_a",
            srcs = ["@repo_a//:src.txt"],
            outs = ["out_a.txt"],
            cmd = "cp $< $@",
        )

        genrule(
            name = "consume_b",
            srcs = ["@repo_b//:own.txt"],
            outs = ["out_b.txt"],
            cmd = "cp $< $@",
        )
        """);

    CountDownLatch lostInputObserved = new CountDownLatch(1);
    AtomicReference<Artifact> lostInput = new AtomicReference<>();
    helper.addSpawnShim(
        "Executing genrule //test:consume_a",
        lostRepoFileShim("src.txt", "content_a.txt", "new", lostInputObserved, lostInput));

    List<SkyKey> rewoundKeys = helper.collectOrderedRewoundKeys();
    // The fetch of repo_b reads (in production: materializes) repo_a, whose file is then lost by
    // the action consuming it and recovered by refetching repo_a.
    buildTarget("//test:consume_a", "//test:consume_b");

    helper.verifyAllSpawnShimsConsumed();
    assertContents("new", "//test:consume_a");
    // repo_b retains the contents it read from repo_a when it was fetched: the refetch of repo_a
    // does not affect repos fetched from it earlier in the build.
    assertContents("old", "//test:consume_b");
    assertThat(rewoundKeys).containsExactlyElementsIn(expectedRewoundChain(lostInput.get()));

    // The next build notices that repo_b's recorded input @repo_a//:src.txt has changed and
    // refetches it.
    helper.clearExecutedSpawnDescriptions();
    buildTarget("//test:consume_b");
    assertContents("new", "//test:consume_b");
    assertThat(helper.getExecutedSpawnDescriptions())
        .containsExactly("Executing genrule //test:consume_b");
  }

  private Path markerFileForRepoOf(Artifact repoFile) {
    String repoName = repoFile.getPath().getParentDirectory().getBaseName();
    return getOutputBase()
        .getRelative("external")
        .getRelative(RepositoryName.createUnvalidated(repoName).getMarkerFileName());
  }

  private static void waitFor(BooleanSupplier condition, String what) throws InterruptedException {
    long deadlineMillis = System.currentTimeMillis() + 60_000;
    while (!condition.getAsBoolean()) {
      checkState(System.currentTimeMillis() < deadlineMillis, "timed out waiting for %s", what);
      Thread.sleep(50);
    }
  }

  /**
   * A {@link DelegateFileSystem} that simulates the {@link RewindableRepoFileSystem} capability of
   * the file system that serves repo contents from the remote repo contents cache.
   */
  private static final class RewindableRepoFileSystemForTesting extends DelegateFileSystem
      implements RewindableRepoFileSystem {
    private final String outputBaseName;
    final List<PathFragment> lostRepoFiles = Collections.synchronizedList(new ArrayList<>());

    RewindableRepoFileSystemForTesting(FileSystem delegateFs, String outputBaseName) {
      super(delegateFs);
      this.outputBaseName = outputBaseName;
    }

    @Override
    @Nullable
    public String markLostRepoFile(PathFragment path) {
      for (int i = 1; i < path.segmentCount() - 1; i++) {
        if (path.getSegment(i).equals("external")
            && path.getSegment(i - 1).equals(outputBaseName)) {
          lostRepoFiles.add(path);
          return path.getSegment(i + 1);
        }
      }
      return null;
    }
  }
}
