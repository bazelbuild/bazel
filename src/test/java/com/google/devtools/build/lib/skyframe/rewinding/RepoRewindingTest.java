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

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.base.Preconditions.checkState;
import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.common.collect.Iterables.getLast;
import static com.google.common.truth.Truth.assertThat;
import static java.nio.charset.StandardCharsets.UTF_8;
import static java.util.concurrent.TimeUnit.SECONDS;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.util.concurrent.Uninterruptibles;
import com.google.devtools.build.lib.actions.ActionLookupData;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.FileStateValue;
import com.google.devtools.build.lib.actions.FileValue;
import com.google.devtools.build.lib.buildtool.util.BuildIntegrationTestCase;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.rules.repository.RepositoryDirectoryValue;
import com.google.devtools.build.lib.runtime.BlazeRuntime;
import com.google.devtools.build.lib.runtime.BlockWaitingModule;
import com.google.devtools.build.lib.skyframe.SkyFunctions;
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
import com.google.devtools.build.lib.vfs.RewindingSynchronizer;
import com.google.devtools.build.lib.vfs.RewindingSynchronizer.TransferableWriteLock;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.testing.junit.testparameterinjector.TestParameter;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
import javax.annotation.Nullable;
import org.junit.Test;
import org.junit.runner.RunWith;

/**
 * Integration tests for rewinding of external repository fetches to recover lost source files.
 *
 * <p>This mirrors the situation in which the contents of a source file in an external repository
 * are served from the remote repo contents cache and lost from the remote cache: the consuming
 * action fails with a lost input that has no generating action and is instead recovered by
 * rewinding the file's metadata nodes together with the repository fetch.
 *
 * <p>These tests are kept separate from {@link RewindingTest}, which disables external repositories
 * to preserve its action graph structure between blaze and bazel.
 *
 * <p>Uses a {@link TestParameter} to run tests with and without {@code
 * --experimental_precise_rewinding}, which only takes a different path through {@link
 * ActionRewindStrategy} for lost inputs owned by an aggregation artifact such as a runfiles tree.
 */
@RunWith(TestParameterInjector.class)
public final class RepoRewindingTest extends BuildIntegrationTestCase {

  @TestParameter private boolean precise;

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
    addOptions(
        "--spawn_strategy=standalone",
        "--rewind_lost_inputs",
        "--jobs=8",
        "--experimental_precise_rewinding=" + precise);
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
   * Writes a repo rule whose repos contain the two files {@code src_1.txt} and {@code src_2.txt},
   * each with the contents of its own workspace file, which are intentionally not watched so that
   * they can be modified mid-build to observe refetches.
   */
  private void writeTwoFileRepoRule() throws Exception {
    write("repo/BUILD");
    write(
        "repo/two_file_repo.bzl",
        """
        def _read_workspace_file(rctx, name):
            return rctx.read(rctx.workspace_root.get_child("repo", name), watch = "no")

        def _two_file_repo_impl(rctx):
            rctx.file("BUILD", "exports_files(['src_1.txt', 'src_2.txt'])")
            rctx.file("src_1.txt", _read_workspace_file(rctx, rctx.attr.content_file_1))
            rctx.file("src_2.txt", _read_workspace_file(rctx, rctx.attr.content_file_2))

        two_file_repo = repository_rule(
            implementation = _two_file_repo_impl,
            attrs = {
                "content_file_1": attr.string(),
                "content_file_2": attr.string(),
            },
        )
        """);
  }

  /** Declares a {@code two_file_repo} named {@code repo_a} backed by the given workspace files. */
  private void useTwoFileRepo(String contentFile1, String contentFile2) throws Exception {
    appendToModuleFile(
        "two_file_repo = use_repo_rule('//repo:two_file_repo.bzl', 'two_file_repo')",
        "two_file_repo(name = 'repo_a', content_file_1 = '%s', content_file_2 = '%s')"
            .formatted(contentFile1, contentFile2));
  }

  /**
   * Returns a spawn shim that simulates the loss of the given source input from the remote repo
   * contents cache: it changes the contents the repo rule would produce on a refetch, deletes the
   * repo's marker file so that a rewound repository fetch actually re-executes the repo rule (as a
   * cache lookup miss would in production), and fails with a lost input.
   *
   * <p>The shim waits on {@code allSpawnsObservedLostInputs} before failing so that all lost inputs
   * are reported concurrently and the resulting rewinds race with each other.
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

    rewindableFs.setExternalDir(getOutputBase().getRelative("external").asFragment());
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
    // Both repos were marked as having lost files in the file system.
    assertThat(rewindableFs.lostRepos)
        .containsExactly(repoOf(lostInputA.get()), repoOf(lostInputB.get()));
    // Both rewinds dirtied the full chain from the source artifact to the repository fetch, with
    // each chain dirtied in reverse dependency order.
    ImmutableList<SkyKey> chainA = expectedRewoundChain(lostInputA.get());
    ImmutableList<SkyKey> chainB = expectedRewoundChain(lostInputB.get());
    assertThat(rewoundKeys)
        .containsExactlyElementsIn(ImmutableList.builder().addAll(chainA).addAll(chainB).build());
    assertRewoundInOrder(rewoundKeys, chainA);
    assertRewoundInOrder(rewoundKeys, chainB);
    actionEventRecorder.assertTotalLostInputCountsFromStats(ImmutableList.of(2));
  }

  @Test
  public void unrelatedActionReadingWhileRepoRefetches_sharedLock() throws Exception {
    runUnrelatedActionReadingWhileRepoRefetches(/* perRepoLocksFromStart= */ false);
  }

  @Test
  public void unrelatedActionReadingWhileRepoRefetches_perRepoLock() throws Exception {
    runUnrelatedActionReadingWhileRepoRefetches(/* perRepoLocksFromStart= */ true);
  }

  private void runUnrelatedActionReadingWhileRepoRefetches(boolean perRepoLocksFromStart)
      throws Exception {
    if (perRepoLocksFromStart) {
      // Makes the initial repo fetches take real write locks, which switches all actions to locks
      // keyed by the repos of their inputs. The refetch below then has to wait for the lock of its
      // own repo instead of the shared one, which only works if both name the same repo.
      rewindableFs.getRewindingSynchronizer().markReplacementsPossible();
    }
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
    CountDownLatch repoRefetchRequested = new CountDownLatch(1);
    AtomicBoolean readerFinished = new AtomicBoolean();
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
          // Observe the refetch attempting to acquire its write lock. The reader already holds a
          // read lock, so the refetch must wait until its action has finished.
          rewindableFs.signalAroundNextRepoWriteLock(
              repoOf(input),
              repoRefetchRequested,
              () ->
                  checkState(
                      readerFinished.get(),
                      "the repo write lock was granted while the reader was still executing"));
          return helper.createLostInputsExecException(context, ImmutableList.of(input));
        });
    helper.addSpawnShim(
        "Executing genrule //test:reader",
        (spawn, context) -> {
          readerStarted.countDown();
          Artifact input = (Artifact) SpawnInputUtils.getInputWithName(spawn, "other.txt");
          checkState(
              Uninterruptibles.awaitUninterruptibly(repoRefetchRequested, 60, SECONDS),
              "timed out waiting for the repo refetch to request its write lock");
          // The refetch is waiting on this action's read lock, so its in-place replacement cannot
          // delete the input before the action has finished reading it.
          assertThat(new String(FileSystemUtils.readContentAsLatin1(input.getPath())))
              .isEqualTo("other");
          readerFinished.set(true);
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

    rewindableFs.setExternalDir(getOutputBase().getRelative("external").asFragment());
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
  public void multipleLostFilesInOneAction_repoRewoundOnce() throws Exception {
    writeTwoFileRepoRule();
    write("repo/content_1.txt", "old_1");
    write("repo/content_2.txt", "old_2");
    useTwoFileRepo("content_1.txt", "content_2.txt");
    write(
        "test/BUILD",
        """
        genrule(
            name = "consume",
            srcs = [
                "@repo_a//:src_1.txt",
                "@repo_a//:src_2.txt",
            ],
            outs = ["out.txt"],
            cmd = "cat $(SRCS) > $@",
        )
        """);

    AtomicReference<Artifact> lostInput1 = new AtomicReference<>();
    AtomicReference<Artifact> lostInput2 = new AtomicReference<>();
    helper.addSpawnShim(
        "Executing genrule //test:consume",
        (spawn, context) -> {
          Artifact input1 = (Artifact) SpawnInputUtils.getInputWithName(spawn, "src_1.txt");
          Artifact input2 = (Artifact) SpawnInputUtils.getInputWithName(spawn, "src_2.txt");
          lostInput1.set(input1);
          lostInput2.set(input2);
          write("repo/content_1.txt", "new_1");
          write("repo/content_2.txt", "new_2");
          Path markerFile = markerFileForRepoOf(input1);
          checkState(markerFile.delete(), "marker file %s did not exist", markerFile);
          return helper.createLostInputsExecException(context, ImmutableList.of(input1, input2));
        });

    rewindableFs.setExternalDir(getOutputBase().getRelative("external").asFragment());
    List<SkyKey> rewoundKeys = helper.collectOrderedRewoundKeys();
    buildTarget("//test:consume");

    helper.verifyAllSpawnShimsConsumed();
    // Both lost files carry their new contents, so the single refetch of the repo containing them
    // recovered both.
    assertContents("new_1\nnew_2", "//test:consume");
    assertThat(helper.getExecutedSpawnDescriptions())
        .containsExactly("Executing genrule //test:consume", "Executing genrule //test:consume");
    // The repo is marked as having lost files once per lost file, which is idempotent since the
    // file system tracks the repos with lost files in a set.
    assertThat(ImmutableSet.copyOf(rewindableFs.lostRepos))
        .containsExactly(repoOf(lostInput1.get()));
    // Each lost file is rewound along its own chain of metadata nodes, but both chains end in the
    // fetch of the repo containing them, which is thus rewound exactly once.
    ImmutableList<SkyKey> chain1 = expectedRewoundChain(lostInput1.get());
    ImmutableList<SkyKey> chain2 = expectedRewoundChain(lostInput2.get());
    assertThat(rewoundKeys)
        .containsExactlyElementsIn(
            ImmutableSet.<SkyKey>builder().addAll(chain1).addAll(chain2).build());
    assertRewoundInOrder(rewoundKeys, chain1);
    assertRewoundInOrder(rewoundKeys, chain2);
    actionEventRecorder.assertTotalLostInputCountsFromStats(ImmutableList.of(2));
  }

  @Test
  public void differentLostFilesFromSameRepo_repoRewoundConcurrently() throws Exception {
    writeTwoFileRepoRule();
    write("repo/content_1.txt", "old_1");
    write("repo/content_2.txt", "old_2");
    useTwoFileRepo("content_1.txt", "content_2.txt");
    write(
        "test/BUILD",
        """
        genrule(
            name = "consume_1",
            srcs = ["@repo_a//:src_1.txt"],
            outs = ["out_1.txt"],
            cmd = "cp $< $@",
        )

        genrule(
            name = "consume_2",
            srcs = ["@repo_a//:src_2.txt"],
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
            "src_1.txt", "content_1.txt", "new_1", allSpawnsObservedLostInputs, lostInput1));
    helper.addSpawnShim(
        "Executing genrule //test:consume_2",
        lostRepoFileShim(
            "src_2.txt", "content_2.txt", "new_2", allSpawnsObservedLostInputs, lostInput2));

    rewindableFs.setExternalDir(getOutputBase().getRelative("external").asFragment());
    List<SkyKey> rewoundKeys = helper.collectOrderedRewoundKeys();
    buildTarget("//test:consume_1", "//test:consume_2");

    helper.verifyAllSpawnShimsConsumed();
    assertContents("new_1", "//test:consume_1");
    assertContents("new_2", "//test:consume_2");
    assertThat(helper.getExecutedSpawnDescriptions())
        .containsExactly(
            "Executing genrule //test:consume_1",
            "Executing genrule //test:consume_2",
            "Executing genrule //test:consume_1",
            "Executing genrule //test:consume_2");
    assertThat(ImmutableSet.copyOf(rewindableFs.lostRepos))
        .containsExactly(repoOf(lostInput1.get()));
    // The two actions lost different files of the same repo, so their chains of rewound nodes are
    // disjoint except for the repo fetch they share. Since the two rewinds are independent, that
    // shared node may be rewound by each of them and thus be reported more than once.
    ImmutableList<SkyKey> chain1 = expectedRewoundChain(lostInput1.get());
    ImmutableList<SkyKey> chain2 = expectedRewoundChain(lostInput2.get());
    assertThat(ImmutableSet.copyOf(rewoundKeys))
        .containsExactlyElementsIn(
            ImmutableSet.<SkyKey>builder().addAll(chain1).addAll(chain2).build());
    actionEventRecorder.assertTotalLostInputCountsFromStats(ImmutableList.of(2));
  }

  @Test
  public void lostFileInRunfiles_repoRewound() throws Exception {
    writeRepoRule();
    write("repo/content_a.txt", "old");
    appendToModuleFile(
        "my_repo = use_repo_rule('//repo:repo.bzl', 'my_repo')",
        "my_repo(name = 'repo_a', content_file = 'content_a.txt')");
    helper.mockFooBinary("test/foo_binary.bzl");
    write(
        "test/BUILD",
        """
        load(":foo_binary.bzl", "foo_binary")

        foo_binary(
            name = "tool",
            srcs = ["tool.sh"],
            data = ["@repo_a//:src.txt"],
        )

        genrule(
            name = "tool_user",
            srcs = [],
            outs = ["out.txt"],
            cmd = "touch $@",
            tools = ["tool"],
        )
        """);
    write("test/tool.sh", "#!/bin/bash").setExecutable(true);

    AtomicReference<Artifact> lostInput = new AtomicReference<>();
    helper.addSpawnShim(
        "Executing genrule //test:tool_user",
        (spawn, context) -> {
          // The lost file is reached through the runfiles tree of the tool, so it is a lost input
          // owned by an aggregation artifact rather than a direct dep of the failed action.
          Artifact input = SpawnInputUtils.getRunfilesArtifactWithName(spawn, context, "src.txt");
          lostInput.set(input);
          write("repo/content_a.txt", "new");
          Path markerFile = markerFileForRepoOf(input);
          checkState(markerFile.delete(), "marker file %s did not exist", markerFile);
          return helper.createLostInputsExecException(context, ImmutableList.of(input));
        });

    List<SkyKey> rewoundKeys = helper.collectOrderedRewoundKeys();
    buildTarget("//test:tool_user");

    helper.verifyAllSpawnShimsConsumed();
    // The refetched repo carries the new contents, so the rewound fetch recovered the lost file.
    assertThat(FileSystemUtils.readContent(lostInput.get().getPath(), UTF_8)).isEqualTo("new\n");
    assertThat(helper.getExecutedSpawnDescriptions())
        .contains("Executing genrule //test:tool_user");
    assertThat(rewindableFs.lostRepos).containsExactly(repoOf(lostInput.get()));
    // The chain from the source artifact to the repo fetch is rewound even though the failed action
    // depends on the file only through the runfiles tree: rewinding the runfiles tree alone would
    // recompute it from the very file that is gone.
    ImmutableList<SkyKey> chain = expectedRewoundChain(lostInput.get());
    assertThat(rewoundKeys).containsAtLeastElementsIn(chain).inOrder();
    // The runfiles tree caches the metadata of the files it contains, so the action creating it is
    // rewound after the source artifact it owns.
    ImmutableList<ActionLookupData> rewoundActions =
        rewoundKeys.stream()
            .filter(ActionLookupData.class::isInstance)
            .map(ActionLookupData.class::cast)
            .collect(toImmutableList());
    assertThat(getLast(rewoundKeys)).isEqualTo(getLast(rewoundActions));
    assertThat(getLast(rewoundActions).getLabel().getCanonicalForm()).isEqualTo("//test:tool");
    if (precise) {
      // Precise rewinding restricts the rewind to the path through the runfiles tree that leads to
      // the lost file, leaving the tool's other actions alone.
      assertThat(rewoundActions).hasSize(1);
      assertThat(rewoundKeys)
          .containsExactlyElementsIn(
              ImmutableList.builder().addAll(chain).addAll(rewoundActions).build());
    } else {
      // Without precise rewinding, all generated inputs of the runfiles tree are rewound, including
      // the tool's actions that have nothing to do with the lost file.
      assertThat(rewoundActions.size()).isGreaterThan(1);
    }
    actionEventRecorder.assertTotalLostInputCountsFromStats(ImmutableList.of(1));
  }

  @Test
  public void lostFileInRunfilesAndDirectDep_repoRewound() throws Exception {
    writeRepoRule();
    write("repo/content_a.txt", "old");
    appendToModuleFile(
        "my_repo = use_repo_rule('//repo:repo.bzl', 'my_repo')",
        "my_repo(name = 'repo_a', content_file = 'content_a.txt')");
    helper.mockFooBinary("test/foo_binary.bzl");
    write(
        "test/BUILD",
        """
        load(":foo_binary.bzl", "foo_binary")

        foo_binary(
            name = "tool",
            srcs = ["tool.sh"],
            data = ["@repo_a//:src.txt"],
        )

        genrule(
            name = "tool_user",
            srcs = ["@repo_a//:src.txt"],
            outs = ["out.txt"],
            cmd = "cp $< $@",
            tools = ["tool"],
        )
        """);
    write("test/tool.sh", "#!/bin/bash").setExecutable(true);

    AtomicReference<Artifact> lostInput = new AtomicReference<>();
    helper.addSpawnShim(
        "Executing genrule //test:tool_user",
        (spawn, context) -> {
          // The lost file is both a direct dep of the failed action and contained in the runfiles
          // tree of its tool, so rewinding must invalidate both paths to it.
          Artifact input = (Artifact) SpawnInputUtils.getInputWithName(spawn, "src.txt");
          checkState(
              input.equals(SpawnInputUtils.getRunfilesArtifactWithName(spawn, context, "src.txt")),
              "%s is not the artifact contained in the runfiles tree",
              input);
          lostInput.set(input);
          write("repo/content_a.txt", "new");
          Path markerFile = markerFileForRepoOf(input);
          checkState(markerFile.delete(), "marker file %s did not exist", markerFile);
          return helper.createLostInputsExecException(context, ImmutableList.of(input));
        });

    List<SkyKey> rewoundKeys = helper.collectOrderedRewoundKeys();
    buildTarget("//test:tool_user");

    helper.verifyAllSpawnShimsConsumed();
    assertContents("new", "//test:tool_user");
    assertThat(rewindableFs.lostRepos).containsExactly(repoOf(lostInput.get()));
    // The lost file is rewound along its chain to the repo fetch exactly once, no matter how many
    // paths lead to it.
    ImmutableList<SkyKey> chain = expectedRewoundChain(lostInput.get());
    assertThat(rewoundKeys).containsAtLeastElementsIn(chain).inOrder();
    assertThat(rewoundKeys).containsNoDuplicates();
    // The runfiles tree is recomputed after the source artifact it owns, so it cannot propagate the
    // metadata of the lost file to the retried action.
    ImmutableList<ActionLookupData> rewoundActions =
        rewoundKeys.stream()
            .filter(ActionLookupData.class::isInstance)
            .map(ActionLookupData.class::cast)
            .collect(toImmutableList());
    assertThat(getLast(rewoundKeys)).isEqualTo(getLast(rewoundActions));
    assertThat(getLast(rewoundActions).getLabel().getCanonicalForm()).isEqualTo("//test:tool");
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

    rewindableFs.setExternalDir(getOutputBase().getRelative("external").asFragment());
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

  @Test
  public void lostFilesInModuleExtension_reposRewound() throws Exception {
    writeRepoRule();
    write("repo/content_a.txt", "old_a");
    write("repo/content_b.txt", "old_b");
    write(
        "repo/ext.bzl",
        """
        def _generated_repo_impl(rctx):
            rctx.file("BUILD", "exports_files(['combined.txt'])")
            rctx.file("combined.txt", rctx.attr.content)

        generated_repo = repository_rule(
            implementation = _generated_repo_impl,
            attrs = {"content": attr.string()},
        )

        def _ext_impl(module_ctx):
            a = module_ctx.read(Label("@repo_a//:src.txt"), watch = "no")
            b = module_ctx.read(Label("@repo_b//:src.txt"), watch = "no")
            generated_repo(name = "generated", content = a + b)

        ext = module_extension(implementation = _ext_impl)
        """);
    appendToModuleFile(
        "my_repo = use_repo_rule('//repo:repo.bzl', 'my_repo')",
        "my_repo(name = 'repo_a', content_file = 'content_a.txt')",
        "my_repo(name = 'repo_b', content_file = 'content_b.txt')",
        "ext = use_extension('//repo:ext.bzl', 'ext')",
        "use_repo(ext, 'generated')");
    write(
        "test/BUILD",
        """
        genrule(
            name = "consume",
            srcs = ["@generated//:combined.txt"],
            outs = ["out.txt"],
            cmd = "cp $< $@",
        )
        """);

    // Warm up the repos so that they are fetched and their marker files exist.
    buildTarget("//test:consume");
    assertContents("old_a\nold_b", "//test:consume");

    // Both repos lose a file, but the extension only ever observes one at a time: it aborts at the
    // first failed read. Change what a refetch would produce and drop the marker files so that a
    // rewound fetch actually re-runs the repo rule, as a cache miss would in production.
    write("repo/content_a.txt", "new_a");
    write("repo/content_b.txt", "new_b");
    RepositoryName repoA = canonicalRepoName("repo_a");
    RepositoryName repoB = canonicalRepoName("repo_b");
    for (RepositoryName repo : ImmutableList.of(repoA, repoB)) {
      rewindableFs.loseOnNextMaterialization(repo.getName());
    }
    // Force the extension to be evaluated again.
    getSkyframeExecutor()
        .getEvaluator()
        .delete(k -> k.functionName().equals(SkyFunctions.SINGLE_EXTENSION_EVAL));

    rewindableFs.setExternalDir(getOutputBase().getRelative("external").asFragment());
    List<SkyKey> rewoundKeys = helper.collectOrderedRewoundKeys();
    buildTarget("//test:consume");

    // Both repos were refetched within this build.
    assertContents("new_a\nnew_b", "//test:consume");
    assertThat(rewoundKeys)
        .containsAtLeast(RepositoryDirectoryValue.key(repoA), RepositoryDirectoryValue.key(repoB));
  }

  @Test
  public void lostFileReadByRepoRule_repoRewound() throws Exception {
    writeRepoRule();
    write("repo/content_a.txt", "old_a");
    appendToModuleFile(
        "my_repo = use_repo_rule('//repo:repo.bzl', 'my_repo')",
        "my_repo(name = 'repo_a', content_file = 'content_a.txt')",
        "dep_repo = use_repo_rule('//repo:repo.bzl', 'dep_repo')",
        "dep_repo(name = 'dep')");
    write(
        "test/BUILD",
        """
        genrule(
            name = "consume",
            srcs = ["@dep//:own.txt"],
            outs = ["out.txt"],
            cmd = "cp $< $@",
        )
        """);

    buildTarget("//test:consume");
    assertContents("old_a", "//test:consume");

    // @dep's repo rule reads @repo_a//:src.txt, which is the analog of materializing a cached repo
    // in production. Lose that file and make a refetch of @repo_a produce new contents.
    write("repo/content_a.txt", "new_a");
    RepositoryName repoA = canonicalRepoName("repo_a");
    RepositoryName dep = canonicalRepoName("dep");
    rewindableFs.loseOnNextMaterialization(repoA.getName());
    // Only @dep's repo rule is made to run again. @repo_a's marker file stays in place, so it is
    // only refetched because the rewind triggered by the lost file invalidates it.
    Path depMarkerFile =
        getOutputBase().getRelative("external").getRelative(dep.getMarkerFileName());
    checkState(depMarkerFile.delete(), "marker file %s did not exist", depMarkerFile);
    getSkyframeExecutor()
        .getEvaluator()
        .delete(k -> k.functionName().equals(SkyFunctions.REPOSITORY_DIRECTORY));

    rewindableFs.setExternalDir(getOutputBase().getRelative("external").asFragment());
    List<SkyKey> rewoundKeys = helper.collectOrderedRewoundKeys();
    buildTarget("//test:consume");

    // @repo_a was refetched within this build and @dep picked up its new contents.
    assertThat(rewindableFs.lostRepoFiles).isNotEmpty();
    assertThat(rewoundKeys).contains(RepositoryDirectoryValue.key(repoA));
    assertContents("new_a", "//test:consume");
  }

  @Test
  public void lostBuildFileDuringLoading_repoRewound() throws Exception {
    writeRepoRule();
    write("repo/content_a.txt", "old_a");
    appendToModuleFile(
        "my_repo = use_repo_rule('//repo:repo.bzl', 'my_repo')",
        "my_repo(name = 'repo_a', content_file = 'content_a.txt')");
    write(
        "test/BUILD",
        """
        genrule(
            name = "consume",
            srcs = ["@repo_a//:src.txt"],
            outs = ["out.txt"],
            cmd = "cp $< $@",
        )
        """);

    buildTarget("//test:consume");
    assertContents("old_a", "//test:consume");

    // @repo_a's BUILD file is lost, which surfaces while loading the package rather than in a repo
    // rule or an action.
    write("repo/content_a.txt", "new_a");
    RepositoryName repoA = canonicalRepoName("repo_a");
    rewindableFs.setExternalDir(getOutputBase().getRelative("external").asFragment());
    rewindableFs.loseOnNextRead(repoA.getName() + "/BUILD");
    getSkyframeExecutor()
        .getEvaluator()
        .delete(
            k ->
                k.functionName().equals(SkyFunctions.PACKAGE)
                    || k.functionName().equals(SkyFunctions.PACKAGE_LOOKUP));

    List<SkyKey> rewoundKeys = helper.collectOrderedRewoundKeys();
    buildTarget("//test:consume");

    assertThat(rewindableFs.lostRepoFiles).isNotEmpty();
    assertThat(rewoundKeys).contains(RepositoryDirectoryValue.key(repoA));
    assertContents("new_a", "//test:consume");
  }

  /** Returns the canonical name of the repo with the given apparent name in the main repo. */
  private RepositoryName canonicalRepoName(String apparentName) throws IOException {
    Path externalDir = getOutputBase().getRelative("external");
    for (Path child : externalDir.getDirectoryEntries()) {
      String name = child.getBaseName();
      if (child.isDirectory() && (name.equals(apparentName) || name.endsWith("+" + apparentName))) {
        return RepositoryName.createUnvalidated(name);
      }
    }
    throw new IllegalStateException(
        "no repo directory for %s in %s"
            .formatted(apparentName, externalDir.getDirectoryEntries()));
  }

  private Path markerFileForRepoOf(Artifact repoFile) {
    return getOutputBase()
        .getRelative("external")
        .getRelative(repoOf(repoFile).getMarkerFileName());
  }

  private static RepositoryName repoOf(Artifact repoFile) {
    return repoFile.getRoot().getExternalRepositoryName();
  }

  /**
   * A {@link DelegateFileSystem} that simulates the {@link RewindableRepoFileSystem} capability of
   * the file system that serves repo contents from the remote repo contents cache.
   */
  private static final class RewindableRepoFileSystemForTesting extends DelegateFileSystem
      implements RewindableRepoFileSystem {
    private static final String LOST_BLOB_DIGEST = "0".repeat(64) + "/1";

    private final String outputBaseName;
    final List<RepositoryName> lostRepos = Collections.synchronizedList(new ArrayList<>());
    final List<PathFragment> lostRepoFiles = Collections.synchronizedList(new ArrayList<>());
    // Repos whose next materialization fails as if the remote cache had lost the contents of one
    // of their files. A later materialization succeeds, just as it does in production once the
    // repo has been fetched again.
    private final Set<String> reposToLoseOnce = ConcurrentHashMap.newKeySet();
    private final Set<String> pathsToLoseOnce = ConcurrentHashMap.newKeySet();
    private final RewindingSynchronizer rewindingSynchronizer = new RewindingSynchronizer();
    private final AtomicReference<PathFragment> externalDirSupplier = new AtomicReference<>();
    private final AtomicReference<RepoWriteLockRequest> repoWriteLockRequest =
        new AtomicReference<>();

    RewindableRepoFileSystemForTesting(FileSystem delegateFs, String outputBaseName) {
      super(delegateFs);
      this.outputBaseName = outputBaseName;
    }

    void loseOnNextMaterialization(String repoName) {
      reposToLoseOnce.add(repoName);
    }

    /**
     * Makes the next read of the given repo-relative file fail as if the remote cache had lost its
     * contents. This is how a lost file surfaces during loading, where only the file's metadata has
     * been injected and reading its contents is what reaches the cache.
     */
    void loseOnNextRead(String repoRelativePath) {
      pathsToLoseOnce.add(repoRelativePath);
    }

    @Override
    public InputStream getInputStream(PathFragment path) throws IOException {
      PathFragment externalDir = externalDirOf(path);
      if (externalDir != null && !pathsToLoseOnce.isEmpty()) {
        String repoName = path.getSegment(externalDir.segmentCount());
        String repoRelativePath = path.relativeTo(externalDir).getPathString();
        if (pathsToLoseOnce.remove(repoRelativePath)) {
          lostRepoFiles.add(path);
          rewindingSynchronizer.markReplacementsPossible();
          var unused =
              getPath(
                      externalDir.getChild(
                          RepositoryName.createUnvalidated(repoName).getMarkerFileName()))
                  .delete();
          throw new LostRemoteRepoFileException(
              "%s is no longer available in the remote cache".formatted(path),
              new IOException("missing blob"),
              RepositoryName.createUnvalidated(repoName),
              LOST_BLOB_DIGEST);
        }
      }
      return super.getInputStream(path);
    }

    @Override
    public RewindingSynchronizer getRewindingSynchronizer() {
      return rewindingSynchronizer;
    }

    @Override
    public TransferableWriteLock acquireRepoWriteLock(RepositoryName repo)
        throws InterruptedException {
      RepoWriteLockRequest request = repoWriteLockRequest.get();
      if (request == null
          || !request.repo().equals(repo)
          || !repoWriteLockRequest.compareAndSet(request, null)) {
        return RewindableRepoFileSystem.super.acquireRepoWriteLock(repo);
      }
      checkState(
          rewindingSynchronizer.hasBlockingReadLockForTesting(repo),
          "repo refetch requested before the executing reader acquired its lock");
      request.requested().countDown();
      TransferableWriteLock lock = RewindableRepoFileSystem.super.acquireRepoWriteLock(repo);
      request.afterAcquired().run();
      return lock;
    }

    /**
     * Makes the next acquisition of the write lock for the given repo count down {@code requested}
     * just before it starts to wait for the readers of that repo and run {@code afterAcquired} once
     * it has been granted.
     */
    void signalAroundNextRepoWriteLock(
        RepositoryName repo, CountDownLatch requested, Runnable afterAcquired) {
      checkState(
          repoWriteLockRequest.compareAndSet(
              null, new RepoWriteLockRequest(repo, requested, afterAcquired)),
          "a repo write lock request is already configured");
    }

    @Override
    public void markLostRepoFile(RepositoryName repo) {
      lostRepos.add(repo);
    }

    @Override
    public void ensureRepoMaterialized(RepositoryName repo, ExtendedEventHandler reporter)
        throws IOException {
      if (!reposToLoseOnce.remove(repo.getName())) {
        return;
      }
      PathFragment externalDir = externalDir();
      PathFragment lostFile = externalDir.getRelative(repo.getName()).getRelative("src.txt");
      lostRepoFiles.add(lostFile);
      rewindingSynchronizer.markReplacementsPossible();
      // A repo with lost files is a cache miss, so its fetch actually re-runs the repo rule.
      var unused = getPath(externalDir.getChild(repo.getMarkerFileName())).delete();
      throw new LostRemoteRepoFileException(
          "%s is no longer available in the remote cache".formatted(lostFile),
          new IOException("missing blob"),
          repo,
          LOST_BLOB_DIGEST);
    }

    @Nullable
    private PathFragment externalDirOf(PathFragment path) {
      for (int i = 1; i < path.segmentCount() - 1; i++) {
        if (path.getSegment(i).equals("external")
            && path.getSegment(i - 1).equals(outputBaseName)) {
          return path.subFragment(0, i + 1);
        }
      }
      return null;
    }

    private PathFragment externalDir() {
      return checkNotNull(externalDirSupplier.get(), "external dir not set");
    }

    void setExternalDir(PathFragment externalDir) {
      externalDirSupplier.set(externalDir);
    }

    private record RepoWriteLockRequest(
        RepositoryName repo, CountDownLatch requested, Runnable afterAcquired) {}
  }
}
