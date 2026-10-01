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
import com.google.devtools.build.lib.rules.repository.RepositoryDirectoryValue;
import com.google.devtools.build.lib.runtime.BlazeRuntime;
import com.google.devtools.build.lib.runtime.BlockWaitingModule;
import com.google.devtools.build.lib.runtime.MemoryPressureModule;
import com.google.devtools.build.lib.testutil.ActionEventRecorder;
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
 * <p>These tests are kept separate from {@link RewindingTest}, which disables external repositories
 * to preserve its action graph structure between blaze and bazel.
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
        .addBlazeModule(new MemoryPressureModule())
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

  /** Writes a rule whose only action symlinks its output to the given source file. */
  private void writeSymlinkRule() throws Exception {
    write(
        "test/symlink.bzl",
        """
        def _symlink_impl(ctx):
            out = ctx.actions.declare_file(ctx.label.name + ".txt")
            ctx.actions.symlink(output = out, target_file = ctx.file.src)
            return [DefaultInfo(files = depset([out]))]

        symlink = rule(
            implementation = _symlink_impl,
            attrs = {"src": attr.label(allow_single_file = True)},
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
    RootedPath rootedPath = lostInput.getRootedPath();
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
  public void lostFileBehindSymlinkAction_repoRewound() throws Exception {
    writeRepoRule();
    writeSymlinkRule();
    write("repo/content_a.txt", "old");
    appendToModuleFile(
        "my_repo = use_repo_rule('//repo:repo.bzl', 'my_repo')",
        "my_repo(name = 'repo_a', content_file = 'content_a.txt')");
    write(
        "test/BUILD",
        """
        load("//test:symlink.bzl", "symlink")

        symlink(
            name = "link",
            src = "@repo_a//:src.txt",
        )

        genrule(
            name = "consume",
            srcs = [":link"],
            outs = ["out.txt"],
            cmd = "cp $< $@",
        )
        """);

    AtomicReference<Artifact> lostInput = new AtomicReference<>();
    helper.addSpawnShim(
        "Executing genrule //test:consume",
        (spawn, context) -> {
          // The symlink action's output is what the action reads and thus what is reported lost,
          // but its contents are those of the source file it points to.
          Artifact input = (Artifact) SpawnInputUtils.getInputWithName(spawn, "link.txt");
          lostInput.set(input);
          write("repo/content_a.txt", "new");
          var unused =
              getOutputBase()
                  .getRelative("external")
                  .getRelative(canonicalRepoName("repo_a").getMarkerFileName())
                  .delete();
          return helper.createLostInputsExecException(context, ImmutableList.of(input));
        });

    List<SkyKey> rewoundKeys = helper.collectOrderedRewoundKeys();
    buildTarget("//test:consume");

    helper.verifyAllSpawnShimsConsumed();
    // Rewinding the symlink action alone would recreate the symlink to the very file whose
    // contents are gone. The repo is refetched instead, after which the symlink action runs again.
    assertContents("new", "//test:consume");
    assertThat(rewoundKeys).contains(RepositoryDirectoryValue.key(canonicalRepoName("repo_a")));
    assertThat(rewoundKeys).contains(Artifact.key(lostInput.get()));
    assertThat(helper.getExecutedSpawnDescriptions())
        .containsExactly("Executing genrule //test:consume", "Executing genrule //test:consume");
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
    // The rewind is restricted to the path through the runfiles tree that leads to the lost file,
    // leaving the tool's other actions alone.
    assertThat(rewoundActions).hasSize(1);
    assertThat(rewoundKeys)
        .containsExactlyElementsIn(
            ImmutableList.builder().addAll(chain).addAll(rewoundActions).build());
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
    return RepositoryName.createUnvalidated(repoFile.getRoot().getRoot().asPath().getBaseName());
  }

  /**
   * A {@link DelegateFileSystem} that simulates the {@link RewindableRepoFileSystem} capability of
   * the file system that serves repo contents from the remote repo contents cache.
   */
  private static final class RewindableRepoFileSystemForTesting extends DelegateFileSystem
      implements RewindableRepoFileSystem {
    final List<RepositoryName> lostRepos = Collections.synchronizedList(new ArrayList<>());
    private final String outputBaseName;

    RewindableRepoFileSystemForTesting(FileSystem delegateFs, String outputBaseName) {
      super(delegateFs);
      this.outputBaseName = outputBaseName;
    }

    @Override
    public void markLostRepoFile(RepositoryName repo) {
      lostRepos.add(repo);
    }

    @Override
    public void repoRefetched(RepositoryName repo) {}

    @Override
    public boolean isRepoPath(PathFragment path) {
      return externalDirOf(path) != null;
    }

    @Override
    public RepositoryName repoContaining(PathFragment path) {
      return RepositoryName.createUnvalidated(
          path.getSegment(checkNotNull(externalDirOf(path)).segmentCount()));
    }

    /** Returns the external directory the given path lies in, or null if it lies outside one. */
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
  }
}
