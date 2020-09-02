// Copyright 2020 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe;

import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.ActionLookupData;
import com.google.devtools.build.lib.actions.Artifact.ArchivedTreeArtifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.vfs.BatchStat;
import com.google.devtools.build.lib.vfs.FileStatusWithDigest;
import com.google.devtools.build.lib.vfs.FileStatusWithDigestAdapter;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.ModifiedFileSet;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Symlinks;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import javax.annotation.Nullable;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameter;
import org.junit.runners.Parameterized.Parameters;

/** {@link FilesystemValueChecker} checker parameterized on {@link BatchStatMode}. */
@RunWith(Parameterized.class)
public final class FilesystemValueCheckerParameterizedTest extends FilesystemValueCheckerTestBase {

  private static final ActionLookupData ACTION_LOOKUP_DATA = actionLookupData(0);

  @Parameter public BatchStatMode batchStat;

  @Parameters(name = "batchStat={0}")
  public static ImmutableList<Object[]> batchStatModes() {
    return Arrays.stream(BatchStatMode.values())
        .map(mode -> new BatchStatMode[] {mode})
        .collect(toImmutableList());
  }

  @Test
  public void getDirtyActionValues_unchangedEmptyTreeArtifactWithArchivedFile_noDirtyKeys()
      throws Exception {
    SpecialArtifact treeArtifact = createTreeArtifact("dir");
    treeArtifact.getPath().createDirectoryAndParents();
    ActionExecutionValue actionExecutionValue =
        actionValueWithTreeArtifacts(
            ImmutableList.of(),
            ImmutableList.of(createArchivedTreeArtifactWithContent(treeArtifact)));

    assertThat(getDirtyActionValues(actionExecutionValue)).isEmpty();
  }

  @Test
  public void getDirtyActionValues_unchangedTreeArtifactWithArchivedFile_noDirtyKeys()
      throws Exception {
    SpecialArtifact treeArtifact = createTreeArtifact("dir");
    ActionExecutionValue actionExecutionValue =
        actionValueWithTreeArtifacts(
            ImmutableList.of(
                createTreeFileArtifactWithContent(treeArtifact, "file1", "content"),
                createTreeFileArtifactWithContent(treeArtifact, "file2", "content2")),
            ImmutableList.of(createArchivedTreeArtifactWithContent(treeArtifact)));

    assertThat(getDirtyActionValues(actionExecutionValue)).isEmpty();
  }

  @Test
  public void getDirtyActionValues_editedArchivedFileForEmptyTreeArtifact_reportsChange()
      throws Exception {
    SpecialArtifact treeArtifact = createTreeArtifact("dir");
    treeArtifact.getPath().createDirectoryAndParents();
    ArchivedTreeArtifact archivedTreeArtifact =
        createArchivedTreeArtifactWithContent(treeArtifact, "old content");
    ActionExecutionValue actionExecutionValue =
        actionValueWithTreeArtifacts(ImmutableList.of(), ImmutableList.of(archivedTreeArtifact));

    writeFile(archivedTreeArtifact.getPath(), "new content");
    assertThat(getDirtyActionValues(actionExecutionValue)).containsExactly(ACTION_LOOKUP_DATA);
  }

  @Test
  public void getDirtyActionValues_editedArchivedFileForTreeArtifact_reportsChange()
      throws Exception {
    SpecialArtifact treeArtifact = createTreeArtifact("dir");
    ArchivedTreeArtifact archivedTreeArtifact =
        createArchivedTreeArtifactWithContent(treeArtifact, "old content");
    ActionExecutionValue actionExecutionValue =
        actionValueWithTreeArtifacts(
            ImmutableList.of(
                createTreeFileArtifactWithContent(
                    treeArtifact, /*parentRelativePath=*/ "file1", "content"),
                createTreeFileArtifactWithContent(
                    treeArtifact, /*parentRelativePath=*/ "file2", "content2")),
            ImmutableList.of(archivedTreeArtifact));

    writeFile(archivedTreeArtifact.getPath(), "new content");
    assertThat(getDirtyActionValues(actionExecutionValue)).containsExactly(ACTION_LOOKUP_DATA);
  }

  @Test
  public void getDirtyActionValues_deletedArchivedFileForTreeArtifact_reportsChange()
      throws Exception {
    SpecialArtifact treeArtifact = createTreeArtifact("dir");
    ArchivedTreeArtifact archivedTreeArtifact = createArchivedTreeArtifactWithContent(treeArtifact);
    ActionExecutionValue actionExecutionValue =
        actionValueWithTreeArtifacts(
            ImmutableList.of(
                createTreeFileArtifactWithContent(
                    treeArtifact, /*parentRelativePath=*/ "file1", "content"),
                createTreeFileArtifactWithContent(
                    treeArtifact, /*parentRelativePath=*/ "file2", "content2")),
            ImmutableList.of(archivedTreeArtifact));

    archivedTreeArtifact.getPath().delete();
    assertThat(getDirtyActionValues(actionExecutionValue)).containsExactly(ACTION_LOOKUP_DATA);
  }

  @Test
  public void getDirtyActionValues_deletedArchivedFileForEmptyTreeArtifact_reportsChange()
      throws Exception {
    SpecialArtifact treeArtifact = createTreeArtifact("dir");
    ArchivedTreeArtifact archivedTreeArtifact = createArchivedTreeArtifactWithContent(treeArtifact);
    ActionExecutionValue actionExecutionValue =
        actionValueWithTreeArtifacts(ImmutableList.of(), ImmutableList.of(archivedTreeArtifact));

    archivedTreeArtifact.getPath().delete();
    assertThat(getDirtyActionValues(actionExecutionValue)).containsExactly(ACTION_LOOKUP_DATA);
  }

  @Test
  public void getDirtyActionValues_editedFileForTreeArtifactWithArchivedFile_reportsChange()
      throws Exception {
    SpecialArtifact treeArtifact = createTreeArtifact("dir");
    TreeFileArtifact child1 =
        createTreeFileArtifactWithContent(
            treeArtifact, /*parentRelativePath=*/ "file1", "old content");
    ActionExecutionValue actionExecutionValue =
        actionValueWithTreeArtifacts(
            ImmutableList.of(
                child1,
                createTreeFileArtifactWithContent(
                    treeArtifact, /*parentRelativePath=*/ "file2", "content2")),
            ImmutableList.of(createArchivedTreeArtifactWithContent(treeArtifact)));

    writeFile(child1.getPath(), "new content");
    assertThat(getDirtyActionValues(actionExecutionValue)).containsExactly(ACTION_LOOKUP_DATA);
  }

  @Test
  public void getDirtyActionValues_treeArtifactWithArchivedArtifact_reportsOnlyChangedKey()
      throws Exception {
    SpecialArtifact unchangedTreeArtifact = createTreeArtifact("dir1");
    ActionExecutionValue unchangedValue =
        actionValueWithTreeArtifacts(
            ImmutableList.of(createTreeFileArtifactWithContent(unchangedTreeArtifact, "child")),
            ImmutableList.of(createArchivedTreeArtifactWithContent(unchangedTreeArtifact)));
    SpecialArtifact changedTreeArtifact = createTreeArtifact("dir2");
    ArchivedTreeArtifact changedArchivedTreeArtifact =
        createArchivedTreeArtifactWithContent(changedTreeArtifact, "old content");
    ActionExecutionValue changedValue =
        actionValueWithTreeArtifacts(
            ImmutableList.of(
                createTreeFileArtifactWithContent(changedTreeArtifact, "file", "content")),
            ImmutableList.of(changedArchivedTreeArtifact));

    writeFile(changedArchivedTreeArtifact.getPath(), "new content");
    assertThat(
            getDirtyActionValues(
                ImmutableMap.of(
                    actionLookupData(0), unchangedValue, actionLookupData(1), changedValue)))
        .containsExactly(actionLookupData(1));
  }

  private Collection<SkyKey> getDirtyActionValues(ActionExecutionValue actionExecutionValue)
      throws InterruptedException {
    return getDirtyActionValues(ImmutableMap.of(ACTION_LOOKUP_DATA, actionExecutionValue));
  }

  private Collection<SkyKey> getDirtyActionValues(ImmutableMap<SkyKey, SkyValue> valuesMap)
      throws InterruptedException {
    return new FilesystemValueChecker(
            /*tsgm=*/ null, /*lastExecutionTimeRange=*/ null, FSVC_THREADS_FOR_TEST)
        .getDirtyActionValues(
            valuesMap,
            batchStat.getBatchStat(fs),
            ModifiedFileSet.EVERYTHING_MODIFIED,
            /*trustRemoteArtifacts=*/ false);
  }

  private TreeFileArtifact createTreeFileArtifactWithContent(
      SpecialArtifact treeArtifact, String parentRelativePath, String... contentLines)
      throws IOException {
    TreeFileArtifact artifact = TreeFileArtifact.createTreeOutput(treeArtifact, parentRelativePath);
    writeFile(artifact.getPath(), contentLines);
    return artifact;
  }

  private ArchivedTreeArtifact createArchivedTreeArtifactWithContent(
      SpecialArtifact treeArtifact, String... contentLines) throws IOException {
    ArchivedTreeArtifact artifact =
        ArchivedTreeArtifact.create(treeArtifact, PathFragment.create("bin"));
    writeFile(artifact.getPath(), contentLines);
    return artifact;
  }

  private static ActionLookupData actionLookupData(int actionIndex) {
    return ActionLookupData.create(ACTION_LOOKUP_KEY, actionIndex);
  }

  private enum BatchStatMode {
    DISABLED {
      @Nullable
      @Override
      BatchStat getBatchStat(FileSystem fileSystem) {
        return null;
      }
    },
    ENABLED {
      @Override
      BatchStat getBatchStat(FileSystem fileSystem) {
        return (useDigest, includeLinks, paths) -> {
          List<FileStatusWithDigest> stats = new ArrayList<>();
          for (PathFragment pathFrag : paths) {
            stats.add(
                FileStatusWithDigestAdapter.adapt(
                    fileSystem.getPath("/").getRelative(pathFrag).statIfFound(Symlinks.NOFOLLOW)));
          }
          return stats;
        };
      }
    };

    @Nullable
    abstract BatchStat getBatchStat(FileSystem fileSystem);
  }
}
