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

import static com.google.common.truth.Truth.assertThat;
import static org.mockito.Mockito.mock;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.ActionLookupData;
import com.google.devtools.build.lib.actions.ActionLookupKey;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.ArchivedTreeArtifact;
import com.google.devtools.build.lib.actions.Artifact.DerivedArtifact;
import com.google.devtools.build.lib.actions.Artifact.OwnerlessArtifactWrapper;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifactType;
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.skyframe.TreeArtifactValue.ArchivedRepresentation;
import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameter;
import org.junit.runners.Parameterized.Parameters;

/**
 * Tests for {@link ActionExecutionValue#transformForSharedAction} for values including tree
 * artifacts.
 */
@RunWith(Parameterized.class)
public class ActionExecutionValueTransformSharedTreeArtifactsTest {

  private static final PathFragment DERIVED_PATH_PREFIX = PathFragment.create("bazel-out");

  private static final ActionLookupKey KEY_1 = mock(ActionLookupKey.class);
  private static final ActionLookupKey KEY_2 = mock(ActionLookupKey.class);

  @Parameter public boolean includeArchivedRepresentationForTreeArtifacts;

  @Parameters(name = "include archived tree artifacts: {0}")
  public static ImmutableList<Boolean> archivedRepresentationOptions() {
    return ImmutableList.of(true, false);
  }

  private final Scratch scratch = new Scratch();
  private ArtifactRoot derivedRoot;

  @Before
  public void createDerivedRoot() throws IOException {
    derivedRoot = ArtifactRoot.asDerivedRoot(scratch.dir("/execroot"), false, DERIVED_PATH_PREFIX);
  }

  @Test
  public void transformForSharedAction_createsCopyOfEmptyTreeArtifact() throws Exception {
    SpecialArtifact tree = createTreeArtifact("dir", KEY_1);
    TreeArtifactValue value = createTreeArtifactValue(tree);
    ActionExecutionValue actionExecutionValue =
        createActionExecutionValue(ImmutableMap.of(tree, value));

    SpecialArtifact tree2 = createTreeArtifact("dir", KEY_2);
    ActionExecutionValue transformedValue =
        actionExecutionValue.transformForSharedAction(ImmutableSet.of(tree2));

    assertThat(transformedValue.getAllFileValues()).isEmpty();
    assertThat(transformedValue.getAllTreeArtifactValues().keySet()).containsExactly(tree2);
    assertEqualsWithNewParent(value, tree2, transformedValue.getTreeArtifactValue(tree2));
  }

  @Test
  public void transformForSharedAction_createsCopyOfTreeArtifact() throws Exception {
    SpecialArtifact tree = createTreeArtifact("dir", KEY_1);
    TreeArtifactValue value = createTreeArtifactValue(tree, "file1", "file2");
    ActionExecutionValue actionExecutionValue =
        createActionExecutionValue(ImmutableMap.of(tree, value));

    SpecialArtifact tree2 = createTreeArtifact("dir", KEY_2);
    ActionExecutionValue transformedValue =
        actionExecutionValue.transformForSharedAction(ImmutableSet.of(tree2));

    assertThat(transformedValue.getAllFileValues()).isEmpty();
    assertThat(transformedValue.getAllTreeArtifactValues().keySet()).containsExactly(tree2);
    assertEqualsWithNewParent(value, tree2, transformedValue.getTreeArtifactValue(tree2));
  }

  @Test
  public void transformForSharedAction_createsCopyOfMultipleTreeArtifacts() throws Exception {
    SpecialArtifact tree1 = createTreeArtifact("dir1", KEY_1);
    SpecialArtifact tree2 = createTreeArtifact("dir2", KEY_1);
    TreeArtifactValue value1 = createTreeArtifactValue(tree1, "file1");
    TreeArtifactValue value2 = createTreeArtifactValue(tree2, "file2", "file3");
    ActionExecutionValue actionExecutionValue =
        createActionExecutionValue(ImmutableMap.of(tree1, value1, tree2, value2));

    SpecialArtifact sharedTree1 = createTreeArtifact("dir1", KEY_2);
    SpecialArtifact sharedTree2 = createTreeArtifact("dir2", KEY_2);
    ActionExecutionValue transformedValue =
        actionExecutionValue.transformForSharedAction(ImmutableSet.of(sharedTree1, sharedTree2));

    assertThat(transformedValue.getAllFileValues()).isEmpty();
    assertThat(transformedValue.getAllTreeArtifactValues().keySet())
        .containsExactly(sharedTree1, sharedTree2);
    assertEqualsWithNewParent(
        value1, sharedTree1, transformedValue.getTreeArtifactValue(sharedTree1));
    assertEqualsWithNewParent(
        value2, sharedTree2, transformedValue.getTreeArtifactValue(sharedTree2));
  }

  @Test
  public void transformForSharedAction_createsCopyForFileAndTreeArtifacts() throws Exception {
    DerivedArtifact file = createFileArtifact("file", KEY_1);
    createFile(file.getPath());
    FileArtifactValue fileValue = FileArtifactValue.createForTesting(file);
    SpecialArtifact tree = createTreeArtifact("dir", KEY_1);
    TreeArtifactValue treeValue = createTreeArtifactValue(tree, "file1", "file2");
    ActionExecutionValue actionExecutionValue =
        createActionExecutionValue(
            ImmutableMap.of(file, fileValue), ImmutableMap.of(tree, treeValue));

    SpecialArtifact sharedTree = createTreeArtifact("dir", KEY_2);
    DerivedArtifact sharedFile = createFileArtifact("file", KEY_2);
    ActionExecutionValue transformedValue =
        actionExecutionValue.transformForSharedAction(ImmutableSet.of(sharedFile, sharedTree));

    assertThat(transformedValue.getAllFileValues().keySet()).containsExactly(sharedFile);
    assertThat(transformedValue.getAllFileValues().get(sharedFile)).isSameInstanceAs(fileValue);
    assertThat(transformedValue.getAllTreeArtifactValues().keySet()).containsExactly(sharedTree);
    assertEqualsWithNewParent(
        treeValue, sharedTree, transformedValue.getTreeArtifactValue(sharedTree));
  }

  /**
   * Checks that provided {@link TreeArtifactValue} has equal metadata to the original one, expected
   * parent for all of the included artifacts and otherwise the same artifacts as the original one.
   */
  private void assertEqualsWithNewParent(
      TreeArtifactValue originalValue,
      SpecialArtifact expectedTree,
      TreeArtifactValue actualValue) {
    assertThat(actualValue.getDigest()).isEqualTo(originalValue.getDigest());
    assertThat(actualValue.getMetadata()).isEqualTo(originalValue.getMetadata());
    assertThat(actualValue.getChildPaths()).isEqualTo(originalValue.getChildPaths());

    assertThat(actualValue.getArchivedRepresentation().isPresent())
        .isEqualTo(includeArchivedRepresentationForTreeArtifacts);

    actualValue
        .getArchivedRepresentation()
        .ifPresent(
            archivedRepresentation -> {
              ArchivedRepresentation originalRepresentation =
                  originalValue.getArchivedRepresentation().get();
              assertEqualsWithNewParent(
                  originalRepresentation, expectedTree, archivedRepresentation);
            });

    actualValue
        .getChildValues()
        .forEach(
            (artifact, metadata) -> {
              TreeFileArtifact originalArtifact =
                  originalValue.getChildren().stream()
                      .filter(
                          original ->
                              original
                                  .getParentRelativePath()
                                  .equals(artifact.getParentRelativePath()))
                      .findAny()
                      .get();
              assertThat(artifact.getParent()).isEqualTo(expectedTree);
              assertOwnerlessEquals(originalArtifact, artifact);
              assertThat(artifact.isChildOfDeclaredDirectory())
                  .isEqualTo(originalArtifact.isChildOfDeclaredDirectory());
              assertThat(metadata)
                  .isSameInstanceAs(originalValue.getChildValues().get(originalArtifact));
            });
  }

  private static void assertEqualsWithNewParent(
      ArchivedRepresentation expectedRepresentation,
      SpecialArtifact expectedTree,
      ArchivedRepresentation actualRepresentation) {
    assertThat(actualRepresentation.archivedTreeFileArtifact().getParent()).isEqualTo(expectedTree);
    assertOwnerlessEquals(
        expectedRepresentation.archivedTreeFileArtifact(),
        actualRepresentation.archivedTreeFileArtifact());
    assertThat(actualRepresentation.archivedFileValue())
        .isSameInstanceAs(expectedRepresentation.archivedFileValue());
  }

  private static void assertOwnerlessEquals(Artifact expectedArtifact, Artifact actualArtifact) {
    assertThat(new OwnerlessArtifactWrapper(actualArtifact))
        .isEqualTo(new OwnerlessArtifactWrapper(expectedArtifact));
  }

  private TreeArtifactValue createTreeArtifactValue(
      SpecialArtifact treeArtifact, String... parentRelativePaths) throws IOException {
    TreeArtifactValue.Builder builder = TreeArtifactValue.newBuilder(treeArtifact);

    for (String parentRelativePath : parentRelativePaths) {
      TreeFileArtifact childArtifact =
          TreeFileArtifact.createTreeOutput(treeArtifact, parentRelativePath);
      createFile(childArtifact.getPath());
      builder.putChild(childArtifact, FileArtifactValue.createForTesting(childArtifact));
    }

    if (includeArchivedRepresentationForTreeArtifacts) {
      ArchivedTreeArtifact archivedArtifact =
          ArchivedTreeArtifact.create(treeArtifact, DERIVED_PATH_PREFIX);
      createFile(archivedArtifact.getPath());
      builder.setArchivedRepresentation(
          archivedArtifact, FileArtifactValue.createForTesting(archivedArtifact));
    }

    return builder.build();
  }

  private DerivedArtifact createFileArtifact(String relativePath, ActionLookupKey owner) {
    return new DerivedArtifact(derivedRoot, DERIVED_PATH_PREFIX.getRelative(relativePath), owner);
  }

  private SpecialArtifact createTreeArtifact(String relativePath, ActionLookupKey owner) {
    SpecialArtifact treeArtifact =
        new SpecialArtifact(
            derivedRoot,
            DERIVED_PATH_PREFIX.getRelative(relativePath),
            owner,
            SpecialArtifactType.TREE);
    treeArtifact.setGeneratingActionKey(ActionLookupData.create(owner, 0));
    return treeArtifact;
  }

  private static ActionExecutionValue createActionExecutionValue(
      ImmutableMap<Artifact, TreeArtifactValue> treeArtifacts) {
    return createActionExecutionValue(/*fileArtifacts=*/ ImmutableMap.of(), treeArtifacts);
  }

  private static ActionExecutionValue createActionExecutionValue(
      ImmutableMap<Artifact, FileArtifactValue> fileArtifacts,
      ImmutableMap<Artifact, TreeArtifactValue> treeArtifacts) {
    return ActionExecutionValue.create(
        fileArtifacts,
        treeArtifacts,
        /*outputSymlinks=*/ null,
        /*discoveredModules=*/ null,
        /*actionDependsOnBuildId=*/ false);
  }

  private static void createFile(Path file) throws IOException {
    FileSystemUtils.writeIsoLatin1(file);
  }
}
