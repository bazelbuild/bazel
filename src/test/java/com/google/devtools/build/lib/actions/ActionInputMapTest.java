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
package com.google.devtools.build.lib.actions;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.truth.Truth.assertThat;
import static java.nio.charset.StandardCharsets.US_ASCII;
import static org.junit.Assert.assertThrows;

import com.google.auto.value.AutoValue;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.actions.ArtifactRoot.RootType;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.skyframe.TreeArtifactValue;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import com.google.testing.junit.testparameterinjector.TestParameter;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.Random;
import org.junit.Test;
import org.junit.runner.RunWith;

/** Unit test for {@link ActionInputMap}. */
@RunWith(TestParameterInjector.class)
public final class ActionInputMapTest {

  // small hint to stress the map
  private final ActionInputMap map = new ActionInputMap(1);
  private final ArtifactRoot artifactRoot =
      ArtifactRoot.asDerivedRoot(
          new InMemoryFileSystem(DigestHashFunction.SHA256).getPath("/execroot"),
          RootType.OUTPUT,
          "bazel-out");

  @Test
  public void basicPutAndLookup() {
    put("/abc/def", 5);
    assertThat(map.sizeForDebugging()).isEqualTo(1);
    assertContains("/abc/def", 5);
    assertThat(map.getMetadata(PathFragment.create("blah"))).isNull();
    assertThat(map.getInput("blah")).isNull();
  }

  @Test
  public void put_ignoresSubsequentPuts() {
    put("/abc/def", 5);
    assertThat(map.sizeForDebugging()).isEqualTo(1);
    put("/abc/def", 6);
    assertThat(map.sizeForDebugging()).isEqualTo(1);
    put("/ghi/jkl", 7);
    assertThat(map.sizeForDebugging()).isEqualTo(2);
    put("/ghi/jkl", 8);
    assertThat(map.sizeForDebugging()).isEqualTo(2);
    assertContains("/abc/def", 5);
    assertContains("/ghi/jkl", 7);
  }

  @Test
  public void clear_removesAllElements() {
    ActionInput input1 = new TestInput("/abc/def");
    ActionInput input2 = new TestInput("/ghi/jkl");
    SpecialArtifact tree = createTreeArtifact("tree");
    TreeFileArtifact treeChild = TreeFileArtifact.createTreeOutput(tree, "child");
    map.put(input1, TestMetadata.create(1));
    map.put(input2, TestMetadata.create(2));
    map.putTreeArtifact(
        tree,
        TreeArtifactValue.newBuilder(tree).putChild(treeChild, TestMetadata.create(3)).build());
    // Sanity check
    assertThat(map.sizeForDebugging()).isEqualTo(3);

    map.clear();

    assertThat(map.sizeForDebugging()).isEqualTo(0);
    assertDoesNotContain(input1);
    assertDoesNotContain(input2);
    assertDoesNotContain(tree);
    assertDoesNotContain(treeChild);
  }

  @Test
  public void putTreeArtifact_addsEmptyTreeArtifact() {
    SpecialArtifact tree = createTreeArtifact("tree");

    map.putTreeArtifact(tree, TreeArtifactValue.empty());

    assertThat(map.sizeForDebugging()).isEqualTo(1);
    assertContainsTree(tree, TreeArtifactValue.empty());
  }

  @Test
  public void putTreeArtifact_addsTreeArtifactAndAllChildren() {
    SpecialArtifact tree = createTreeArtifact("tree");
    TreeFileArtifact child1 = TreeFileArtifact.createTreeOutput(tree, "child1");
    FileArtifactValue child1Metadata = TestMetadata.create(1);
    TreeFileArtifact child2 = TreeFileArtifact.createTreeOutput(tree, "child2");
    FileArtifactValue child2Metadata = TestMetadata.create(2);
    TreeArtifactValue treeValue =
        TreeArtifactValue.newBuilder(tree)
            .putChild(child1, child1Metadata)
            .putChild(child2, child2Metadata)
            .build();

    map.putTreeArtifact(tree, treeValue);

    assertThat(map.sizeForDebugging()).isEqualTo(1);
    assertContainsTree(tree, treeValue);
    assertContainsFile(child1, child1Metadata);
    assertContainsFile(child2, child2Metadata);
  }

  @Test
  public void putTreeArtifact_mixedTreeAndFiles_addsTreeAndChildren() {
    SpecialArtifact tree = createTreeArtifact("tree");
    TreeFileArtifact child = TreeFileArtifact.createTreeOutput(tree, "child");
    FileArtifactValue childMetadata = TestMetadata.create(1);
    ActionInput file = ActionInputHelper.fromPath("file");
    FileArtifactValue fileMetadata = TestMetadata.create(2);
    map.put(file, fileMetadata);
    TreeArtifactValue treeValue =
        TreeArtifactValue.newBuilder(tree).putChild(child, childMetadata).build();

    map.putTreeArtifact(tree, treeValue);

    assertContainsTree(tree, treeValue);
    assertContainsFile(child, childMetadata);
    assertContainsFile(file, fileMetadata);
  }

  @Test
  public void putTreeArtifact_multipleTrees_addsAllTreesAndChildren() {
    SpecialArtifact tree1 = createTreeArtifact("tree1");
    TreeFileArtifact tree1Child = TreeFileArtifact.createTreeOutput(tree1, "child");
    FileArtifactValue tree1ChildMetadata = TestMetadata.create(1);
    SpecialArtifact tree2 = createTreeArtifact("tree2");
    TreeFileArtifact tree2Child = TreeFileArtifact.createTreeOutput(tree2, "child");
    FileArtifactValue tree2ChildMetadata = TestMetadata.create(2);
    TreeArtifactValue tree1Value =
        TreeArtifactValue.newBuilder(tree1).putChild(tree1Child, tree1ChildMetadata).build();
    TreeArtifactValue tree2Value =
        TreeArtifactValue.newBuilder(tree2).putChild(tree2Child, tree2ChildMetadata).build();

    map.putTreeArtifact(tree1, tree1Value);
    map.putTreeArtifact(tree2, tree2Value);

    assertContainsTree(tree1, tree1Value);
    assertContainsFile(tree1Child, tree1ChildMetadata);
    assertContainsTree(tree2, tree2Value);
    assertContainsFile(tree2Child, tree2ChildMetadata);
  }

  @Test
  public void putTreeArtifact_multipleTreesUnderSameDirectory_addsAllTrees() {
    SpecialArtifact tree1 = createTreeArtifact("dir/tree1");
    SpecialArtifact tree2 = createTreeArtifact("dir/tree2");
    SpecialArtifact tree3 = createTreeArtifact("dir/tree3");

    map.putTreeArtifact(tree1, TreeArtifactValue.empty());
    map.putTreeArtifact(tree2, TreeArtifactValue.empty());
    map.putTreeArtifact(tree3, TreeArtifactValue.empty());

    assertContainsTree(tree1, TreeArtifactValue.empty());
    assertContainsTree(tree2, TreeArtifactValue.empty());
    assertContainsTree(tree3, TreeArtifactValue.empty());
  }

  @Test
  public void putTreeArtifact_afterPutTreeArtifactWithSameExecPath_doesNothing() {
    SpecialArtifact tree1 = createTreeArtifact("tree");
    SpecialArtifact tree2 = createTreeArtifact("tree");
    TreeFileArtifact tree2File = TreeFileArtifact.createTreeOutput(tree2, "file");
    TreeArtifactValue tree1Value = TreeArtifactValue.empty();
    TreeArtifactValue tree2Value =
        TreeArtifactValue.newBuilder(tree2).putChild(tree2File, TestMetadata.create(1)).build();
    map.putTreeArtifact(tree1, tree1Value);

    map.putTreeArtifact(tree2, tree2Value);

    assertContainsTree(tree1, tree1Value);
    // Cannot assertContainsTree since the execpath will point to tree1 instead.
    assertThat(map.getInputMetadata(tree2)).isEqualTo(tree1Value.getMetadata());
    assertThat(map.getTreeMetadata(tree2.getExecPath())).isSameInstanceAs(tree1Value);
    assertThat(map.getInput(tree2.getExecPathString())).isSameInstanceAs(tree1);
    assertDoesNotContain(tree2File);
  }

  @Test
  public void putTreeArtifact_sameExecPathAsARegularFile_fails() {
    SpecialArtifact tree = createTreeArtifact("tree");
    ActionInput file = ActionInputHelper.fromPath(tree.getExecPath());
    map.put(file, TestMetadata.create(1));

    assertThrows(
        IllegalArgumentException.class, () -> map.putTreeArtifact(tree, TreeArtifactValue.empty()));
  }

  private enum PutOrder {
    DECLARED,
    REVERSE {
      @Override
      void runPuts(Runnable put1, Runnable put2) {
        super.runPuts(put2, put1);
      }
    };

    void runPuts(Runnable put1, Runnable put2) {
      put1.run();
      put2.run();
    }
  }

  @Test
  public void putTreeArtifact_nestedFile_returnsNestedFileFromExecPath(
      @TestParameter PutOrder putOrder) {
    SpecialArtifact tree = createTreeArtifact("tree");
    TreeFileArtifact treeFile = TreeFileArtifact.createTreeOutput(tree, "file");
    FileArtifactValue treeFileMetadata = TestMetadata.create(1);
    ActionInput file = ActionInputHelper.fromPath(treeFile.getExecPath());
    FileArtifactValue fileMetadata = TestMetadata.create(1); // identical to `tree/file` file.
    TreeArtifactValue treeValue =
        TreeArtifactValue.newBuilder(tree).putChild(treeFile, treeFileMetadata).build();

    putOrder.runPuts(() -> map.put(file, fileMetadata), () -> map.putTreeArtifact(tree, treeValue));

    assertThat(map.getInputMetadata(file)).isSameInstanceAs(fileMetadata);
    assertThat(map.getInputMetadata(treeFile)).isSameInstanceAs(treeFileMetadata);
    assertThat(map.getMetadata(treeFile.getExecPath())).isSameInstanceAs(fileMetadata);
    assertThat(map.getInput(treeFile.getExecPathString())).isSameInstanceAs(file);
  }

  @Test
  public void put_treeFileArtifact_addsEntry() {
    TreeFileArtifact treeFile =
        TreeFileArtifact.createTreeOutput(createTreeArtifact("tree"), "file");
    FileArtifactValue metadata = TestMetadata.create(1);

    map.put(treeFile, metadata);

    assertContainsFile(treeFile, metadata);
  }

  @Test
  public void put_sameExecPathAsATree_fails() {
    SpecialArtifact tree = createTreeArtifact("tree");
    ActionInput file = ActionInputHelper.fromPath(tree.getExecPath());
    FileArtifactValue fileMetadata = TestMetadata.create(1);
    map.putTreeArtifact(tree, TreeArtifactValue.empty());

    assertThrows(IllegalArgumentException.class, () -> map.put(file, fileMetadata));
  }

  @Test
  public void put_treeArtifact_fails() {
    SpecialArtifact tree = createTreeArtifact("tree");
    FileArtifactValue metadata = TestMetadata.create(1);

    assertThrows(IllegalArgumentException.class, () -> map.put(tree, metadata));
  }

  @Test
  public void getMetadata_actionInputWithTreeExecPath_returnsTreeArtifactEntries() {
    SpecialArtifact tree = createTreeArtifact("tree");
    map.putTreeArtifact(tree, TreeArtifactValue.empty());
    ActionInput input = ActionInputHelper.fromPath(tree.getExecPath());

    assertThat(map.getInputMetadata(input)).isEqualTo(TreeArtifactValue.empty().getMetadata());
  }

  @Test
  public void getMetadata_actionInputWithTreeFileExecPath_returnsTreeArtifactEntries() {
    ActionInputMap inputMap = new ActionInputMap(/* sizeHint= */ 1);
    SpecialArtifact tree = createTreeArtifact("tree");
    TreeFileArtifact treeFile = TreeFileArtifact.createTreeOutput(tree, "file");
    FileArtifactValue treeFileMetadata = TestMetadata.create(1);
    TreeArtifactValue treeValue =
        TreeArtifactValue.newBuilder(tree).putChild(treeFile, treeFileMetadata).build();
    inputMap.putTreeArtifact(tree, treeValue);
    ActionInput input = ActionInputHelper.fromPath(treeFile.getExecPath());

    FileArtifactValue metadata = inputMap.getInputMetadata(input);

    assertThat(metadata).isSameInstanceAs(treeFileMetadata);
  }

  @Test
  public void getMetadata_artifactWithTreeFileExecPath_returnsNull() {
    SpecialArtifact tree = createTreeArtifact("tree");
    TreeFileArtifact treeFile = TreeFileArtifact.createTreeOutput(tree, "file");
    TreeArtifactValue treeValue =
        TreeArtifactValue.newBuilder(tree).putChild(treeFile, TestMetadata.create(1)).build();
    map.putTreeArtifact(tree, treeValue);
    Artifact artifact =
        ActionsTestUtil.createArtifactWithExecPath(artifactRoot, treeFile.getExecPath());

    // Even though we could match the artifact by exec path, it was not registered as a nested
    // artifact -- only the tree file was.
    assertThat(map.getInputMetadata(artifact)).isNull();
  }

  @Test
  public void getMetadata_missingFileWithinTree_returnsNull() {
    SpecialArtifact tree = createTreeArtifact("tree");
    map.putTreeArtifact(
        tree,
        TreeArtifactValue.newBuilder(tree)
            .putChild(TreeFileArtifact.createTreeOutput(tree, "file"), TestMetadata.create(1))
            .build());
    TreeFileArtifact nonexistentTreeFile = TreeFileArtifact.createTreeOutput(tree, "nonexistent");

    assertDoesNotContain(nonexistentTreeFile);
  }

  @Test
  public void getInputMetadata_treeFileUnderFile_fails() {
    SpecialArtifact tree = createTreeArtifact("tree");
    TreeFileArtifact child = TreeFileArtifact.createTreeOutput(tree, "file");
    ActionInput file = ActionInputHelper.fromPath(tree.getExecPath());
    map.put(file, TestMetadata.create(1));

    assertThrows(IllegalArgumentException.class, () -> map.getInputMetadata(child));
  }

  @Test
  public void getTreeMetadataForPrefix_nonTree() {
    ActionInput file = ActionInputHelper.fromPath("some/file");
    map.put(file, TestMetadata.create(1));

    assertThat(map.getEnclosingTreeMetadata(file.getExecPath())).isNull();
    assertThat(map.getEnclosingTreeMetadata(file.getExecPath().getParentDirectory())).isNull();
    assertThat(map.getEnclosingTreeMetadata(file.getExecPath().getChild("under"))).isNull();
  }

  @Test
  public void getTreeMetadataForPrefix_emptyTree() {
    SpecialArtifact tree = createTreeArtifact("a/tree");
    TreeArtifactValue treeValue = TreeArtifactValue.newBuilder(tree).build();
    map.putTreeArtifact(tree, treeValue);

    assertThat(map.getEnclosingTreeMetadata(tree.getExecPath().getParentDirectory())).isNull();
    assertThat(map.getEnclosingTreeMetadata(tree.getExecPath())).isEqualTo(treeValue);
    assertThat(map.getEnclosingTreeMetadata(tree.getExecPath().getChild("under")))
        .isEqualTo(treeValue);
  }

  @Test
  public void getTreeMetadataForPrefix_nonEmptyTree() {
    SpecialArtifact tree = createTreeArtifact("a/tree");
    TreeFileArtifact child = TreeFileArtifact.createTreeOutput(tree, "some/child");
    TreeArtifactValue treeValue =
        TreeArtifactValue.newBuilder(tree).putChild(child, TestMetadata.create(1)).build();
    map.putTreeArtifact(tree, treeValue);

    assertThat(map.getEnclosingTreeMetadata(tree.getExecPath().getParentDirectory())).isNull();
    assertThat(map.getEnclosingTreeMetadata(tree.getExecPath())).isEqualTo(treeValue);
    assertThat(map.getEnclosingTreeMetadata(child.getExecPath())).isEqualTo(treeValue);
    assertThat(map.getEnclosingTreeMetadata(child.getExecPath().getParentDirectory()))
        .isEqualTo(treeValue);
    assertThat(map.getEnclosingTreeMetadata(child.getExecPath().getChild("under")))
        .isEqualTo(treeValue);
  }

  @Test
  public void getters_missingTree_returnNull() {
    map.putTreeArtifact(createTreeArtifact("tree"), TreeArtifactValue.empty());
    SpecialArtifact otherTree = createTreeArtifact("other");

    assertDoesNotContain(otherTree);
    assertDoesNotContain(TreeFileArtifact.createTreeOutput(otherTree, "child"));
  }

  @Test
  public void stress() {
    ArrayList<TestEntry> data = new ArrayList<>();
    {
      Random rng = new Random();
      HashSet<TestInput> deduper = new HashSet<>();
      for (int i = 0; i < 100000; ++i) {
        byte[] bytes = new byte[80];
        rng.nextBytes(bytes);
        for (int j = 0; j < bytes.length; ++j) {
          bytes[j] &= ((byte) 0x7f);
        }
        TestInput nextInput = new TestInput(new String(bytes, US_ASCII));
        if (deduper.add(nextInput)) {
          data.add(new TestEntry(nextInput, TestMetadata.create(i)));
        }
      }
    }
    for (int iteration = 0; iteration < 20; ++iteration) {
      map.clear();
      Collections.shuffle(data);
      for (int i = 0; i < data.size(); ++i) {
        TestEntry entry = data.get(i);
        map.put(entry.input, entry.metadata);
      }
      assertThat(map.sizeForDebugging()).isEqualTo(data.size());
      for (int i = 0; i < data.size(); ++i) {
        TestEntry entry = data.get(i);
        assertThat(map.getInputMetadata(entry.input)).isEqualTo(entry.metadata);
      }
    }
  }

  private void put(String execPath, int value) {
    map.put(new TestInput(execPath), TestMetadata.create(value));
  }

  private void assertContains(String execPath, int value) {
    assertThat(map.getInputMetadata(new TestInput(execPath))).isEqualTo(TestMetadata.create(value));
    assertThat(map.getMetadata(PathFragment.create(execPath)))
        .isEqualTo(TestMetadata.create(value));
    assertThat(map.getInput(execPath)).isEqualTo(new TestInput(execPath));
  }

  private void assertDoesNotContain(ActionInput input) {
    assertThat(map.getInputMetadata(input)).isNull();
    assertThat(map.getMetadata(input.getExecPath())).isNull();
    assertThat(map.getTreeMetadata(input.getExecPath())).isNull();
    assertThat(map.getInput(input.getExecPathString())).isNull();
  }

  private void assertContainsFile(ActionInput input, FileArtifactValue fileValue) {
    checkArgument(!(input instanceof SpecialArtifact), "use assertContainsTree for tree artifacts");
    assertThat(map.getInputMetadata(input)).isSameInstanceAs(fileValue);
    assertThat(map.getMetadata(input.getExecPath())).isSameInstanceAs(fileValue);
    assertThat(map.getTreeMetadata(input.getExecPath())).isNull();
    assertThat(map.getInput(input.getExecPathString())).isSameInstanceAs(input);
  }

  private void assertContainsTree(SpecialArtifact input, TreeArtifactValue treeValue) {
    // TreeArtifactValue#getMetadata returns a freshly allocated instance.
    assertThat(map.getInputMetadata(input)).isEqualTo(treeValue.getMetadata());
    assertThat(map.getMetadata(input.getExecPath())).isEqualTo(treeValue.getMetadata());
    assertThat(map.getTreeMetadata(input.getExecPath())).isSameInstanceAs(treeValue);
    assertThat(map.getInput(input.getExecPathString())).isSameInstanceAs(input);
  }

  private static class TestEntry {
    public final TestInput input;
    public final TestMetadata metadata;

    public TestEntry(TestInput input, TestMetadata metadata) {
      this.input = input;
      this.metadata = metadata;
    }
  }

  private static class TestInput implements ActionInput {
    private final PathFragment fragment;

    public TestInput(String fragment) {
      this.fragment = PathFragment.create(fragment);
    }

    @Override
    public boolean isDirectory() {
      return false;
    }

    @Override
    public boolean isSymlink() {
      return false;
    }

    @Override
    public PathFragment getExecPath() {
      return fragment;
    }

    @Override
    public String getExecPathString() {
      return fragment.toString();
    }

    @Override
    public boolean equals(Object other) {
      if (!(other instanceof TestInput)) {
        return false;
      }
      if (this == other) {
        return true;
      }
      return fragment.equals(((TestInput) other).fragment);
    }

    @Override
    public int hashCode() {
      return fragment.hashCode();
    }
  }

  private SpecialArtifact createTreeArtifact(String relativeExecPath) {
    return ActionsTestUtil.createTreeArtifactWithGeneratingAction(
        artifactRoot, artifactRoot.getExecPath().getRelative(relativeExecPath));
  }

  @AutoValue
  abstract static class TestMetadata extends FileArtifactValue {
    abstract int id();

    static TestMetadata create(int id) {
      return new AutoValue_ActionInputMapTest_TestMetadata(id);
    }

    @Override
    public FileStateType getType() {
      return FileStateType.REGULAR_FILE;
    }

    @Override
    public byte[] getDigest() {
      return DigestHashFunction.SHA256.getHashFunction().hashInt(id()).asBytes();
    }

    @Override
    public long getSize() {
      return id();
    }

    @Override
    public long getModifiedTime() {
      throw new UnsupportedOperationException();
    }

    @Override
    public boolean wasModifiedSinceDigest(Path path) {
      throw new UnsupportedOperationException();
    }

    @Override
    public FileContentsProxy getContentsProxy() {
      throw new UnsupportedOperationException();
    }
  }
}
