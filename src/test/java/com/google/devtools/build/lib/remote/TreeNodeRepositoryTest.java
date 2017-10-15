// Copyright 2015 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.remote;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionInputFileCache;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Root;
import com.google.devtools.build.lib.exec.SingleBuildFileCache;
import com.google.devtools.build.lib.remote.TreeNodeRepository.TreeNode;
import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystem.HashFunction;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.remoteexecution.v1test.Digest;
import com.google.devtools.remoteexecution.v1test.Directory;
import java.util.ArrayList;
import java.util.SortedMap;
import java.util.TreeMap;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link TreeNodeRepository}. */
@RunWith(JUnit4.class)
public class TreeNodeRepositoryTest {
  private Scratch scratch;
  private Root rootDir;
  private Path rootPath;

  @Before
  public final void setRootDir() throws Exception {
    FileSystem.setDigestFunctionForTesting(HashFunction.SHA1);
    scratch = new Scratch();
    rootDir = Root.asDerivedRoot(scratch.dir("/exec/root"));
    rootPath = rootDir.getPath();
  }

  private TreeNodeRepository createTestTreeNodeRepository() {
    ActionInputFileCache inputFileCache =
        new SingleBuildFileCache(rootPath.getPathString(), scratch.getFileSystem());
    return new TreeNodeRepository(rootPath, inputFileCache);
  }

  @Test
  @SuppressWarnings("ReferenceEquality")
  public void testSubtreeReusage() throws Exception {
    Artifact fooCc = new Artifact(scratch.file("/exec/root/a/foo.cc"), rootDir);
    Artifact fooH = new Artifact(scratch.file("/exec/root/a/foo.h"), rootDir);
    Artifact bar = new Artifact(scratch.file("/exec/root/b/bar.txt"), rootDir);
    Artifact baz = new Artifact(scratch.file("/exec/root/c/baz.txt"), rootDir);
    TreeNodeRepository repo = createTestTreeNodeRepository();
    TreeNode root1 = repo.buildFromActionInputs(ImmutableList.<ActionInput>of(fooCc, fooH, bar));
    TreeNode root2 = repo.buildFromActionInputs(ImmutableList.<ActionInput>of(fooCc, fooH, baz));
    // Reusing same node for the "a" subtree.
    assertThat(
            root1.getChildEntries().get(0).getChild() == root2.getChildEntries().get(0).getChild())
        .isTrue();
  }

  @Test
  public void testMerkleDigests() throws Exception {
    Artifact foo = new Artifact(scratch.file("/exec/root/a/foo", "1"), rootDir);
    Artifact bar = new Artifact(scratch.file("/exec/root/a/bar", "11"), rootDir);
    TreeNodeRepository repo = createTestTreeNodeRepository();
    TreeNode root = repo.buildFromActionInputs(ImmutableList.<ActionInput>of(foo, bar));
    TreeNode aNode = root.getChildEntries().get(0).getChild();
    TreeNode fooNode = aNode.getChildEntries().get(1).getChild(); // foo > bar in sort order!
    TreeNode barNode = aNode.getChildEntries().get(0).getChild();

    repo.computeMerkleDigests(root);
    ImmutableCollection<Digest> digests = repo.getAllDigests(root);
    Digest rootDigest = repo.getMerkleDigest(root);
    Digest aDigest = repo.getMerkleDigest(aNode);
    Digest fooDigest = repo.getMerkleDigest(fooNode); // The contents digest.
    Digest barDigest = repo.getMerkleDigest(barNode);
    assertThat(digests).containsExactly(rootDigest, aDigest, barDigest, fooDigest);

    ArrayList<Directory> directories = new ArrayList<>();
    ArrayList<ActionInput> actionInputs = new ArrayList<>();
    repo.getDataFromDigests(digests, actionInputs, directories);
    assertThat(actionInputs).containsExactly(bar, foo);
    assertThat(directories).hasSize(2);
    Directory rootDirectory = directories.get(0);
    assertThat(rootDirectory.getDirectories(0).getName()).isEqualTo("a");
    assertThat(rootDirectory.getDirectories(0).getDigest()).isEqualTo(aDigest);
    Directory aDirectory = directories.get(1);
    assertThat(aDirectory.getFiles(0).getName()).isEqualTo("bar");
    assertThat(aDirectory.getFiles(0).getDigest()).isEqualTo(barDigest);
    assertThat(aDirectory.getFiles(1).getName()).isEqualTo("foo");
    assertThat(aDirectory.getFiles(1).getDigest()).isEqualTo(fooDigest);
  }

  @Test
  public void testGetAllDigests() throws Exception {
    Artifact foo1 = new Artifact(scratch.file("/exec/root/a/foo", "1"), rootDir);
    Artifact foo2 = new Artifact(scratch.file("/exec/root/b/foo", "1"), rootDir);
    Artifact foo3 = new Artifact(scratch.file("/exec/root/c/foo", "1"), rootDir);
    TreeNodeRepository repo = createTestTreeNodeRepository();
    TreeNode root = repo.buildFromActionInputs(ImmutableList.<ActionInput>of(foo1, foo2, foo3));
    repo.computeMerkleDigests(root);
    // Reusing same node for the "foo" subtree: only need the root, root child, and foo contents:
    assertThat(repo.getAllDigests(root)).hasSize(3);
  }

  @Test
  public void testNullArtifacts() throws Exception {
    Artifact foo = new Artifact(scratch.file("/exec/root/a/foo", "1"), rootDir);
    SortedMap<PathFragment, ActionInput> inputs = new TreeMap<>();
    inputs.put(foo.getExecPath(), foo);
    inputs.put(PathFragment.create("a/bar"), null);
    TreeNodeRepository repo = createTestTreeNodeRepository();
    TreeNode root = repo.buildFromActionInputs(inputs);
    repo.computeMerkleDigests(root);

    TreeNode aNode = root.getChildEntries().get(0).getChild();
    TreeNode fooNode = aNode.getChildEntries().get(1).getChild(); // foo > bar in sort order!
    TreeNode barNode = aNode.getChildEntries().get(0).getChild();
    ImmutableCollection<Digest> digests = repo.getAllDigests(root);
    Digest rootDigest = repo.getMerkleDigest(root);
    Digest aDigest = repo.getMerkleDigest(aNode);
    Digest fooDigest = repo.getMerkleDigest(fooNode);
    Digest barDigest = repo.getMerkleDigest(barNode);
    assertThat(digests).containsExactly(rootDigest, aDigest, barDigest, fooDigest);
  }

  @Test
  public void testEmptyTree() throws Exception {
    SortedMap<PathFragment, ActionInput> inputs = new TreeMap<>();
    TreeNodeRepository repo = createTestTreeNodeRepository();
    TreeNode root = repo.buildFromActionInputs(inputs);
    repo.computeMerkleDigests(root);

    assertThat(root.getChildEntries()).isEmpty();
  }
}
