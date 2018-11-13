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
import static org.junit.Assert.fail;

import build.bazel.remote.execution.v2.Digest;
import build.bazel.remote.execution.v2.Directory;
import build.bazel.remote.execution.v2.DirectoryNode;
import build.bazel.remote.execution.v2.FileNode;
import build.bazel.remote.execution.v2.SymlinkNode;
import com.google.common.collect.ImmutableCollection;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionInputHelper;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.MetadataProvider;
import com.google.devtools.build.lib.clock.BlazeClock;
import com.google.devtools.build.lib.exec.SingleBuildFileCache;
import com.google.devtools.build.lib.remote.TreeNodeRepository.TreeNode;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
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
  private DigestUtil digestUtil;
  private Path execRoot;
  private ArtifactRoot rootDir;

  @Before
  public final void setRootDir() throws Exception {
    digestUtil = new DigestUtil(DigestHashFunction.SHA256);
    scratch = new Scratch(new InMemoryFileSystem(BlazeClock.instance(), DigestHashFunction.SHA256));
    execRoot = scratch.getFileSystem().getPath("/exec/root");
    rootDir = ArtifactRoot.asSourceRoot(Root.fromPath(scratch.dir("/exec/root")));
  }

  private TreeNodeRepository createTestTreeNodeRepository(boolean uploadSymlinks) {
    MetadataProvider inputFileCache =
        new SingleBuildFileCache(execRoot.getPathString(), scratch.getFileSystem());
    return new TreeNodeRepository(execRoot, inputFileCache, digestUtil, uploadSymlinks);
  }

  private TreeNodeRepository createTestTreeNodeRepository() {
    return createTestTreeNodeRepository(true);
  }

  private TreeNode buildFromActionInputs(TreeNodeRepository repo, ActionInput... inputs)
      throws IOException {
    TreeMap<PathFragment, ActionInput> sortedMap = new TreeMap<>();
    for (ActionInput input : inputs) {
      sortedMap.put(PathFragment.create(input.getExecPathString()), input);
    }
    return repo.buildFromActionInputs(sortedMap);
  }

  @Test
  @SuppressWarnings("ReferenceEquality")
  public void testSubtreeReusage() throws Exception {
    Artifact fooCc = new Artifact(scratch.file("/exec/root/a/foo.cc"), rootDir);
    Artifact fooH = new Artifact(scratch.file("/exec/root/a/foo.h"), rootDir);
    Artifact bar = new Artifact(scratch.file("/exec/root/b/bar.txt"), rootDir);
    Artifact baz = new Artifact(scratch.file("/exec/root/c/baz.txt"), rootDir);
    TreeNodeRepository repo = createTestTreeNodeRepository();
    TreeNode root1 = buildFromActionInputs(repo, fooCc, fooH, bar);
    TreeNode root2 = buildFromActionInputs(repo, fooCc, fooH, baz);
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
    TreeNode root = buildFromActionInputs(repo, foo, bar);
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

    Map<Digest, Directory> directories = new HashMap<>();
    Map<Digest, ActionInput> actionInputs = new HashMap<>();
    repo.getDataFromDigests(digests, actionInputs, directories);
    assertThat(actionInputs.values()).containsExactly(bar, foo);
    assertThat(directories).hasSize(2);
    Directory rootDirectory = directories.get(rootDigest);
    assertThat(rootDirectory.getDirectories(0).getName()).isEqualTo("a");
    assertThat(rootDirectory.getDirectories(0).getDigest()).isEqualTo(aDigest);
    Directory aDirectory = directories.get(aDigest);
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
    TreeNode root = buildFromActionInputs(repo, foo1, foo2, foo3);
    repo.computeMerkleDigests(root);
    // Reusing same node for the "foo" subtree: only need the root, root child, and foo contents:
    assertThat(repo.getAllDigests(root)).hasSize(3);
  }

  @Test
  public void testEmptyTree() throws Exception {
    SortedMap<PathFragment, ActionInput> inputs = new TreeMap<>();
    TreeNodeRepository repo = createTestTreeNodeRepository();
    TreeNode root = repo.buildFromActionInputs(inputs);
    repo.computeMerkleDigests(root);

    assertThat(root.getChildEntries()).isEmpty();
  }

  @Test
  public void testDirectoryInput() throws Exception {
    Artifact foo = new Artifact(scratch.dir("/exec/root/a/foo"), rootDir);
    scratch.file("/exec/root/a/foo/foo.h", "1");
    ActionInput fooH = ActionInputHelper.fromPath("/exec/root/a/foo/foo.h");
    scratch.file("/exec/root/a/foo/foo.cc", "2");
    ActionInput fooCc = ActionInputHelper.fromPath("/exec/root/a/foo/foo.cc");

    Artifact bar = new Artifact(scratch.file("/exec/root/a/bar.txt"), rootDir);
    TreeNodeRepository repo = createTestTreeNodeRepository();

    Artifact aClient = new Artifact(scratch.dir("/exec/root/a-client"), rootDir);
    scratch.file("/exec/root/a-client/baz.txt", "3");
    ActionInput baz = ActionInputHelper.fromPath("/exec/root/a-client/baz.txt");

    TreeNode root = buildFromActionInputs(repo, foo, aClient, bar);
    TreeNode aNode = root.getChildEntries().get(0).getChild();
    TreeNode fooNode = aNode.getChildEntries().get(1).getChild(); // foo > bar in sort order!
    TreeNode barNode = aNode.getChildEntries().get(0).getChild();
    TreeNode aClientNode = root.getChildEntries().get(1).getChild(); // a-client > a in sort order
    TreeNode bazNode = aClientNode.getChildEntries().get(0).getChild();

    TreeNode fooHNode =
        fooNode.getChildEntries().get(1).getChild(); // foo.h > foo.cc in sort order!
    TreeNode fooCcNode = fooNode.getChildEntries().get(0).getChild();

    repo.computeMerkleDigests(root);
    ImmutableCollection<Digest> digests = repo.getAllDigests(root);
    Digest rootDigest = repo.getMerkleDigest(root);
    Digest aDigest = repo.getMerkleDigest(aNode);
    Digest fooDigest = repo.getMerkleDigest(fooNode);
    Digest fooHDigest = repo.getMerkleDigest(fooHNode);
    Digest fooCcDigest = repo.getMerkleDigest(fooCcNode);
    Digest aClientDigest = repo.getMerkleDigest(aClientNode);
    Digest bazDigest = repo.getMerkleDigest(bazNode);
    Digest barDigest = repo.getMerkleDigest(barNode);
    assertThat(digests)
        .containsExactly(
            rootDigest,
            aDigest,
            barDigest,
            fooDigest,
            fooCcDigest,
            fooHDigest,
            aClientDigest,
            bazDigest);

    Map<Digest, Directory> directories = new HashMap<>();
    Map<Digest, ActionInput> actionInputs = new HashMap<>();
    repo.getDataFromDigests(digests, actionInputs, directories);
    assertThat(actionInputs.values()).containsExactly(bar, fooH, fooCc, baz);
    assertThat(directories).hasSize(4); // root, root/a, root/a/foo, and root/a-client
    Directory rootDirectory = directories.get(rootDigest);
    assertThat(rootDirectory.getDirectories(0).getName()).isEqualTo("a");
    assertThat(rootDirectory.getDirectories(0).getDigest()).isEqualTo(aDigest);
    assertThat(rootDirectory.getDirectories(1).getName()).isEqualTo("a-client");
    assertThat(rootDirectory.getDirectories(1).getDigest()).isEqualTo(aClientDigest);
    Directory aDirectory = directories.get(aDigest);
    assertThat(aDirectory.getFiles(0).getName()).isEqualTo("bar.txt");
    assertThat(aDirectory.getFiles(0).getDigest()).isEqualTo(barDigest);
    assertThat(aDirectory.getDirectories(0).getName()).isEqualTo("foo");
    assertThat(aDirectory.getDirectories(0).getDigest()).isEqualTo(fooDigest);
    Directory fooDirectory = directories.get(fooDigest);
    assertThat(fooDirectory.getFiles(0).getName()).isEqualTo("foo.cc");
    assertThat(fooDirectory.getFiles(0).getDigest()).isEqualTo(fooCcDigest);
    assertThat(fooDirectory.getFiles(1).getName()).isEqualTo("foo.h");
    assertThat(fooDirectory.getFiles(1).getDigest()).isEqualTo(fooHDigest);
    Directory aClientDirectory = directories.get(aClientDigest);
    assertThat(aClientDirectory.getFiles(0).getName()).isEqualTo("baz.txt");
    assertThat(aClientDirectory.getFiles(0).getDigest()).isEqualTo(bazDigest);
  }

  @Test
  public void testAbsoluteFileSymlinkAsFile() throws Exception {
    Path link = scratch.getFileSystem().getPath("/exec/root/link");
    Path target = scratch.file("/exec/root/target", "bla");
    link.createSymbolicLink(target);
    Artifact linkInput = new Artifact(link, rootDir);
    TreeNodeRepository repo = createTestTreeNodeRepository(true);
    TreeNode root = buildFromActionInputs(repo, linkInput);
    repo.computeMerkleDigests(root);
    Digest digest = digestUtil.compute(target);
    Directory rootDir =
        Directory.newBuilder()
            .addFiles(FileNode.newBuilder().setName("link").setDigest(digest).setIsExecutable(true))
            .build();
    assertThat(repo.treeToDirectories(root)).containsExactly(rootDir);
  }

  @Test
  public void testAbsoluteFileSymlinkInDirectoryAsFile() throws Exception {
    Path foo = scratch.file("/exec/root/foo", "bla");
    Path dir = scratch.dir("/exec/root/dir");
    Path link = scratch.getFileSystem().getPath("/exec/root/dir/link");
    link.createSymbolicLink(foo);
    Artifact dirInput = new Artifact(dir, rootDir);
    TreeNodeRepository repo = createTestTreeNodeRepository(true);
    TreeNode root = buildFromActionInputs(repo, dirInput);
    repo.computeMerkleDigests(root);
    Digest digest = digestUtil.compute(foo);
    Directory childDir =
        Directory.newBuilder()
            .addFiles(FileNode.newBuilder().setName("link").setDigest(digest).setIsExecutable(true))
            .build();
    Digest dirDigest = digestUtil.compute(childDir);
    Directory rootDir =
        Directory.newBuilder()
            .addDirectories(DirectoryNode.newBuilder().setName("dir").setDigest(dirDigest))
            .build();
    assertThat(repo.treeToDirectories(root)).containsExactly(rootDir, childDir);
  }

  @Test
  public void testRelativeFileSymlinkAsFile() throws Exception {
    Path link = scratch.getFileSystem().getPath("/exec/root/link");
    Path target = scratch.file("/exec/root/target", "bla");
    link.createSymbolicLink(target.relativeTo(execRoot));
    Artifact linkInput = new Artifact(link, rootDir);
    TreeNodeRepository repo = createTestTreeNodeRepository(false);
    TreeNode root = buildFromActionInputs(repo, linkInput);
    repo.computeMerkleDigests(root);
    Digest digest = digestUtil.compute(target);
    Directory rootDir =
        Directory.newBuilder()
            .addFiles(FileNode.newBuilder().setName("link").setDigest(digest).setIsExecutable(true))
            .build();
    assertThat(repo.treeToDirectories(root)).containsExactly(rootDir);
  }

  @Test
  public void testRelativeFileSymlinkInDirectoryAsFile() throws Exception {
    Path foo = scratch.file("/exec/root/foo", "bla");
    Path dir = scratch.dir("/exec/root/dir");
    Path link = scratch.getFileSystem().getPath("/exec/root/dir/link");
    link.createSymbolicLink(PathFragment.create("../foo"));
    Artifact dirInput = new Artifact(dir, rootDir);
    TreeNodeRepository repo = createTestTreeNodeRepository(false);
    TreeNode root = buildFromActionInputs(repo, dirInput);
    repo.computeMerkleDigests(root);
    Digest digest = digestUtil.compute(foo);
    Directory childDir =
        Directory.newBuilder()
            .addFiles(FileNode.newBuilder().setName("link").setDigest(digest).setIsExecutable(true))
            .build();
    Digest dirDigest = digestUtil.compute(childDir);
    Directory rootDir =
        Directory.newBuilder()
            .addDirectories(DirectoryNode.newBuilder().setName("dir").setDigest(dirDigest))
            .build();
    assertThat(repo.treeToDirectories(root)).containsExactly(rootDir, childDir);
  }

  @Test
  public void testRelativeFileSymlink() throws Exception {
    Path link = scratch.getFileSystem().getPath("/exec/root/link");
    Path target = scratch.file("/exec/root/target", "bla");
    link.createSymbolicLink(target.relativeTo(execRoot));
    Artifact linkInput = new Artifact(link, rootDir);
    TreeNodeRepository repo = createTestTreeNodeRepository(true);
    TreeNode root = buildFromActionInputs(repo, linkInput);
    repo.computeMerkleDigests(root);
    Directory rootDir =
        Directory.newBuilder()
            .addSymlinks(SymlinkNode.newBuilder().setName("link").setTarget("target"))
            .build();
    assertThat(repo.treeToDirectories(root)).containsExactly(rootDir);
  }

  @Test
  public void testRelativeFileSymlinkInDirectory() throws Exception {
    scratch.file("/exec/root/foo", "bla");
    Path dir = scratch.dir("/exec/root/dir");
    Path link = scratch.getFileSystem().getPath("/exec/root/dir/link");
    link.createSymbolicLink(PathFragment.create("../foo"));
    Artifact dirInput = new Artifact(dir, rootDir);
    TreeNodeRepository repo = createTestTreeNodeRepository(true);
    TreeNode root = buildFromActionInputs(repo, dirInput);
    repo.computeMerkleDigests(root);
    Directory childDir =
        Directory.newBuilder()
            .addSymlinks(SymlinkNode.newBuilder().setName("link").setTarget("../foo"))
            .build();
    Digest dirDigest = digestUtil.compute(childDir);
    Directory rootDir =
        Directory.newBuilder()
            .addDirectories(DirectoryNode.newBuilder().setName("dir").setDigest(dirDigest))
            .build();
    assertThat(repo.treeToDirectories(root)).containsExactly(rootDir, childDir);
  }

  @Test
  public void testDanglingSymlinkFail() throws Exception {
    Path link = scratch.getFileSystem().getPath("/exec/root/link");
    Path target = scratch.getFileSystem().getPath("/exec/root/target");
    link.createSymbolicLink(target.relativeTo(execRoot));
    Artifact linkInput = new Artifact(link, rootDir);
    TreeNodeRepository repo = createTestTreeNodeRepository(true);
    try {
      buildFromActionInputs(repo, linkInput);
      fail("Expected exception");
    } catch (Exception e) {
      assertThat(e).hasMessageThat().contains("dangling");
      assertThat(e).hasMessageThat().contains("/exec/root/link");
      assertThat(e).hasMessageThat().contains("target");
    }
  }

  @Test
  public void testDanglingSymlinkInDirectoryFail() throws Exception {
    scratch.getFileSystem().getPath("/exec/root/foo");
    Path dir = scratch.dir("/exec/root/dir");
    Path link = scratch.getFileSystem().getPath("/exec/root/dir/link");
    link.createSymbolicLink(PathFragment.create("../foo"));
    Artifact dirInput = new Artifact(dir, rootDir);
    TreeNodeRepository repo = createTestTreeNodeRepository(true);
    try {
      buildFromActionInputs(repo, dirInput);
      fail("Expected exception");
    } catch (Exception e) {
      assertThat(e).hasMessageThat().contains("Dangling");
      assertThat(e).hasMessageThat().contains("/exec/root/dir/link");
      assertThat(e).hasMessageThat().contains("../foo");
    }
  }

  @Test
  public void testAbsoluteDirectorySymlinkAsDirectory() throws Exception {
    Path dir = scratch.dir("/exec/root/dir");
    Path foo = scratch.file("/exec/root/dir/foo", "bla");
    Path link = scratch.getFileSystem().getPath("/exec/root/link");
    link.createSymbolicLink(dir);
    Artifact linkInput = new Artifact(link, rootDir);
    TreeNodeRepository repo = createTestTreeNodeRepository(true);
    TreeNode root = buildFromActionInputs(repo, linkInput);
    repo.computeMerkleDigests(root);
    Digest digest = digestUtil.compute(foo);
    Directory childDir =
        Directory.newBuilder()
            .addFiles(FileNode.newBuilder().setName("foo").setDigest(digest).setIsExecutable(true))
            .build();
    Digest dirDigest = digestUtil.compute(childDir);
    Directory rootDir =
        Directory.newBuilder()
            .addDirectories(DirectoryNode.newBuilder().setName("link").setDigest(dirDigest))
            .build();
    assertThat(repo.treeToDirectories(root)).containsExactly(rootDir, childDir);
  }

  @Test
  public void testAbsoluteDirectorySymlinkInDirectoryAsDirectory() throws Exception {
    Path dir = scratch.dir("/exec/root/dir");
    Path bardir = scratch.dir("/exec/root/bardir");
    Path foo = scratch.file("/exec/root/bardir/foo", "bla");
    Path link = scratch.getFileSystem().getPath("/exec/root/dir/link");
    link.createSymbolicLink(bardir);
    Artifact dirInput = new Artifact(dir, rootDir);
    TreeNodeRepository repo = createTestTreeNodeRepository(true);
    TreeNode root = buildFromActionInputs(repo, dirInput);
    repo.computeMerkleDigests(root);
    Directory barDir =
        Directory.newBuilder()
            .addFiles(
                FileNode.newBuilder()
                    .setName("foo")
                    .setDigest(digestUtil.compute(foo))
                    .setIsExecutable(true))
            .build();
    Digest barDigest = digestUtil.compute(barDir);
    Directory dirNode =
        Directory.newBuilder()
            .addDirectories(DirectoryNode.newBuilder().setName("link").setDigest(barDigest))
            .build();
    Digest dirDigest = digestUtil.compute(dirNode);
    Directory rootDir =
        Directory.newBuilder()
            .addDirectories(DirectoryNode.newBuilder().setName("dir").setDigest(dirDigest))
            .build();
    assertThat(repo.treeToDirectories(root)).containsExactly(rootDir, dirNode, barDir);
  }

  @Test
  public void testRelativeDirectorySymlinkAsDirectory() throws Exception {
    Path dir = scratch.dir("/exec/root/dir");
    Path foo = scratch.file("/exec/root/dir/foo", "bla");
    Path link = scratch.getFileSystem().getPath("/exec/root/link");
    link.createSymbolicLink(dir.relativeTo(execRoot));
    Artifact linkInput = new Artifact(link, rootDir);
    TreeNodeRepository repo = createTestTreeNodeRepository(false);
    TreeNode root = buildFromActionInputs(repo, linkInput);
    repo.computeMerkleDigests(root);
    Digest digest = digestUtil.compute(foo);
    Directory childDir =
        Directory.newBuilder()
            .addFiles(FileNode.newBuilder().setName("foo").setDigest(digest).setIsExecutable(true))
            .build();
    Digest dirDigest = digestUtil.compute(childDir);
    Directory rootDir =
        Directory.newBuilder()
            .addDirectories(DirectoryNode.newBuilder().setName("link").setDigest(dirDigest))
            .build();
    assertThat(repo.treeToDirectories(root)).containsExactly(rootDir, childDir);
  }

  @Test
  public void testRelativeDirectorySymlinkInDirectoryAsDirectory() throws Exception {
    Path dir = scratch.dir("/exec/root/dir");
    scratch.dir("/exec/root/bardir");
    Path foo = scratch.file("/exec/root/bardir/foo", "bla");
    Path link = scratch.getFileSystem().getPath("/exec/root/dir/link");
    link.createSymbolicLink(PathFragment.create("../bardir"));
    Artifact dirInput = new Artifact(dir, rootDir);
    TreeNodeRepository repo = createTestTreeNodeRepository(false);
    TreeNode root = buildFromActionInputs(repo, dirInput);
    repo.computeMerkleDigests(root);
    Directory barDir =
        Directory.newBuilder()
            .addFiles(
                FileNode.newBuilder()
                    .setName("foo")
                    .setDigest(digestUtil.compute(foo))
                    .setIsExecutable(true))
            .build();
    Digest barDigest = digestUtil.compute(barDir);
    Directory dirNode =
        Directory.newBuilder()
            .addDirectories(DirectoryNode.newBuilder().setName("link").setDigest(barDigest))
            .build();
    Digest dirDigest = digestUtil.compute(dirNode);
    Directory rootDir =
        Directory.newBuilder()
            .addDirectories(DirectoryNode.newBuilder().setName("dir").setDigest(dirDigest))
            .build();
    assertThat(repo.treeToDirectories(root)).containsExactly(rootDir, dirNode, barDir);
  }

  @Test
  public void testRelativeDirectorySymlink() throws Exception {
    Path dir = scratch.dir("/exec/root/dir");
    scratch.file("/exec/root/dir/foo", "bla");
    Path link = scratch.getFileSystem().getPath("/exec/root/link");
    link.createSymbolicLink(dir.relativeTo(execRoot));
    Artifact linkInput = new Artifact(link, rootDir);
    TreeNodeRepository repo = createTestTreeNodeRepository(true);
    TreeNode root = buildFromActionInputs(repo, linkInput);
    repo.computeMerkleDigests(root);
    Directory rootDir =
        Directory.newBuilder()
            .addSymlinks(SymlinkNode.newBuilder().setName("link").setTarget("dir"))
            .build();
    assertThat(repo.treeToDirectories(root)).containsExactly(rootDir);
  }

  @Test
  public void testRelativeDirectorySymlinkInDirectory() throws Exception {
    Path dir = scratch.dir("/exec/root/dir");
    scratch.dir("/exec/root/bardir");
    scratch.file("/exec/root/bardir/foo", "bla");
    Path link = scratch.getFileSystem().getPath("/exec/root/dir/link");
    link.createSymbolicLink(PathFragment.create("../bardir"));
    Artifact dirInput = new Artifact(dir, rootDir);
    TreeNodeRepository repo = createTestTreeNodeRepository(true);
    TreeNode root = buildFromActionInputs(repo, dirInput);
    repo.computeMerkleDigests(root);
    Directory dirNode =
        Directory.newBuilder()
            .addSymlinks(SymlinkNode.newBuilder().setName("link").setTarget("../bardir"))
            .build();
    Digest dirDigest = digestUtil.compute(dirNode);
    Directory rootDir =
        Directory.newBuilder()
            .addDirectories(DirectoryNode.newBuilder().setName("dir").setDigest(dirDigest))
            .build();
    assertThat(repo.treeToDirectories(root)).containsExactly(rootDir, dirNode);
  }
}
