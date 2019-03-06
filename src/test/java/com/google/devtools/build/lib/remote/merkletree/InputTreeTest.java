// Copyright 2019 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.remote.merkletree;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.ActionInputHelper;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.cache.VirtualActionInput;
import com.google.devtools.build.lib.clock.JavaClock;
import com.google.devtools.build.lib.remote.merkletree.InputTree.DirectoryNode;
import com.google.devtools.build.lib.remote.merkletree.InputTree.FileNode;
import com.google.devtools.build.lib.remote.merkletree.InputTree.Node;
import com.google.devtools.build.lib.remote.util.DigestUtil;
import com.google.devtools.build.lib.remote.util.StringActionInput;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.SortedMap;
import java.util.TreeMap;
import java.util.stream.Collectors;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link InputTree}. */
@RunWith(JUnit4.class)
public class InputTreeTest {

  private Path execRoot;
  private ArtifactRoot artifactRoot;
  private DigestUtil digestUtil;

  @Before
  public void setup() {
    FileSystem fs = new InMemoryFileSystem(new JavaClock(), DigestHashFunction.SHA256);
    execRoot = fs.getPath("/exec");
    artifactRoot = ArtifactRoot.asDerivedRoot(execRoot, execRoot.getRelative("srcs"));
    digestUtil = new DigestUtil(fs.getDigestFunction());
  }

  @Test
  public void emptyTreeShouldWork() throws Exception {
    InputTree tree =
        InputTree.build(
            new TreeMap<>(),
            new StaticMetadataProvider(Collections.emptyMap()),
            execRoot,
            digestUtil);
    assertThat(directoryNodesAtDepth(tree, 0)).isEmpty();
    assertThat(fileNodesAtDepth(tree, 0)).isEmpty();
  }

  @Test
  public void buildingATreeOfFilesShouldWork() throws Exception {
    SortedMap<PathFragment, ActionInput> sortedInputs = new TreeMap<>();
    Map<ActionInput, FileArtifactValue> metadata = new HashMap<>();

    Artifact foo = addFile("srcs/foo.cc", "foo", sortedInputs, metadata);
    Artifact bar = addFile("srcs/bar.cc", "bar", sortedInputs, metadata);
    Artifact buzz = addFile("srcs/fizz/buzz.cc", "buzz", sortedInputs, metadata);

    InputTree tree =
        InputTree.build(sortedInputs, new StaticMetadataProvider(metadata), execRoot, digestUtil);
    assertLexicographicalOrder(tree);

    assertThat(directoriesAtDepth(0, tree)).containsExactly("srcs");
    assertThat(directoriesAtDepth(1, tree)).containsExactly("fizz");
    assertThat(directoriesAtDepth(2, tree)).isEmpty();

    FileNode expectedFooNode = new FileNode("foo.cc", foo, digestUtil.computeAsUtf8("foo"));
    FileNode expectedBarNode = new FileNode("bar.cc", bar, digestUtil.computeAsUtf8("bar"));
    FileNode expectedBuzzNode = new FileNode("buzz.cc", buzz, digestUtil.computeAsUtf8("buzz"));
    assertThat(fileNodesAtDepth(tree, 0)).isEmpty();
    assertThat(fileNodesAtDepth(tree, 1)).containsExactly(expectedFooNode, expectedBarNode);
    assertThat(fileNodesAtDepth(tree, 2)).containsExactly(expectedBuzzNode);
  }

  @Test
  public void virtualActionInputShouldWork() throws Exception {
    SortedMap<PathFragment, ActionInput> sortedInputs = new TreeMap<>();
    Map<ActionInput, FileArtifactValue> metadata = new HashMap<>();

    Artifact foo = addFile("srcs/foo.cc", "foo", sortedInputs, metadata);
    VirtualActionInput bar = addVirtualFile("srcs/bar.cc", "bar", sortedInputs);

    InputTree tree =
        InputTree.build(sortedInputs, new StaticMetadataProvider(metadata), execRoot, digestUtil);
    assertLexicographicalOrder(tree);

    assertThat(directoriesAtDepth(0, tree)).containsExactly("srcs");
    assertThat(directoriesAtDepth(1, tree)).isEmpty();

    FileNode expectedFooNode = new FileNode("foo.cc", foo, digestUtil.computeAsUtf8("foo"));
    FileNode expectedBarNode = new FileNode("bar.cc", bar, digestUtil.computeAsUtf8("bar"));
    assertThat(fileNodesAtDepth(tree, 0)).isEmpty();
    assertThat(fileNodesAtDepth(tree, 1)).containsExactly(expectedFooNode, expectedBarNode);
  }

  @Test
  public void directoryInputShouldBeExpanded() throws Exception {
    // Test that directory inputs are fully expanded and added to the input tree.
    // Note that this test is not about tree artifacts, but normal artifacts that point to
    // a directory on disk.

    SortedMap<PathFragment, ActionInput> sortedInputs = new TreeMap<>();
    Map<ActionInput, FileArtifactValue> metadata = new HashMap<>();

    Artifact foo = addFile("srcs/foo.cc", "foo", sortedInputs, metadata);

    Path dirPath = execRoot.getRelative("srcs/dir");
    dirPath.createDirectoryAndParents();

    Path barPath = dirPath.getRelative("bar.cc");
    FileSystemUtils.writeContentAsLatin1(barPath, "bar");
    ActionInput bar = ActionInputHelper.fromPath(barPath.relativeTo(execRoot));
    metadata.put(bar, FileArtifactValue.createShareable(barPath));

    dirPath.getRelative("fizz").createDirectoryAndParents();
    Path buzzPath = dirPath.getRelative("fizz/buzz.cc");
    FileSystemUtils.writeContentAsLatin1(dirPath.getRelative("fizz/buzz.cc"), "buzz");
    ActionInput buzz = ActionInputHelper.fromPath(buzzPath.relativeTo(execRoot));
    metadata.put(buzz, FileArtifactValue.createShareable(buzzPath));

    Artifact dir = new Artifact(dirPath, artifactRoot);
    sortedInputs.put(dirPath.relativeTo(execRoot), dir);
    metadata.put(dir, FileArtifactValue.createShareable(dirPath));

    InputTree tree =
        InputTree.build(sortedInputs, new StaticMetadataProvider(metadata), execRoot, digestUtil);
    assertLexicographicalOrder(tree);

    assertThat(directoriesAtDepth(0, tree)).containsExactly("srcs");
    assertThat(directoriesAtDepth(1, tree)).containsExactly("dir");
    assertThat(directoriesAtDepth(2, tree)).containsExactly("fizz");
    assertThat(directoriesAtDepth(3, tree)).isEmpty();

    FileNode expectedFooNode = new FileNode("foo.cc", foo, digestUtil.computeAsUtf8("foo"));
    FileNode expectedBarNode = new FileNode("bar.cc", bar, digestUtil.computeAsUtf8("bar"));
    FileNode expectedBuzzNode = new FileNode("buzz.cc", buzz, digestUtil.computeAsUtf8("buzz"));
    assertThat(fileNodesAtDepth(tree, 0)).isEmpty();
    assertThat(fileNodesAtDepth(tree, 1)).containsExactly(expectedFooNode);
    assertThat(fileNodesAtDepth(tree, 2)).containsExactly(expectedBarNode);
    assertThat(fileNodesAtDepth(tree, 3)).containsExactly(expectedBuzzNode);
  }

  private Artifact addFile(
      String path,
      String content,
      SortedMap<PathFragment, ActionInput> sortedInputs,
      Map<ActionInput, FileArtifactValue> metadata)
      throws IOException {
    Path p = execRoot.getRelative(path);
    p.getParentDirectory().createDirectoryAndParents();
    FileSystemUtils.writeContentAsLatin1(p, content);
    Artifact a = new Artifact(p, artifactRoot);

    sortedInputs.put(PathFragment.create(path), a);
    metadata.put(a, FileArtifactValue.create(a));
    return a;
  }

  private VirtualActionInput addVirtualFile(
      String path, String content, SortedMap<PathFragment, ActionInput> sortedInputs) {
    VirtualActionInput input = new StringActionInput(content, PathFragment.create(path));
    sortedInputs.put(PathFragment.create(path), input);
    return input;
  }

  private static void assertLexicographicalOrder(InputTree tree) {
    int depth = 0;
    while (true) {
      // String::compareTo implements lexicographical order
      List<String> files = filesAtDepth(depth, tree);
      assertThat(files).isStrictlyOrdered();
      List<String> directories = directoriesAtDepth(depth, tree);
      assertThat(directories).isStrictlyOrdered();
      if (directories.isEmpty()) {
        break;
      }
      depth++;
    }
  }

  private static List<String> directoriesAtDepth(int depth, InputTree tree) {
    return asPathSegments(directoryNodesAtDepth(tree, depth));
  }

  private static List<String> filesAtDepth(int depth, InputTree tree) {
    return asPathSegments(fileNodesAtDepth(tree, depth));
  }

  private static List<String> asPathSegments(List<? extends Node> nodes) {
    return nodes.stream().map(Node::getPathSegment).collect(Collectors.toList());
  }

  private static List<DirectoryNode> directoryNodesAtDepth(InputTree tree, int depth) {
    List<DirectoryNode> directoryNodes = new ArrayList<>();
    tree.visit(
        (PathFragment dirname, List<FileNode> files, List<DirectoryNode> dirs) -> {
          int currDepth = dirname.segmentCount();
          if (currDepth == depth) {
            directoryNodes.addAll(dirs);
          }
        });
    return directoryNodes;
  }

  private static List<FileNode> fileNodesAtDepth(InputTree tree, int depth) {
    List<FileNode> fileNodes = new ArrayList<>();
    tree.visit(
        (PathFragment dirname, List<FileNode> files, List<DirectoryNode> dirs) -> {
          int currDepth = dirname.segmentCount();
          if (currDepth == depth) {
            fileNodes.addAll(files);
          }
        });
    return fileNodes;
  }
}
