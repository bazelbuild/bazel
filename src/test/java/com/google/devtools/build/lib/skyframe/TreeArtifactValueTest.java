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
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.fail;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.actions.ArtifactRoot;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.actions.FileArtifactValue.RemoteFileArtifactValue;
import com.google.devtools.build.lib.actions.cache.MetadataInjector;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.vfs.Dirent;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.inmemoryfs.InMemoryFileSystem;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameter;
import org.junit.runners.Parameterized.Parameters;

/** Tests for {@link TreeArtifactValue}. */
@RunWith(JUnit4.class)
public final class TreeArtifactValueTest {

  private final Scratch scratch = new Scratch();

  @Test
  public void visitTree_visitsEachChild() throws Exception {
    Path treeDir = scratch.dir("tree");
    scratch.file("tree/file1");
    scratch.file("tree/a/file2");
    scratch.file("tree/a/b/file3");
    scratch.resolve("tree/file_link").createSymbolicLink(PathFragment.create("file1"));
    scratch.resolve("tree/a/dir_link").createSymbolicLink(PathFragment.create("c"));
    scratch.resolve("tree/a/b/dangling_link").createSymbolicLink(PathFragment.create("?"));
    List<Pair<PathFragment, Dirent.Type>> children = new ArrayList<>();

    TreeArtifactValue.visitTree(treeDir, (child, type) -> children.add(Pair.of(child, type)));

    assertThat(children)
        .containsExactly(
            Pair.of(PathFragment.create("a"), Dirent.Type.DIRECTORY),
            Pair.of(PathFragment.create("a/b"), Dirent.Type.DIRECTORY),
            Pair.of(PathFragment.create("file1"), Dirent.Type.FILE),
            Pair.of(PathFragment.create("a/file2"), Dirent.Type.FILE),
            Pair.of(PathFragment.create("a/b/file3"), Dirent.Type.FILE),
            Pair.of(PathFragment.create("file_link"), Dirent.Type.SYMLINK),
            Pair.of(PathFragment.create("a/dir_link"), Dirent.Type.SYMLINK),
            Pair.of(PathFragment.create("a/b/dangling_link"), Dirent.Type.SYMLINK));
  }

  @Test
  public void visitTree_throwsOnUnknownDirentType() {
    FileSystem fs =
        new InMemoryFileSystem() {
          @Override
          public ImmutableList<Dirent> readdir(Path path, boolean followSymlinks) {
            return ImmutableList.of(new Dirent("?", Dirent.Type.UNKNOWN));
          }
        };
    Path treeDir = fs.getPath("/tree");

    Exception e =
        assertThrows(
            IOException.class,
            () ->
                TreeArtifactValue.visitTree(
                    treeDir, (child, type) -> fail("Should not be called")));
    assertThat(e).hasMessageThat().contains("Could not determine type of file for ? under /tree");
  }

  @Test
  public void visitTree_propagatesIoExceptionFromVisitor() throws Exception {
    Path treeDir = scratch.dir("tree");
    scratch.file("tree/file");
    IOException e = new IOException("From visitor");

    IOException thrown =
        assertThrows(
            IOException.class,
            () ->
                TreeArtifactValue.visitTree(
                    treeDir,
                    (child, type) -> {
                      assertThat(child).isEqualTo(PathFragment.create("file"));
                      assertThat(type).isEqualTo(Dirent.Type.FILE);
                      throw e;
                    }));
    assertThat(thrown).isSameInstanceAs(e);
  }

  @Test
  public void visitTree_pemitsUpLevelSymlinkInsideTree() throws Exception {
    Path treeDir = scratch.dir("tree");
    scratch.file("tree/file");
    scratch.dir("tree/a");
    scratch.resolve("tree/a/up_link").createSymbolicLink(PathFragment.create("../file"));
    List<Pair<PathFragment, Dirent.Type>> children = new ArrayList<>();

    TreeArtifactValue.visitTree(treeDir, (child, type) -> children.add(Pair.of(child, type)));

    assertThat(children)
        .containsExactly(
            Pair.of(PathFragment.create("file"), Dirent.Type.FILE),
            Pair.of(PathFragment.create("a"), Dirent.Type.DIRECTORY),
            Pair.of(PathFragment.create("a/up_link"), Dirent.Type.SYMLINK));
  }

  @Test
  public void visitTree_permitsAbsoluteSymlink() throws Exception {
    Path treeDir = scratch.dir("tree");
    scratch.resolve("tree/absolute_link").createSymbolicLink(PathFragment.create("/tmp"));
    List<Pair<PathFragment, Dirent.Type>> children = new ArrayList<>();

    TreeArtifactValue.visitTree(treeDir, (child, type) -> children.add(Pair.of(child, type)));

    assertThat(children)
        .containsExactly(Pair.of(PathFragment.create("absolute_link"), Dirent.Type.SYMLINK));
  }

  @Test
  public void visitTree_throwsOnSymlinkPointingOutsideTree() throws Exception {
    Path treeDir = scratch.dir("tree");
    scratch.file("outside");
    scratch.resolve("tree/link").createSymbolicLink(PathFragment.create("../outside"));

    Exception e =
        assertThrows(
            IOException.class,
            () ->
                TreeArtifactValue.visitTree(
                    treeDir, (child, type) -> fail("Should not be called")));
    assertThat(e).hasMessageThat().contains("/tree/link pointing to ../outside");
  }

  @Test
  public void visitTree_throwsOnSymlinkTraversingOutsideThenBackInsideTree() throws Exception {
    Path treeDir = scratch.dir("tree");
    scratch.file("tree/file");
    scratch.resolve("tree/link").createSymbolicLink(PathFragment.create("../tree/file"));

    Exception e =
        assertThrows(
            IOException.class,
            () ->
                TreeArtifactValue.visitTree(
                    treeDir,
                    (child, type) -> {
                      assertThat(child).isEqualTo(PathFragment.create("file"));
                      assertThat(type).isEqualTo(Dirent.Type.FILE);
                    }));
    assertThat(e).hasMessageThat().contains("/tree/link pointing to ../tree/file");
  }

  /** Parameterized tests for {@link TreeArtifactValue.MultiBuilder}. */
  @RunWith(Parameterized.class)
  public static final class MultiBuilderTest {

    private static final ArtifactRoot ROOT =
        ArtifactRoot.asDerivedRoot(new InMemoryFileSystem().getPath("/root"), "bin");

    private enum MultiBuilderType {
      BASIC {
        @Override
        TreeArtifactValue.MultiBuilder newMultiBuilder() {
          return TreeArtifactValue.newMultiBuilder();
        }
      },
      CONCURRENT {
        @Override
        TreeArtifactValue.MultiBuilder newMultiBuilder() {
          return TreeArtifactValue.newConcurrentMultiBuilder();
        }
      };

      abstract TreeArtifactValue.MultiBuilder newMultiBuilder();
    }

    @Parameter public MultiBuilderType multiBuilderType;
    private final FakeMetadataInjector metadataInjector = new FakeMetadataInjector();

    @Parameters(name = "{0}")
    public static MultiBuilderType[] params() {
      return MultiBuilderType.values();
    }

    @Test
    public void empty() {
      TreeArtifactValue.MultiBuilder treeArtifacts = multiBuilderType.newMultiBuilder();

      treeArtifacts.injectTo(metadataInjector);

      assertThat(metadataInjector.injectedTreeArtifacts).isEmpty();
    }

    @Test
    public void singleTreeArtifact() {
      TreeArtifactValue.MultiBuilder treeArtifacts = multiBuilderType.newMultiBuilder();
      SpecialArtifact parent = createTreeArtifact("tree");
      TreeFileArtifact child1 = TreeFileArtifact.createTreeOutput(parent, "child1");
      TreeFileArtifact child2 = TreeFileArtifact.createTreeOutput(parent, "child2");

      treeArtifacts.putChild(child1, metadataWithId(1));
      treeArtifacts.putChild(child2, metadataWithId(2));
      treeArtifacts.injectTo(metadataInjector);

      assertThat(metadataInjector.injectedTreeArtifacts)
          .containsExactly(
              parent,
              TreeArtifactValue.create(
                  ImmutableMap.of(child1, metadataWithId(1), child2, metadataWithId(2))));
    }

    @Test
    public void multipleTreeArtifacts() {
      TreeArtifactValue.MultiBuilder treeArtifacts = multiBuilderType.newMultiBuilder();
      SpecialArtifact parent1 = createTreeArtifact("tree1");
      TreeFileArtifact parent1Child1 = TreeFileArtifact.createTreeOutput(parent1, "child1");
      TreeFileArtifact parent1Child2 = TreeFileArtifact.createTreeOutput(parent1, "child2");
      SpecialArtifact parent2 = createTreeArtifact("tree2");
      TreeFileArtifact parent2Child = TreeFileArtifact.createTreeOutput(parent2, "child");

      treeArtifacts.putChild(parent1Child1, metadataWithId(1));
      treeArtifacts.putChild(parent2Child, metadataWithId(3));
      treeArtifacts.putChild(parent1Child2, metadataWithId(2));
      treeArtifacts.injectTo(metadataInjector);

      assertThat(metadataInjector.injectedTreeArtifacts)
          .containsExactly(
              parent1,
              TreeArtifactValue.create(
                  ImmutableMap.of(
                      parent1Child1, metadataWithId(1), parent1Child2, metadataWithId(2))),
              parent2,
              TreeArtifactValue.create(ImmutableMap.of(parent2Child, metadataWithId(3))));
    }

    private static SpecialArtifact createTreeArtifact(String execPath) {
      return ActionsTestUtil.createTreeArtifactWithGeneratingAction(
          ROOT, PathFragment.create(execPath));
    }

    private static FileArtifactValue metadataWithId(int id) {
      return new RemoteFileArtifactValue(new byte[] {(byte) id}, id, id);
    }

    private static final class FakeMetadataInjector implements MetadataInjector {
      private final Map<SpecialArtifact, TreeArtifactValue> injectedTreeArtifacts = new HashMap<>();

      @Override
      public void injectFile(Artifact output, FileArtifactValue metadata) {
        throw new UnsupportedOperationException();
      }

      @Override
      public void injectDirectory(
          SpecialArtifact output, Map<TreeFileArtifact, FileArtifactValue> children) {
        injectedTreeArtifacts.put(output, TreeArtifactValue.create(children));
      }

      @Override
      public void markOmitted(Artifact output) {
        throw new UnsupportedOperationException();
      }
    }
  }
}
