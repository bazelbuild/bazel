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
package com.google.devtools.build.lib.actions;


import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.devtools.build.lib.actions.Action.MiddlemanType;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.actions.util.LabelArtifactOwner;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.rules.cpp.CppFileTypes;
import com.google.devtools.build.lib.rules.java.JavaSemantics;
import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

@RunWith(JUnit4.class)
public class ArtifactTest {
  private Scratch scratch;
  private Path execDir;
  private Root rootDir;

  @Before
  public final void setRootDir() throws Exception  {
    scratch = new Scratch();
    execDir = scratch.dir("/exec");
    rootDir = Root.asDerivedRoot(scratch.dir("/exec/root"));
  }

  @Test
  public void testConstruction_badRootDir() throws IOException {
    Path f1 = scratch.file("/exec/dir/file.ext");
    Path bogusDir = scratch.file("/exec/dir/bogus");
    try {
      new Artifact(f1, Root.asDerivedRoot(bogusDir), f1.relativeTo(execDir));
      fail("Expected IllegalArgumentException constructing artifact with a bad root dir");
    } catch (IllegalArgumentException expected) {}
  }

  @Test
  public void testEquivalenceRelation() throws Exception {
    PathFragment aPath = new PathFragment("src/a");
    PathFragment bPath = new PathFragment("src/b");
    assertEquals(new Artifact(aPath, rootDir),
                 new Artifact(aPath, rootDir));
    assertEquals(new Artifact(bPath, rootDir),
                 new Artifact(bPath, rootDir));
    assertFalse(new Artifact(aPath, rootDir).equals(
                new Artifact(bPath, rootDir)));
  }

  @Test
  public void testEmptyLabelIsNone() throws Exception {
    Artifact artifact = new Artifact(new PathFragment("src/a"), rootDir);
    assertThat(artifact.getOwnerLabel()).isNull();
  }

  @Test
  public void testComparison() throws Exception {
    PathFragment aPath = new PathFragment("src/a");
    PathFragment bPath = new PathFragment("src/b");
    Artifact aArtifact = new Artifact(aPath, rootDir);
    Artifact bArtifact = new Artifact(bPath, rootDir);
    assertEquals(-1, Artifact.EXEC_PATH_COMPARATOR.compare(aArtifact, bArtifact));
    assertEquals(0, Artifact.EXEC_PATH_COMPARATOR.compare(aArtifact, aArtifact));
    assertEquals(0, Artifact.EXEC_PATH_COMPARATOR.compare(bArtifact, bArtifact));
    assertEquals(1, Artifact.EXEC_PATH_COMPARATOR.compare(bArtifact, aArtifact));
  }

  @Test
  public void testRootPrefixedExecPath_normal() throws IOException {
    Path f1 = scratch.file("/exec/root/dir/file.ext");
    Artifact a1 = new Artifact(f1, rootDir, f1.relativeTo(execDir));
    assertEquals("root:dir/file.ext", Artifact.asRootPrefixedExecPath(a1));
  }

  @Test
  public void testRootPrefixedExecPath_noRoot() throws IOException {
    Path f1 = scratch.file("/exec/dir/file.ext");
    Artifact a1 = new Artifact(f1.relativeTo(execDir), Root.asDerivedRoot(execDir));
    assertEquals(":dir/file.ext", Artifact.asRootPrefixedExecPath(a1));
  }

  @Test
  public void testRootPrefixedExecPath_nullRootDir() throws IOException {
    Path f1 = scratch.file("/exec/dir/file.ext");
    try {
      new Artifact(f1, null, f1.relativeTo(execDir));
      fail("Expected IllegalArgumentException creating artifact with null root");
    } catch (IllegalArgumentException expected) {}
  }

  @Test
  public void testRootPrefixedExecPaths() throws IOException {
    Path f1 = scratch.file("/exec/root/dir/file1.ext");
    Path f2 = scratch.file("/exec/root/dir/dir/file2.ext");
    Path f3 = scratch.file("/exec/root/dir/dir/dir/file3.ext");
    Artifact a1 = new Artifact(f1, rootDir, f1.relativeTo(execDir));
    Artifact a2 = new Artifact(f2, rootDir, f2.relativeTo(execDir));
    Artifact a3 = new Artifact(f3, rootDir, f3.relativeTo(execDir));
    List<String> strings = new ArrayList<>();
    Artifact.addRootPrefixedExecPaths(Lists.newArrayList(a1, a2, a3), strings);
    assertThat(strings).containsExactly(
        "root:dir/file1.ext",
        "root:dir/dir/file2.ext",
        "root:dir/dir/dir/file3.ext").inOrder();
  }

  @Test
  public void testGetFilename() throws Exception {
    Root root = Root.asSourceRoot(scratch.dir("/foo"));
    Artifact javaFile = new Artifact(scratch.file("/foo/Bar.java"), root);
    Artifact generatedHeader = new Artifact(scratch.file("/foo/bar.proto.h"), root);
    Artifact generatedCc = new Artifact(scratch.file("/foo/bar.proto.cc"), root);
    Artifact aCPlusPlusFile = new Artifact(scratch.file("/foo/bar.cc"), root);
    assertTrue(JavaSemantics.JAVA_SOURCE.matches(javaFile.getFilename()));
    assertTrue(CppFileTypes.CPP_HEADER.matches(generatedHeader.getFilename()));
    assertTrue(CppFileTypes.CPP_SOURCE.matches(generatedCc.getFilename()));
    assertTrue(CppFileTypes.CPP_SOURCE.matches(aCPlusPlusFile.getFilename()));
  }

  @Test
  public void testMangledPath() {
    String path = "dir/sub_dir/name:end";
    assertEquals("dir_Ssub_Udir_Sname_Cend", Actions.escapedPath(path));
  }

  private List<Artifact> getFooBarArtifacts(MutableActionGraph actionGraph, boolean collapsedList)
      throws Exception {
    Root root = Root.asSourceRoot(scratch.dir("/foo"));
    Artifact aHeader1 = new Artifact(scratch.file("/foo/bar1.h"), root);
    Artifact aHeader2 = new Artifact(scratch.file("/foo/bar2.h"), root);
    Artifact aHeader3 = new Artifact(scratch.file("/foo/bar3.h"), root);
    Artifact middleman = new Artifact(new PathFragment("middleman"),
        Root.middlemanRoot(scratch.dir("/foo"), scratch.dir("/foo/out")));
    actionGraph.registerAction(new MiddlemanAction(ActionsTestUtil.NULL_ACTION_OWNER,
        ImmutableList.of(aHeader1, aHeader2, aHeader3), middleman, "desc",
        MiddlemanType.AGGREGATING_MIDDLEMAN));
    return collapsedList ? Lists.newArrayList(aHeader1, middleman) :
        Lists.newArrayList(aHeader1, aHeader2, middleman);
  }

  @Test
  public void testAddExecPaths() throws Exception {
    List<String> paths = new ArrayList<>();
    MutableActionGraph actionGraph = new MapBasedActionGraph();
    Artifact.addExecPaths(getFooBarArtifacts(actionGraph, false), paths);
    assertThat(paths).containsExactlyElementsIn(ImmutableList.of("bar1.h", "bar2.h"));
  }

  @Test
  public void testAddExpandedExecPathStrings() throws Exception {
    List<String> paths = new ArrayList<>();
    MutableActionGraph actionGraph = new MapBasedActionGraph();
    Artifact.addExpandedExecPathStrings(getFooBarArtifacts(actionGraph, true), paths,
        ActionInputHelper.actionGraphArtifactExpander(actionGraph));
    assertThat(paths).containsExactly("bar1.h", "bar1.h", "bar2.h", "bar3.h");
  }

  @Test
  public void testAddExpandedExecPaths() throws Exception {
    List<PathFragment> paths = new ArrayList<>();
    MutableActionGraph actionGraph = new MapBasedActionGraph();
    Artifact.addExpandedExecPaths(getFooBarArtifacts(actionGraph, true), paths,
        ActionInputHelper.actionGraphArtifactExpander(actionGraph));
    assertThat(paths).containsExactly(
        new PathFragment("bar1.h"),
        new PathFragment("bar1.h"),
        new PathFragment("bar2.h"),
        new PathFragment("bar3.h"));
  }

  @Test
  public void testAddExpandedArtifacts() throws Exception {
    List<ArtifactFile> expanded = new ArrayList<>();
    MutableActionGraph actionGraph = new MapBasedActionGraph();
    List<Artifact> original = getFooBarArtifacts(actionGraph, true);
    Artifact.addExpandedArtifacts(original, expanded,
        ActionInputHelper.actionGraphArtifactExpander(actionGraph));

    List<Artifact> manuallyExpanded = new ArrayList<>();
    for (Artifact artifact : original) {
      Action action = actionGraph.getGeneratingAction(artifact);
      if (artifact.isMiddlemanArtifact()) {
        Iterables.addAll(manuallyExpanded, action.getInputs());
      } else {
        manuallyExpanded.add(artifact);
      }
    }
    assertThat(expanded).containsExactlyElementsIn(manuallyExpanded);
  }

  @Test
  public void testAddExecPathsNewActionGraph() throws Exception {
    List<String> paths = new ArrayList<>();
    MutableActionGraph actionGraph = new MapBasedActionGraph();
    Artifact.addExecPaths(getFooBarArtifacts(actionGraph, false), paths);
    assertThat(paths).containsExactlyElementsIn(ImmutableList.of("bar1.h", "bar2.h"));
  }

  @Test
  public void testAddExpandedExecPathStringsNewActionGraph() throws Exception {
    List<String> paths = new ArrayList<>();
    MutableActionGraph actionGraph = new MapBasedActionGraph();
    Artifact.addExpandedExecPathStrings(getFooBarArtifacts(actionGraph, true), paths,
        ActionInputHelper.actionGraphArtifactExpander(actionGraph));
    assertThat(paths).containsExactly("bar1.h", "bar1.h", "bar2.h", "bar3.h");
  }

  @Test
  public void testAddExpandedExecPathsNewActionGraph() throws Exception {
    List<PathFragment> paths = new ArrayList<>();
    MutableActionGraph actionGraph = new MapBasedActionGraph();
    Artifact.addExpandedExecPaths(getFooBarArtifacts(actionGraph, true), paths,
        ActionInputHelper.actionGraphArtifactExpander(actionGraph));
    assertThat(paths).containsExactly(
        new PathFragment("bar1.h"),
        new PathFragment("bar1.h"),
        new PathFragment("bar2.h"),
        new PathFragment("bar3.h"));
  }

  // TODO consider tests for the future
  @Test
  public void testAddExpandedArtifactsNewActionGraph() throws Exception {
    List<ArtifactFile> expanded = new ArrayList<>();
    MutableActionGraph actionGraph = new MapBasedActionGraph();
    List<Artifact> original = getFooBarArtifacts(actionGraph, true);
    Artifact.addExpandedArtifacts(original, expanded,
        ActionInputHelper.actionGraphArtifactExpander(actionGraph));

    List<Artifact> manuallyExpanded = new ArrayList<>();
    for (Artifact artifact : original) {
      Action action = actionGraph.getGeneratingAction(artifact);
      if (artifact.isMiddlemanArtifact()) {
        Iterables.addAll(manuallyExpanded, action.getInputs());
      } else {
        manuallyExpanded.add(artifact);
      }
    }
    assertThat(expanded).containsExactlyElementsIn(manuallyExpanded);
  }

  @Test
  public void testRootRelativePathIsSameAsExecPath() throws Exception {
    Root root = Root.asSourceRoot(scratch.dir("/foo"));
    Artifact a = new Artifact(scratch.file("/foo/bar1.h"), root);
    assertSame(a.getExecPath(), a.getRootRelativePath());
  }

  @Test
  public void testToDetailString() throws Exception {
    Artifact a = new Artifact(scratch.file("/a/b/c"), Root.asDerivedRoot(scratch.dir("/a/b")),
        new PathFragment("b/c"));
    assertEquals("[[/a]b]c", a.toDetailString());
  }

  @Test
  public void testWeirdArtifact() throws Exception {
    try {
      new Artifact(scratch.file("/a/b/c"), Root.asDerivedRoot(scratch.dir("/a")),
          new PathFragment("c"));
      fail();
    } catch (IllegalArgumentException e) {
      assertThat(e).hasMessage(
          "c: illegal execPath doesn't end with b/c at /a/b/c with root /a[derived]");
    }
  }

  @Test
  public void testSerializeToString() throws Exception {
    assertEquals("b/c /3",
        new Artifact(scratch.file("/a/b/c"),
            Root.asDerivedRoot(scratch.dir("/a"))).serializeToString());
  }

  @Test
  public void testSerializeToStringWithExecPath() throws Exception {
    Path path = scratch.file("/aaa/bbb/ccc");
    Root root = Root.asDerivedRoot(scratch.dir("/aaa/bbb"));
    PathFragment execPath = new PathFragment("bbb/ccc");

    assertEquals("bbb/ccc /3", new Artifact(path, root, execPath).serializeToString());
  }

  @Test
  public void testSerializeToStringWithOwner() throws Exception {
    assertEquals("b/c /3 //foo:bar",
        new Artifact(scratch.file("/aa/b/c"), Root.asDerivedRoot(scratch.dir("/aa")),
            new PathFragment("b/c"),
            new LabelArtifactOwner(Label.parseAbsoluteUnchecked("//foo:bar"))).serializeToString());
  }

  @Test
  public void testLongDirname() throws Exception {
    String dirName = createDirNameArtifact().getDirname();

    assertThat(dirName).isEqualTo("aaa/bbb/ccc");
  }

  @Test
  public void testDirnameInExecutionDir() throws Exception {
    Artifact artifact = new Artifact(scratch.file("/foo/bar.txt"),
        Root.asDerivedRoot(scratch.dir("/foo")));

    assertThat(artifact.getDirname()).isEqualTo(".");
  }

  @Test
  public void testCanConstructPathFromDirAndFilename() throws Exception {
    Artifact artifact = createDirNameArtifact();
    String constructed =
        String.format("%s/%s", artifact.getDirname(), artifact.getFilename());

    assertThat(constructed).isEqualTo("aaa/bbb/ccc/ddd");
  }

  @Test
  public void testIsSourceArtifact() throws Exception {
    assertThat(
        new Artifact(scratch.file("/src/foo.cc"), Root.asSourceRoot(scratch.dir("/")),
            new PathFragment("src/foo.cc"))
            .isSourceArtifact())
        .isTrue();
    assertThat(
        new Artifact(scratch.file("/genfiles/aaa/bar.out"),
            Root.asDerivedRoot(scratch.dir("/genfiles"), scratch.dir("/genfiles/aaa")))
            .isSourceArtifact())
        .isFalse();

  }

  @Test
  public void testGetRoot() throws Exception {
    Root root = Root.asDerivedRoot(scratch.dir("/newRoot"));
    assertThat(new Artifact(scratch.file("/newRoot/foo"), root).getRoot()).isEqualTo(root);
  }

  private Artifact createDirNameArtifact() throws Exception {
    return new Artifact(scratch.file("/aaa/bbb/ccc/ddd"), Root.asDerivedRoot(scratch.dir("/")));
  }
}
