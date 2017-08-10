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
import static org.junit.Assert.fail;

import com.google.common.base.Function;
import com.google.common.collect.ImmutableList;
import com.google.common.testing.NullPointerTester;
import com.google.devtools.build.lib.actions.Artifact.ArtifactExpander;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifactType;
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.analysis.actions.CommandLine;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine.CustomArgv;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine.CustomMultiArgv;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine.VectorArg;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.Collection;
import javax.annotation.Nullable;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for CustomCommandLine.
 */
@RunWith(JUnit4.class)
public class CustomCommandLineTest {

  private Scratch scratch;
  private Root rootDir;
  private Artifact artifact1;
  private Artifact artifact2;

  @Before
  public void createArtifacts() throws Exception  {
    scratch = new Scratch();
    rootDir = Root.asDerivedRoot(scratch.dir("/exec/root"));
    artifact1 = new Artifact(scratch.file("/exec/root/dir/file1.txt"), rootDir);
    artifact2 = new Artifact(scratch.file("/exec/root/dir/file2.txt"), rootDir);
  }

  @Test
  public void testStringArgs() {
    CustomCommandLine cl = CustomCommandLine.builder().add("--arg1").add("--arg2").build();
    assertThat(cl.arguments()).isEqualTo(ImmutableList.of("--arg1", "--arg2"));
  }

  @Test
  public void testLabelArgs() throws LabelSyntaxException {
    CustomCommandLine cl = CustomCommandLine.builder().add(Label.parseAbsolute("//a:b")).build();
    assertThat(cl.arguments()).isEqualTo(ImmutableList.of("//a:b"));
  }

  @Test
  public void testStringsArgs() {
    CustomCommandLine cl = CustomCommandLine.builder().add("--arg",
        ImmutableList.of("a", "b")).build();
    assertThat(cl.arguments()).isEqualTo(ImmutableList.of("--arg", "a", "b"));
  }

  @Test
  public void testArtifactJoinStringArgs() {
    CustomCommandLine cl =
        CustomCommandLine.builder()
            .add("--path", VectorArg.of(ImmutableList.of("foo", "bar")).joinWith(":"))
            .build();
    assertThat(cl.arguments()).isEqualTo(ImmutableList.of("--path", "foo:bar"));
  }

  @Test
  public void testJoinValues() {
    CustomCommandLine cl =
        CustomCommandLine.builder()
            .add(
                "--path",
                VectorArg.of(ImmutableList.of("foo", "bar", "baz"))
                    .joinWith(":")
                    .mapEach(
                        new Function<String, String>() {
                          @Nullable
                          @Override
                          public String apply(@Nullable String s) {
                            return s.toUpperCase();
                          }
                        }))
            .build();
    assertThat(cl.arguments()).isEqualTo(ImmutableList.of("--path", "FOO:BAR:BAZ"));
  }

  @Test
  public void testArtifactExecPathArgs() {
    CustomCommandLine cl = CustomCommandLine.builder().add("--path", artifact1).build();
    assertThat(cl.arguments()).isEqualTo(ImmutableList.of("--path", "dir/file1.txt"));
  }

  @Test
  public void testArtifactExecPathsArgs() {
    CustomCommandLine cl =
        CustomCommandLine.builder().add("--path", ImmutableList.of(artifact1, artifact2)).build();
    assertThat(cl.arguments())
        .isEqualTo(ImmutableList.of("--path", "dir/file1.txt", "dir/file2.txt"));
  }

  @Test
  public void testNestedSetArtifactExecPathsArgs() {
    CustomCommandLine cl =
        CustomCommandLine.builder()
            .add(NestedSetBuilder.<Artifact>stableOrder().add(artifact1).add(artifact2).build())
            .build();
    assertThat(cl.arguments()).isEqualTo(ImmutableList.of("dir/file1.txt", "dir/file2.txt"));
  }

  @Test
  public void testArtifactJoinExecPathArgs() {
    CustomCommandLine cl =
        CustomCommandLine.builder()
            .add("--path", VectorArg.of(ImmutableList.of(artifact1, artifact2)).joinWith(":"))
            .build();
    assertThat(cl.arguments()).isEqualTo(ImmutableList.of("--path", "dir/file1.txt:dir/file2.txt"));
  }

  @Test
  public void testPathArgs() {
    CustomCommandLine cl = CustomCommandLine.builder().add(artifact1.getExecPath()).build();
    assertThat(cl.arguments()).isEqualTo(ImmutableList.of("dir/file1.txt"));
  }

  @Test
  public void testJoinPathArgs() {
    CustomCommandLine cl =
        CustomCommandLine.builder()
            .add(
                VectorArg.of(ImmutableList.of(artifact1.getExecPath(), artifact2.getExecPath()))
                    .joinWith(":"))
            .build();
    assertThat(cl.arguments()).isEqualTo(ImmutableList.of("dir/file1.txt:dir/file2.txt"));
  }

  @Test
  public void testPathsArgs() {
    CustomCommandLine cl =
        CustomCommandLine.builder()
            .addFormatted("%s:%s", artifact1.getExecPath(), artifact1.getRootRelativePath())
            .build();
    assertThat(cl.arguments()).isEqualTo(ImmutableList.of("dir/file1.txt:dir/file1.txt"));
  }

  @Test
  public void testCustomArgs() {
    CustomCommandLine cl = CustomCommandLine.builder().add(new CustomArgv() {
      @Override
      public String argv() {
        return "--arg";
      }
    }).build();
    assertThat(cl.arguments()).isEqualTo(ImmutableList.of("--arg"));
  }

  @Test
  public void testCustomMultiArgs() {
    CustomCommandLine cl = CustomCommandLine.builder().add(new CustomMultiArgv() {
      @Override
      public ImmutableList<String> argv() {
        return ImmutableList.of("--arg1", "--arg2");
      }
    }).build();
    assertThat(cl.arguments()).isEqualTo(ImmutableList.of("--arg1", "--arg2"));
  }

  @Test
  public void testCombinedArgs() {
    CustomCommandLine cl =
        CustomCommandLine.builder()
            .add("--arg")
            .add("--args", ImmutableList.of("abc"))
            .add("--path1", ImmutableList.of(artifact1))
            .add("--path2", artifact2)
            .build();
    assertThat(cl.arguments())
        .isEqualTo(
            ImmutableList.of(
                "--arg", "--args", "abc", "--path1", "dir/file1.txt", "--path2", "dir/file2.txt"));
  }

  @Test
  public void testAddNulls() throws Exception {
    Artifact treeArtifact = createTreeArtifact("myTreeArtifact");
    assertThat(treeArtifact).isNotNull();

    CustomCommandLine cl =
        CustomCommandLine.builder()
            .add((Object) null)
            .add("foo", (Object) null)
            .add((ImmutableList<String>) null)
            .add(ImmutableList.<String>of())
            .add((NestedSet<String>) null)
            .add(NestedSetBuilder.<String>emptySet(Order.STABLE_ORDER))
            .add("foo", (ImmutableList<String>) null)
            .add("foo", ImmutableList.<String>of())
            .add("foo", (NestedSet<String>) null)
            .add("foo", NestedSetBuilder.<String>emptySet(Order.STABLE_ORDER))
            .add(VectorArg.of((ImmutableList<String>) null))
            .add(VectorArg.of(ImmutableList.<String>of()))
            .add(VectorArg.of((NestedSet<String>) null))
            .add(VectorArg.of(NestedSetBuilder.<String>emptySet(Order.STABLE_ORDER)))
            .add("foo", VectorArg.of((ImmutableList<String>) null))
            .add("foo", VectorArg.of(ImmutableList.<String>of()))
            .add("foo", VectorArg.of((NestedSet<String>) null))
            .add("foo", VectorArg.of(NestedSetBuilder.<String>emptySet(Order.STABLE_ORDER)))
            .addPlaceholderTreeArtifactExecPath("foo", null)
            .addPlaceholderTreeArtifactFormattedExecPath("foo", null)
            .add((CustomArgv) null)
            .add((CustomMultiArgv) null)
            .build();
    assertThat(cl.arguments()).isEmpty();

    CustomCommandLine.Builder obj = CustomCommandLine.builder();
    Class<CustomCommandLine.Builder> clazz = CustomCommandLine.Builder.class;
    NullPointerTester npt =
        new NullPointerTester()
            .setDefault(Artifact.class, artifact1)
            .setDefault(String.class, "foo")
            .setDefault(PathFragment[].class, new PathFragment[] {PathFragment.create("foo")});

    npt.testMethod(obj, clazz.getMethod("add", String.class, Object.class));
    npt.testMethod(
        obj, clazz.getMethod("addPlaceholderTreeArtifactExecPath", String.class, Artifact.class));
    npt.testMethod(
        obj,
        clazz.getMethod(
            "addPlaceholderTreeArtifactFormattedExecPath", String.class, Artifact.class));
    npt.testMethod(
        obj, clazz.getMethod("addJoinExpandedTreeArtifactExecPath", String.class, Artifact.class));
    npt.testMethod(obj, clazz.getMethod("addExpandedTreeArtifactExecPaths", Artifact.class));

    npt.setDefault(Iterable.class, ImmutableList.of("foo"));

    npt.setDefault(Iterable.class, ImmutableList.of(artifact1));

    npt.setDefault(Iterable.class, ImmutableList.of(PathFragment.create("foo")));
    npt.setDefault(Artifact.class, treeArtifact);
    npt.testMethod(
        obj, clazz.getMethod("addJoinExpandedTreeArtifactExecPath", String.class, Artifact.class));
    npt.testMethod(obj, clazz.getMethod("addExpandedTreeArtifactExecPaths", Artifact.class));
  }

  @Test
  public void testTreeFileArtifactExecPathArgs() {
    Artifact treeArtifactOne = createTreeArtifact("myArtifact/treeArtifact1");
    Artifact treeArtifactTwo = createTreeArtifact("myArtifact/treeArtifact2");

    CustomCommandLine commandLineTemplate = CustomCommandLine.builder()
        .addPlaceholderTreeArtifactExecPath("--argOne", treeArtifactOne)
        .addPlaceholderTreeArtifactExecPath("--argTwo", treeArtifactTwo)
        .build();

    TreeFileArtifact treeFileArtifactOne = createTreeFileArtifact(
        treeArtifactOne, "children/child1");
    TreeFileArtifact treeFileArtifactTwo = createTreeFileArtifact(
        treeArtifactTwo, "children/child2");

    CustomCommandLine commandLine = commandLineTemplate.evaluateTreeFileArtifacts(
        ImmutableList.of(treeFileArtifactOne, treeFileArtifactTwo));

    assertThat(commandLine.arguments())
        .containsExactly(
            "--argOne",
            "myArtifact/treeArtifact1/children/child1",
            "--argTwo",
            "myArtifact/treeArtifact2/children/child2")
        .inOrder();
  }

  @Test
  public void testTreeFileArtifactExecPathWithTemplateArgs() {
    Artifact treeArtifact = createTreeArtifact("myArtifact/treeArtifact1");

    CustomCommandLine commandLineTemplate = CustomCommandLine.builder()
        .addPlaceholderTreeArtifactFormattedExecPath("path:%s", treeArtifact)
        .build();

    TreeFileArtifact treeFileArtifact = createTreeFileArtifact(
        treeArtifact, "children/child1");

    CustomCommandLine commandLine = commandLineTemplate.evaluateTreeFileArtifacts(
        ImmutableList.of(treeFileArtifact));

    assertThat(commandLine.arguments()).containsExactly(
        "path:myArtifact/treeArtifact1/children/child1");
  }

  @Test
  public void testTreeFileArtifactArgThrowWithoutSubstitution() {
    Artifact treeArtifactOne = createTreeArtifact("myArtifact/treeArtifact1");
    Artifact treeArtifactTwo = createTreeArtifact("myArtifact/treeArtifact2");

    CustomCommandLine commandLineTemplate = CustomCommandLine.builder()
        .addPlaceholderTreeArtifactExecPath("--argOne", treeArtifactOne)
        .addPlaceholderTreeArtifactExecPath("--argTwo", treeArtifactTwo)
        .build();

    try {
      commandLineTemplate.arguments();
      fail("No substitution map provided, expected NullPointerException");
    } catch (NullPointerException e) {
      // expected
    }

  }

  @Test
  public void testJoinExpandedTreeArtifactExecPath() {
    Artifact treeArtifact = createTreeArtifact("myTreeArtifact");

    CommandLine commandLine = CustomCommandLine.builder()
        .add("hello")
        .addJoinExpandedTreeArtifactExecPath(":", treeArtifact)
        .build();

    assertThat(commandLine.arguments()).containsExactly(
        "hello",
        "JoinExpandedTreeArtifactExecPathsArg{ delimiter: :, treeArtifact: myTreeArtifact}");

    final Iterable<TreeFileArtifact> treeFileArtifacts = ImmutableList.of(
        createTreeFileArtifact(treeArtifact, "children/child1"),
        createTreeFileArtifact(treeArtifact, "children/child2"));

    ArtifactExpander artifactExpander = new ArtifactExpander() {
      @Override
      public void expand(Artifact artifact, Collection<? super Artifact> output) {
        for (TreeFileArtifact treeFileArtifact : treeFileArtifacts) {
          if (treeFileArtifact.getParent().equals(artifact)) {
            output.add(treeFileArtifact);
          }
        }
      }
    };

    assertThat(commandLine.arguments(artifactExpander)).containsExactly(
        "hello",
        "myTreeArtifact/children/child1:myTreeArtifact/children/child2");
  }

  private Artifact createTreeArtifact(String rootRelativePath) {
    PathFragment relpath = PathFragment.create(rootRelativePath);
    return new SpecialArtifact(
        rootDir.getPath().getRelative(relpath),
        rootDir,
        rootDir.getExecPath().getRelative(relpath),
        ArtifactOwner.NULL_OWNER,
        SpecialArtifactType.TREE);
  }

  private TreeFileArtifact createTreeFileArtifact(
      Artifact inputTreeArtifact, String parentRelativePath) {
    return ActionInputHelper.treeFileArtifact(
        inputTreeArtifact,
        PathFragment.create(parentRelativePath));
  }
}
