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
import static com.google.devtools.build.lib.analysis.actions.CustomCommandLine.builder;
import static org.junit.Assert.fail;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifactType;
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine.CustomMultiArgv;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.lib.util.LazyString;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.Arrays;
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
  public void testScalarAdds() throws Exception {
    assertThat(builder().add("--arg").build().arguments()).containsExactly("--arg").inOrder();
    assertThat(builder().addDynamicString("--arg").build().arguments())
        .containsExactly("--arg")
        .inOrder();
    assertThat(builder().addLabel(Label.parseAbsolute("//a:b")).build().arguments())
        .containsExactly("//a:b")
        .inOrder();
    assertThat(builder().addPath(PathFragment.create("path")).build().arguments())
        .containsExactly("path")
        .inOrder();
    assertThat(builder().addExecPath(artifact1).build().arguments())
        .containsExactly("dir/file1.txt")
        .inOrder();
    assertThat(
            builder()
                .addLazyString(
                    new LazyString() {
                      @Override
                      public String toString() {
                        return "foo";
                      }
                    })
                .build()
                .arguments())
        .containsExactly("foo")
        .inOrder();

    assertThat(builder().add("--arg", "val").build().arguments())
        .containsExactly("--arg", "val")
        .inOrder();
    assertThat(builder().addLabel("--arg", Label.parseAbsolute("//a:b")).build().arguments())
        .containsExactly("--arg", "//a:b")
        .inOrder();
    assertThat(builder().addPath("--arg", PathFragment.create("path")).build().arguments())
        .containsExactly("--arg", "path")
        .inOrder();
    assertThat(builder().addExecPath("--arg", artifact1).build().arguments())
        .containsExactly("--arg", "dir/file1.txt")
        .inOrder();
    assertThat(
            builder()
                .addLazyString(
                    "--arg",
                    new LazyString() {
                      @Override
                      public String toString() {
                        return "foo";
                      }
                    })
                .build()
                .arguments())
        .containsExactly("--arg", "foo")
        .inOrder();
  }

  @Test
  public void testAddFormatted() throws Exception {
    assertThat(builder().addFormatted("%s%s", "hello", "world").build().arguments())
        .containsExactly("helloworld")
        .inOrder();
  }

  @Test
  public void testAddPrefixed() throws Exception {
    assertThat(builder().addPrefixed("prefix-", "foo").build().arguments())
        .containsExactly("prefix-foo")
        .inOrder();
    assertThat(
            builder().addPrefixedLabel("prefix-", Label.parseAbsolute("//a:b")).build().arguments())
        .containsExactly("prefix-//a:b")
        .inOrder();
    assertThat(
            builder().addPrefixedPath("prefix-", PathFragment.create("path")).build().arguments())
        .containsExactly("prefix-path")
        .inOrder();
    assertThat(builder().addPrefixedExecPath("prefix-", artifact1).build().arguments())
        .containsExactly("prefix-dir/file1.txt")
        .inOrder();
  }

  @Test
  public void testVectorAdds() throws Exception {
    assertThat(builder().addAll(list("val1", "val2")).build().arguments())
        .containsExactly("val1", "val2")
        .inOrder();
    assertThat(builder().addAll(nestedSet("val1", "val2")).build().arguments())
        .containsExactly("val1", "val2")
        .inOrder();
    assertThat(
            builder()
                .addPaths(list(PathFragment.create("path1"), PathFragment.create("path2")))
                .build()
                .arguments())
        .containsExactly("path1", "path2")
        .inOrder();
    assertThat(
            builder()
                .addPaths(nestedSet(PathFragment.create("path1"), PathFragment.create("path2")))
                .build()
                .arguments())
        .containsExactly("path1", "path2")
        .inOrder();
    assertThat(builder().addExecPaths(list(artifact1, artifact2)).build().arguments())
        .containsExactly("dir/file1.txt", "dir/file2.txt")
        .inOrder();
    assertThat(builder().addExecPaths(nestedSet(artifact1, artifact2)).build().arguments())
        .containsExactly("dir/file1.txt", "dir/file2.txt")
        .inOrder();
    assertThat(builder().addAll(list(foo("1"), foo("2")), Foo::str).build().arguments())
        .containsExactly("1", "2")
        .inOrder();
    assertThat(builder().addAll(nestedSet(foo("1"), foo("2")), Foo::str).build().arguments())
        .containsExactly("1", "2")
        .inOrder();

    assertThat(builder().addAll("--arg", list("val1", "val2")).build().arguments())
        .containsExactly("--arg", "val1", "val2")
        .inOrder();
    assertThat(builder().addAll("--arg", nestedSet("val1", "val2")).build().arguments())
        .containsExactly("--arg", "val1", "val2")
        .inOrder();
    assertThat(
            builder()
                .addPaths("--arg", list(PathFragment.create("path1"), PathFragment.create("path2")))
                .build()
                .arguments())
        .containsExactly("--arg", "path1", "path2")
        .inOrder();
    assertThat(
            builder()
                .addPaths(
                    "--arg", nestedSet(PathFragment.create("path1"), PathFragment.create("path2")))
                .build()
                .arguments())
        .containsExactly("--arg", "path1", "path2")
        .inOrder();
    assertThat(builder().addExecPaths("--arg", list(artifact1, artifact2)).build().arguments())
        .containsExactly("--arg", "dir/file1.txt", "dir/file2.txt")
        .inOrder();
    assertThat(builder().addExecPaths("--arg", nestedSet(artifact1, artifact2)).build().arguments())
        .containsExactly("--arg", "dir/file1.txt", "dir/file2.txt")
        .inOrder();
    assertThat(builder().addAll("--arg", list(foo("1"), foo("2")), Foo::str).build().arguments())
        .containsExactly("--arg", "1", "2")
        .inOrder();
    assertThat(
            builder().addAll("--arg", nestedSet(foo("1"), foo("2")), Foo::str).build().arguments())
        .containsExactly("--arg", "1", "2")
        .inOrder();
  }

  @Test
  public void testAddJoined() throws Exception {
    assertThat(builder().addJoined(":", list("val1", "val2")).build().arguments())
        .containsExactly("val1:val2")
        .inOrder();
    assertThat(builder().addJoined(":", nestedSet("val1", "val2")).build().arguments())
        .containsExactly("val1:val2")
        .inOrder();
    assertThat(
            builder()
                .addJoinedPaths(
                    ":", list(PathFragment.create("path1"), PathFragment.create("path2")))
                .build()
                .arguments())
        .containsExactly("path1:path2")
        .inOrder();
    assertThat(
            builder()
                .addJoinedPaths(
                    ":", nestedSet(PathFragment.create("path1"), PathFragment.create("path2")))
                .build()
                .arguments())
        .containsExactly("path1:path2")
        .inOrder();
    assertThat(builder().addJoinedExecPaths(":", list(artifact1, artifact2)).build().arguments())
        .containsExactly("dir/file1.txt:dir/file2.txt")
        .inOrder();
    assertThat(
            builder().addJoinedExecPaths(":", nestedSet(artifact1, artifact2)).build().arguments())
        .containsExactly("dir/file1.txt:dir/file2.txt")
        .inOrder();
    assertThat(builder().addJoined(":", list(foo("1"), foo("2")), Foo::str).build().arguments())
        .containsExactly("1:2")
        .inOrder();
    assertThat(
            builder().addJoined(":", nestedSet(foo("1"), foo("2")), Foo::str).build().arguments())
        .containsExactly("1:2")
        .inOrder();

    assertThat(builder().addJoined("--arg", ":", list("val1", "val2")).build().arguments())
        .containsExactly("--arg", "val1:val2")
        .inOrder();
    assertThat(builder().addJoined("--arg", ":", nestedSet("val1", "val2")).build().arguments())
        .containsExactly("--arg", "val1:val2")
        .inOrder();
    assertThat(
            builder()
                .addJoinedPaths(
                    "--arg", ":", list(PathFragment.create("path1"), PathFragment.create("path2")))
                .build()
                .arguments())
        .containsExactly("--arg", "path1:path2")
        .inOrder();
    assertThat(
            builder()
                .addJoinedPaths(
                    "--arg",
                    ":",
                    nestedSet(PathFragment.create("path1"), PathFragment.create("path2")))
                .build()
                .arguments())
        .containsExactly("--arg", "path1:path2")
        .inOrder();
    assertThat(
            builder()
                .addJoinedExecPaths("--arg", ":", list(artifact1, artifact2))
                .build()
                .arguments())
        .containsExactly("--arg", "dir/file1.txt:dir/file2.txt")
        .inOrder();
    assertThat(
            builder()
                .addJoinedExecPaths("--arg", ":", nestedSet(artifact1, artifact2))
                .build()
                .arguments())
        .containsExactly("--arg", "dir/file1.txt:dir/file2.txt")
        .inOrder();
    assertThat(
            builder()
                .addJoined("--arg", ":", list(foo("1"), foo("2")), Foo::str)
                .build()
                .arguments())
        .containsExactly("--arg", "1:2")
        .inOrder();
    assertThat(
            builder()
                .addJoined("--arg", ":", nestedSet(foo("1"), foo("2")), Foo::str)
                .build()
                .arguments())
        .containsExactly("--arg", "1:2")
        .inOrder();
  }

  @Test
  public void testAddFormatEach() throws Exception {
    assertThat(builder().addFormatEach("-D%s", list("val1", "val2")).build().arguments())
        .containsExactly("-Dval1", "-Dval2")
        .inOrder();
    assertThat(builder().addFormatEach("-D%s", nestedSet("val1", "val2")).build().arguments())
        .containsExactly("-Dval1", "-Dval2")
        .inOrder();
    assertThat(
            builder()
                .addFormatEachPath(
                    "-D%s", list(PathFragment.create("path1"), PathFragment.create("path2")))
                .build()
                .arguments())
        .containsExactly("-Dpath1", "-Dpath2")
        .inOrder();
    assertThat(
            builder()
                .addFormatEachPath(
                    "-D%s", nestedSet(PathFragment.create("path1"), PathFragment.create("path2")))
                .build()
                .arguments())
        .containsExactly("-Dpath1", "-Dpath2")
        .inOrder();
    assertThat(
            builder().addFormatEachExecPath("-D%s", list(artifact1, artifact2)).build().arguments())
        .containsExactly("-Ddir/file1.txt", "-Ddir/file2.txt")
        .inOrder();
    assertThat(
            builder()
                .addFormatEachExecPath("-D%s", nestedSet(artifact1, artifact2))
                .build()
                .arguments())
        .containsExactly("-Ddir/file1.txt", "-Ddir/file2.txt")
        .inOrder();
    assertThat(
            builder().addFormatEach("-D%s", list(foo("1"), foo("2")), Foo::str).build().arguments())
        .containsExactly("-D1", "-D2")
        .inOrder();
    assertThat(
            builder()
                .addFormatEach("-D%s", nestedSet(foo("1"), foo("2")), Foo::str)
                .build()
                .arguments())
        .containsExactly("-D1", "-D2")
        .inOrder();

    assertThat(builder().addFormatEach("--arg", "-D%s", list("val1", "val2")).build().arguments())
        .containsExactly("--arg", "-Dval1", "-Dval2")
        .inOrder();
    assertThat(
            builder().addFormatEach("--arg", "-D%s", nestedSet("val1", "val2")).build().arguments())
        .containsExactly("--arg", "-Dval1", "-Dval2")
        .inOrder();
    assertThat(
            builder()
                .addFormatEachPath(
                    "--arg",
                    "-D%s",
                    list(PathFragment.create("path1"), PathFragment.create("path2")))
                .build()
                .arguments())
        .containsExactly("--arg", "-Dpath1", "-Dpath2")
        .inOrder();
    assertThat(
            builder()
                .addFormatEachPath(
                    "--arg",
                    "-D%s",
                    nestedSet(PathFragment.create("path1"), PathFragment.create("path2")))
                .build()
                .arguments())
        .containsExactly("--arg", "-Dpath1", "-Dpath2")
        .inOrder();
    assertThat(
            builder()
                .addFormatEachExecPath("--arg", "-D%s", list(artifact1, artifact2))
                .build()
                .arguments())
        .containsExactly("--arg", "-Ddir/file1.txt", "-Ddir/file2.txt")
        .inOrder();
    assertThat(
            builder()
                .addFormatEachExecPath("--arg", "-D%s", nestedSet(artifact1, artifact2))
                .build()
                .arguments())
        .containsExactly("--arg", "-Ddir/file1.txt", "-Ddir/file2.txt")
        .inOrder();
    assertThat(
            builder()
                .addFormatEach("--arg", "-D%s", list(foo("1"), foo("2")), Foo::str)
                .build()
                .arguments())
        .containsExactly("--arg", "-D1", "-D2")
        .inOrder();
    assertThat(
            builder()
                .addFormatEach("--arg", "-D%s", nestedSet(foo("1"), foo("2")), Foo::str)
                .build()
                .arguments())
        .containsExactly("--arg", "-D1", "-D2")
        .inOrder();
  }

  @Test
  public void testAddFormatEachJoined() throws Exception {
    assertThat(builder().addFormatEachJoined("-D%s", ":", list("val1", "val2")).build().arguments())
        .containsExactly("-Dval1:-Dval2")
        .inOrder();
    assertThat(
            builder()
                .addFormatEachJoined("-D%s", ":", nestedSet("val1", "val2"))
                .build()
                .arguments())
        .containsExactly("-Dval1:-Dval2")
        .inOrder();
    assertThat(
            builder()
                .addFormatEachPathJoined(
                    "-D%s", ":", list(PathFragment.create("path1"), PathFragment.create("path2")))
                .build()
                .arguments())
        .containsExactly("-Dpath1:-Dpath2")
        .inOrder();
    assertThat(
            builder()
                .addFormatEachPathJoined(
                    "-D%s",
                    ":", nestedSet(PathFragment.create("path1"), PathFragment.create("path2")))
                .build()
                .arguments())
        .containsExactly("-Dpath1:-Dpath2")
        .inOrder();
    assertThat(
            builder()
                .addFormatEachExecPathJoined("-D%s", ":", list(artifact1, artifact2))
                .build()
                .arguments())
        .containsExactly("-Ddir/file1.txt:-Ddir/file2.txt")
        .inOrder();
    assertThat(
            builder()
                .addFormatEachExecPathJoined("-D%s", ":", nestedSet(artifact1, artifact2))
                .build()
                .arguments())
        .containsExactly("-Ddir/file1.txt:-Ddir/file2.txt")
        .inOrder();
    assertThat(
            builder()
                .addFormatEachJoined("-D%s", ":", list(foo("1"), foo("2")), Foo::str)
                .build()
                .arguments())
        .containsExactly("-D1:-D2")
        .inOrder();
    assertThat(
            builder()
                .addFormatEachJoined("-D%s", ":", nestedSet(foo("1"), foo("2")), Foo::str)
                .build()
                .arguments())
        .containsExactly("-D1:-D2")
        .inOrder();

    assertThat(
            builder()
                .addFormatEachJoined("--arg", "-D%s", ":", list("val1", "val2"))
                .build()
                .arguments())
        .containsExactly("--arg", "-Dval1:-Dval2")
        .inOrder();
    assertThat(
            builder()
                .addFormatEachJoined("--arg", "-D%s", ":", nestedSet("val1", "val2"))
                .build()
                .arguments())
        .containsExactly("--arg", "-Dval1:-Dval2")
        .inOrder();
    assertThat(
            builder()
                .addFormatEachPathJoined(
                    "--arg",
                    "-D%s",
                    ":",
                    list(PathFragment.create("path1"), PathFragment.create("path2")))
                .build()
                .arguments())
        .containsExactly("--arg", "-Dpath1:-Dpath2")
        .inOrder();
    assertThat(
            builder()
                .addFormatEachPathJoined(
                    "--arg",
                    "-D%s",
                    ":",
                    nestedSet(PathFragment.create("path1"), PathFragment.create("path2")))
                .build()
                .arguments())
        .containsExactly("--arg", "-Dpath1:-Dpath2")
        .inOrder();
    assertThat(
            builder()
                .addFormatEachExecPathJoined("--arg", "-D%s", ":", list(artifact1, artifact2))
                .build()
                .arguments())
        .containsExactly("--arg", "-Ddir/file1.txt:-Ddir/file2.txt")
        .inOrder();
    assertThat(
            builder()
                .addFormatEachExecPathJoined("--arg", "-D%s", ":", nestedSet(artifact1, artifact2))
                .build()
                .arguments())
        .containsExactly("--arg", "-Ddir/file1.txt:-Ddir/file2.txt")
        .inOrder();
    assertThat(
            builder()
                .addFormatEachJoined("--arg", "-D%s", ":", list(foo("1"), foo("2")), Foo::str)
                .build()
                .arguments())
        .containsExactly("--arg", "-D1:-D2")
        .inOrder();
    assertThat(
            builder()
                .addFormatEachJoined("--arg", "-D%s", ":", nestedSet(foo("1"), foo("2")), Foo::str)
                .build()
                .arguments())
        .containsExactly("--arg", "-D1:-D2")
        .inOrder();
  }

  @Test
  public void testAddBeforeEach() throws Exception {
    assertThat(builder().addBeforeEach("-D", list("val1", "val2")).build().arguments())
        .containsExactly("-D", "val1", "-D", "val2")
        .inOrder();
    assertThat(builder().addBeforeEach("-D", nestedSet("val1", "val2")).build().arguments())
        .containsExactly("-D", "val1", "-D", "val2")
        .inOrder();
    assertThat(
            builder()
                .addBeforeEachPath(
                    "-D", list(PathFragment.create("path1"), PathFragment.create("path2")))
                .build()
                .arguments())
        .containsExactly("-D", "path1", "-D", "path2")
        .inOrder();
    assertThat(
            builder()
                .addBeforeEachPath(
                    "-D", nestedSet(PathFragment.create("path1"), PathFragment.create("path2")))
                .build()
                .arguments())
        .containsExactly("-D", "path1", "-D", "path2")
        .inOrder();
    assertThat(
            builder().addBeforeEachExecPath("-D", list(artifact1, artifact2)).build().arguments())
        .containsExactly("-D", "dir/file1.txt", "-D", "dir/file2.txt")
        .inOrder();
    assertThat(
            builder()
                .addBeforeEachExecPath("-D", nestedSet(artifact1, artifact2))
                .build()
                .arguments())
        .containsExactly("-D", "dir/file1.txt", "-D", "dir/file2.txt")
        .inOrder();
    assertThat(
            builder().addBeforeEach("-D", list(foo("1"), foo("2")), Foo::str).build().arguments())
        .containsExactly("-D", "1", "-D", "2")
        .inOrder();
    assertThat(
            builder()
                .addBeforeEach("-D", nestedSet(foo("1"), foo("2")), Foo::str)
                .build()
                .arguments())
        .containsExactly("-D", "1", "-D", "2")
        .inOrder();
  }

  @Test
  public void testAddBeforeEachFormatted() throws Exception {
    assertThat(
            builder().addBeforeEachFormatted("-D", "D%s", list("val1", "val2")).build().arguments())
        .containsExactly("-D", "Dval1", "-D", "Dval2")
        .inOrder();
    assertThat(
            builder()
                .addBeforeEachFormatted("-D", "D%s", nestedSet("val1", "val2"))
                .build()
                .arguments())
        .containsExactly("-D", "Dval1", "-D", "Dval2")
        .inOrder();
    assertThat(
            builder()
                .addBeforeEachPathFormatted(
                    "-D", "D%s", list(PathFragment.create("path1"), PathFragment.create("path2")))
                .build()
                .arguments())
        .containsExactly("-D", "Dpath1", "-D", "Dpath2")
        .inOrder();
    assertThat(
            builder()
                .addBeforeEachPathFormatted(
                    "-D",
                    "D%s",
                    nestedSet(PathFragment.create("path1"), PathFragment.create("path2")))
                .build()
                .arguments())
        .containsExactly("-D", "Dpath1", "-D", "Dpath2")
        .inOrder();
    assertThat(
            builder()
                .addBeforeEachExecPathFormatted("-D", "D%s", list(artifact1, artifact2))
                .build()
                .arguments())
        .containsExactly("-D", "Ddir/file1.txt", "-D", "Ddir/file2.txt")
        .inOrder();
    assertThat(
            builder()
                .addBeforeEachExecPathFormatted("-D", "D%s", nestedSet(artifact1, artifact2))
                .build()
                .arguments())
        .containsExactly("-D", "Ddir/file1.txt", "-D", "Ddir/file2.txt")
        .inOrder();
    assertThat(
            builder()
                .addBeforeEachFormatted("-D", "D%s", list(foo("1"), foo("2")), Foo::str)
                .build()
                .arguments())
        .containsExactly("-D", "D1", "-D", "D2")
        .inOrder();
    assertThat(
            builder()
                .addBeforeEachFormatted("-D", "D%s", nestedSet(foo("1"), foo("2")), Foo::str)
                .build()
                .arguments())
        .containsExactly("-D", "D1", "-D", "D2")
        .inOrder();
  }

  @Test
  public void testCustomMultiArgs() {
    CustomCommandLine cl =
        builder()
            .addCustomMultiArgv(
                new CustomMultiArgv() {
                  @Override
                  public ImmutableList<String> argv() {
                    return ImmutableList.of("--arg1", "--arg2");
                  }
                })
            .build();
    assertThat(cl.arguments()).containsExactly("--arg1", "--arg2").inOrder();
  }

  @Test
  public void testCombinedArgs() {
    CustomCommandLine cl =
        builder()
            .add("--arg")
            .addAll("--args", ImmutableList.of("abc"))
            .addExecPaths("--path1", ImmutableList.of(artifact1))
            .addExecPath("--path2", artifact2)
            .build();
    assertThat(cl.arguments())
        .containsExactly(
            "--arg", "--args", "abc", "--path1", "dir/file1.txt", "--path2", "dir/file2.txt")
        .inOrder();
  }

  @Test
  public void testAddNulls() throws Exception {
    Artifact treeArtifact = createTreeArtifact("myTreeArtifact");
    assertThat(treeArtifact).isNotNull();

    CustomCommandLine cl =
        builder()
            .addDynamicString(null)
            .addLabel(null)
            .addPath(null)
            .addExecPath(null)
            .addLazyString(null)
            .add("foo", null)
            .addLabel("foo", null)
            .addPath("foo", null)
            .addExecPath("foo", null)
            .addLazyString("foo", null)
            .addPrefixed("prefix", null)
            .addPrefixedLabel("prefix", null)
            .addPrefixedPath("prefix", null)
            .addPrefixedExecPath("prefix", null)
            .addAll((ImmutableList<String>) null)
            .addAll(ImmutableList.of())
            .addPaths((ImmutableList<PathFragment>) null)
            .addPaths(ImmutableList.of())
            .addExecPaths((ImmutableList<Artifact>) null)
            .addExecPaths(ImmutableList.of())
            .addAll((ImmutableList<Foo>) null, Foo::str)
            .addAll(ImmutableList.of(), Foo::str)
            .addAll((NestedSet<String>) null)
            .addAll(NestedSetBuilder.emptySet(Order.STABLE_ORDER))
            .addPaths((NestedSet<PathFragment>) null)
            .addPaths(NestedSetBuilder.emptySet(Order.STABLE_ORDER))
            .addExecPaths((NestedSet<Artifact>) null)
            .addExecPaths(NestedSetBuilder.emptySet(Order.STABLE_ORDER))
            .addAll((NestedSet<Foo>) null, Foo::str)
            .addAll(NestedSetBuilder.emptySet(Order.STABLE_ORDER), Foo::str)
            .addAll("foo", (ImmutableList<String>) null)
            .addAll("foo", ImmutableList.of())
            .addPaths("foo", (ImmutableList<PathFragment>) null)
            .addPaths("foo", ImmutableList.of())
            .addAll("foo", (ImmutableList<Foo>) null, Foo::str)
            .addAll("foo", ImmutableList.of(), Foo::str)
            .addExecPaths("foo", (ImmutableList<Artifact>) null)
            .addExecPaths("foo", ImmutableList.of())
            .addAll("foo", (NestedSet<String>) null)
            .addAll("foo", NestedSetBuilder.emptySet(Order.STABLE_ORDER))
            .addPaths("foo", (NestedSet<PathFragment>) null)
            .addPaths("foo", NestedSetBuilder.emptySet(Order.STABLE_ORDER))
            .addExecPaths("foo", (NestedSet<Artifact>) null)
            .addExecPaths("foo", NestedSetBuilder.emptySet(Order.STABLE_ORDER))
            .addAll("foo", (NestedSet<Foo>) null, Foo::str)
            .addAll("foo", NestedSetBuilder.emptySet(Order.STABLE_ORDER), Foo::str)
            .addCustomMultiArgv(null)
            .addPlaceholderTreeArtifactExecPath("foo", null)
            .build();
    assertThat(cl.arguments()).isEmpty();
  }

  @Test
  public void testTreeFileArtifactExecPathArgs() {
    Artifact treeArtifactOne = createTreeArtifact("myArtifact/treeArtifact1");
    Artifact treeArtifactTwo = createTreeArtifact("myArtifact/treeArtifact2");

    CustomCommandLine commandLineTemplate =
        builder()
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
  public void testTreeFileArtifactArgThrowWithoutSubstitution() {
    Artifact treeArtifactOne = createTreeArtifact("myArtifact/treeArtifact1");
    Artifact treeArtifactTwo = createTreeArtifact("myArtifact/treeArtifact2");

    CustomCommandLine commandLineTemplate =
        builder()
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

  private static <T> ImmutableList<T> list(T... objects) {
    return ImmutableList.<T>builder().addAll(Arrays.asList(objects)).build();
  }

  private static <T> NestedSet<T> nestedSet(T... objects) {
    return NestedSetBuilder.<T>stableOrder().addAll(Arrays.asList(objects)).build();
  }

  private static Foo foo(String str) {
    return new Foo(str);
  }

  private static class Foo {
    private final String str;

    Foo(String str) {
      this.str = str;
    }

    static String str(Foo foo) {
      return foo.str;
    }
  }
}
