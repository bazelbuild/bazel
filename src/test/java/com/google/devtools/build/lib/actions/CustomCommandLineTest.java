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
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.fail;

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.Artifact.SpecialArtifact;
import com.google.devtools.build.lib.actions.Artifact.TreeFileArtifact;
import com.google.devtools.build.lib.actions.ArtifactRoot.RootType;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine.VectorArg;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine.VectorArg.SimpleVectorArg;
import com.google.devtools.build.lib.analysis.config.CoreOptions;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.RepositoryMapping;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.util.OnDemandString;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.testing.junit.testparameterinjector.TestParameter;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import java.util.HashMap;
import java.util.Map;
import java.util.function.Consumer;
import net.starlark.java.eval.EvalException;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;

/** Tests for {@link CustomCommandLine}. */
@RunWith(TestParameterInjector.class)
public final class CustomCommandLineTest {
  private ArtifactRoot rootDir;
  private Artifact artifact1;
  private Artifact artifact2;

  @Before
  public void createArtifacts() throws Exception {
    Scratch scratch = new Scratch();
    rootDir = ArtifactRoot.asDerivedRoot(scratch.dir("/exec/root"), RootType.OUTPUT, "dir");
    artifact1 = ActionsTestUtil.createArtifact(rootDir, scratch.file("/exec/root/dir/file1.txt"));
    artifact2 = ActionsTestUtil.createArtifact(rootDir, scratch.file("/exec/root/dir/file2.txt"));
  }

  @Test
  public void addScalar_addsSingleArgument() throws Exception {
    assertThat(builder().add("--arg").build().arguments()).containsExactly("--arg");
    assertThat(builder().addDynamicString("--arg").build().arguments()).containsExactly("--arg");
    assertThat(builder().addLabel(Label.parseCanonical("//a:b")).build().arguments())
        .containsExactly("//a:b");
    assertThat(builder().addPath(PathFragment.create("path")).build().arguments())
        .containsExactly("path");
    assertThat(builder().addExecPath(artifact1).build().arguments())
        .containsExactly("dir/file1.txt");
    assertThat(
            builder()
                .addLazyString(
                    new OnDemandString() {
                      @Override
                      public String toString() {
                        return "foo";
                      }
                    })
                .build()
                .arguments())
        .containsExactly("foo");
  }

  @Test
  public void addScalar_withConstantArg_addsStringPrependedByArg() throws Exception {
    assertThat(builder().add("--arg", "val").build().arguments())
        .containsExactly("--arg", "val")
        .inOrder();
    assertThat(builder().addLabel("--arg", Label.parseCanonical("//a:b")).build().arguments())
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
                    new OnDemandString() {
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
  public void addFormatted_addsCorrectlyFormattedArgument() throws Exception {
    assertThat(builder().addFormatted("%s%s", "hello", "world").build().arguments())
        .containsExactly("helloworld");
  }

  @Test
  public void addPrefixed_addsPrefixForArguments() throws Exception {
    assertThat(builder().addPrefixed("prefix-", "foo").build().arguments())
        .containsExactly("prefix-foo");
    assertThat(
            builder()
                .addPrefixedLabel(
                    "prefix-", Label.parseCanonical("//a:b"), /* mainRepoMapping= */ null)
                .build()
                .arguments())
        .containsExactly("prefix-//a:b");
    assertThat(
            builder().addPrefixedPath("prefix-", PathFragment.create("path")).build().arguments())
        .containsExactly("prefix-path");
    assertThat(builder().addPrefixedExecPath("prefix-", artifact1).build().arguments())
        .containsExactly("prefix-dir/file1.txt");
  }

  @Test
  public void addPrefixedLabel_emitsExternalLabelInDisplayForm() throws Exception {
    assertThat(
            builder()
                .addPrefixedLabel(
                    "prefix-",
                    Label.parseCanonical("@@canonical_name//a:b"),
                    RepositoryMapping.create(
                        ImmutableMap.of(
                            "apparent_name", RepositoryName.createUnvalidated("canonical_name")),
                        RepositoryName.MAIN))
                .build()
                .arguments())
        .containsExactly("prefix-@apparent_name//a:b");
  }

  @Test
  public void addAll_addsAllArguments() throws Exception {
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
  }

  @Test
  public void vectorAdds_withCompileTimeArg_addsAllValuesPrependedByArg() throws Exception {
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
  }

  private enum CustomCommandLineMode {
    REGULAR {
      @Override
      CustomCommandLine addAll(VectorArg<String> vectorArg) {
        return builder().addAll(vectorArg).build();
      }

      @Override
      CustomCommandLine addPaths(VectorArg<PathFragment> vectorArg) {
        return builder().addPaths(vectorArg).build();
      }

      @Override
      CustomCommandLine addExecPaths(VectorArg<Artifact> vectorArg) {
        return builder().addExecPaths(vectorArg).build();
      }

      @Override
      ImmutableList<String> expected(String... values) {
        return ImmutableList.copyOf(values);
      }
    },
    WITH_CONSTANT_ARG {
      @Override
      CustomCommandLine addAll(VectorArg<String> vectorArg) {
        return builder().addAll("--arg", vectorArg).build();
      }

      @Override
      CustomCommandLine addPaths(VectorArg<PathFragment> vectorArg) {
        return builder().addPaths("--arg", vectorArg).build();
      }

      @Override
      CustomCommandLine addExecPaths(VectorArg<Artifact> vectorArg) {
        return builder().addExecPaths("--arg", vectorArg).build();
      }

      @Override
      ImmutableList<String> expected(String... values) {
        return ImmutableList.<String>builderWithExpectedSize(values.length + 1)
            .add("--arg")
            .add(values)
            .build();
      }
    };

    abstract CustomCommandLine addAll(VectorArg<String> vectorArg);

    abstract CustomCommandLine addPaths(VectorArg<PathFragment> vectorArg);

    abstract CustomCommandLine addExecPaths(VectorArg<Artifact> vectorArg);

    abstract ImmutableList<String> expected(String... values);
  }

  private enum VectorArgMode {
    LIST {
      @Override
      <T> SimpleVectorArg<T> of(T... objects) {
        return VectorArg.of(list(objects));
      }

      @Override
      <T> SimpleVectorArg<T> each(VectorArg.Builder vectorArg, T... objects) {
        return vectorArg.each(list(objects));
      }
    },
    NESTED_SET {
      @Override
      <T> SimpleVectorArg<T> of(T... objects) {
        return VectorArg.of(nestedSet(objects));
      }

      @Override
      <T> SimpleVectorArg<T> each(VectorArg.Builder vectorArg, T... objects) {
        return vectorArg.each(nestedSet(objects));
      }
    };

    abstract <T> SimpleVectorArg<T> of(T... objects);

    abstract <T> SimpleVectorArg<T> each(VectorArg.Builder vectorArg, T... objects);
  }

  @Test
  public void addAllVector_addsAllArguments(
      @TestParameter CustomCommandLineMode customCommandLine,
      @TestParameter VectorArgMode vectorArg)
      throws Exception {
    assertThat(customCommandLine.addAll(vectorArg.of("1", "2")).arguments())
        .containsExactlyElementsIn(customCommandLine.expected("1", "2"))
        .inOrder();
    assertThat(
            customCommandLine
                .addAll(vectorArg.of(foo("1"), foo("2")).mapped(Foo::expandToStr))
                .arguments())
        .containsExactlyElementsIn(customCommandLine.expected("1", "2"))
        .inOrder();
  }

  @Test
  public void addJoinedVector_addsJoinedArguments(
      @TestParameter CustomCommandLineMode customCommandLine,
      @TestParameter VectorArgMode vectorArg)
      throws Exception {
    assertThat(
            customCommandLine
                .addAll(vectorArg.each(VectorArg.join(":"), "val1", "val2"))
                .arguments())
        .containsExactlyElementsIn(customCommandLine.expected("val1:val2"))
        .inOrder();
    assertThat(
            customCommandLine
                .addPaths(
                    vectorArg.each(
                        VectorArg.join(":"),
                        PathFragment.create("path1"),
                        PathFragment.create("path2")))
                .arguments())
        .containsExactlyElementsIn(customCommandLine.expected("path1:path2"))
        .inOrder();
    assertThat(
            customCommandLine
                .addExecPaths(vectorArg.each(VectorArg.join(":"), artifact1, artifact2))
                .arguments())
        .containsExactlyElementsIn(customCommandLine.expected("dir/file1.txt:dir/file2.txt"))
        .inOrder();
    assertThat(
            customCommandLine
                .addAll(
                    vectorArg
                        .each(VectorArg.join(":"), foo("1"), foo("2"))
                        .mapped(Foo::expandToStr))
                .arguments())
        .containsExactlyElementsIn(customCommandLine.expected("1:2"))
        .inOrder();
  }

  @Test
  public void addFormatEachVector_addsFormattedStrings(
      @TestParameter CustomCommandLineMode customCommandLine,
      @TestParameter VectorArgMode vectorArg)
      throws Exception {
    assertThat(
            customCommandLine
                .addAll(vectorArg.each(VectorArg.format("-D%s"), "val1", "val2"))
                .arguments())
        .containsExactlyElementsIn(customCommandLine.expected("-Dval1", "-Dval2"))
        .inOrder();
    assertThat(
            customCommandLine
                .addPaths(
                    vectorArg.each(
                        VectorArg.format("-D%s"),
                        PathFragment.create("path1"),
                        PathFragment.create("path2")))
                .arguments())
        .containsExactlyElementsIn(customCommandLine.expected("-Dpath1", "-Dpath2"))
        .inOrder();
    assertThat(
            customCommandLine
                .addExecPaths(vectorArg.each(VectorArg.format("-D%s"), artifact1, artifact2))
                .arguments())
        .containsExactlyElementsIn(customCommandLine.expected("-Ddir/file1.txt", "-Ddir/file2.txt"))
        .inOrder();
    assertThat(
            customCommandLine
                .addAll(
                    vectorArg
                        .each(VectorArg.format("-D%s"), foo("1"), foo("2"))
                        .mapped(Foo::expandToStr))
                .arguments())
        .containsExactlyElementsIn(customCommandLine.expected("-D1", "-D2"))
        .inOrder();
  }

  @Test
  public void addFormatEachJoinedVector_addsJoinedFormattedStrings(
      @TestParameter CustomCommandLineMode customCommandLine,
      @TestParameter VectorArgMode vectorArg)
      throws Exception {
    assertThat(
            customCommandLine
                .addAll(vectorArg.each(VectorArg.format("-D%s").join(":"), "val1", "val2"))
                .arguments())
        .containsExactlyElementsIn(customCommandLine.expected("-Dval1:-Dval2"))
        .inOrder();
    assertThat(
            customCommandLine
                .addPaths(
                    vectorArg.each(
                        VectorArg.format("-D%s").join(":"),
                        PathFragment.create("path1"),
                        PathFragment.create("path2")))
                .arguments())
        .containsExactlyElementsIn(customCommandLine.expected("-Dpath1:-Dpath2"))
        .inOrder();
    assertThat(
            customCommandLine
                .addExecPaths(
                    vectorArg.each(VectorArg.format("-D%s").join(":"), artifact1, artifact2))
                .arguments())
        .containsExactlyElementsIn(customCommandLine.expected("-Ddir/file1.txt:-Ddir/file2.txt"))
        .inOrder();
    assertThat(
            customCommandLine
                .addAll(
                    vectorArg
                        .each(VectorArg.format("-D%s").join(":"), foo("1"), foo("2"))
                        .mapped(Foo::expandToStr))
                .arguments())
        .containsExactlyElementsIn(customCommandLine.expected("-D1:-D2"))
        .inOrder();
  }

  @Test
  public void addBeforeEachVector_addsArgumentsEachPrependedWithArg(
      @TestParameter CustomCommandLineMode customCommandLine,
      @TestParameter VectorArgMode vectorArg)
      throws Exception {
    assertThat(
            customCommandLine
                .addAll(vectorArg.each(VectorArg.addBefore("-D"), "val1", "val2"))
                .arguments())
        .containsExactlyElementsIn(customCommandLine.expected("-D", "val1", "-D", "val2"))
        .inOrder();
    assertThat(
            customCommandLine
                .addPaths(
                    vectorArg.each(
                        VectorArg.addBefore("-D"),
                        PathFragment.create("path1"),
                        PathFragment.create("path2")))
                .arguments())
        .containsExactlyElementsIn(customCommandLine.expected("-D", "path1", "-D", "path2"))
        .inOrder();
    assertThat(
            customCommandLine
                .addExecPaths(vectorArg.each(VectorArg.addBefore("-D"), artifact1, artifact2))
                .arguments())
        .containsExactlyElementsIn(
            customCommandLine.expected("-D", "dir/file1.txt", "-D", "dir/file2.txt"))
        .inOrder();
    assertThat(
            customCommandLine
                .addAll(
                    vectorArg
                        .each(VectorArg.addBefore("-D"), foo("1"), foo("2"))
                        .mapped(Foo::expandToStr))
                .arguments())
        .containsExactlyElementsIn(customCommandLine.expected("-D", "1", "-D", "2"))
        .inOrder();
  }

  @Test
  public void addBeforeEachFormattedVector_addsFormattedStringsPrependedWithArg(
      @TestParameter VectorArgMode vectorArg) throws Exception {
    assertThat(
            builder()
                .addAll(vectorArg.each(VectorArg.addBefore("-D").format("D%s"), "val1", "val2"))
                .build()
                .arguments())
        .containsExactly("-D", "Dval1", "-D", "Dval2")
        .inOrder();
    assertThat(
            builder()
                .addPaths(
                    vectorArg.each(
                        VectorArg.addBefore("-D").format("D%s"),
                        PathFragment.create("path1"),
                        PathFragment.create("path2")))
                .build()
                .arguments())
        .containsExactly("-D", "Dpath1", "-D", "Dpath2")
        .inOrder();
    assertThat(
            builder()
                .addExecPaths(
                    vectorArg.each(VectorArg.addBefore("-D").format("D%s"), artifact1, artifact2))
                .build()
                .arguments())
        .containsExactly("-D", "Ddir/file1.txt", "-D", "Ddir/file2.txt")
        .inOrder();
    assertThat(
            builder()
                .addAll(
                    vectorArg
                        .each(VectorArg.addBefore("-D").format("D%s"), foo("1"), foo("2"))
                        .mapped(Foo::expandToStr))
                .build()
                .arguments())
        .containsExactly("-D", "D1", "-D", "D2")
        .inOrder();
  }

  @Test
  public void addCombinedArgs_addsAllArguments() throws Exception {
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
  public void addNulls_addsNothing() throws Exception {
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
            .addPrefixedLabel("prefix", null, /* mainRepoMapping= */ null)
            .addPrefixedPath("prefix", null)
            .addPrefixedExecPath("prefix", null)
            .addAll((ImmutableList<String>) null)
            .addAll(ImmutableList.of())
            .addPaths((ImmutableList<PathFragment>) null)
            .addPaths(ImmutableList.of())
            .addExecPaths((ImmutableList<Artifact>) null)
            .addExecPaths(ImmutableList.of())
            .addAll((NestedSet<String>) null)
            .addAll(NestedSetBuilder.emptySet(Order.STABLE_ORDER))
            .addPaths((NestedSet<PathFragment>) null)
            .addPaths(NestedSetBuilder.emptySet(Order.STABLE_ORDER))
            .addExecPaths((NestedSet<Artifact>) null)
            .addExecPaths(NestedSetBuilder.emptySet(Order.STABLE_ORDER))
            .addAll(VectorArg.of((NestedSet<String>) null))
            .addAll(VectorArg.of(NestedSetBuilder.<String>emptySet(Order.STABLE_ORDER)))
            .addAll("foo", (ImmutableList<String>) null)
            .addAll("foo", ImmutableList.of())
            .addPaths("foo", (ImmutableList<PathFragment>) null)
            .addPaths("foo", ImmutableList.of())
            .addExecPaths("foo", (ImmutableList<Artifact>) null)
            .addExecPaths("foo", ImmutableList.of())
            .addAll("foo", (NestedSet<String>) null)
            .addAll("foo", NestedSetBuilder.emptySet(Order.STABLE_ORDER))
            .addPaths("foo", (NestedSet<PathFragment>) null)
            .addPaths("foo", NestedSetBuilder.emptySet(Order.STABLE_ORDER))
            .addExecPaths("foo", (NestedSet<Artifact>) null)
            .addExecPaths("foo", NestedSetBuilder.emptySet(Order.STABLE_ORDER))
            .addAll("foo", VectorArg.of((NestedSet<String>) null))
            .addAll("foo", VectorArg.of(NestedSetBuilder.<String>emptySet(Order.STABLE_ORDER)))
            .addPlaceholderTreeArtifactExecPath("foo", null)
            .build();

    assertThat(cl.arguments()).isEmpty();
  }

  @Test
  public void evaluateTreeFileArtifacts_replacesTreeArtifactsWithChildrenExecPaths()
      throws Exception {
    SpecialArtifact treeArtifactOne = createTreeArtifact("myArtifact/treeArtifact1");
    SpecialArtifact treeArtifactTwo = createTreeArtifact("myArtifact/treeArtifact2");

    CustomCommandLine commandLineTemplate =
        builder()
            .addPlaceholderTreeArtifactExecPath("--argOne", treeArtifactOne)
            .addPlaceholderTreeArtifactExecPath("--argTwo", treeArtifactTwo)
            .build();

    TreeFileArtifact treeFileArtifactOne =
        TreeFileArtifact.createTreeOutput(treeArtifactOne, "children/child1");
    TreeFileArtifact treeFileArtifactTwo =
        TreeFileArtifact.createTreeOutput(treeArtifactTwo, "children/child2");

    CustomCommandLine commandLine =
        commandLineTemplate.evaluateTreeFileArtifacts(
            ImmutableList.of(treeFileArtifactOne, treeFileArtifactTwo));

    assertThat(commandLine.arguments())
        .containsExactly(
            "--argOne",
            "dir/myArtifact/treeArtifact1/children/child1",
            "--argTwo",
            "dir/myArtifact/treeArtifact2/children/child2")
        .inOrder();
  }

  @Test
  public void addAllMappedTreeFileArtifacts_mapToRelativePath_addsTreeFileRelativePaths()
      throws Exception {
    SpecialArtifact treeArtifact = createTreeArtifact("myArtifact/treeArtifact");

    TreeFileArtifact treeFileArtifactOne =
        TreeFileArtifact.createTreeOutput(treeArtifact, "children/child1");
    TreeFileArtifact treeFileArtifactTwo =
        TreeFileArtifact.createTreeOutput(treeArtifact, "children/child2");

    CommandLineItem.MapFn<Artifact> expandParentRelativePath =
        (src, args) -> {
          try {
            args.accept(src.getTreeRelativePathString());
          } catch (EvalException e) {
            throw new IllegalStateException("Unexpected EvalException thown.", e);
          }
        };

    CustomCommandLine commandLineTemplate =
        builder()
            .addAll(
                VectorArg.SimpleVectorArg.of(
                        ImmutableList.of(treeFileArtifactOne, treeFileArtifactTwo))
                    .mapped(expandParentRelativePath))
            .build();

    assertThat(commandLineTemplate.arguments())
        .containsExactly("children/child1", "children/child2")
        .inOrder();
  }

  @Test
  public void arguments_unsubstitutedTreeArtifactPlaceholder_fails() {
    Artifact treeArtifactOne = createTreeArtifact("myArtifact/treeArtifact1");
    Artifact treeArtifactTwo = createTreeArtifact("myArtifact/treeArtifact2");

    CustomCommandLine commandLineTemplate =
        builder()
            .addPlaceholderTreeArtifactExecPath("--argOne", treeArtifactOne)
            .addPlaceholderTreeArtifactExecPath("--argTwo", treeArtifactTwo)
            .build();

    assertThrows(RuntimeException.class, commandLineTemplate::arguments);
  }

  @Test
  public void addToFingerPrint_computesUniqueKeyForDifferentCommandLines() throws Exception {
    NestedSet<String> values = NestedSetBuilder.<String>stableOrder().add("a").add("b").build();
    ImmutableList<CustomCommandLine> commandLines =
        ImmutableList.<CustomCommandLine>builder()
            .add(builder().add("arg").build())
            .add(builder().addFormatted("--foo=%s", "arg").build())
            .add(builder().addPrefixed("--foo=%s", "arg").build())
            .add(builder().addAll(values).build())
            .add(builder().addAll(VectorArg.addBefore("--foo=%s").each(values)).build())
            .add(builder().addAll(VectorArg.join("--foo=%s").each(values)).build())
            .add(builder().addAll(VectorArg.format("--foo=%s").each(values)).build())
            .add(
                builder()
                    .addAll(VectorArg.of(values).mapped((s, args) -> args.accept(s + "_mapped")))
                    .build())
            .build();

    // Ensure all these command lines have distinct keys
    ActionKeyContext actionKeyContext = new ActionKeyContext();
    Map<String, CustomCommandLine> digests = new HashMap<>();
    for (CustomCommandLine commandLine : commandLines) {
      Fingerprint fingerprint = new Fingerprint();
      commandLine.addToFingerprint(
          actionKeyContext,
          /* inputMetadataProvider= */ null,
          CoreOptions.OutputPathsMode.OFF,
          fingerprint);
      String digest = fingerprint.hexDigestAndReset();
      CustomCommandLine previous = digests.putIfAbsent(digest, commandLine);
      if (previous != null) {
        fail(
            String.format(
                "Found two command lines with identical digest %s: '%s' and '%s'",
                digest,
                Joiner.on(' ').join(previous.arguments()),
                Joiner.on(' ').join(commandLine.arguments())));
      }
    }
  }

  private SpecialArtifact createTreeArtifact(String rootRelativePath) {
    return ActionsTestUtil.createTreeArtifactWithGeneratingAction(
        rootDir, rootDir.getExecPath().getRelative(rootRelativePath));
  }

  private static <T> ImmutableList<T> list(T... objects) {
    return ImmutableList.copyOf(objects);
  }

  private static <T> NestedSet<T> nestedSet(T... objects) {
    return NestedSetBuilder.create(Order.STABLE_ORDER, objects);
  }

  private static Foo foo(String str) {
    return new Foo(str);
  }

  private static class Foo {
    private final String str;

    Foo(String str) {
      this.str = str;
    }

    static void expandToStr(Foo foo, Consumer<String> args) {
      args.accept(foo.str);
    }
  }
}
