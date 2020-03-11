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

package com.google.devtools.build.lib.bazel.rules.ninja;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.testutil.MoreAsserts.assertThrows;
import static junit.framework.TestCase.fail;

import com.google.common.collect.ImmutableList;
import com.google.common.util.concurrent.ListeningExecutorService;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.common.util.concurrent.ThreadFactoryBuilder;
import com.google.devtools.build.lib.bazel.rules.ninja.file.GenericParsingException;
import com.google.devtools.build.lib.bazel.rules.ninja.parser.NinjaPool;
import com.google.devtools.build.lib.bazel.rules.ninja.parser.NinjaRule;
import com.google.devtools.build.lib.bazel.rules.ninja.parser.NinjaRuleVariable;
import com.google.devtools.build.lib.bazel.rules.ninja.parser.NinjaTarget;
import com.google.devtools.build.lib.bazel.rules.ninja.parser.NinjaVariableValue;
import com.google.devtools.build.lib.bazel.rules.ninja.pipeline.NinjaPipeline;
import com.google.devtools.build.lib.concurrent.ExecutorUtil;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.DigestHashFunction.DefaultHashFunctionNotSetException;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.JavaIoFileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Executors;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link com.google.devtools.build.lib.bazel.rules.ninja.pipeline.NinjaPipeline}. */
@RunWith(JUnit4.class)
public class NinjaPipelineTest {
  private static class Tester {
    private final Path dir;
    private final ListeningExecutorService service;

    Tester() throws IOException, DefaultHashFunctionNotSetException {
      service =
          MoreExecutors.listeningDecorator(
              Executors.newFixedThreadPool(
                  25,
                  new ThreadFactoryBuilder()
                      .setNameFormat(NinjaPipelineTest.class.getSimpleName() + "-%d")
                      .build()));
      java.nio.file.Path tmpDir = Files.createTempDirectory("test");
      dir = new JavaIoFileSystem().getPath(tmpDir.toString());
    }

    ListeningExecutorService getService() {
      return service;
    }

    Path writeTmpFile(String name, String... lines) throws IOException {
      Path path = dir.getRelative(name);
      if (lines.length > 0) {
        FileSystemUtils.writeContent(
            path, String.join("\n", lines).getBytes(StandardCharsets.ISO_8859_1));
      } else {
        FileSystemUtils.createEmptyFile(path);
      }
      return path;
    }

    public void tearDown() throws IOException {
      ExecutorUtil.interruptibleShutdown(service);
      dir.deleteTree();
    }
  }

  private Tester tester;

  @Before
  public void setUp() throws Exception {
    try {
      DigestHashFunction.setDefault(DigestHashFunction.SHA256);
    } catch (DigestHashFunction.DefaultAlreadySetException e) {
      // Do nothing.
    }
    tester = new Tester();
  }

  @After
  public void tearDown() throws Exception {
    tester.tearDown();
  }

  @Test
  public void testOneFilePipeline() throws Exception {
    Path vfsPath =
        tester.writeTmpFile(
            "test.ninja",
            "rule r1",
            "  command = c $in $out",
            "build t1: r1 in1 in2",
            "build t2: r1 in3");
    NinjaPipeline pipeline =
        new NinjaPipeline(
            vfsPath.getParentDirectory(), tester.getService(), ImmutableList.of(), "ninja_target");
    List<NinjaTarget> targets = pipeline.pipeline(vfsPath);
    checkTargets(targets);
  }

  @Test
  public void testOneFilePipelineWithNewlines() throws Exception {
    Path vfsPath =
        tester.writeTmpFile(
            "test.ninja",
            "",
            "",
            "",
            "",
            "rule r1",
            "  command = c $in $out",
            "",
            "",
            "",
            "build t1: r1 in1 in2",
            "",
            "",
            "",
            "build t2: r1 in3",
            "");
    NinjaPipeline pipeline =
        new NinjaPipeline(
            vfsPath.getParentDirectory(), tester.getService(), ImmutableList.of(), "ninja_target");
    List<NinjaTarget> targets = pipeline.pipeline(vfsPath);
    checkTargets(targets);
  }

  @Test
  public void testWithIncluded() throws Exception {
    Path vfsPath =
        tester.writeTmpFile(
            "test.ninja", "rule r1", "  command = c $in $out", "include child.ninja");
    Path childFile = tester.writeTmpFile("child.ninja", "build t1: r1 in1 in2", "build t2: r1 in3");
    NinjaPipeline pipeline =
        new NinjaPipeline(
            vfsPath.getParentDirectory(),
            tester.getService(),
            ImmutableList.of(childFile),
            "ninja_target");
    List<NinjaTarget> targets = pipeline.pipeline(vfsPath);
    checkTargets(targets);
  }

  @Test
  public void testComputedSubNinja() throws Exception {
    Path vfsPath =
        tester.writeTmpFile(
            "test.ninja",
            "subfile=child",
            "rule r1",
            "  command = c $in $out",
            "include ${subfile}.ninja");
    Path childFile = tester.writeTmpFile("child.ninja", "build t1: r1 in1 in2", "build t2: r1 in3");
    NinjaPipeline pipeline =
        new NinjaPipeline(
            vfsPath.getParentDirectory(),
            tester.getService(),
            ImmutableList.of(childFile),
            "ninja_target");
    List<NinjaTarget> targets = pipeline.pipeline(vfsPath);
    checkTargets(targets);
  }

  @Test
  public void testComputedDeeplyIncluded() throws Exception {
    Path vfsPath =
        tester.writeTmpFile(
            "test.ninja",
            "subfile=child",
            "subninja_file=sub",
            "rule r1",
            "  command = c $in $out",
            "include ${subfile}.ninja",
            "build t1: r1 ${top_scope_var} in2");
    Path childFile =
        tester.writeTmpFile(
            "child.ninja",
            "top_scope_var=in1",
            "var_for_sub=in3",
            "subninja ${subninja_file}.ninja");
    Path subFile = tester.writeTmpFile("sub.ninja", "build t2: r1 ${var_for_sub}");
    NinjaPipeline pipeline =
        new NinjaPipeline(
            vfsPath.getParentDirectory(),
            tester.getService(),
            ImmutableList.of(childFile, subFile),
            "ninja_target");
    List<NinjaTarget> targets = pipeline.pipeline(vfsPath);
    checkTargets(targets);
  }

  @Test
  public void testEmptyFile() throws Exception {
    Path vfsPath = tester.writeTmpFile("test.ninja");
    NinjaPipeline pipeline =
        new NinjaPipeline(
            vfsPath.getParentDirectory(), tester.getService(), ImmutableList.of(), "ninja_target");
    List<NinjaTarget> targets = pipeline.pipeline(vfsPath);
    assertThat(targets).isEmpty();
  }

  @Test
  public void testIncludedNinjaFileIsNotDeclared() throws Exception {
    Path vfsPath = tester.writeTmpFile("test.ninja", "include subfile.ninja");
    GenericParsingException exception =
        assertThrows(
            GenericParsingException.class,
            () ->
                new NinjaPipeline(
                        vfsPath.getParentDirectory(),
                        tester.getService(),
                        ImmutableList.of(),
                        "ninja_target")
                    .pipeline(vfsPath));
    assertThat(exception)
        .hasMessageThat()
        .isEqualTo(
            "Ninja file requested from 'test.ninja' "
                + "not declared in 'srcs' attribute of 'ninja_target'.");
  }

  @Test
  public void testIncludeCycle() throws Exception {
    Path vfsPath = tester.writeTmpFile("test.ninja", "include one.ninja");
    Path oneFile = tester.writeTmpFile("one.ninja", "include two.ninja");
    Path twoFile = tester.writeTmpFile("two.ninja", "include one.ninja");
    NinjaPipeline pipeline =
        new NinjaPipeline(
            vfsPath.getParentDirectory(),
            tester.getService(),
            ImmutableList.of(oneFile, twoFile),
            "ninja_target");
    GenericParsingException exception =
        assertThrows(GenericParsingException.class, () -> pipeline.pipeline(vfsPath));
    assertThat(exception)
        .hasMessageThat()
        .isEqualTo(
            "Detected cycle or duplicate inclusion in Ninja files dependencies, "
                + "including 'one.ninja'.");
  }

  @Test
  public void testSpacesInEmptyLine() throws Exception {
    Path vfsPath =
        tester.writeTmpFile(
            "test.ninja",
            "subfile=child",
            "   ",
            "subninja_file=sub",
            "   ",
            "rule r1",
            "  command = c $in $out",
            // This could be interpreted as indent, but should be skipped as the empty line.
            "   ",
            "   \n\n\n\n",
            "include ${subfile}.ninja",
            "   ",
            "build t2: r1 in3",
            "   ",
            "build t1: r1 in1 in2");
    Path childFile = tester.writeTmpFile("child.ninja");
    NinjaPipeline pipeline =
        new NinjaPipeline(
            vfsPath.getParentDirectory(),
            tester.getService(),
            ImmutableList.of(childFile),
            "ninja_target");
    List<NinjaTarget> targets = pipeline.pipeline(vfsPath);
    checkTargets(targets);
  }

  @Test
  public void testBigFile() throws Exception {
    String[] lines = new String[1000];
    for (int i = 0; i < lines.length - 2; i++) {
      lines[i] = "rule rule" + i + "\n command = echo 'Hello' > ${out}";
    }
    lines[998] = "build out: rule1";
    lines[999] = "pool link_pool\n  depth = 4";
    Path path = tester.writeTmpFile("big_file.ninja", lines);
    NinjaPipeline pipeline =
        new NinjaPipeline(
            path.getParentDirectory(), tester.getService(), ImmutableList.of(), "ninja_target");
    // Test specifically that all manipulations with connecting buffers are working fine:
    // for that, have relatively long file and small buffer size.
    pipeline.setReadBlockSize(100);
    List<NinjaTarget> targets = pipeline.pipeline(path);
    assertThat(targets).hasSize(1);
    Map<String, List<Pair<Long, NinjaPool>>> pools = targets.get(0).getScope().getPools();
    assertThat(pools).hasSize(1);
    Map<String, List<Pair<Long, NinjaRule>>> rules = targets.get(0).getScope().getRules();
    assertThat(rules).hasSize(998);
    assertThat(rules.get("rule1")).hasSize(1);
    NinjaVariableValue expectedValue =
        NinjaVariableValue.builder().addText("echo 'Hello' > ").addVariable("out").build();
    assertThat(
            rules
                .get("rule1")
                .get(0)
                .getSecond()
                .getVariables()
                .get(NinjaRuleVariable.COMMAND)
                .getRawText())
        .isEqualTo(expectedValue.getRawText());
  }

  private static void checkTargets(List<NinjaTarget> targets) {
    assertThat(targets).hasSize(2);
    for (NinjaTarget target : targets) {
      if (target.getAllOutputs().contains(PathFragment.create("t1"))) {
        assertThat(target.getAllInputs())
            .containsExactly(PathFragment.create("in1"), PathFragment.create("in2"));
      } else if (target.getAllOutputs().contains(PathFragment.create("t2"))) {
        assertThat(target.getAllInputs()).containsExactly(PathFragment.create("in3"));
      } else {
        fail();
      }
    }
  }
}
