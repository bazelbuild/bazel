// Copyright 2026 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.starlark;

import static com.google.common.truth.Truth.assertThat;
import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.buildtool.util.BuildIntegrationTestCase;
import com.google.devtools.build.lib.testutil.TestConstants;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.Symlinks;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Execution-level tests for {@code ctx.actions.copy} on a local build. */
@RunWith(JUnit4.class)
public final class CopyActionIntegrationTest extends BuildIntegrationTestCase {

  /** One rule with a source file, generated file, trees, copies and a copy of a copy. */
  private void writeRule() throws Exception {
    write(
        "test/rule.bzl",
        """
        def _impl(ctx):
            gen = ctx.actions.declare_file("gen.txt")
            ctx.actions.write(output = gen, content = "generated")
            gen_copy = ctx.actions.declare_file("gen_copy.txt")
            ctx.actions.copy(output = gen_copy, src = gen)
            gen_copy_copy = ctx.actions.declare_file("gen_copy_copy.txt")
            ctx.actions.copy(output = gen_copy_copy, src = gen_copy)

            src_copy = ctx.actions.declare_file("src_copy.txt")
            ctx.actions.copy(output = src_copy, src = ctx.file.src)

            tree = ctx.actions.declare_directory("tree")
            ctx.actions.run_shell(
                outputs = [tree],
                command = "mkdir -p %s/a/b && echo one > %s/f1.txt && echo deep > %s/a/b/f2.txt"
                          % (tree.path, tree.path, tree.path) + " && : > %s/empty.txt" % tree.path,
            )
            tree_copy = ctx.actions.declare_directory("tree_copy")
            ctx.actions.copy(output = tree_copy, src = tree)
            tree_copy_copy = ctx.actions.declare_directory("tree_copy_copy")
            ctx.actions.copy(output = tree_copy_copy, src = tree_copy)

            empty_tree = ctx.actions.declare_directory("empty_tree")
            ctx.actions.run_shell(outputs = [empty_tree], command = "mkdir -p " + empty_tree.path)
            empty_tree_copy = ctx.actions.declare_directory("empty_tree_copy")
            ctx.actions.copy(output = empty_tree_copy, src = empty_tree)

            return [DefaultInfo(files = depset([
                gen_copy,
                gen_copy_copy,
                src_copy,
                tree_copy,
                tree_copy_copy,
                empty_tree_copy,
            ]))]

        copy_rule = rule(
            implementation = _impl,
            attrs = {"src": attr.label(allow_single_file = True)},
        )
        """);
    write(
        "test/BUILD",
        """
        load(":rule.bzl", "copy_rule")

        copy_rule(name = "target", src = "source.txt")
        """);
    write("test/source.txt", "source content");
  }

  private Path out(String name) throws Exception {
    for (Artifact a : getArtifacts("//test:target")) {
      if (a.getRootRelativePathString().equals("test/" + name)) {
        return a.getPath();
      }
    }
    throw new AssertionError("no output " + name);
  }

  private static String read(Path p) throws Exception {
    return new String(FileSystemUtils.readContent(p), UTF_8);
  }

  private void assertRegularFile(Path p, String content) throws Exception {
    assertThat(p.isSymbolicLink()).isFalse();
    assertThat(p.isFile(Symlinks.NOFOLLOW)).isTrue();
    assertThat(p.resolveSymbolicLinks()).isEqualTo(p);
    assertThat(read(p)).isEqualTo(content);
  }

  private void assertNormalizedPermissions(Path p) throws Exception {
    assertThat(p.isReadable()).isTrue();
    assertThat(p.isWritable()).isFalse();
    assertThat(p.isExecutable()).isTrue();
  }

  @Test
  public void fileCopies_areRegularFilesWithSourceContent() throws Exception {
    writeRule();
    buildTarget("//test:target");

    assertRegularFile(out("gen_copy.txt"), "generated");
    assertRegularFile(out("gen_copy_copy.txt"), "generated");
    assertRegularFile(out("src_copy.txt"), "source content\n");
    assertNormalizedPermissions(out("src_copy.txt"));
    assertNormalizedPermissions(out("gen_copy_copy.txt"));
  }

  @Test
  public void copy_hasSameDigestAsSource() throws Exception {
    writeRule();
    buildTarget("//test:target");

    FileArtifactValue copy = FileArtifactValue.createForTesting(out("src_copy.txt"));
    FileArtifactValue source =
        FileArtifactValue.createForTesting(getWorkspace().getRelative("test/source.txt"));
    assertThat(copy.getDigest()).isEqualTo(source.getDigest());
    assertThat(copy.getSize()).isEqualTo(source.getSize());
    FileArtifactValue copyOfCopy = FileArtifactValue.createForTesting(out("gen_copy_copy.txt"));
    FileArtifactValue copyOfGen = FileArtifactValue.createForTesting(out("gen_copy.txt"));
    assertThat(copyOfCopy.getDigest()).isEqualTo(copyOfGen.getDigest());
  }

  @Test
  public void treeCopies_haveRecordedChildrenAsRegularFiles() throws Exception {
    writeRule();
    buildTarget("//test:target");

    for (String name : new String[] {"tree_copy", "tree_copy_copy"}) {
      Path dir = out(name);
      assertThat(dir.isDirectory(Symlinks.NOFOLLOW)).isTrue();
      assertRegularFile(dir.getRelative("f1.txt"), "one\n");
      assertRegularFile(dir.getRelative("a/b/f2.txt"), "deep\n");
      // A child that is an empty file is still copied.
      assertRegularFile(dir.getRelative("empty.txt"), "");
      assertThat(dir.getRelative("empty.txt").getFileSize()).isEqualTo(0);
      assertNormalizedPermissions(dir.getRelative("f1.txt"));
      assertNormalizedPermissions(dir.getRelative("a/b/f2.txt"));
    }
  }

  @Test
  public void emptyTreeCopy_isEmptyDirectory() throws Exception {
    writeRule();
    buildTarget("//test:target");

    Path dir = out("empty_tree_copy");
    assertThat(dir.isDirectory(Symlinks.NOFOLLOW)).isTrue();
    assertThat(dir.getDirectoryEntries()).isEmpty();
  }

  @Test
  public void incremental_noRerunWhenNothingChanged() throws Exception {
    writeRule();
    buildTarget("//test:target");
    Path copy = out("src_copy.txt");
    Path treeChild = out("tree_copy").getRelative("f1.txt");
    long copyCtime = copy.stat().getLastChangeTime();
    long childCtime = treeChild.stat().getLastChangeTime();

    buildTarget("//test:target");

    assertThat(copy.stat().getLastChangeTime()).isEqualTo(copyCtime);
    assertThat(treeChild.stat().getLastChangeTime()).isEqualTo(childCtime);
  }

  @Test
  public void incremental_rerunsWhenSourceChanges() throws Exception {
    writeRule();
    buildTarget("//test:target");

    write("test/source.txt", "changed source");
    buildTarget("//test:target");

    assertRegularFile(out("src_copy.txt"), "changed source\n");
  }

  @Test
  public void incremental_rerunsWhenTreeSourceChanges() throws Exception {
    writeRule();
    buildTarget("//test:target");
    write(
        "test/BUILD",
        """
        load(":rule.bzl", "copy_rule")

        copy_rule(name = "target", src = "other.txt")
        """);
    write("test/other.txt", "other");
    buildTarget("//test:target");

    assertRegularFile(out("src_copy.txt"), "other\n");
    assertRegularFile(out("tree_copy").getRelative("f1.txt"), "one\n");
  }

  @Test
  public void incremental_rerunsWhenOutputEdited() throws Exception {
    writeRule();
    buildTarget("//test:target");
    Path copy = out("src_copy.txt");
    copy.setWritable(true);
    FileSystemUtils.writeContent(copy, UTF_8, "tampered");
    Path child = out("tree_copy").getRelative("f1.txt");
    child.setWritable(true);
    FileSystemUtils.writeContent(child, UTF_8, "tampered");

    buildTarget("//test:target");

    assertRegularFile(copy, "source content\n");
    assertRegularFile(child, "one\n");
  }

  @Test
  public void incremental_rerunsWhenOutputDeleted() throws Exception {
    writeRule();
    buildTarget("//test:target");
    out("src_copy.txt").delete();
    out("tree_copy").deleteTree();
    out("gen_copy_copy.txt").delete();

    buildTarget("//test:target");

    assertRegularFile(out("src_copy.txt"), "source content\n");
    assertRegularFile(out("tree_copy").getRelative("a/b/f2.txt"), "deep\n");
    assertRegularFile(out("gen_copy_copy.txt"), "generated");
  }

  @Test
  public void readOnlySource_givesReadOnlyOutput() throws Exception {
    writeRule();
    getWorkspace().getRelative("test/source.txt").setWritable(false);
    buildTarget("//test:target");

    Path copy = out("src_copy.txt");
    assertThat(copy.isWritable()).isFalse();
    // Re-running over a read-only output works.
    getWorkspace().getRelative("test/source.txt").setWritable(true);
    write("test/source.txt", "changed");
    buildTarget("//test:target");
    assertRegularFile(copy, "changed\n");
    assertThat(copy.isWritable()).isFalse();
  }

  @Test
  public void treeCopy_doesNotCopyLinksOrEmptyDirsLeftInSource() throws Exception {
    writeRule();
    buildTarget("//test:target");
    // Corrupt the on-disk source tree behind Bazel's back; the recorded children stay the same
    // until the producer re-runs, which it will not as its inputs are unchanged.
    Path srcTree = out("tree_copy").getParentDirectory().getRelative("tree");
    srcTree.setWritable(true);
    srcTree.getRelative("stray_dir").createDirectory();
    srcTree
        .getRelative("stray_link")
        .createSymbolicLink(srcTree.getRelative("f1.txt").asFragment());
    out("tree_copy").deleteTree();

    buildTarget("//test:target");

    assertThat(out("tree_copy").getRelative("stray_dir").exists()).isFalse();
    assertThat(out("tree_copy").getRelative("stray_link").exists()).isFalse();
    assertRegularFile(out("tree_copy").getRelative("f1.txt"), "one\n");
  }

  @Test
  public void consumer_seesCopyContent_defaultStrategy() throws Exception {
    writeRule();
    write(
        "consumer/BUILD",
        """
        load("//test:rule.bzl", "copy_rule")

        copy_rule(name = "copies", src = "//test:source.txt")

        genrule(
            name = "consume",
            srcs = [":copies"],
            outs = ["consumed.txt"],
            cmd = "for f in $(SRCS); do case $$f in"
                + " *src_copy.txt) cat $$f > $@;;"
                + " */tree_copy) cat $$f/a/b/f2.txt >> $@;;"
                + " esac; done",
        )
        """);
    write(
        "test/BUILD",
        """
        load(":rule.bzl", "copy_rule")

        exports_files(["source.txt"])
        copy_rule(name = "target", src = "source.txt")
        """);

    buildTarget("//consumer:consume");

    assertThat(read(getArtifacts("//consumer:consume").get(0).getPath()))
        .isEqualTo("source content\ndeep\n");
  }

  @Test
  public void consumer_seesCopyContent_localStrategy() throws Exception {
    addOptions("--spawn_strategy=local");
    consumer_seesCopyContent_defaultStrategy();
  }
}
