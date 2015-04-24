// Copyright 2014 Google Inc. All rights reserved.
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
package com.google.devtools.build.lib.util;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

import com.google.common.collect.Sets;
import com.google.devtools.build.lib.testutil.MoreAsserts;
import com.google.devtools.build.lib.testutil.Scratch;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.util.Collection;

@RunWith(JUnit4.class)
public class DependencySetTest {

  private Scratch scratch = new Scratch();

  private DependencySet newDependencySet() {
    return new DependencySet(scratch.resolve("/"));
  }

  @Test
  public void dotDParser_simple() throws Exception {
    PathFragment file1 = new PathFragment("/usr/local/blah/blah/genhello/hello.cc");
    PathFragment file2 = new PathFragment("/usr/local/blah/blah/genhello/hello.h");
    String filename = "hello.o";
    Path dotd = scratch.file("/tmp/foo.d",
        filename + ": \\",
        " " + file1 + " \\",
        " " + file2 + " ");
    DependencySet depset = newDependencySet().read(dotd);
    MoreAsserts.assertSameContents(Sets.newHashSet(file1, file2),
                       depset.getDependencies());
    assertEquals(depset.getOutputFileName(), filename);
  }

  @Test
  public void dotDParser_simple_crlf() throws Exception {
    PathFragment file1 = new PathFragment("/usr/local/blah/blah/genhello/hello.cc");
    PathFragment file2 = new PathFragment("/usr/local/blah/blah/genhello/hello.h");
    String filename = "hello.o";
    Path dotd = scratch.file("/tmp/foo.d",
        filename + ": \\\r",
        " " + file1 + " \\\r",
        " " + file2 + " ");
    DependencySet depset = newDependencySet().read(dotd);
    MoreAsserts.assertSameContents(Sets.newHashSet(file1, file2),
                       depset.getDependencies());
    assertEquals(depset.getOutputFileName(), filename);
  }

  @Test
  public void dotDParser_simple_cr() throws Exception {
    PathFragment file1 = new PathFragment("/usr/local/blah/blah/genhello/hello.cc");
    PathFragment file2 = new PathFragment("/usr/local/blah/blah/genhello/hello.h");
    String filename = "hello.o";
    Path dotd = scratch.file("/tmp/foo.d",
        filename + ": \\\r"
        + " " + file1 + " \\\r"
        + " " + file2 + " ");
    DependencySet depset = newDependencySet().read(dotd);
    MoreAsserts.assertSameContents(Sets.newHashSet(file1, file2),
                       depset.getDependencies());
    assertEquals(depset.getOutputFileName(), filename);
  }

  @Test
  public void dotDParser_leading_crlf() throws Exception {
    PathFragment file1 = new PathFragment("/usr/local/blah/blah/genhello/hello.cc");
    PathFragment file2 = new PathFragment("/usr/local/blah/blah/genhello/hello.h");
    String filename = "hello.o";
    Path dotd = scratch.file("/tmp/foo.d",
        "\r\n" + filename + ": \\\r\n"
        + " " + file1 + " \\\r\n"
        + " " + file2 + " ");
    DependencySet depset = newDependencySet().read(dotd);
    MoreAsserts.assertSameContents(Sets.newHashSet(file1, file2),
                       depset.getDependencies());
    assertEquals(depset.getOutputFileName(), filename);
  }

  @Test
  public void dotDParser_oddFormatting() throws Exception {
    PathFragment file1 = new PathFragment("/usr/local/blah/blah/genhello/hello.cc");
    PathFragment file2 = new PathFragment("/usr/local/blah/blah/genhello/hello.h");
    PathFragment file3 = new PathFragment("/usr/local/blah/blah/genhello/other.h");
    PathFragment file4 = new PathFragment("/usr/local/blah/blah/genhello/onemore.h");
    String filename = "hello.o";
    Path dotd = scratch.file("/tmp/foo.d",
        filename + ": " + file1 + " \\",
        " " + file2 + "\\",
        " " + file3 + " " + file4);
    DependencySet depset = newDependencySet().read(dotd);
    MoreAsserts.assertSameContents(Sets.newHashSet(file1, file2, file3, file4),
                       depset.getDependencies());
    assertEquals(depset.getOutputFileName(), filename);
  }

  @Test
  public void dotDParser_relativeFilenames() throws Exception {
    PathFragment file1 = new PathFragment("hello.cc");
    PathFragment file2 = new PathFragment("hello.h");
    String filename = "hello.o";
    Path dotd = scratch.file("/tmp/foo.d",
        filename + ": \\",
        " " + file1 + " \\",
        " " + file2 + " ");
    DependencySet depset = newDependencySet().read(dotd);
    MoreAsserts.assertSameContents(Sets.newHashSet(file1, file2),
                       depset.getDependencies());
    assertEquals(depset.getOutputFileName(), filename);
  }

  @Test
  public void dotDParser_emptyFile() throws Exception {
    Path dotd = scratch.file("/tmp/empty.d");
    DependencySet depset = newDependencySet().read(dotd);
    Collection<PathFragment> headers = depset.getDependencies();
    if (!headers.isEmpty()) {
      fail("Not empty: " + headers.size() + " " + headers);
    }
    assertEquals(depset.getOutputFileName(), null);
  }

  @Test
  public void dotDParser_multipleTargets() throws Exception {
    PathFragment file1 = new PathFragment("/usr/local/blah/blah/genhello/hello.cc");
    PathFragment file2 = new PathFragment("/usr/local/blah/blah/genhello/hello.h");
    Path dotd = scratch.file("/tmp/foo.d",
        "hello.o: \\",
        " " + file1,
        "hello2.o: \\",
        " " + file2);
    MoreAsserts.assertSameContents(Sets.newHashSet(file1, file2),
        newDependencySet().read(dotd).getDependencies());
  }

  /*
   * Regression test: if gcc fails to execute remotely, and we retry locally, then the behavior
   * of gcc's DEPENDENCIES_OUTPUT option is to append, not overwrite, the .d file. As a result,
   * during retry, a second stanza is written to the file.
   *
   * We handle this by merging all of the stanzas.
   */
  @Test
  public void dotDParser_duplicateStanza() throws Exception {
    PathFragment file1 = new PathFragment("/usr/local/blah/blah/genhello/hello.cc");
    PathFragment file2 = new PathFragment("/usr/local/blah/blah/genhello/hello.h");
    PathFragment file3 = new PathFragment("/usr/local/blah/blah/genhello/other.h");
    Path dotd = scratch.file("/tmp/foo.d",
        "hello.o: \\",
        " " + file1 + " \\",
        " " + file2 + " ",
        "hello.o: \\",
        " " + file1 + " \\",
        " " + file3 + " ");
    MoreAsserts.assertSameContents(Sets.newHashSet(file1, file2, file3),
                       newDependencySet().read(dotd).getDependencies());
  }

  @Test
  public void writeSet() throws Exception {
    PathFragment file1 = new PathFragment("/usr/local/blah/blah/genhello/hello.cc");
    PathFragment file2 = new PathFragment("/usr/local/blah/blah/genhello/hello.h");
    PathFragment file3 = new PathFragment("/usr/local/blah/blah/genhello/other.h");
    String filename = "/usr/local/blah/blah/genhello/hello.o";

    DependencySet depSet1 = newDependencySet();
    depSet1.addDependency(file1);
    depSet1.addDependency(file2);
    depSet1.addDependency(file3);
    depSet1.setOutputFileName(filename);
    
    Path outfile = scratch.resolve(filename);
    Path dotd = scratch.resolve("/usr/local/blah/blah/genhello/hello.d");
    FileSystemUtils.createDirectoryAndParents(dotd.getParentDirectory());
    depSet1.write(outfile, ".d");

    String dotdContents = new String(FileSystemUtils.readContentAsLatin1(dotd));
    String expected =
        "usr/local/blah/blah/genhello/hello.o:  \\\n" +
        "  /usr/local/blah/blah/genhello/hello.cc \\\n" +
        "  /usr/local/blah/blah/genhello/hello.h \\\n" +
        "  /usr/local/blah/blah/genhello/other.h\n";
    assertEquals(expected, dotdContents);
    assertEquals(filename, depSet1.getOutputFileName());
  }

  @Test
  public void writeReadSet() throws Exception {
    String filename = "/usr/local/blah/blah/genhello/hello.d";
    PathFragment file1 = new PathFragment("/usr/local/blah/blah/genhello/hello.cc");
    PathFragment file2 = new PathFragment("/usr/local/blah/blah/genhello/hello.h");
    PathFragment file3 = new PathFragment("/usr/local/blah/blah/genhello/other.h");
    DependencySet depSet1 = newDependencySet();
    depSet1.addDependency(file1);
    depSet1.addDependency(file2);
    depSet1.addDependency(file3);
    depSet1.setOutputFileName(filename);

    Path dotd = scratch.resolve(filename);
    FileSystemUtils.createDirectoryAndParents(dotd.getParentDirectory());
    depSet1.write(dotd, ".d");
    
    DependencySet depSet2 = newDependencySet().read(dotd);
    assertEquals(depSet1, depSet2);
    // due to how pic.d files are written, absolute paths are changed into relatives
    assertEquals(depSet1.getOutputFileName(), "/" + depSet2.getOutputFileName());
  }

}
