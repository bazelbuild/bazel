// Copyright 2006 The Bazel Authors.  All Rights Reserved.
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
package com.google.devtools.build.lib.syntax;


import com.google.devtools.build.lib.skyframe.serialization.testutils.SerializationTester;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link FileLocations}. */
// TODO(adonovan): express this test in terms of the public API.
@RunWith(JUnit4.class)
public class FileLocationsTest {

  private static FileLocations create(String buffer) {
    return FileLocations.create(buffer.toCharArray(), "/fake/file");
  }

  // Asserts that the specified offset results in a line/column pair of the form "1:2".
  private static void checkOffset(FileLocations table, int offset, String wantLineCol) {
    Location loc = table.getLocation(offset);
    String got = String.format("%d:%d", loc.line(), loc.column());
    if (!got.equals(wantLineCol)) {
      throw new AssertionError(
          String.format("location(%d) = %s, want %s", offset, got, wantLineCol));
    }
  }

  @Test
  public void testEmpty() {
    FileLocations table = create("");
    checkOffset(table, 0, "1:1");
  }

  @Test
  public void testNewline() {
    FileLocations table = create("\n");
    checkOffset(table, 0, "1:1");
    checkOffset(table, 1, "2:1"); // EOF
  }

  @Test
  public void testOneLiner() {
    FileLocations table = create("foo");
    checkOffset(table, 0, "1:1");
    checkOffset(table, 1, "1:2");
    checkOffset(table, 2, "1:3");
    checkOffset(table, 3, "1:4"); // EOF
  }

  @Test
  public void testMultiLiner() {
    FileLocations table = create("\ntwo\nthree\n\nfive\n");

    // \n
    checkOffset(table, 0, "1:1");

    // two\n
    checkOffset(table, 1, "2:1");
    checkOffset(table, 2, "2:2");
    checkOffset(table, 3, "2:3");
    checkOffset(table, 4, "2:4");

    // three\n
    checkOffset(table, 5, "3:1");
    checkOffset(table, 10, "3:6");

    // \n
    checkOffset(table, 11, "4:1");

    // five\n
    checkOffset(table, 12, "5:1");
    checkOffset(table, 16, "5:5");

    // start of final empty line
    checkOffset(table, 17, "6:1"); // EOF
  }

  @Test
  public void testCodec() throws Exception {
    new SerializationTester(
            create(
                "#\n"
                    + "#line 67 \"/foo\"\n"
                    + "cc_binary(name='a',\n"
                    + "          srcs=[])\n"
                    + "#line 23 \"/ba.r\"\n"
                    + "vardef(x,y)\n"),
            create("\ntwo\nthree\n\nfive\n"))
        .runTests();
  }
}
