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

import static org.junit.Assert.assertEquals;

import com.google.devtools.build.lib.events.Location.LineAndColumn;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.vfs.PathFragment;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for {@link LineNumberTable}.
 */
@RunWith(JUnit4.class)
public class LineNumberTableTest {
  private LineNumberTable create(String buffer) {
    return LineNumberTable.create(buffer.toCharArray(), PathFragment.create("/fake/file"));
  }

  @Test
  public void testEmpty() {
    LineNumberTable table = create("");
    assertEquals(new LineAndColumn(1, 1), table.getLineAndColumn(0));
  }

  @Test
  public void testNewline() {
    LineNumberTable table = create("\n");
    assertEquals(new LineAndColumn(1, 1), table.getLineAndColumn(0));
    assertEquals(new LineAndColumn(2, 1), table.getLineAndColumn(1));
  }

  @Test
  public void testOneLiner() {
    LineNumberTable table = create("foo");
    assertEquals(new LineAndColumn(1, 1), table.getLineAndColumn(0));
    assertEquals(new LineAndColumn(1, 2), table.getLineAndColumn(1));
    assertEquals(new LineAndColumn(1, 3), table.getLineAndColumn(2));
    assertEquals(Pair.of(0, 3), table.getOffsetsForLine(1));
  }

  @Test
  public void testMultiLiner() {
    LineNumberTable table = create("\ntwo\nthree\n\nfive\n");

    // \n
    assertEquals(new LineAndColumn(1, 1), table.getLineAndColumn(0));
    assertEquals(Pair.of(0, 1), table.getOffsetsForLine(1));

    // two\n
    assertEquals(new LineAndColumn(2, 1), table.getLineAndColumn(1));
    assertEquals(new LineAndColumn(2, 2), table.getLineAndColumn(2));
    assertEquals(new LineAndColumn(2, 3), table.getLineAndColumn(3));
    assertEquals(new LineAndColumn(2, 4), table.getLineAndColumn(4));
    assertEquals(Pair.of(1, 5), table.getOffsetsForLine(2));

    // three\n
    assertEquals(new LineAndColumn(3, 1), table.getLineAndColumn(5));
    assertEquals(new LineAndColumn(3, 6), table.getLineAndColumn(10));
    assertEquals(Pair.of(5, 11), table.getOffsetsForLine(3));

    // \n
    assertEquals(new LineAndColumn(4, 1), table.getLineAndColumn(11));
    assertEquals(Pair.of(11, 12), table.getOffsetsForLine(4));

    // five\n
    assertEquals(new LineAndColumn(5, 1), table.getLineAndColumn(12));
    assertEquals(new LineAndColumn(5, 5), table.getLineAndColumn(16));
    assertEquals(Pair.of(12, 17), table.getOffsetsForLine(5));
  }

  @Test
  public void testHashLine() {
    String data = "#\n"
        + "#line 67 \"/foo\"\n"
        + "cc_binary(name='a',\n"
        + "          srcs=[])\n"
        + "#line 23 \"/ba.r\"\n"
        + "vardef(x,y)\n";

    LineNumberTable table = create(data);

    // Note: no attempt is made to return accurate column information.
    assertEquals(new LineAndColumn(67, 1), table.getLineAndColumn(data.indexOf("cc_binary")));
    assertEquals(new LineAndColumn(67, 1), table.getLineAndColumn(data.indexOf("name='a'")));
    assertEquals("/fake/file", table.getPath(0).toString());
    // Note: newlines ignored; "srcs" is still (intentionally) considered to be
    // on L67.  Consider the alternative, and assume that rule 'a' is 50 lines
    // when pretty-printed: the last line of 'a' would be reported as line 67 +
    // 50, which may be in a part of the original BUILD file that has nothing
    // to do with this rule.  In other words, the size of rules before and
    // after pretty printing are essentially unrelated.
    assertEquals(new LineAndColumn(67, 1), table.getLineAndColumn(data.indexOf("srcs")));
    assertEquals("/foo", table.getPath(data.indexOf("srcs")).toString());
    assertEquals(Pair.of(2, 57), table.getOffsetsForLine(67));

    assertEquals(new LineAndColumn(23, 1), table.getLineAndColumn(data.indexOf("vardef")));
    assertEquals(new LineAndColumn(23, 1), table.getLineAndColumn(data.indexOf("x,y")));
    assertEquals("/ba.r", table.getPath(data.indexOf("x,y")).toString());
    assertEquals(Pair.of(57, 86), table.getOffsetsForLine(23));

    assertEquals(Pair.of(0, 0), table.getOffsetsForLine(42));
  }

}
