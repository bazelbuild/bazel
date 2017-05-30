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

import static com.google.common.truth.Truth.assertThat;

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
    assertThat(table.getLineAndColumn(0)).isEqualTo(new LineAndColumn(1, 1));
  }

  @Test
  public void testNewline() {
    LineNumberTable table = create("\n");
    assertThat(table.getLineAndColumn(0)).isEqualTo(new LineAndColumn(1, 1));
    assertThat(table.getLineAndColumn(1)).isEqualTo(new LineAndColumn(2, 1));
  }

  @Test
  public void testOneLiner() {
    LineNumberTable table = create("foo");
    assertThat(table.getLineAndColumn(0)).isEqualTo(new LineAndColumn(1, 1));
    assertThat(table.getLineAndColumn(1)).isEqualTo(new LineAndColumn(1, 2));
    assertThat(table.getLineAndColumn(2)).isEqualTo(new LineAndColumn(1, 3));
    assertThat(table.getOffsetsForLine(1)).isEqualTo(Pair.of(0, 3));
  }

  @Test
  public void testMultiLiner() {
    LineNumberTable table = create("\ntwo\nthree\n\nfive\n");

    // \n
    assertThat(table.getLineAndColumn(0)).isEqualTo(new LineAndColumn(1, 1));
    assertThat(table.getOffsetsForLine(1)).isEqualTo(Pair.of(0, 1));

    // two\n
    assertThat(table.getLineAndColumn(1)).isEqualTo(new LineAndColumn(2, 1));
    assertThat(table.getLineAndColumn(2)).isEqualTo(new LineAndColumn(2, 2));
    assertThat(table.getLineAndColumn(3)).isEqualTo(new LineAndColumn(2, 3));
    assertThat(table.getLineAndColumn(4)).isEqualTo(new LineAndColumn(2, 4));
    assertThat(table.getOffsetsForLine(2)).isEqualTo(Pair.of(1, 5));

    // three\n
    assertThat(table.getLineAndColumn(5)).isEqualTo(new LineAndColumn(3, 1));
    assertThat(table.getLineAndColumn(10)).isEqualTo(new LineAndColumn(3, 6));
    assertThat(table.getOffsetsForLine(3)).isEqualTo(Pair.of(5, 11));

    // \n
    assertThat(table.getLineAndColumn(11)).isEqualTo(new LineAndColumn(4, 1));
    assertThat(table.getOffsetsForLine(4)).isEqualTo(Pair.of(11, 12));

    // five\n
    assertThat(table.getLineAndColumn(12)).isEqualTo(new LineAndColumn(5, 1));
    assertThat(table.getLineAndColumn(16)).isEqualTo(new LineAndColumn(5, 5));
    assertThat(table.getOffsetsForLine(5)).isEqualTo(Pair.of(12, 17));
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
    assertThat(table.getLineAndColumn(data.indexOf("cc_binary")))
        .isEqualTo(new LineAndColumn(67, 1));
    assertThat(table.getLineAndColumn(data.indexOf("name='a'")))
        .isEqualTo(new LineAndColumn(67, 1));
    assertThat(table.getPath(0).toString()).isEqualTo("/fake/file");
    // Note: newlines ignored; "srcs" is still (intentionally) considered to be
    // on L67.  Consider the alternative, and assume that rule 'a' is 50 lines
    // when pretty-printed: the last line of 'a' would be reported as line 67 +
    // 50, which may be in a part of the original BUILD file that has nothing
    // to do with this rule.  In other words, the size of rules before and
    // after pretty printing are essentially unrelated.
    assertThat(table.getLineAndColumn(data.indexOf("srcs"))).isEqualTo(new LineAndColumn(67, 1));
    assertThat(table.getPath(data.indexOf("srcs")).toString()).isEqualTo("/foo");
    assertThat(table.getOffsetsForLine(67)).isEqualTo(Pair.of(2, 57));

    assertThat(table.getLineAndColumn(data.indexOf("vardef"))).isEqualTo(new LineAndColumn(23, 1));
    assertThat(table.getLineAndColumn(data.indexOf("x,y"))).isEqualTo(new LineAndColumn(23, 1));
    assertThat(table.getPath(data.indexOf("x,y")).toString()).isEqualTo("/ba.r");
    assertThat(table.getOffsetsForLine(23)).isEqualTo(Pair.of(57, 86));

    assertThat(table.getOffsetsForLine(42)).isEqualTo(Pair.of(0, 0));
  }

}
