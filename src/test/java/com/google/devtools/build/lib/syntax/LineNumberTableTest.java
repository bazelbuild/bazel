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

import com.google.devtools.build.lib.skyframe.serialization.testutils.SerializationTester;
import com.google.devtools.build.lib.syntax.Location.LineAndColumn;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for {@link LineNumberTable}.
 */
@RunWith(JUnit4.class)
public class LineNumberTableTest {

  private LineNumberTable create(String buffer) {
    return LineNumberTable.create(buffer.toCharArray(), "/fake/file");
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
  }

  @Test
  public void testMultiLiner() {
    LineNumberTable table = create("\ntwo\nthree\n\nfive\n");

    // \n
    assertThat(table.getLineAndColumn(0)).isEqualTo(new LineAndColumn(1, 1));

    // two\n
    assertThat(table.getLineAndColumn(1)).isEqualTo(new LineAndColumn(2, 1));
    assertThat(table.getLineAndColumn(2)).isEqualTo(new LineAndColumn(2, 2));
    assertThat(table.getLineAndColumn(3)).isEqualTo(new LineAndColumn(2, 3));
    assertThat(table.getLineAndColumn(4)).isEqualTo(new LineAndColumn(2, 4));

    // three\n
    assertThat(table.getLineAndColumn(5)).isEqualTo(new LineAndColumn(3, 1));
    assertThat(table.getLineAndColumn(10)).isEqualTo(new LineAndColumn(3, 6));

    // \n
    assertThat(table.getLineAndColumn(11)).isEqualTo(new LineAndColumn(4, 1));

    // five\n
    assertThat(table.getLineAndColumn(12)).isEqualTo(new LineAndColumn(5, 1));
    assertThat(table.getLineAndColumn(16)).isEqualTo(new LineAndColumn(5, 5));
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
