// Copyright 2006-2015 Google Inc. All Rights Reserved.
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
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Lists;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.util.Arrays;
import java.util.HashMap;
import java.util.IllegalFormatException;
import java.util.List;
import java.util.Map;

/**
 *  Test properties of the evaluator's datatypes and utility functions
 *  without actually creating any parse trees.
 */
@RunWith(JUnit4.class)
public class PrinterTest {

  private static List<?> makeList(Object... args) {
    return EvalUtils.makeSequence(Arrays.<Object>asList(args), false);
  }

  private static List<?> makeTuple(Object... args) {
    return EvalUtils.makeSequence(Arrays.<Object>asList(args), true);
  }

  private static FilesetEntry makeFilesetEntry() {
    try {
      return new FilesetEntry(Label.parseAbsolute("//foo:bar"),
                              Lists.<Label>newArrayList(), Lists.newArrayList("xyz"), "",
                              FilesetEntry.SymlinkBehavior.COPY, ".");
    } catch (LabelSyntaxException e) {
      throw new RuntimeException("Bad label: ", e);
    }
  }

  @Test
  public void testPrinter() throws Exception {
    // Note that prettyPrintValue and printValue only differ on behaviour of
    // labels and strings at toplevel.
    assertEquals("foo\nbar", Printer.str("foo\nbar"));
    assertEquals("\"foo\\nbar\"", Printer.repr("foo\nbar"));
    assertEquals("'", Printer.str("'"));
    assertEquals("\"'\"", Printer.repr("'"));
    assertEquals("\"", Printer.str("\""));
    assertEquals("\"\\\"\"", Printer.repr("\""));
    assertEquals("3", Printer.str(3));
    assertEquals("3", Printer.repr(3));
    assertEquals("None", Printer.repr(Runtime.NONE));

    assertEquals("//x:x", Printer.str(Label.parseAbsolute("//x")));
    assertEquals("\"//x:x\"", Printer.repr(Label.parseAbsolute("//x")));

    List<?> list = makeList("foo", "bar");
    List<?> tuple = makeTuple("foo", "bar");

    assertEquals("(1, [\"foo\", \"bar\"], 3)",
                 Printer.str(makeTuple(1, list, 3)));
    assertEquals("(1, [\"foo\", \"bar\"], 3)",
                 Printer.repr(makeTuple(1, list, 3)));
    assertEquals("[1, (\"foo\", \"bar\"), 3]",
                 Printer.str(makeList(1, tuple, 3)));
    assertEquals("[1, (\"foo\", \"bar\"), 3]",
                 Printer.repr(makeList(1, tuple, 3)));

    Map<Object, Object> dict = ImmutableMap.<Object, Object>of(
        1, tuple,
        2, list,
        "foo", makeList());
    assertEquals("{1: (\"foo\", \"bar\"), 2: [\"foo\", \"bar\"], \"foo\": []}",
                Printer.str(dict));
    assertEquals("{1: (\"foo\", \"bar\"), 2: [\"foo\", \"bar\"], \"foo\": []}",
                Printer.repr(dict));
    assertEquals("FilesetEntry(srcdir = \"//foo:bar\", files = [], "
               + "excludes = [\"xyz\"], destdir = \"\", "
               + "strip_prefix = \".\", symlinks = \"copy\")",
                 Printer.repr(makeFilesetEntry()));
  }

  private void checkFormatPositionalFails(String errorMessage, String format, Object... arguments) {
    try {
      Printer.format(format, arguments);
      fail();
    } catch (IllegalFormatException e) {
      assertThat(e).hasMessage(errorMessage);
    }
  }

  @Test
  public void testSortedOutputOfUnsortedMap() throws Exception {
    Map<Integer, Integer> map = new HashMap<>();
    int[] data = {5, 7, 3};

    for (int current : data) {
      map.put(current, current);
    }
    assertThat(Printer.str(map)).isEqualTo("{3: 3, 5: 5, 7: 7}");
  }

  @Test
  public void testFormatPositional() throws Exception {
    assertEquals("foo 3", Printer.formatToString("%s %d", makeTuple("foo", 3)));
    assertEquals("foo 3", Printer.format("%s %d", "foo", 3));

    // Note: formatToString doesn't perform scalar x -> (x) conversion;
    // The %-operator is responsible for that.
    assertThat(Printer.formatToString("", makeTuple())).isEmpty();
    assertEquals("foo", Printer.format("%s", "foo"));
    assertEquals("3.14159", Printer.format("%s", 3.14159));
    checkFormatPositionalFails("not all arguments converted during string formatting",
        "%s", 1, 2, 3);
    assertEquals("%foo", Printer.format("%%%s", "foo"));
    checkFormatPositionalFails("not all arguments converted during string formatting",
        "%%s", "foo");
    checkFormatPositionalFails("unsupported format character \" \" at index 1 in \"% %s\"",
        "% %s", "foo");
    assertEquals("[1, 2, 3]", Printer.format("%s", makeList(1, 2, 3)));
    assertEquals("(1, 2, 3)", Printer.format("%s", makeTuple(1, 2, 3)));
    assertEquals("[]", Printer.format("%s", makeList()));
    assertEquals("()", Printer.format("%s", makeTuple()));
    assertEquals("% 1 \"2\" 3", Printer.format("%% %d %r %s", 1, "2", "3"));

    checkFormatPositionalFails(
        "invalid argument \"1\" for format pattern %d",
        "%d", "1");
    checkFormatPositionalFails("unsupported format character \".\" at index 1 in \"%.3g\"",
        "%.3g");
    checkFormatPositionalFails("unsupported format character \".\" at index 1 in \"%.3g\"",
        "%.3g", 1, 2);
    checkFormatPositionalFails("unsupported format character \".\" at index 1 in \"%.s\"",
        "%.s");
  }

  @Test
  public void testFilesetEntrySymlinkAttr() throws Exception {
    FilesetEntry entryDereference =
      createTestFilesetEntry(FilesetEntry.SymlinkBehavior.DEREFERENCE);

    assertEquals(
        createExpectedFilesetEntryString(FilesetEntry.SymlinkBehavior.DEREFERENCE, '"'),
        Printer.repr(entryDereference));
  }

  private FilesetEntry createStripPrefixFilesetEntry(String stripPrefix)  throws Exception {
    Label label = Label.parseAbsolute("//x");
    return new FilesetEntry(
        label,
        Arrays.asList(label),
        Arrays.<String>asList(),
        "",
        FilesetEntry.SymlinkBehavior.DEREFERENCE,
        stripPrefix);
  }

  @Test
  public void testFilesetEntryStripPrefixAttr() throws Exception {
    FilesetEntry withoutStripPrefix = createStripPrefixFilesetEntry(".");
    FilesetEntry withStripPrefix = createStripPrefixFilesetEntry("orange");

    String prettyWithout = Printer.repr(withoutStripPrefix);
    String prettyWith = Printer.repr(withStripPrefix);

    assertThat(prettyWithout).contains("strip_prefix = \".\"");
    assertThat(prettyWith).contains("strip_prefix = \"orange\"");
  }

  @Test
  public void testRegressionCrashInPrettyPrintValue() throws Exception {
    // Would cause crash in code such as this:
    //  Fileset(name='x', entries=[], out=[FilesetEntry(files=['a'])])
    // While formatting the "expected x, got y" message for the 'out'
    // attribute, prettyPrintValue(FilesetEntry) would be recursively called
    // with a List<Label> even though this isn't a valid datatype in the
    // interpreter.
    // Fileset isn't part of bazel, even though FilesetEntry is.
    assertEquals(createExpectedFilesetEntryString('"'), Printer.repr(createTestFilesetEntry()));
  }

  @Test
  public void testSingleQuotes() throws Exception {
    assertThat(Printer.str("test", '\'')).isEqualTo("test");
    assertThat(Printer.repr("test", '\'')).isEqualTo("'test'");

    assertEquals("'\\''", Printer.repr("'", '\''));
    assertEquals("\"", Printer.str("\"", '\''));
    assertEquals("'\"'", Printer.repr("\"", '\''));

    List<?> list = makeList("foo", "bar");
    List<?> tuple = makeTuple("foo", "bar");

    assertThat(Printer.str(makeTuple(1, list, 3), '\'')).isEqualTo("(1, ['foo', 'bar'], 3)");
    assertThat(Printer.repr(makeTuple(1, list, 3), '\'')).isEqualTo("(1, ['foo', 'bar'], 3)");
    assertThat(Printer.str(makeList(1, tuple, 3), '\'')).isEqualTo("[1, ('foo', 'bar'), 3]");
    assertThat(Printer.repr(makeList(1, tuple, 3), '\'')).isEqualTo("[1, ('foo', 'bar'), 3]");

    Map<Object, Object> dict =
        ImmutableMap.<Object, Object>of(1, tuple, 2, list, "foo", makeList());

    assertThat(Printer.str(dict, '\''))
        .isEqualTo("{1: ('foo', 'bar'), 2: ['foo', 'bar'], 'foo': []}");
    assertThat(Printer.repr(dict, '\''))
        .isEqualTo("{1: ('foo', 'bar'), 2: ['foo', 'bar'], 'foo': []}");

    assertThat(Printer.repr(createTestFilesetEntry(), '\''))
        .isEqualTo(createExpectedFilesetEntryString('\''));
  }

  private String createExpectedFilesetEntryString(
      FilesetEntry.SymlinkBehavior symlinkBehavior, char quotationMark) {
    return String.format(
        "FilesetEntry(srcdir = %1$c//x:x%1$c,"
        + " files = [%1$c//x:x%1$c],"
        + " excludes = [],"
        + " destdir = %1$c%1$c,"
        + " strip_prefix = %1$c.%1$c,"
        + " symlinks = %1$c%2$s%1$c)",
        quotationMark, symlinkBehavior.toString().toLowerCase());
  }

  private String createExpectedFilesetEntryString(char quotationMark) {
    return createExpectedFilesetEntryString(FilesetEntry.SymlinkBehavior.COPY, quotationMark);
  }

  private FilesetEntry createTestFilesetEntry(FilesetEntry.SymlinkBehavior symlinkBehavior)
      throws LabelSyntaxException {
    Label label = Label.parseAbsolute("//x");
    return new FilesetEntry(
        label, Arrays.asList(label), Arrays.<String>asList(), "", symlinkBehavior, ".");
  }

  private FilesetEntry createTestFilesetEntry() throws LabelSyntaxException {
    return createTestFilesetEntry(FilesetEntry.SymlinkBehavior.COPY);
  }
}
