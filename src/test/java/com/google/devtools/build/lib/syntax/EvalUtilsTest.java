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
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import com.google.common.collect.Lists;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.util.Arrays;
import java.util.IllegalFormatException;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 *  Test properties of the evaluator's datatypes and utility functions
 *  without actually creating any parse trees.
 */
@RunWith(JUnit4.class)
public class EvalUtilsTest {

  private static List<?> makeList(Object ...args) {
    return EvalUtils.makeSequence(Arrays.<Object>asList(args), false);
  }
  private static List<?> makeTuple(Object ...args) {
    return EvalUtils.makeSequence(Arrays.<Object>asList(args), true);
  }
  private static Map<Object, Object> makeDict() {
    return new LinkedHashMap<>();
  }
  private static FilesetEntry makeFilesetEntry() {
    try {
      return new FilesetEntry(Label.parseAbsolute("//foo:bar"),
                              Lists.<Label>newArrayList(), Lists.newArrayList("xyz"), "",
                              FilesetEntry.SymlinkBehavior.COPY, ".");
    } catch (Label.SyntaxException e) {
      throw new RuntimeException("Bad label: ", e);
    }
  }

  @Test
  public void testDataTypeNames() throws Exception {
    assertEquals("string", EvalUtils.getDataTypeName("foo"));
    assertEquals("int", EvalUtils.getDataTypeName(3));
    assertEquals("Tuple", EvalUtils.getDataTypeName(makeTuple(1, 2, 3)));
    assertEquals("List",  EvalUtils.getDataTypeName(makeList(1, 2, 3)));
    assertEquals("dict",  EvalUtils.getDataTypeName(makeDict()));
    assertEquals("FilesetEntry",  EvalUtils.getDataTypeName(makeFilesetEntry()));
    assertEquals("NoneType", EvalUtils.getDataTypeName(Environment.NONE));
  }

  @Test
  public void testDatatypeMutability() throws Exception {
    assertTrue(EvalUtils.isImmutable("foo"));
    assertTrue(EvalUtils.isImmutable(3));
    assertTrue(EvalUtils.isImmutable(makeTuple(1, 2, 3)));
    assertFalse(EvalUtils.isImmutable(makeList(1, 2, 3)));
    assertFalse(EvalUtils.isImmutable(makeDict()));
    assertFalse(EvalUtils.isImmutable(makeFilesetEntry()));
  }

  @Test
  public void testPrintValue() throws Exception {
    // Note that prettyPrintValue and printValue only differ on behaviour of
    // labels and strings at toplevel.
    assertEquals("foo\nbar", EvalUtils.printValue("foo\nbar"));
    assertEquals("\"foo\\nbar\"", EvalUtils.prettyPrintValue("foo\nbar"));
    assertEquals("foo\nbar", EvalUtils.printValue("foo\nbar"));
    assertEquals("'", EvalUtils.printValue("'"));
    assertEquals("\"'\"", EvalUtils.prettyPrintValue("'"));
    assertEquals("\"", EvalUtils.printValue("\""));
    assertEquals("\"\\\"\"", EvalUtils.prettyPrintValue("\""));
    assertEquals("a\\b", EvalUtils.printValue("a\\b"));
    assertEquals("\"a\\\\b\"", EvalUtils.prettyPrintValue("a\\b"));
    assertEquals("3", EvalUtils.printValue(3));
    assertEquals("3", EvalUtils.prettyPrintValue(3));
    assertEquals("None", EvalUtils.printValue(Environment.NONE));
    assertEquals("None", EvalUtils.prettyPrintValue(Environment.NONE));

    assertEquals("//x:x", EvalUtils.printValue(Label.parseAbsolute("//x")));
    assertEquals("\"//x:x\"", EvalUtils.prettyPrintValue(Label.parseAbsolute("//x")));

    List<?> list = makeList("foo", "bar");
    List<?> tuple = makeTuple("foo", "bar");

    assertEquals("(1, [\"foo\", \"bar\"], 3)",
                 EvalUtils.printValue(makeTuple(1, list, 3)));
    assertEquals("(1, [\"foo\", \"bar\"], 3)",
                 EvalUtils.prettyPrintValue(makeTuple(1, list, 3)));
    assertEquals("[1, (\"foo\", \"bar\"), 3]",
                 EvalUtils.printValue(makeList(1, tuple, 3)));
    assertEquals("[1, (\"foo\", \"bar\"), 3]",
                 EvalUtils.prettyPrintValue(makeList(1, tuple, 3)));

    Map<Object, Object> dict = makeDict();
    dict.put(1, tuple);
    dict.put(2, list);
    dict.put("foo", makeList());
    assertEquals("{1: (\"foo\", \"bar\"), 2: [\"foo\", \"bar\"], \"foo\": []}",
                EvalUtils.printValue(dict));
    assertEquals("{1: (\"foo\", \"bar\"), 2: [\"foo\", \"bar\"], \"foo\": []}",
                EvalUtils.prettyPrintValue(dict));
    assertEquals("FilesetEntry(srcdir = \"//foo:bar\", files = [], "
               + "excludes = [\"xyz\"], destdir = \"\", "
               + "strip_prefix = \".\", symlinks = \"copy\")",
                 EvalUtils.prettyPrintValue(makeFilesetEntry()));
  }

  private void checkFormatPositionalFails(String format, List<?> tuple,
                                          String errorMessage) {
    try {
      EvalUtils.formatString(format, tuple);
      fail();
    } catch (IllegalFormatException e) {
      assertThat(e).hasMessage(errorMessage);
    }
  }

  @Test
  public void testFormatPositional() throws Exception {
    assertEquals("foo 3", EvalUtils.formatString("%s %d", makeTuple("foo", 3)));

    // Note: formatString doesn't perform scalar x -> (x) conversion;
    // The %-operator is responsible for that.
    assertThat(EvalUtils.formatString("", makeTuple())).isEmpty();
    assertEquals("foo", EvalUtils.formatString("%s", makeTuple("foo")));
    assertEquals("3.14159", EvalUtils.formatString("%s", makeTuple(3.14159)));
    checkFormatPositionalFails("%s", makeTuple(1, 2, 3),
        "not all arguments converted during string formatting");
    assertEquals("%foo", EvalUtils.formatString("%%%s", makeTuple("foo")));
    checkFormatPositionalFails("%%s", makeTuple("foo"),
        "not all arguments converted during string formatting");
    checkFormatPositionalFails("% %s", makeTuple("foo"),
        "invalid arguments for format string");
    assertEquals("[1, 2, 3]", EvalUtils.formatString("%s", makeTuple(makeList(1, 2, 3))));
    assertEquals("(1, 2, 3)", EvalUtils.formatString("%s", makeTuple(makeTuple(1, 2, 3))));
    assertEquals("[]", EvalUtils.formatString("%s", makeTuple(makeList())));
    assertEquals("()", EvalUtils.formatString("%s", makeTuple(makeTuple())));

    checkFormatPositionalFails("%.3g", makeTuple(), "invalid arguments for format string");
    checkFormatPositionalFails("%.3g", makeTuple(1, 2), "invalid arguments for format string");
    checkFormatPositionalFails("%.s", makeTuple(), "invalid arguments for format string");
  }

  private String createExpectedFilesetEntryString(FilesetEntry.SymlinkBehavior symlinkBehavior) {
    return "FilesetEntry(srcdir = \"//x:x\","
           + " files = [\"//x:x\"],"
           + " excludes = [],"
           + " destdir = \"\","
           + " strip_prefix = \".\","
           + " symlinks = \"" + symlinkBehavior.toString().toLowerCase() + "\")";
  }

  private FilesetEntry createTestFilesetEntry(FilesetEntry.SymlinkBehavior symlinkBehavior)
    throws Exception {
    Label label = Label.parseAbsolute("//x");
    return new FilesetEntry(label,
                            Arrays.asList(label),
                            Arrays.<String>asList(),
                            "",
                            symlinkBehavior,
                            ".");
  }

  @Test
  public void testFilesetEntrySymlinkAttr() throws Exception {
    FilesetEntry entryDereference =
      createTestFilesetEntry(FilesetEntry.SymlinkBehavior.DEREFERENCE);

    assertEquals(createExpectedFilesetEntryString(FilesetEntry.SymlinkBehavior.DEREFERENCE),
                 EvalUtils.prettyPrintValue(entryDereference));
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

    String prettyWithout = EvalUtils.prettyPrintValue(withoutStripPrefix);
    String prettyWith = EvalUtils.prettyPrintValue(withStripPrefix);

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
    Label label = Label.parseAbsolute("//x");
    assertEquals("FilesetEntry(srcdir = \"//x:x\","
                 + " files = [\"//x:x\"],"
                 + " excludes = [],"
                 + " destdir = \"\","
                 + " strip_prefix = \".\","
                 + " symlinks = \"copy\")",
                 EvalUtils.prettyPrintValue(
                     new FilesetEntry(label,
                                      Arrays.asList(label),
                                      Arrays.<String>asList(),
                                      "",
                                      FilesetEntry.SymlinkBehavior.COPY,
                                      ".")));
  }
}
