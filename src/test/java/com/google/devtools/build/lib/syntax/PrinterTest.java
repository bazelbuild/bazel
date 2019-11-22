// Copyright 2015 The Bazel Authors. All Rights Reserved.
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
import static com.google.devtools.build.lib.testutil.MoreAsserts.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.skylarkinterface.SkylarkPrinter;
import com.google.devtools.build.lib.skylarkinterface.SkylarkValue;
import java.util.IllegalFormatException;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 *  Test properties of the evaluator's datatypes and utility functions
 *  without actually creating any parse trees.
 */
@RunWith(JUnit4.class)
public class PrinterTest {

  @Test
  public void testPrinter() throws Exception {
    // Note that prettyPrintValue and printValue only differ on behaviour of
    // labels and strings at toplevel.
    assertThat(Starlark.str(createObjWithStr())).isEqualTo("<str marker>");
    assertThat(Starlark.repr(createObjWithStr())).isEqualTo("<repr marker>");

    assertThat(Starlark.str("foo\nbar")).isEqualTo("foo\nbar");
    assertThat(Starlark.repr("foo\nbar")).isEqualTo("\"foo\\nbar\"");
    assertThat(Starlark.str("'")).isEqualTo("'");
    assertThat(Starlark.repr("'")).isEqualTo("\"'\"");
    assertThat(Starlark.str("\"")).isEqualTo("\"");
    assertThat(Starlark.repr("\"")).isEqualTo("\"\\\"\"");
    assertThat(Starlark.str(3)).isEqualTo("3");
    assertThat(Starlark.repr(3)).isEqualTo("3");
    assertThat(Starlark.repr(Starlark.NONE)).isEqualTo("None");

    assertThat(Starlark.str(Label.parseAbsolute("//x", ImmutableMap.of()))).isEqualTo("//x:x");
    assertThat(Starlark.repr(Label.parseAbsolute("//x", ImmutableMap.of())))
        .isEqualTo("Label(\"//x:x\")");

    List<?> list = StarlarkList.of(null, "foo", "bar");
    List<?> tuple = Tuple.of("foo", "bar");

    assertThat(Starlark.str(Tuple.of(1, list, 3))).isEqualTo("(1, [\"foo\", \"bar\"], 3)");
    assertThat(Starlark.repr(Tuple.of(1, list, 3))).isEqualTo("(1, [\"foo\", \"bar\"], 3)");
    assertThat(Starlark.str(StarlarkList.of(null, 1, tuple, 3)))
        .isEqualTo("[1, (\"foo\", \"bar\"), 3]");
    assertThat(Starlark.repr(StarlarkList.of(null, 1, tuple, 3)))
        .isEqualTo("[1, (\"foo\", \"bar\"), 3]");

    Map<Object, Object> dict =
        ImmutableMap.<Object, Object>of(1, tuple, 2, list, "foo", StarlarkList.of(null));
    assertThat(Starlark.str(dict))
        .isEqualTo("{1: (\"foo\", \"bar\"), 2: [\"foo\", \"bar\"], \"foo\": []}");
    assertThat(Starlark.repr(dict))
        .isEqualTo("{1: (\"foo\", \"bar\"), 2: [\"foo\", \"bar\"], \"foo\": []}");
  }

  private void checkFormatPositionalFails(String errorMessage, String format, Object... arguments) {
    IllegalFormatException e =
        assertThrows(IllegalFormatException.class, () -> Starlark.format(format, arguments));
    assertThat(e).hasMessageThat().isEqualTo(errorMessage);
  }

  @Test
  public void testOutputOrderOfMap() throws Exception {
    Map<Object, Object> map = new LinkedHashMap<>();
    map.put(5, 5);
    map.put(3, 3);
    map.put("foo", 42);
    map.put(7, "bar");
    assertThat(Starlark.str(map)).isEqualTo("{5: 5, 3: 3, \"foo\": 42, 7: \"bar\"}");
  }

  @Test
  public void testFormatPositional() throws Exception {
    assertThat(Starlark.formatWithList("%s %d", Tuple.of("foo", 3))).isEqualTo("foo 3");
    assertThat(Starlark.format("%s %d", "foo", 3)).isEqualTo("foo 3");

    assertThat(Starlark.format("%s %s %s", 1, null, 3)).isEqualTo("1 null 3");

    // Note: formatToString doesn't perform scalar x -> (x) conversion;
    // The %-operator is responsible for that.
    assertThat(Starlark.formatWithList("", Tuple.of())).isEmpty();
    assertThat(Starlark.format("%s", "foo")).isEqualTo("foo");
    assertThat(Starlark.format("%s", 3.14159)).isEqualTo("3.14159");
    checkFormatPositionalFails("not all arguments converted during string formatting",
        "%s", 1, 2, 3);
    assertThat(Starlark.format("%%%s", "foo")).isEqualTo("%foo");
    checkFormatPositionalFails("not all arguments converted during string formatting",
        "%%s", "foo");
    checkFormatPositionalFails("unsupported format character \" \" at index 1 in \"% %s\"",
        "% %s", "foo");
    assertThat(Starlark.format("%s", StarlarkList.of(null, 1, 2, 3))).isEqualTo("[1, 2, 3]");
    assertThat(Starlark.format("%s", Tuple.of(1, 2, 3))).isEqualTo("(1, 2, 3)");
    assertThat(Starlark.format("%s", StarlarkList.of(null))).isEqualTo("[]");
    assertThat(Starlark.format("%s", Tuple.of())).isEqualTo("()");
    assertThat(Starlark.format("%% %d %r %s", 1, "2", "3")).isEqualTo("% 1 \"2\" 3");

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
  public void testPrettyPrinter() throws Exception {
    assertThat(Printer.getPrettyPrinter().repr(ImmutableList.of(1, 2, 3)).toString())
        .isEqualTo(
            "[\n" +
            "    1,\n" +
            "    2,\n" +
            "    3\n" +
            "]");
    assertThat(Printer.getPrettyPrinter().repr(ImmutableList.<String>of()).toString())
        .isEqualTo("[]");
    assertThat(Printer.getPrettyPrinter().repr(ImmutableList.of("foo")).toString())
        .isEqualTo("[\n    \"foo\"\n]");
    assertThat(
            Printer.getPrettyPrinter()
                .repr(ImmutableMap.<Object, Object>of("foo", "bar", "baz", ImmutableList.of(1, 2)))
                .toString())
        .isEqualTo(
            "{\n" +
            "    \"foo\": \"bar\",\n" +
            "    \"baz\": [\n" +
            "        1,\n" +
            "        2\n" +
            "    ]\n" +
            "}");
    assertThat(
            Printer.getPrettyPrinter()
                .repr(ImmutableMap.<Object, Object>of(
                        "foo", "bar", "empty", ImmutableList.of(), "a", "b"))
                .toString())
        .isEqualTo(
            "{\n" +
            "    \"foo\": \"bar\",\n" +
            "    \"empty\": [],\n" +
            "    \"a\": \"b\"\n" +
            "}");
  }

  private SkylarkPrinter makeSimplifiedFormatPrinter() {
    return new Printer.BasePrinter(new StringBuilder(), /*simplifiedFormatStrings=*/ true);
  }

  @Test
  public void testSimplifiedDisallowsPlaceholdersBesidesPercentS() {
    assertThat(makeSimplifiedFormatPrinter().format("Allowed: %%").toString())
        .isEqualTo("Allowed: %");
    assertThat(makeSimplifiedFormatPrinter().format("Allowed: %s", "abc").toString())
        .isEqualTo("Allowed: abc");
    assertThrows(
        IllegalFormatException.class,
        () -> makeSimplifiedFormatPrinter().format("Disallowed: %r", "abc"));
    assertThrows(
        IllegalFormatException.class,
        () -> makeSimplifiedFormatPrinter().format("Disallowed: %d", 5));
  }

  private SkylarkValue createObjWithStr() {
    return new SkylarkValue() {
      @Override
      public void repr(SkylarkPrinter printer) {
        printer.append("<repr marker>");
      }

      @Override
      public void str(SkylarkPrinter printer) {
        printer.append("<str marker>");
      }
    };
  }
}
