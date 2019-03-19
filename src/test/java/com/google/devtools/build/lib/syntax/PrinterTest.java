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
import static org.junit.Assert.fail;

import com.google.common.base.Joiner;
import com.google.common.base.Strings;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.skylarkinterface.SkylarkPrinter;
import com.google.devtools.build.lib.skylarkinterface.SkylarkValue;
import com.google.devtools.build.lib.syntax.SkylarkList.MutableList;
import com.google.devtools.build.lib.syntax.SkylarkList.Tuple;
import java.util.Arrays;
import java.util.IllegalFormatException;
import java.util.LinkedHashMap;
import java.util.LinkedList;
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
    assertThat(Printer.str(createObjWithStr())).isEqualTo("<str marker>");
    assertThat(Printer.repr(createObjWithStr())).isEqualTo("<repr marker>");

    assertThat(Printer.str("foo\nbar")).isEqualTo("foo\nbar");
    assertThat(Printer.repr("foo\nbar")).isEqualTo("\"foo\\nbar\"");
    assertThat(Printer.str("'")).isEqualTo("'");
    assertThat(Printer.repr("'")).isEqualTo("\"'\"");
    assertThat(Printer.str("\"")).isEqualTo("\"");
    assertThat(Printer.repr("\"")).isEqualTo("\"\\\"\"");
    assertThat(Printer.str(3)).isEqualTo("3");
    assertThat(Printer.repr(3)).isEqualTo("3");
    assertThat(Printer.repr(Runtime.NONE)).isEqualTo("None");

    assertThat(Printer.str(Label.parseAbsolute("//x", ImmutableMap.of()))).isEqualTo("//x:x");
    assertThat(Printer.repr(Label.parseAbsolute("//x", ImmutableMap.of())))
        .isEqualTo("Label(\"//x:x\")");

    List<?> list = MutableList.of(null, "foo", "bar");
    List<?> tuple = Tuple.of("foo", "bar");

    assertThat(Printer.str(Tuple.of(1, list, 3))).isEqualTo("(1, [\"foo\", \"bar\"], 3)");
    assertThat(Printer.repr(Tuple.of(1, list, 3))).isEqualTo("(1, [\"foo\", \"bar\"], 3)");
    assertThat(Printer.str(MutableList.of(null, 1, tuple, 3)))
        .isEqualTo("[1, (\"foo\", \"bar\"), 3]");
    assertThat(Printer.repr(MutableList.of(null, 1, tuple, 3)))
        .isEqualTo("[1, (\"foo\", \"bar\"), 3]");

    Map<Object, Object> dict = ImmutableMap.<Object, Object>of(
        1, tuple,
        2, list,
        "foo", MutableList.of(null));
    assertThat(Printer.str(dict))
        .isEqualTo("{1: (\"foo\", \"bar\"), 2: [\"foo\", \"bar\"], \"foo\": []}");
    assertThat(Printer.repr(dict))
        .isEqualTo("{1: (\"foo\", \"bar\"), 2: [\"foo\", \"bar\"], \"foo\": []}");
  }

  private void checkFormatPositionalFails(String errorMessage, String format, Object... arguments) {
    try {
      Printer.format(format, arguments);
      fail();
    } catch (IllegalFormatException e) {
      assertThat(e).hasMessageThat().isEqualTo(errorMessage);
    }
  }

  @Test
  public void testOutputOrderOfMap() throws Exception {
    Map<Object, Object> map = new LinkedHashMap<>();
    map.put(5, 5);
    map.put(3, 3);
    map.put("foo", 42);
    map.put(7, "bar");
    assertThat(Printer.str(map)).isEqualTo("{5: 5, 3: 3, \"foo\": 42, 7: \"bar\"}");
  }

  @Test
  public void testFormatPositional() throws Exception {
    assertThat(Printer.formatWithList("%s %d", Tuple.of("foo", 3))).isEqualTo("foo 3");
    assertThat(Printer.format("%s %d", "foo", 3)).isEqualTo("foo 3");

    assertThat(Printer.format("%s %s %s", 1, null, 3)).isEqualTo("1 null 3");

    // Note: formatToString doesn't perform scalar x -> (x) conversion;
    // The %-operator is responsible for that.
    assertThat(Printer.formatWithList("", Tuple.of())).isEmpty();
    assertThat(Printer.format("%s", "foo")).isEqualTo("foo");
    assertThat(Printer.format("%s", 3.14159)).isEqualTo("3.14159");
    checkFormatPositionalFails("not all arguments converted during string formatting",
        "%s", 1, 2, 3);
    assertThat(Printer.format("%%%s", "foo")).isEqualTo("%foo");
    checkFormatPositionalFails("not all arguments converted during string formatting",
        "%%s", "foo");
    checkFormatPositionalFails("unsupported format character \" \" at index 1 in \"% %s\"",
        "% %s", "foo");
    assertThat(Printer.format("%s", MutableList.of(null, 1, 2, 3))).isEqualTo("[1, 2, 3]");
    assertThat(Printer.format("%s", Tuple.of(1, 2, 3))).isEqualTo("(1, 2, 3)");
    assertThat(Printer.format("%s", MutableList.of(null))).isEqualTo("[]");
    assertThat(Printer.format("%s", Tuple.of())).isEqualTo("()");
    assertThat(Printer.format("%% %d %r %s", 1, "2", "3")).isEqualTo("% 1 \"2\" 3");

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

  @Test
  public void testListLimitStringLength() throws Exception {
    int lengthDivisibleByTwo = Printer.SUGGESTED_CRITICAL_LIST_ELEMENTS_STRING_LENGTH;
    if (lengthDivisibleByTwo % 2 == 1) {
      ++lengthDivisibleByTwo;
    }
    String limit = Strings.repeat("x", lengthDivisibleByTwo);
    String half = Strings.repeat("x", lengthDivisibleByTwo / 2);

    List<String> list = Arrays.asList(limit + limit);

    // String is way too long -> shorten.
    assertThat(printListWithLimit(list)).isEqualTo("[\"" + limit + "...\"]");

    LinkedList<List<String>> nestedList = new LinkedList<>();
    nestedList.add(list);

    // Same as above, but with one additional level of indirection.
    assertThat(printListWithLimit(nestedList)).isEqualTo("[[\"" + limit + "...\"]]");

    // The inner list alone would meet the limit, but because of the first element, it has to be
    // shortened.
    assertThat(printListWithLimit(Arrays.asList(half, Arrays.asList(limit))))
        .isEqualTo("[\"" + half + "\", [\"" + half + "...\"]]");

    // String is too long, but the ellipsis make it even longer.
    assertThat(printListWithLimit(Arrays.asList(limit + "x"))).isEqualTo("[\"" + limit + "...\"]");

    // We hit the limit exactly -> everything is printed.
    assertThat(printListWithLimit(Arrays.asList(limit))).isEqualTo("[\"" + limit + "\"]");

    // Exact hit, but with two arguments -> everything is printed.
    assertThat(printListWithLimit(Arrays.asList(half, half)))
        .isEqualTo("[\"" + half + "\", \"" + half + "\"]");

    // First argument hits the limit -> remaining argument is shortened.
    assertThat(printListWithLimit(Arrays.asList(limit, limit)))
        .isEqualTo("[\"" + limit + "\", \"...\"]");

    String limitMinusOne = limit.substring(0, limit.length() - 1);

    // First arguments is one below the limit -> print first character of remaining argument.
    assertThat(printListWithLimit(Arrays.asList(limitMinusOne, limit)))
        .isEqualTo("[\"" + limitMinusOne + "\", \"x...\"]");

    // First argument hits the limit -> we skip  the remaining two arguments.
    assertThat(printListWithLimit(Arrays.asList(limit, limit, limit)))
        .isEqualTo("[\"" + limit + "\", <2 more arguments>]");
  }

  @Test
  public void testListLimitTooManyArgs() throws Exception {
    StringBuilder builder = new StringBuilder();
    List<Integer> maxLength = new LinkedList<>();

    int next;
    for (next = 0; next < Printer.SUGGESTED_CRITICAL_LIST_ELEMENTS_COUNT; ++next) {
      maxLength.add(next);
      if (next > 0) {
        builder.append(", ");
      }
      builder.append(next);
    }

    // There is one too many, but we print every argument nonetheless.
    maxLength.add(next);
    assertThat(printListWithLimit(maxLength)).isEqualTo("[" + builder + ", " + next + "]");

    // There are two too many, hence we don't print them.
    ++next;
    maxLength.add(next);
    assertThat(printListWithLimit(maxLength)).isEqualTo("[" + builder + ", <2 more arguments>]");
  }

  @Test
  public void testPrintListDefaultNoLimit() throws Exception {
    List<Integer> list = new LinkedList<>();
    // Make sure that the resulting string is longer than the suggestion. This should also lead to
    // way more items than suggested.
    for (int i = 0; i < Printer.SUGGESTED_CRITICAL_LIST_ELEMENTS_STRING_LENGTH * 2; ++i) {
      list.add(i);
    }
    assertThat(Printer.str(list)).isEqualTo(String.format("[%s]", Joiner.on(", ").join(list)));
  }

  private String printListWithLimit(List<?> list) {
    return printList(list, Printer.SUGGESTED_CRITICAL_LIST_ELEMENTS_COUNT,
        Printer.SUGGESTED_CRITICAL_LIST_ELEMENTS_STRING_LENGTH);
  }

  private String printList(List<?> list, int criticalElementsCount, int criticalStringLength) {
    return Printer.printAbbreviatedList(
        list, "[", ", ", "]", "", criticalElementsCount, criticalStringLength);
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
