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
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;

import com.google.common.base.Joiner;
import com.google.common.base.Strings;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.syntax.SkylarkList.MutableList;
import com.google.devtools.build.lib.syntax.SkylarkList.Tuple;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.util.Arrays;
import java.util.HashMap;
import java.util.IllegalFormatException;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

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
    assertEquals("foo\nbar", Printer.str("foo\nbar"));
    assertEquals("\"foo\\nbar\"", Printer.repr("foo\nbar"));
    assertEquals("'", Printer.str("'"));
    assertEquals("\"'\"", Printer.repr("'"));
    assertEquals("\"", Printer.str("\""));
    assertEquals("\"\\\"\"", Printer.repr("\""));
    assertEquals("3", Printer.str(3));
    assertEquals("3", Printer.repr(3));
    assertEquals("None", Printer.repr(Runtime.NONE));

    assertEquals("//x:x", Printer.str(
        Label.parseAbsolute("//x")));
    assertEquals("\"//x:x\"", Printer.repr(
        Label.parseAbsolute("//x")));

    List<?> list = MutableList.of(null, "foo", "bar");
    List<?> tuple = Tuple.of("foo", "bar");

    assertEquals("(1, [\"foo\", \"bar\"], 3)",
                 Printer.str(Tuple.of(1, list, 3)));
    assertEquals("(1, [\"foo\", \"bar\"], 3)",
                 Printer.repr(Tuple.of(1, list, 3)));
    assertEquals("[1, (\"foo\", \"bar\"), 3]",
                 Printer.str(MutableList.of(null, 1, tuple, 3)));
    assertEquals("[1, (\"foo\", \"bar\"), 3]",
                 Printer.repr(MutableList.of(null, 1, tuple, 3)));

    Map<Object, Object> dict = ImmutableMap.<Object, Object>of(
        1, tuple,
        2, list,
        "foo", MutableList.of(null));
    assertEquals("{1: (\"foo\", \"bar\"), 2: [\"foo\", \"bar\"], \"foo\": []}",
                Printer.str(dict));
    assertEquals("{1: (\"foo\", \"bar\"), 2: [\"foo\", \"bar\"], \"foo\": []}",
                Printer.repr(dict));
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
    assertEquals("foo 3", Printer.formatToString("%s %d", Tuple.of("foo", 3)));
    assertEquals("foo 3", Printer.format("%s %d", "foo", 3));

    // Note: formatToString doesn't perform scalar x -> (x) conversion;
    // The %-operator is responsible for that.
    assertThat(Printer.formatToString("", Tuple.of())).isEmpty();
    assertEquals("foo", Printer.format("%s", "foo"));
    assertEquals("3.14159", Printer.format("%s", 3.14159));
    checkFormatPositionalFails("not all arguments converted during string formatting",
        "%s", 1, 2, 3);
    assertEquals("%foo", Printer.format("%%%s", "foo"));
    checkFormatPositionalFails("not all arguments converted during string formatting",
        "%%s", "foo");
    checkFormatPositionalFails("unsupported format character \" \" at index 1 in \"% %s\"",
        "% %s", "foo");
    assertEquals("[1, 2, 3]", Printer.format("%s", MutableList.of(null, 1, 2, 3)));
    assertEquals("(1, 2, 3)", Printer.format("%s", Tuple.of(1, 2, 3)));
    assertEquals("[]", Printer.format("%s", MutableList.of(null)));
    assertEquals("()", Printer.format("%s", Tuple.of()));
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
  public void testSingleQuotes() throws Exception {
    assertThat(Printer.str("test", '\'')).isEqualTo("test");
    assertThat(Printer.repr("test", '\'')).isEqualTo("'test'");

    assertEquals("'\\''", Printer.repr("'", '\''));
    assertEquals("\"", Printer.str("\"", '\''));
    assertEquals("'\"'", Printer.repr("\"", '\''));

    List<?> list = MutableList.of(null, "foo", "bar");
    List<?> tuple = Tuple.of("foo", "bar");

    assertThat(Printer.str(Tuple.of(1, list, 3), '\'')).isEqualTo("(1, ['foo', 'bar'], 3)");
    assertThat(Printer.repr(Tuple.of(1, list, 3), '\'')).isEqualTo("(1, ['foo', 'bar'], 3)");
    assertThat(Printer.str(MutableList.of(null, 1, tuple, 3), '\''))
        .isEqualTo("[1, ('foo', 'bar'), 3]");
    assertThat(Printer.repr(MutableList.of(null, 1, tuple, 3), '\''))
        .isEqualTo("[1, ('foo', 'bar'), 3]");

    Map<Object, Object> dict =
        ImmutableMap.<Object, Object>of(1, tuple, 2, list, "foo", MutableList.of(null));

    assertThat(Printer.str(dict, '\''))
        .isEqualTo("{1: ('foo', 'bar'), 2: ['foo', 'bar'], 'foo': []}");
    assertThat(Printer.repr(dict, '\''))
        .isEqualTo("{1: ('foo', 'bar'), 2: ['foo', 'bar'], 'foo': []}");
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
    StringBuilder builder = new StringBuilder();
    Printer.printList(
        builder, list, "[", ", ", "]", "", '"', criticalElementsCount, criticalStringLength);
    return builder.toString();
  }
}
