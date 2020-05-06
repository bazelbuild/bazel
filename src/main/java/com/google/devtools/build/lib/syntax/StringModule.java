// Copyright 2018 The Bazel Authors. All rights reserved.
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

import com.google.common.base.Ascii;
import com.google.common.base.CharMatcher;
import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.skylarkinterface.Param;
import com.google.devtools.build.lib.skylarkinterface.ParamType;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.StarlarkBuiltin;
import com.google.devtools.build.lib.skylarkinterface.StarlarkDocumentationCategory;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Starlark String module.
 *
 * <p>This module has special treatment in Starlark, as its methods represent methods represent for
 * any 'string' objects in the language.
 *
 * <p>Methods of this class annotated with {@link SkylarkCallable} must have a positional-only
 * 'String self' parameter as the first parameter of the method.
 */
@StarlarkBuiltin(
    name = "string",
    category = StarlarkDocumentationCategory.BUILTIN,
    doc =
        "A language built-in type to support strings. "
            + "Examples of string literals:<br>"
            + "<pre class=\"language-python\">a = 'abc\\ndef'\n"
            + "b = \"ab'cd\"\n"
            + "c = \"\"\"multiline string\"\"\"\n"
            + "\n"
            + "# Strings support slicing (negative index starts from the end):\n"
            + "x = \"hello\"[2:4]  # \"ll\"\n"
            + "y = \"hello\"[1:-1]  # \"ell\"\n"
            + "z = \"hello\"[:4]  # \"hell\""
            + "# Slice steps can be used, too:\n"
            + "s = \"hello\"[::2] # \"hlo\"\n"
            + "t = \"hello\"[3:0:-1] # \"lle\"\n</pre>"
            + "Strings are iterable and support the <code>in</code> operator. Examples:<br>"
            + "<pre class=\"language-python\">\"bc\" in \"abcd\"   # evaluates to True\n"
            + "x = [s for s in \"abc\"]  # x == [\"a\", \"b\", \"c\"]</pre>\n"
            + "Implicit concatenation of strings is not allowed; use the <code>+</code> "
            + "operator instead. Comparison operators perform a lexicographical comparison; "
            + "use <code>==</code> to test for equality.")
final class StringModule implements StarlarkValue {

  static final StringModule INSTANCE = new StringModule();

  private StringModule() {}

  // Returns s[start:stop:step], as if by Sequence.getSlice.
  static String slice(String s, int start, int stop, int step) {
    RangeList indices = new RangeList(start, stop, step);
    int n = indices.size();
    if (step == 1) { // common case
      return s.substring(indices.at(0), indices.at(n));
    } else {
      char[] res = new char[n];
      for (int i = 0; i < n; ++i) {
        res[i] = s.charAt(indices.at(i));
      }
      return new String(res);
    }
  }

  // Emulate Python substring function
  // It converts out of range indices, and never fails
  //
  // TODO(adonovan): opt: avoid this function, as String.substring now allocates a copy (!)
  private static String pythonSubstring(String str, int start, Object end, String what)
      throws EvalException {
    if (start == 0 && EvalUtils.isNullOrNone(end)) {
      return str;
    }
    start = EvalUtils.toIndex(start, str.length());
    int stop;
    if (EvalUtils.isNullOrNone(end)) {
      stop = str.length();
    } else if (end instanceof Integer) {
      stop = EvalUtils.toIndex((Integer) end, str.length());
    } else {
      throw new EvalException(null, "expected int for " + what + ", got " + Starlark.type(end));
    }
    if (start >= stop) {
      return "";
    }
    return str.substring(start, stop);
  }

  @SkylarkCallable(
      name = "join",
      doc =
          "Returns a string in which the string elements of the argument have been "
              + "joined by this string as a separator. Example:<br>"
              + "<pre class=\"language-python\">\"|\".join([\"a\", \"b\", \"c\"]) == \"a|b|c\""
              + "</pre>",
      parameters = {
        @Param(name = "self", type = String.class),
        @Param(
            name = "elements",
            type = Object.class,
            doc = "The objects to join.")
      })
  public String join(String self, Object elements) throws EvalException {
    Iterable<?> items = Starlark.toIterable(elements);
    int i = 0;
    for (Object item : items) {
      if (!(item instanceof String)) {
        throw Starlark.errorf(
            "expected string for sequence element %d, got '%s'", i, Starlark.type(item));
      }
      i++;
    }
    return Joiner.on(self).join(items);
  }

  @SkylarkCallable(
      name = "lower",
      doc = "Returns the lower case version of this string.",
      parameters = {@Param(name = "self", type = String.class)})
  public String lower(String self) {
    return Ascii.toLowerCase(self);
  }

  @SkylarkCallable(
      name = "upper",
      doc = "Returns the upper case version of this string.",
      parameters = {@Param(name = "self", type = String.class)})
  public String upper(String self) {
    return Ascii.toUpperCase(self);
  }

  /**
   * For consistency with Python we recognize the same whitespace characters as they do over the
   * range 0x00-0xFF. See https://hg.python.org/cpython/file/3.6/Objects/unicodetype_db.h#l5738 This
   * list is a consequence of Unicode character information.
   *
   * <p>Note that this differs from Python 2.7, which uses ctype.h#isspace(), and from
   * java.lang.Character#isWhitespace(), which does not recognize U+00A0.
   */
  private static final String LATIN1_WHITESPACE =
      ("\u0009" + "\n" + "\u000B" + "\u000C" + "\r" + "\u001C" + "\u001D" + "\u001E" + "\u001F"
          + "\u0020" + "\u0085" + "\u00A0");

  private static String stringLStrip(String self, String chars) {
    CharMatcher matcher = CharMatcher.anyOf(chars);
    for (int i = 0; i < self.length(); i++) {
      if (!matcher.matches(self.charAt(i))) {
        return self.substring(i);
      }
    }
    return ""; // All characters were stripped.
  }

  private static String stringRStrip(String self, String chars) {
    CharMatcher matcher = CharMatcher.anyOf(chars);
    for (int i = self.length() - 1; i >= 0; i--) {
      if (!matcher.matches(self.charAt(i))) {
        return self.substring(0, i + 1);
      }
    }
    return ""; // All characters were stripped.
  }

  private static String stringStrip(String self, String chars) {
    return stringLStrip(stringRStrip(self, chars), chars);
  }

  @SkylarkCallable(
      name = "lstrip",
      doc =
          "Returns a copy of the string where leading characters that appear in "
              + "<code>chars</code> are removed. Note that <code>chars</code> "
              + "is not a prefix: all combinations of its value are removed:"
              + "<pre class=\"language-python\">"
              + "\"abcba\".lstrip(\"ba\") == \"cba\""
              + "</pre>",
      parameters = {
        @Param(name = "self", type = String.class),
        @Param(
            name = "chars",
            type = String.class,
            noneable = true,
            doc = "The characters to remove, or all whitespace if None.",
            defaultValue = "None")
      })
  public String lstrip(String self, Object charsOrNone) {
    String chars = charsOrNone != Starlark.NONE ? (String) charsOrNone : LATIN1_WHITESPACE;
    return stringLStrip(self, chars);
  }

  @SkylarkCallable(
      name = "rstrip",
      doc =
          "Returns a copy of the string where trailing characters that appear in "
              + "<code>chars</code> are removed. Note that <code>chars</code> "
              + "is not a suffix: all combinations of its value are removed:"
              + "<pre class=\"language-python\">"
              + "\"abcbaa\".rstrip(\"ab\") == \"abc\""
              + "</pre>",
      parameters = {
        @Param(name = "self", type = String.class, doc = "This string."),
        @Param(
            name = "chars",
            type = String.class,
            noneable = true,
            doc = "The characters to remove, or all whitespace if None.",
            defaultValue = "None")
      })
  public String rstrip(String self, Object charsOrNone) {
    String chars = charsOrNone != Starlark.NONE ? (String) charsOrNone : LATIN1_WHITESPACE;
    return stringRStrip(self, chars);
  }

  @SkylarkCallable(
      name = "strip",
      doc =
          "Returns a copy of the string where leading or trailing characters that appear in "
              + "<code>chars</code> are removed. Note that <code>chars</code> "
              + "is neither a prefix nor a suffix: all combinations of its value "
              + "are removed:"
              + "<pre class=\"language-python\">"
              + "\"aabcbcbaa\".strip(\"ab\") == \"cbc\""
              + "</pre>",
      parameters = {
        @Param(name = "self", type = String.class, doc = "This string."),
        @Param(
            name = "chars",
            type = String.class,
            noneable = true,
            doc = "The characters to remove, or all whitespace if None.",
            defaultValue = "None")
      })
  public String strip(String self, Object charsOrNone) {
    String chars = charsOrNone != Starlark.NONE ? (String) charsOrNone : LATIN1_WHITESPACE;
    return stringStrip(self, chars);
  }

  @SkylarkCallable(
      name = "replace",
      doc =
          "Returns a copy of the string in which the occurrences "
              + "of <code>old</code> have been replaced with <code>new</code>, optionally "
              + "restricting the number of replacements to <code>maxsplit</code>.",
      parameters = {
        @Param(name = "self", type = String.class, doc = "This string."),
        @Param(name = "old", type = String.class, doc = "The string to be replaced."),
        @Param(name = "new", type = String.class, doc = "The string to replace with."),
        @Param(
            name = "count",
            type = Integer.class,
            noneable = true, // TODO(#11244): Set false once incompatible flag is deleted.
            defaultValue = "unbound",
            doc =
                "The maximum number of replacements. If omitted, there is no limit."
                    + "<p>If <code>--incompatible_string_replace_count</code> is true, a negative "
                    + "value is ignored (so there's no limit) and a <code>None</code> value is an "
                    + "error. Otherwise, a negative value is treated as 0 and a <code>None</code> "
                    + "value is ignored. (See also issue <a "
                    + "href='https://github.com/bazelbuild/bazel/issues/11244'>#11244</a>.)")
      },
      useStarlarkThread = true)
  public String replace(
      String self, String oldString, String newString, Object countUnchecked, StarlarkThread thread)
      throws EvalException {
    int count = Integer.MAX_VALUE;

    StarlarkSemantics semantics = thread.getSemantics();
    if (semantics.incompatibleStringReplaceCount()) {
      if (countUnchecked == Starlark.NONE) {
        throw Starlark.errorf(
            "Cannot pass a None count to string.replace(); omit the count argument instead. (You "
                + "can temporarily opt out of this change by setting "
                + "--incompatible_string_replace_count=false.)");
      }
      if (countUnchecked != Starlark.UNBOUND && (Integer) countUnchecked >= 0) {
        count = (Integer) countUnchecked;
      }
    } else {
      if (countUnchecked != Starlark.UNBOUND && countUnchecked != Starlark.NONE) {
        // Negative has same effect as 0 below.
        count = (Integer) countUnchecked;
      }
    }

    StringBuilder sb = new StringBuilder();
    int start = 0;
    for (int i = 0; i < count; i++) {
      if (oldString.isEmpty()) {
        sb.append(newString);
        if (start < self.length()) {
          sb.append(self.charAt(start++));
        } else {
          break;
        }
      } else {
        int end = self.indexOf(oldString, start);
        if (end < 0) {
          break;
        }
        sb.append(self, start, end).append(newString);
        start = end + oldString.length();
      }
    }
    sb.append(self, start, self.length());
    return sb.toString();
  }

  @SkylarkCallable(
      name = "split",
      doc =
          "Returns a list of all the words in the string, using <code>sep</code> as the "
              + "separator, optionally limiting the number of splits to <code>maxsplit</code>.",
      parameters = {
        @Param(name = "self", type = String.class, doc = "This string."),
        @Param(
            name = "sep",
            type = String.class,
            doc = "The string to split on."),
        @Param(
            name = "maxsplit",
            type = Integer.class,
            noneable = true,
            defaultValue = "None",
            doc = "The maximum number of splits.")
      },
      useStarlarkThread = true)
  public StarlarkList<String> split(
      String self, String sep, Object maxSplitO, StarlarkThread thread) throws EvalException {
    if (sep.isEmpty()) {
      throw Starlark.errorf("Empty separator");
    }
    int maxSplit = Integer.MAX_VALUE;
    if (maxSplitO != Starlark.NONE) {
      maxSplit = (Integer) maxSplitO;
    }
    ArrayList<String> res = new ArrayList<>();
    int start = 0;
    while (true) {
      int end = self.indexOf(sep, start);
      if (end < 0 || maxSplit-- == 0) {
        res.add(self.substring(start));
        break;
      }
      res.add(self.substring(start, end));
      start = end + sep.length();
    }
    return StarlarkList.copyOf(thread.mutability(), res);
  }

  @SkylarkCallable(
      name = "rsplit",
      doc =
          "Returns a list of all the words in the string, using <code>sep</code> as the "
              + "separator, optionally limiting the number of splits to <code>maxsplit</code>. "
              + "Except for splitting from the right, this method behaves like split().",
      parameters = {
        @Param(name = "self", type = String.class, doc = "This string."),
        @Param(
            name = "sep",
            type = String.class,
            doc = "The string to split on."),
        @Param(
            name = "maxsplit",
            type = Integer.class,
            noneable = true,
            defaultValue = "None",
            doc = "The maximum number of splits.")
      },
      useStarlarkThread = true)
  public StarlarkList<String> rsplit(
      String self, String sep, Object maxSplitO, StarlarkThread thread) throws EvalException {
    if (sep.isEmpty()) {
      throw Starlark.errorf("Empty separator");
    }
    int maxSplit = Integer.MAX_VALUE;
    if (maxSplitO != Starlark.NONE) {
      maxSplit = (Integer) maxSplitO;
    }
    ArrayList<String> res = new ArrayList<>();
    int end = self.length();
    while (true) {
      int start = self.lastIndexOf(sep, end - 1);
      if (start < 0 || maxSplit-- == 0) {
        res.add(self.substring(0, end));
        break;
      }
      res.add(self.substring(start + sep.length(), end));
      end = start;
    }
    Collections.reverse(res);
    return StarlarkList.copyOf(thread.mutability(), res);
  }

  @SkylarkCallable(
      name = "partition",
      doc =
          "Splits the input string at the first occurrence of the separator "
              + "<code>sep</code> and returns the resulting partition as a three-element "
              + "tuple of the form (substring_before, separator, substring_after).",
      parameters = {
        @Param(name = "self", type = String.class),
        @Param(
            name = "sep",
            type = Object.class,
            defaultValue = "unbound",
            doc = "The string to split on.")
      },
      useStarlarkThread = true)
  public Tuple<String> partition(String self, Object sep, StarlarkThread thread)
      throws EvalException {
    if (sep == Starlark.UNBOUND) {
      throw Starlark.errorf(
          "parameter 'sep' has no default value, for call to method 'partition(sep)' of 'string'");
    } else if (!(sep instanceof String)) {
      throw Starlark.errorf(
          "expected value of type 'string' for parameter 'sep', for call to method 'partition()' of"
              + " 'string'");
    }

    return partitionWrapper(self, (String) sep, true);
  }

  @SkylarkCallable(
      name = "rpartition",
      doc =
          "Splits the input string at the last occurrence of the separator "
              + "<code>sep</code> and returns the resulting partition as a three-element "
              + "tuple of the form (substring_before, separator, substring_after).",
      parameters = {
        @Param(name = "self", type = String.class, doc = "This string."),
        @Param(
            name = "sep",
            type = String.class,
            defaultValue = "unbound",
            doc = "The string to split on.")
      },
      useStarlarkThread = true)
  public Tuple<String> rpartition(String self, Object sep, StarlarkThread thread)
      throws EvalException {
    if (sep == Starlark.UNBOUND) {
      throw Starlark.errorf(
          "parameter 'sep' has no default value, "
              + "for call to method partition(sep) of 'string'");
    } else if (!(sep instanceof String)) {
      throw Starlark.errorf(
          "expected value of type 'string' for parameter 'sep', for call to method partition(sep ="
              + " unbound) of 'string'");
    }
    return partitionWrapper(self, (String) sep, false);
  }

  /**
   * Splits the input string at the first/last occurrence of the given separator and returns the
   * resulting partition as a three-tuple of Strings.
   *
   * <p>If the input string does not contain the separator, the tuple will consist of the original
   * input string and two empty strings.
   *
   * <p>This method emulates the behavior of Python's str.partition() and str.rpartition(),
   * depending on the value of the {@code forward} flag.
   *
   * @param input The input string
   * @param separator The string to split on
   * @param forward A flag that controls whether the input string is split around the first ({@code
   *     true}) or last ({@code false}) occurrence of the separator.
   * @return a 3-Tuple of the form (part_before_separator, separator, part_after_separator).
   */
  private static Tuple<String> partitionWrapper(String input, String separator, boolean forward)
      throws EvalException {
    if (separator.isEmpty()) {
      throw Starlark.errorf("empty separator");
    }

    String a = "";
    String b = "";
    String c = "";

    int pos = forward ? input.indexOf(separator) : input.lastIndexOf(separator);
    if (pos < 0) {
      // Following Python's implementation of str.partition() and str.rpartition(),
      // the input string is copied to either the first or the last position in the
      // list, depending on the value of the forward flag.
      if (forward) {
        a = input;
      } else {
        c = input;
      }
    } else {
      a = input.substring(0, pos);
      b = separator;
      // pos + sep.length() is at most equal to input.length(). This worst-case
      // happens when the separator is at the end of the input string. However,
      // substring() will return an empty string in this scenario, thus making
      // any additional safety checks obsolete.
      c = input.substring(pos + separator.length());
    }

    return Tuple.triple(a, b, c);
  }

  @SkylarkCallable(
      name = "capitalize",
      doc =
          "Returns a copy of the string with its first character (if any) capitalized and the rest "
              + "lowercased. This method does not support non-ascii characters. ",
      parameters = {@Param(name = "self", type = String.class, doc = "This string.")})
  public String capitalize(String self) throws EvalException {
    if (self.isEmpty()) {
      return self;
    }
    return Character.toUpperCase(self.charAt(0)) + Ascii.toLowerCase(self.substring(1));
  }

  @SkylarkCallable(
      name = "title",
      doc =
          "Converts the input string into title case, i.e. every word starts with an "
              + "uppercase letter while the remaining letters are lowercase. In this "
              + "context, a word means strictly a sequence of letters. This method does "
              + "not support supplementary Unicode characters.",
      parameters = {@Param(name = "self", type = String.class, doc = "This string.")})
  public String title(String self) throws EvalException {
    char[] data = self.toCharArray();
    boolean previousWasLetter = false;

    for (int pos = 0; pos < data.length; ++pos) {
      char current = data[pos];
      boolean currentIsLetter = Character.isLetter(current);

      if (currentIsLetter) {
        if (previousWasLetter && Character.isUpperCase(current)) {
          data[pos] = Character.toLowerCase(current);
        } else if (!previousWasLetter && Character.isLowerCase(current)) {
          data[pos] = Character.toUpperCase(current);
        }
      }
      previousWasLetter = currentIsLetter;
    }

    return new String(data);
  }

  /**
   * Common implementation for find, rfind, index, rindex.
   *
   * @param forward true if we want to return the last matching index.
   */
  private static int stringFind(
      boolean forward, String self, String sub, int start, Object end, String msg)
      throws EvalException {
    String substr = pythonSubstring(self, start, end, msg);
    int subpos = forward ? substr.indexOf(sub) : substr.lastIndexOf(sub);
    return subpos < 0
        ? subpos //
        : subpos + EvalUtils.toIndex(start, self.length());
  }

  private static final Pattern SPLIT_LINES_PATTERN =
      Pattern.compile("(?<line>.*)(?<break>(\\r\\n|\\r|\\n)?)");

  @SkylarkCallable(
      name = "rfind",
      doc =
          "Returns the last index where <code>sub</code> is found, or -1 if no such index exists, "
              + "optionally restricting to <code>[start:end]</code>, "
              + "<code>start</code> being inclusive and <code>end</code> being exclusive.",
      parameters = {
        @Param(name = "self", type = String.class, doc = "This string."),
        @Param(
            name = "sub",
            type = String.class,
            doc = "The substring to find."),
        @Param(
            name = "start",
            type = Integer.class,
            defaultValue = "0",
            doc = "Restrict to search from this position."),
        @Param(
            name = "end",
            type = Integer.class,
            noneable = true,
            defaultValue = "None",
            doc = "optional position before which to restrict to search.")
      })
  public Integer rfind(String self, String sub, Integer start, Object end) throws EvalException {
    return stringFind(false, self, sub, start, end, "'end' argument to rfind");
  }

  @SkylarkCallable(
      name = "find",
      doc =
          "Returns the first index where <code>sub</code> is found, or -1 if no such index exists, "
              + "optionally restricting to <code>[start:end]</code>, "
              + "<code>start</code> being inclusive and <code>end</code> being exclusive.",
      parameters = {
        @Param(name = "self", type = String.class, doc = "This string."),
        @Param(
            name = "sub",
            type = String.class,
            doc = "The substring to find."),
        @Param(
            name = "start",
            type = Integer.class,
            defaultValue = "0",
            doc = "Restrict to search from this position."),
        @Param(
            name = "end",
            type = Integer.class,
            noneable = true,
            defaultValue = "None",
            doc = "optional position before which to restrict to search.")
      })
  public Integer find(String self, String sub, Integer start, Object end) throws EvalException {
    return stringFind(true, self, sub, start, end, "'end' argument to find");
  }

  @SkylarkCallable(
      name = "rindex",
      doc =
          "Returns the last index where <code>sub</code> is found, or raises an error if no such "
              + "index exists, optionally restricting to <code>[start:end]</code>, "
              + "<code>start</code> being inclusive and <code>end</code> being exclusive.",
      parameters = {
        @Param(name = "self", type = String.class, doc = "This string."),
        @Param(
            name = "sub",
            type = String.class,
            doc = "The substring to find."),
        @Param(
            name = "start",
            type = Integer.class,
            defaultValue = "0",
            doc = "Restrict to search from this position."),
        @Param(
            name = "end",
            type = Integer.class,
            noneable = true,
            defaultValue = "None",
            doc = "optional position before which to restrict to search.")
      })
  public Integer rindex(String self, String sub, Integer start, Object end) throws EvalException {
    int res = stringFind(false, self, sub, start, end, "'end' argument to rindex");
    if (res < 0) {
      throw Starlark.errorf(
          "substring %s not found in %s", Starlark.repr(sub), Starlark.repr(self));
    }
    return res;
  }

  @SkylarkCallable(
      name = "index",
      doc =
          "Returns the first index where <code>sub</code> is found, or raises an error if no such "
              + " index exists, optionally restricting to <code>[start:end]</code>"
              + "<code>start</code> being inclusive and <code>end</code> being exclusive.",
      parameters = {
        @Param(name = "self", type = String.class, doc = "This string."),
        @Param(
            name = "sub",
            type = String.class,
            doc = "The substring to find."),
        @Param(
            name = "start",
            type = Integer.class,
            defaultValue = "0",
            doc = "Restrict to search from this position."),
        @Param(
            name = "end",
            type = Integer.class,
            noneable = true,
            defaultValue = "None",
            doc = "optional position before which to restrict to search.")
      })
  public Integer index(String self, String sub, Integer start, Object end) throws EvalException {
    int res = stringFind(true, self, sub, start, end, "'end' argument to index");
    if (res < 0) {
      throw Starlark.errorf(
          "substring %s not found in %s", Starlark.repr(sub), Starlark.repr(self));
    }
    return res;
  }

  @SkylarkCallable(
      name = "splitlines",
      doc =
          "Splits the string at line boundaries ('\\n', '\\r\\n', '\\r') "
              + "and returns the result as a list.",
      parameters = {
        @Param(name = "self", type = String.class, doc = "This string."),
        @Param(
            name = "keepends",
            type = Boolean.class,
            defaultValue = "False",
            doc = "Whether the line breaks should be included in the resulting list.")
      })
  public Sequence<String> splitLines(String self, Boolean keepEnds) throws EvalException {
    List<String> result = new ArrayList<>();
    Matcher matcher = SPLIT_LINES_PATTERN.matcher(self);
    while (matcher.find()) {
      String line = matcher.group("line");
      String lineBreak = matcher.group("break");
      boolean trailingBreak = lineBreak.isEmpty();
      if (line.isEmpty() && trailingBreak) {
        break;
      }
      if (keepEnds && !trailingBreak) {
        result.add(line + lineBreak);
      } else {
        result.add(line);
      }
    }
    return StarlarkList.immutableCopyOf(result);
  }

  @SkylarkCallable(
      name = "isalpha",
      doc =
          "Returns True if all characters in the string are alphabetic ([a-zA-Z]) and there is "
              + "at least one character.",
      parameters = {@Param(name = "self", type = String.class, doc = "This string.")})
  public Boolean isAlpha(String self) throws EvalException {
    return matches(self, ALPHA, false);
  }

  @SkylarkCallable(
      name = "isalnum",
      doc =
          "Returns True if all characters in the string are alphanumeric ([a-zA-Z0-9]) and there "
              + "is at least one character.",
      parameters = {@Param(name = "self", type = String.class, doc = "This string.")})
  public Boolean isAlnum(String self) throws EvalException {
    return matches(self, ALNUM, false);
  }

  @SkylarkCallable(
      name = "isdigit",
      doc =
          "Returns True if all characters in the string are digits ([0-9]) and there is "
              + "at least one character.",
      parameters = {@Param(name = "self", type = String.class, doc = "This string.")})
  public Boolean isDigit(String self) throws EvalException {
    return matches(self, DIGIT, false);
  }

  @SkylarkCallable(
      name = "isspace",
      doc =
          "Returns True if all characters are white space characters and the string "
              + "contains at least one character.",
      parameters = {@Param(name = "self", type = String.class, doc = "This string.")})
  public Boolean isSpace(String self) throws EvalException {
    return matches(self, SPACE, false);
  }

  @SkylarkCallable(
      name = "islower",
      doc =
          "Returns True if all cased characters in the string are lowercase and there is "
              + "at least one character.",
      parameters = {@Param(name = "self", type = String.class, doc = "This string.")})
  public Boolean isLower(String self) throws EvalException {
    // Python also accepts non-cased characters, so we cannot use LOWER.
    return matches(self, UPPER.negate(), true);
  }

  @SkylarkCallable(
      name = "isupper",
      doc =
          "Returns True if all cased characters in the string are uppercase and there is "
              + "at least one character.",
      parameters = {@Param(name = "self", type = String.class, doc = "This string.")})
  public Boolean isUpper(String self) throws EvalException {
    // Python also accepts non-cased characters, so we cannot use UPPER.
    return matches(self, LOWER.negate(), true);
  }

  @SkylarkCallable(
      name = "istitle",
      doc =
          "Returns True if the string is in title case and it contains at least one character. "
              + "This means that every uppercase character must follow an uncased one (e.g. "
              + "whitespace) and every lowercase character must follow a cased one (e.g. "
              + "uppercase or lowercase).",
      parameters = {@Param(name = "self", type = String.class, doc = "This string.")})
  public Boolean isTitle(String self) throws EvalException {
    if (self.isEmpty()) {
      return false;
    }
    // From the Python documentation: "uppercase characters may only follow uncased characters
    // and lowercase characters only cased ones".
    char[] data = self.toCharArray();
    CharMatcher matcher = CharMatcher.any();
    char leftMostCased = ' ';
    for (int pos = data.length - 1; pos >= 0; --pos) {
      char current = data[pos];
      // 1. Check condition that was determined by the right neighbor.
      if (!matcher.matches(current)) {
        return false;
      }
      // 2. Determine condition for the left neighbor.
      if (LOWER.matches(current)) {
        matcher = CASED;
      } else if (UPPER.matches(current)) {
        matcher = CASED.negate();
      } else {
        matcher = CharMatcher.any();
      }
      // 3. Store character if it is cased.
      if (CASED.matches(current)) {
        leftMostCased = current;
      }
    }
    // The leftmost cased letter must be uppercase. If leftMostCased is not a cased letter here,
    // then the string doesn't have any cased letter, so UPPER.test will return false.
    return UPPER.matches(leftMostCased);
  }

  private static boolean matches(
      String str, CharMatcher matcher, boolean requiresAtLeastOneCasedLetter) {
    if (str.isEmpty()) {
      return false;
    } else if (!requiresAtLeastOneCasedLetter) {
      return matcher.matchesAllOf(str);
    }
    int casedLetters = 0;
    for (char current : str.toCharArray()) {
      if (!matcher.matches(current)) {
        return false;
      } else if (requiresAtLeastOneCasedLetter && CASED.matches(current)) {
        ++casedLetters;
      }
    }
    return casedLetters > 0;
  }

  private static final CharMatcher DIGIT = CharMatcher.javaDigit();
  private static final CharMatcher LOWER = CharMatcher.inRange('a', 'z');
  private static final CharMatcher UPPER = CharMatcher.inRange('A', 'Z');
  private static final CharMatcher ALPHA = LOWER.or(UPPER);
  private static final CharMatcher ALNUM = ALPHA.or(DIGIT);
  private static final CharMatcher CASED = ALPHA;
  private static final CharMatcher SPACE = CharMatcher.whitespace();

  @SkylarkCallable(
      name = "count",
      doc =
          "Returns the number of (non-overlapping) occurrences of substring <code>sub</code> in "
              + "string, optionally restricting to <code>[start:end]</code>, <code>start</code> "
              + "being inclusive and <code>end</code> being exclusive.",
      parameters = {
        @Param(name = "self", type = String.class, doc = "This string."),
        @Param(
            name = "sub",
            type = String.class,
            doc = "The substring to count."),
        @Param(
            name = "start",
            type = Integer.class,
            defaultValue = "0",
            doc = "Restrict to search from this position."),
        @Param(
            name = "end",
            type = Integer.class,
            noneable = true,
            defaultValue = "None",
            doc = "optional position before which to restrict to search.")
      })
  public Integer count(String self, String sub, Integer start, Object end) throws EvalException {
    String str = pythonSubstring(self, start, end, "'end' operand of 'find'");
    if (sub.isEmpty()) {
      return str.length() + 1;
    }
    int count = 0;
    int index = 0;
    while ((index = str.indexOf(sub, index)) >= 0) {
      count++;
      index += sub.length();
    }
    return count;
  }

  @SkylarkCallable(
      name = "elems",
      doc =
          "Returns an iterable value containing successive 1-element substrings of the string. "
              + "Equivalent to <code>[s[i] for i in range(len(s))]</code>, except that the "
              + "returned value might not be a list.",
      parameters = {@Param(name = "self", type = String.class, doc = "This string.")})
  public Sequence<String> elems(String self) throws EvalException {
    ImmutableList.Builder<String> builder = new ImmutableList.Builder<>();
    for (char c : self.toCharArray()) {
      builder.add(String.valueOf(c));
    }
    return StarlarkList.immutableCopyOf(builder.build());
  }

  @SkylarkCallable(
      name = "endswith",
      doc =
          "Returns True if the string ends with <code>sub</code>, otherwise False, optionally "
              + "restricting to <code>[start:end]</code>, <code>start</code> being inclusive "
              + "and <code>end</code> being exclusive.",
      parameters = {
        @Param(name = "self", type = String.class, doc = "This string."),
        @Param(
            name = "sub",
            allowedTypes = {
              @ParamType(type = String.class),
              @ParamType(type = Tuple.class, generic1 = String.class),
            },
            doc = "The substring to check."),
        @Param(
            name = "start",
            type = Integer.class,
            defaultValue = "0",
            doc = "Test beginning at this position."),
        @Param(
            name = "end",
            type = Integer.class,
            noneable = true,
            defaultValue = "None",
            doc = "optional position at which to stop comparing.")
      })
  public Boolean endsWith(String self, Object sub, Integer start, Object end) throws EvalException {
    String str = pythonSubstring(self, start, end, "'end' operand of 'endswith'");
    if (sub instanceof String) {
      return str.endsWith((String) sub);
    }
    for (String s : Sequence.cast(sub, String.class, "sub")) {
      if (str.endsWith(s)) {
        return true;
      }
    }
    return false;
  }

  // In Python, formatting is very complex.
  // We handle here the simplest case which provides most of the value of the function.
  // https://docs.python.org/3/library/string.html#formatstrings
  @SkylarkCallable(
      name = "format",
      doc =
          "Perform string interpolation. Format strings contain replacement fields "
              + "surrounded by curly braces <code>{}</code>. Anything that is not contained "
              + "in braces is considered literal text, which is copied unchanged to the output."
              + "If you need to include a brace character in the literal text, it can be "
              + "escaped by doubling: <code>{{</code> and <code>}}</code>"
              + "A replacement field can be either a name, a number, or empty. Values are "
              + "converted to strings using the <a href=\"globals.html#str\">str</a> function."
              + "<pre class=\"language-python\">"
              + "# Access in order:\n"
              + "\"{} < {}\".format(4, 5) == \"4 < 5\"\n"
              + "# Access by position:\n"
              + "\"{1}, {0}\".format(2, 1) == \"1, 2\"\n"
              + "# Access by name:\n"
              + "\"x{key}x\".format(key = 2) == \"x2x\"</pre>\n",
      parameters = {
        @Param(name = "self", type = String.class, doc = "This string."),
      },
      extraPositionals =
          @Param(
              name = "args",
              type = Sequence.class,
              defaultValue = "()",
              doc = "List of arguments."),
      extraKeywords =
          @Param(
              name = "kwargs",
              type = Dict.class,
              defaultValue = "{}",
              doc = "Dictionary of arguments."))
  public String format(String self, Sequence<?> args, Dict<?, ?> kwargs) throws EvalException {
    @SuppressWarnings("unchecked")
    List<Object> argObjects = (List<Object>) args.getImmutableList();
    return new FormatParser()
        .format(self, argObjects, Dict.cast(kwargs, String.class, Object.class, "kwargs"));
  }

  @SkylarkCallable(
      name = "startswith",
      doc =
          "Returns True if the string starts with <code>sub</code>, otherwise False, optionally "
              + "restricting to <code>[start:end]</code>, <code>start</code> being inclusive and "
              + "<code>end</code> being exclusive.",
      parameters = {
        @Param(name = "self", type = String.class, doc = "This string."),
        @Param(
            name = "sub",
            allowedTypes = {
              @ParamType(type = String.class),
              @ParamType(type = Tuple.class, generic1 = String.class),
            },
            doc = "The substring(s) to check."),
        @Param(
            name = "start",
            type = Integer.class,
            defaultValue = "0",
            doc = "Test beginning at this position."),
        @Param(
            name = "end",
            type = Integer.class,
            noneable = true,
            defaultValue = "None",
            doc = "Stop comparing at this position.")
      })
  public Boolean startsWith(String self, Object sub, Integer start, Object end)
      throws EvalException {
    String str = pythonSubstring(self, start, end, "'end' operand of 'startswith'");
    if (sub instanceof String) {
      return str.startsWith((String) sub);
    }
    for (String s : Sequence.cast(sub, String.class, "sub")) {
      if (str.startsWith(s)) {
        return true;
      }
    }
    return false;
  }
}
