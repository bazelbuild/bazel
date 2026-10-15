// Copyright 2025 The Bazel Authors. All rights reserved.
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

package net.starlark.java.lib.toml;

import static java.nio.charset.StandardCharsets.ISO_8859_1;
import static java.nio.charset.StandardCharsets.UTF_8;

import java.nio.ByteBuffer;
import java.nio.charset.CharacterCodingException;
import java.nio.charset.CodingErrorAction;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.LocalTime;
import java.time.OffsetDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import javax.annotation.Nullable;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.ParamType;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Mutability;
import net.starlark.java.eval.NoneType;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkCallable;
import net.starlark.java.eval.StarlarkFloat;
import net.starlark.java.eval.StarlarkInt;
import net.starlark.java.eval.StarlarkIterable;
import net.starlark.java.eval.StarlarkList;
import net.starlark.java.eval.StarlarkSemantics;
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.eval.StarlarkValue;
import net.starlark.java.eval.Structure;
import net.starlark.java.lib.StarlarkEncodable;
import org.tomlj.Toml;
import org.tomlj.TomlArray;
import org.tomlj.TomlParseResult;
import org.tomlj.TomlTable;

// Tests at //src/test/java/net/starlark/java/eval:testdata/toml.star

/** The Starlark {@code toml} module for encoding and decoding TOML. */
@StarlarkBuiltin(
    name = "toml",
    category = "core.lib",
    doc = "Module toml is a Starlark module of TOML-related functions.")
public final class TomlParser implements StarlarkValue {

  private TomlParser() {}

  /** The module instance, suitable for a predeclared environment under the name {@code toml}. */
  public static final TomlParser INSTANCE = new TomlParser();

  private static final char[] HEX = "0123456789ABCDEF".toCharArray();

  /** Encodes a Starlark value as TOML. */
  @StarlarkMethod(
      name = "encode",
      doc =
          "Encodes a dict, struct, or native provider as a TOML document.\n"
              + "<ul>\n"
              + "<li>Booleans are encoded as <code>true</code> or <code>false</code>.\n"
              + "<li>Integers are encoded in decimal. Some decoders cannot read integers outside"
              + " the signed 64-bit range.\n"
              + "<li>Finite floats are encoded with a decimal point or exponent. Non-finite floats"
              + " are encoded as <code>+inf</code>, <code>-inf</code>, or <code>nan</code>.\n"
              + "<li>Strings use single-quoted literal strings unless they contain a single quote,"
              + " backslash, or control character. In those cases they use double-quoted basic"
              + " strings with TOML escape sequences.\n"
              + "<li>Dicts and struct-like values are encoded as tables. Dict keys must be strings."
              + " Keys containing characters other than ASCII letters, digits, underscores, and"
              + " hyphens are quoted like string values.\n"
              + "<li>Lists, tuples, and sets are encoded as arrays, in iteration order.\n"
              + "<li>Table entries whose value is <code>None</code> are omitted. <code>None</code>"
              + " in an array, or any other unsupported value, is an error.\n"
              + "</ul>\n"
              + "Within each table, entries other than nested tables and arrays of tables appear"
              + " first. Nested tables use <code>[path]</code> headers; non-empty arrays containing"
              + " only tables use <code>[[path]]</code> headers. Tables inside other arrays use"
              + " inline <code>{key = value}</code> syntax, including all their descendants."
              + " Headers that only introduce nested tables are omitted; empty nested tables and"
              + " array elements retain their headers.\n"
              + "Dict entries retain their iteration order within each group; struct fields use"
              + " alphabetical order. Grouping means dict key order may change on round trip."
              + " Comments and original TOML formatting are not preserved.",
      parameters = {@Param(name = "x")},
      useStarlarkThread = true)
  public String encode(Object x, StarlarkThread thread) throws EvalException, InterruptedException {
    try {
      Encoder encoder = new Encoder(thread.getSemantics());
      Object value = encoder.unpack(x);
      if (!isTable(value)) {
        throw Starlark.errorf(
            "TOML encode requires a dict, struct, or native provider at the top level, got %s",
            Starlark.type(x));
      }
      encoder.encodeTable(value, /* arrayElement= */ false);
      return encoder.out.toString();
    } catch (StackOverflowError unused) {
      throw Starlark.errorf("nesting depth limit exceeded");
    }
  }

  /** Parses a TOML string as a Starlark value. */
  @StarlarkMethod(
      name = "decode",
      doc =
          "Decodes a TOML document as a dict. Tables become dicts, arrays become lists, and"
              + " booleans, strings, integers, and floats become the corresponding Starlark values."
              + " The returned dicts and lists are mutable.\n"
              + "Dates and times become strings in the RFC 3339 format that TOML uses, always"
              + " including seconds, such as <code>1979-05-27T07:32:00Z</code>,"
              + " <code>1979-05-27T07:32:00</code>, <code>1979-05-27</code>, or"
              + " <code>07:32:00</code>. Re-encoding these values produces TOML strings, not TOML"
              + " dates or times. Supply <code>decode_date</code> to customize their conversion or"
              + " reject them.\n"
              + "If the input cannot be decoded (it is invalid TOML, is not valid UTF-8, or is"
              + " nested too deeply) and <code>default</code> is specified (including"
              + " <code>None</code>), returns that value; otherwise such input causes an error."
              + " Errors raised by <code>decode_date</code> are propagated even if a default is"
              + " given.",
      parameters = {
        @Param(name = "x", doc = "TOML string to decode."),
        @Param(
            name = "default",
            named = true,
            doc = "If specified, the value to return when the input cannot be decoded.",
            defaultValue = "unbound"),
        @Param(
            name = "decode_date",
            named = true,
            positional = false,
            allowedTypes = {
              @ParamType(type = StarlarkCallable.class),
              @ParamType(type = NoneType.class)
            },
            doc =
                "An optional function called with the RFC 3339 string for each date or time. Its"
                    + " return value replaces that date or time, including when it returns None."
                    + " It may call fail() to reject dates. With None, dates remain strings.",
            defaultValue = "None")
      },
      useStarlarkThread = true)
  public Object decode(String x, Object defaultValue, Object decodeDate, StarlarkThread thread)
      throws EvalException, InterruptedException {
    boolean byteStrings =
        thread.getSemantics().getBool(StarlarkSemantics.INTERNAL_BAZEL_ONLY_UTF_8_BYTE_STRINGS);
    TomlParseResult result;
    try {
      result = Toml.parse(byteStrings ? bytesToUnicode(x) : x);
    } catch (CharacterCodingException unused) {
      return decodeError(defaultValue, "input is not valid UTF-8");
    } catch (StackOverflowError unused) {
      return decodeError(defaultValue, "nesting depth limit exceeded");
    } catch (RuntimeException ex) {
      // tomlj throws instead of reporting an error for some malformed input, such as the offset
      // in 1979-05-27T07:32:00-07'00.
      return decodeError(defaultValue, ex.toString());
    }
    if (result.hasErrors()) {
      return decodeError(
          defaultValue,
          result.errors().stream().map(Object::toString).collect(Collectors.joining("; ")));
    }
    Decoder decoder =
        new Decoder(
            thread, byteStrings, decodeDate instanceof StarlarkCallable callback ? callback : null);
    try {
      return decoder.table(result);
    } catch (StackOverflowError unused) {
      return decodeError(defaultValue, "nesting depth limit exceeded");
    }
  }

  private static Object decodeError(Object defaultValue, String message) throws EvalException {
    if (defaultValue != Starlark.UNBOUND) {
      return defaultValue;
    }
    throw Starlark.errorf("TOML decode error: %s", message);
  }

  /** Converts the values of a parsed TOML document to Starlark values. */
  private static final class Decoder {
    private final StarlarkThread thread;
    private final boolean byteStrings;
    @Nullable private final StarlarkCallable decodeDate;

    private Decoder(
        StarlarkThread thread, boolean byteStrings, @Nullable StarlarkCallable decodeDate) {
      this.thread = thread;
      this.byteStrings = byteStrings;
      this.decodeDate = decodeDate;
    }

    private Dict<String, Object> table(TomlTable table) throws EvalException, InterruptedException {
      Dict<String, Object> dict = Dict.of(thread.mutability());
      for (Map.Entry<String, Object> entry : table.entrySet()) {
        dict.putEntry(string(entry.getKey()), value(entry.getValue()));
      }
      return dict;
    }

    private Object value(Object x) throws EvalException, InterruptedException {
      // tomlj produces exactly the types listed in org.tomlj.TomlType, and never null.
      return switch (x) {
        case TomlTable table -> table(table);
        case TomlArray array -> {
          StarlarkList<Object> list = StarlarkList.newList(thread.mutability());
          for (int i = 0; i < array.size(); i++) {
            list.addElement(value(array.get(i)));
          }
          yield list;
        }
        case String s -> string(s);
        // The ISO formatters always include seconds, unlike toString(), so that every date and
        // time is spelled as TOML would spell it.
        case OffsetDateTime t -> date(DateTimeFormatter.ISO_OFFSET_DATE_TIME.format(t));
        case LocalDateTime t -> date(DateTimeFormatter.ISO_LOCAL_DATE_TIME.format(t));
        case LocalDate t -> date(DateTimeFormatter.ISO_LOCAL_DATE.format(t));
        case LocalTime t -> date(DateTimeFormatter.ISO_LOCAL_TIME.format(t));
        // All remaining parser values are booleans, longs, or doubles.
        default -> Starlark.fromJava(x, thread.mutability());
      };
    }

    private Object date(String date) throws EvalException, InterruptedException {
      return decodeDate == null ? date : Starlark.positionalOnlyCall(thread, decodeDate, date);
    }

    private String string(String s) {
      return byteStrings ? unicodeToBytes(s) : s;
    }
  }

  // In UTF-8 byte-string mode, each char of a Starlark string holds one byte of the string's UTF-8
  // encoding, whereas tomlj needs Unicode. ASCII strings have the same form in both.

  private static boolean isAscii(String s) {
    for (int i = 0; i < s.length(); i++) {
      if (s.charAt(i) >= 0x80) {
        return false;
      }
    }
    return true;
  }

  private static String bytesToUnicode(String s) throws CharacterCodingException {
    if (isAscii(s)) {
      return s;
    }
    return UTF_8
        .newDecoder()
        .onMalformedInput(CodingErrorAction.REPORT)
        .onUnmappableCharacter(CodingErrorAction.REPORT)
        .decode(ByteBuffer.wrap(s.getBytes(ISO_8859_1)))
        .toString();
  }

  private static String unicodeToBytes(String s) {
    return isAscii(s) ? s : new String(s.getBytes(UTF_8), ISO_8859_1);
  }

  private static boolean isTable(Object x) {
    return x instanceof Map || x instanceof Structure;
  }

  /**
   * Writes Starlark values as TOML.
   *
   * <p>Each value is unpacked (see {@link #unpack}) exactly once, when it is read from its
   * container; the encoding methods expect unpacked values.
   */
  private static final class Encoder {
    private final StringBuilder out = new StringBuilder();
    private final StarlarkSemantics semantics;
    // Encoded keys of the table being written, extended on descent and truncated on return.
    private final StringBuilder path = new StringBuilder();

    /** A table entry written after the scalar entries: a table, or a non-empty array of tables. */
    private record Nested(String key, Object value, @Nullable List<Object> tables) {}

    private Encoder(StarlarkSemantics semantics) {
      this.semantics = semantics;
    }

    /** Applies any application-defined encoding of a value. */
    private Object unpack(Object x) {
      return x instanceof StarlarkEncodable encodable ? encodable.objectForEncoding(semantics) : x;
    }

    private Map<?, ?> fields(Object table) throws EvalException, InterruptedException {
      if (table instanceof Map<?, ?> map) {
        return map;
      }
      // Snapshot only this struct's fields, leaving their values for the encoder to visit directly.
      Map<String, Object> fields = new LinkedHashMap<>();
      for (String field : Starlark.dir(Mutability.IMMUTABLE, semantics, table)) {
        try {
          fields.put(field, Starlark.getattr(Mutability.IMMUTABLE, semantics, table, field, null));
        } catch (EvalException ex) {
          throw fieldError(table, field, ex);
        }
      }
      return fields;
    }

    private EvalException fieldError(Object table, String key, EvalException ex) {
      return table instanceof Map
          ? Starlark.errorf(
              "in %s key %s: %s",
              Starlark.type(table), Starlark.repr(key, semantics), ex.getMessage())
          : Starlark.errorf("in %s field .%s: %s", Starlark.type(table), key, ex.getMessage());
    }

    private static EvalException indexError(Object list, int index, EvalException ex) {
      return Starlark.errorf("at %s index %d: %s", Starlark.type(list), index, ex.getMessage());
    }

    private static String key(Object table, Object key) throws EvalException {
      if (!(key instanceof String s)) {
        throw Starlark.errorf(
            "%s has %s key, want string", Starlark.type(table), Starlark.type(key));
      }
      return s;
    }

    /**
     * Returns the unpacked elements of a non-empty iterable of tables, or null for any other value.
     */
    @Nullable
    private List<Object> arrayOfTables(Object x) {
      if (!(x instanceof StarlarkIterable<?> iterable)) {
        return null;
      }
      List<Object> tables = new ArrayList<>();
      for (Object item : iterable) {
        Object table = unpack(item);
        if (!isTable(table)) {
          return null;
        }
        tables.add(table);
      }
      return tables.isEmpty() ? null : tables;
    }

    private void header(boolean arrayElement) {
      if (out.length() > 0) {
        out.append('\n');
      }
      out.append(arrayElement ? "[[" : "[");
      out.append(path);
      out.append(arrayElement ? "]]\n" : "]\n");
    }

    private void encodeTable(Object table, boolean arrayElement)
        throws EvalException, InterruptedException {
      if (arrayElement) {
        header(true);
      }
      boolean needsHeader = !arrayElement && !path.isEmpty();
      List<Nested> nested = new ArrayList<>();
      // Write scalar entries first, so that they cannot be mistaken for entries of a nested table.
      for (Map.Entry<?, ?> entry : fields(table).entrySet()) {
        String key = key(table, entry.getKey());
        Object value = unpack(entry.getValue());
        if (value == Starlark.NONE) {
          continue;
        }
        if (isTable(value)) {
          nested.add(new Nested(key, value, null));
          continue;
        }
        List<Object> tables = arrayOfTables(value);
        if (tables != null) {
          nested.add(new Nested(key, value, tables));
          continue;
        }
        if (needsHeader) {
          header(false);
          needsHeader = false;
        }
        encodeField(table, key, value, /* inline= */ false);
        out.append('\n');
      }
      // Preserve empty tables, but omit headers that would only introduce nested tables.
      if (needsHeader && nested.isEmpty()) {
        header(false);
      }
      for (Nested entry : nested) {
        int parentLength = path.length();
        if (parentLength > 0) {
          path.append('.');
        }
        encodeKey(path, entry.key());
        try {
          if (entry.tables() == null) {
            encodeTable(entry.value(), /* arrayElement= */ false);
          } else {
            int i = 0;
            for (Object item : entry.tables()) {
              try {
                encodeTable(item, /* arrayElement= */ true);
              } catch (EvalException ex) {
                throw indexError(entry.value(), i, ex);
              }
              i++;
            }
          }
        } catch (EvalException ex) {
          throw fieldError(table, entry.key(), ex);
        } finally {
          path.setLength(parentLength);
        }
      }
    }

    private void encodeList(StarlarkIterable<?> list, int indent, boolean inline)
        throws EvalException, InterruptedException {
      out.append('[');
      String prefix = inline ? "" : "\n" + "  ".repeat(indent + 1);
      String separator = inline ? ", " : ",";
      int i = 0;
      for (Object item : list) {
        if (i > 0) {
          out.append(separator);
        }
        out.append(prefix);
        try {
          encodeValue(unpack(item), indent + 1, inline);
        } catch (EvalException ex) {
          throw indexError(list, i, ex);
        }
        i++;
      }
      if (i > 0 && !inline) {
        out.append(",\n").append("  ".repeat(indent));
      }
      out.append(']');
    }

    private void encodeInlineTable(Object table) throws EvalException, InterruptedException {
      out.append('{');
      boolean first = true;
      for (Map.Entry<?, ?> entry : fields(table).entrySet()) {
        String key = key(table, entry.getKey());
        Object value = unpack(entry.getValue());
        if (value == Starlark.NONE) {
          continue;
        }
        out.append(first ? " " : ", ");
        first = false;
        encodeField(table, key, value, /* inline= */ true);
      }
      out.append(first ? "}" : " }");
    }

    private void encodeField(Object table, String key, Object value, boolean inline)
        throws EvalException, InterruptedException {
      encodeKey(out, key);
      out.append(" = ");
      try {
        encodeValue(value, 0, inline);
      } catch (EvalException ex) {
        throw fieldError(table, key, ex);
      }
    }

    private void encodeValue(Object x, int indent, boolean inline)
        throws EvalException, InterruptedException {
      if (x instanceof String s) {
        encodeString(out, s);
      } else if (x instanceof Boolean || x instanceof StarlarkInt) {
        out.append(x);
      } else if (x instanceof StarlarkFloat f) {
        // Starlark spells non-finite floats as +inf, -inf, and nan, as required by TOML.
        out.append(Double.isFinite(f.toDouble()) ? Double.toString(f.toDouble()) : f.toString());
      } else if (isTable(x)) {
        encodeInlineTable(x);
      } else if (x instanceof StarlarkIterable<?> iterable) {
        encodeList(iterable, indent, inline);
      } else {
        throw Starlark.errorf("cannot encode %s as TOML", Starlark.type(x));
      }
    }
  }

  private static void encodeKey(StringBuilder out, String key) {
    if (isBareKey(key)) {
      out.append(key);
    } else {
      encodeString(out, key);
    }
  }

  private static boolean isBareKey(String key) {
    if (key.isEmpty()) {
      return false;
    }
    for (int i = 0; i < key.length(); i++) {
      char c = key.charAt(i);
      boolean bare =
          ('A' <= c && c <= 'Z')
              || ('a' <= c && c <= 'z')
              || ('0' <= c && c <= '9')
              || c == '_'
              || c == '-';
      if (!bare) {
        return false;
      }
    }
    return true;
  }

  private static boolean isUnpairedSurrogate(int codePoint) {
    return Character.MIN_SURROGATE <= codePoint && codePoint <= Character.MAX_SURROGATE;
  }

  private static void encodeString(StringBuilder out, String s) {
    // Prefer a literal string, which needs no escaping but cannot contain single quotes or
    // control characters. Backslashes are escaped too, for readers who expect them to be.
    boolean literal = true;
    for (int i = 0, n = s.length(); i < n; ) {
      int cp = s.codePointAt(i);
      if (cp == '\'' || cp == '\\' || cp < 0x20 || cp == 0x7f || isUnpairedSurrogate(cp)) {
        literal = false;
        break;
      }
      i += Character.charCount(cp);
    }
    if (literal) {
      out.append('\'').append(s).append('\'');
      return;
    }
    out.append('"');
    for (int i = 0, n = s.length(); i < n; ) {
      int cp = s.codePointAt(i);
      i += Character.charCount(cp);
      switch (cp) {
        case '\\' -> out.append("\\\\");
        case '"' -> out.append("\\\"");
        case '\b' -> out.append("\\b");
        case '\t' -> out.append("\\t");
        case '\n' -> out.append("\\n");
        case '\f' -> out.append("\\f");
        case '\r' -> out.append("\\r");
        default -> {
          if (cp < 0x20 || cp == 0x7f) {
            out.append("\\u00").append(HEX[cp >> 4]).append(HEX[cp & 0xF]);
          } else if (isUnpairedSurrogate(cp)) {
            out.append('�');
          } else {
            out.appendCodePoint(cp);
          }
        }
      }
    }
    out.append('"');
  }
}
