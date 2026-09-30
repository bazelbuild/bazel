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

import java.time.temporal.TemporalAccessor;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
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

  private static final Pattern BARE_KEY = Pattern.compile("[A-Za-z0-9_-]+");

  /** The module instance, suitable for a predeclared environment under the name {@code toml}. */
  public static final TomlParser INSTANCE = new TomlParser();

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
              + "Dates and times become normalized ISO-8601 strings by default. Re-encoding these"
              + " values produces TOML strings, not TOML dates or times. Supply"
              + " <code>decode_date</code> to customize their conversion or reject them.\n"
              + "If the input is invalid TOML and <code>default</code> is specified (including"
              + " <code>None</code>), returns that value; otherwise invalid TOML causes an error."
              + " Errors raised by <code>decode_date</code> are propagated even if a default is"
              + " given.",
      parameters = {
        @Param(name = "x", doc = "TOML string to decode."),
        @Param(
            name = "default",
            named = true,
            doc = "If specified, the value to return when the input is invalid TOML.",
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
                "An optional function called with the normalized ISO-8601 string for each date or"
                    + " time. Its return value replaces that date or time, including when it"
                    + " returns None. It may call fail() to reject dates. With None, dates remain"
                    + " strings.",
            defaultValue = "None")
      },
      useStarlarkThread = true)
  public Object decode(String x, Object defaultValue, Object decodeDate, StarlarkThread thread)
      throws EvalException, InterruptedException {
    try {
      boolean byteStrings =
          thread.getSemantics().getBool(StarlarkSemantics.INTERNAL_BAZEL_ONLY_UTF_8_BYTE_STRINGS);
      TomlParseResult result =
          Toml.parse(byteStrings ? new String(x.getBytes(ISO_8859_1), UTF_8) : x);
      if (result.hasErrors()) {
        if (defaultValue != Starlark.UNBOUND) {
          return defaultValue;
        }
        String errorMsg =
            result.errors().stream().map(Object::toString).collect(Collectors.joining("; "));
        throw Starlark.errorf("TOML decode error: %s", errorMsg);
      }
      return convertToStarlark(result, decodeDate, thread);
    } catch (StackOverflowError unused) {
      throw Starlark.errorf("nesting depth limit exceeded");
    }
  }

  private static Object convertToStarlark(Object x, Object decodeDate, StarlarkThread thread)
      throws EvalException, InterruptedException {
    if (x instanceof TemporalAccessor) {
      String date = x.toString();
      return decodeDate instanceof StarlarkCallable callback
          ? Starlark.positionalOnlyCall(thread, callback, date)
          : date;
    } else if (x instanceof TomlArray array) {
      StarlarkList<Object> values = StarlarkList.newList(thread.mutability());
      for (int i = 0; i < array.size(); i++) {
        values.addElement(convertToStarlark(array.get(i), decodeDate, thread));
      }
      return values;
    } else if (x instanceof TomlTable table) {
      Dict<String, Object> values = Dict.of(thread.mutability());
      for (Map.Entry<String, Object> entry : table.entrySet()) {
        values.putEntry(
            (String) convertToStarlark(entry.getKey(), decodeDate, thread),
            convertToStarlark(entry.getValue(), decodeDate, thread));
      }
      return values;
    } else if (x instanceof String s
        && thread
            .getSemantics()
            .getBool(StarlarkSemantics.INTERNAL_BAZEL_ONLY_UTF_8_BYTE_STRINGS)) {
      return new String(s.getBytes(UTF_8), ISO_8859_1);
    }
    // TOML has no null value. All remaining parser values are strings, booleans, longs, or doubles.
    return Starlark.fromJava(x, thread.mutability());
  }

  private static boolean isTable(Object x) {
    return x instanceof Map || x instanceof Structure;
  }

  private static final class Encoder {
    final StringBuilder out = new StringBuilder();
    final StarlarkSemantics semantics;
    // Append encoded keys on descent and restore the parent's length on return.
    final StringBuilder path = new StringBuilder();

    Encoder(StarlarkSemantics semantics) {
      this.semantics = semantics;
    }

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

    private String key(Object table, Object key) throws EvalException {
      if (!(key instanceof String s)) {
        throw Starlark.errorf(
            "%s has %s key, want string", Starlark.type(table), Starlark.type(key));
      }
      return s;
    }

    private boolean isArrayOfTables(Object x) {
      if (!(x instanceof StarlarkIterable<?> iterable)) {
        return false;
      }
      boolean nonempty = false;
      for (Object item : iterable) {
        if (!isTable(unpack(item))) {
          return false;
        }
        nonempty = true;
      }
      return nonempty;
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
      List<Map.Entry<String, Object>> nested = new ArrayList<>();
      // Emit scalar fields first, retaining the original values of nested tables for the second
      // pass.
      for (Map.Entry<?, ?> entry : fields(table).entrySet()) {
        String key = key(table, entry.getKey());
        Object value = unpack(entry.getValue());
        if (value == Starlark.NONE) {
          continue;
        }
        if (isTable(value) || isArrayOfTables(value)) {
          nested.add(Map.entry(key, value));
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
      for (Map.Entry<String, Object> entry : nested) {
        int parentLength = path.length();
        if (parentLength > 0) {
          path.append('.');
        }
        encodeKey(path, entry.getKey());
        try {
          Object value = entry.getValue();
          if (isTable(value)) {
            encodeTable(value, /* arrayElement= */ false);
          } else {
            int i = 0;
            for (Object item : (StarlarkIterable<?>) value) {
              try {
                encodeTable(unpack(item), /* arrayElement= */ true);
              } catch (EvalException ex) {
                throw Starlark.errorf(
                    "at %s index %d: %s", Starlark.type(value), i, ex.getMessage());
              }
              i++;
            }
          }
        } catch (EvalException ex) {
          throw fieldError(table, entry.getKey(), ex);
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
          encodeValue(item, indent + 1, inline);
        } catch (EvalException ex) {
          throw Starlark.errorf("at %s index %d: %s", Starlark.type(list), i, ex.getMessage());
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
      x = unpack(x);
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
    if (BARE_KEY.matcher(key).matches()) {
      out.append(key);
    } else {
      encodeString(out, key);
    }
  }

  private static void encodeString(StringBuilder out, String s) {
    boolean needsEscaping = false;
    for (int i = 0; i < s.length(); i++) {
      char c = s.charAt(i);
      if (c == '\'' || c == '\\' || c < 0x20 || c == 0x7f) {
        needsEscaping = true;
        break;
      }
    }
    if (!needsEscaping) {
      out.append('\'').append(s).append('\'');
      return;
    }
    out.append('"');
    for (int i = 0; i < s.length(); i++) {
      char c = s.charAt(i);
      switch (c) {
        case '\\' -> out.append("\\\\");
        case '"' -> out.append("\\\"");
        case '\b' -> out.append("\\b");
        case '\t' -> out.append("\\t");
        case '\n' -> out.append("\\n");
        case '\f' -> out.append("\\f");
        case '\r' -> out.append("\\r");
        default -> {
          if (c < 0x20 || c == 0x7f) {
            out.append(String.format("\\u%04X", (int) c));
          } else {
            out.append(c);
          }
        }
      }
    }
    out.append('"');
  }
}
