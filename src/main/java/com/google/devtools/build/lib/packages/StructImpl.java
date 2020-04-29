// Copyright 2016 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.packages;

import com.google.common.base.Joiner;
import com.google.common.base.Objects;
import com.google.common.collect.Ordering;
import com.google.devtools.build.lib.skylarkbuildapi.core.StructApi;
import com.google.devtools.build.lib.syntax.ClassObject;
import com.google.devtools.build.lib.syntax.Dict;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.Location;
import com.google.devtools.build.lib.syntax.Printer;
import com.google.devtools.build.lib.syntax.Sequence;
import com.google.devtools.build.lib.syntax.Starlark;
import com.google.protobuf.TextFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * An abstract base class for Starlark values that have fields, have to_json and to_proto methods,
 * have an associated provider (type symbol), and may be returned as the result of analysis from one
 * target to another.
 *
 * <p>StructImpl does not specify how the fields are represented; subclasses must define {@code
 * getValue} and {@code getFieldNames}. For example, {@code NativeInfo} supplies fields from the
 * subclass's {@code SkylarkCallable(structField=true)} annotations, and {@code SkylarkInfo}
 * supplies fields from the map provided at its construction.
 *
 * <p>Two StructImpls are equivalent if they have the same provider and, for each field name
 * reported by {@code getFieldNames} their corresponding field values are equivalent, or accessing
 * them both returns an error.
 */
public abstract class StructImpl implements Info, ClassObject, StructApi {

  private final Provider provider;
  private final Location location;

  /**
   * Constructs an {@link StructImpl}.
   *
   * @param provider the provider describing the type of this instance
   * @param location the Starlark location where this instance is created. If null, defaults to
   *     {@link Location#BUILTIN}.
   */
  protected StructImpl(Provider provider, @Nullable Location location) {
    this.provider = provider;
    this.location = location != null ? location : Location.BUILTIN;
  }

  @Override
  public Provider getProvider() {
    return provider;
  }

  @Override
  public Location getCreationLoc() {
    return location;
  }

  /**
   * Returns the result of {@link #getValue(String)}, cast as the given type, throwing {@link
   * EvalException} if the cast fails.
   */
  public final <T> T getValue(String key, Class<T> type) throws EvalException {
    Object obj = getValue(key);
    if (obj == null) {
      return null;
    }
    try {
      return type.cast(obj);
    } catch (
        @SuppressWarnings("UnusedException")
        ClassCastException unused) {
      throw Starlark.errorf(
          "for %s field, got %s, want %s", key, Starlark.type(obj), Starlark.classType(type));
    }
  }

  /**
   * Returns the error message format to use for unknown fields.
   *
   * <p>By default, it is the one specified by the provider.
   */
  protected String getErrorMessageFormatForUnknownField() {
    return getProvider().getErrorMessageFormatForUnknownField();
  }

  @Override
  public String getErrorMessageForUnknownField(String name) {
    String suffix = "Available attributes: "
        + Joiner.on(", ").join(Ordering.natural().sortedCopy(getFieldNames()));
    return String.format(getErrorMessageFormatForUnknownField(), name) + "\n" + suffix;
  }

  @Override
  public boolean equals(Object otherObject) {
    if (!(otherObject instanceof StructImpl)) {
      return false;
    }
    StructImpl other = (StructImpl) otherObject;
    if (this == other) {
      return true;
    }
    if (!this.getProvider().equals(other.getProvider())) {
      return false;
    }
    // Compare objects' fields and their values
    if (!this.getFieldNames().equals(other.getFieldNames())) {
      return false;
    }
    for (String field : getFieldNames()) {
      if (!Objects.equal(this.getValueOrNull(field), other.getValueOrNull(field))) {
        return false;
      }
    }
    return true;
  }

  @Override
  public int hashCode() {
    List<String> fields = new ArrayList<>(getFieldNames());
    Collections.sort(fields);
    List<Object> objectsToHash = new ArrayList<>();
    objectsToHash.add(getProvider());
    for (String field : fields) {
      objectsToHash.add(field);
      objectsToHash.add(getValueOrNull(field));
    }
    return Objects.hashCode(objectsToHash.toArray());
  }

  /**
   * Convert the object to string using Starlark syntax. The output tries to be reversible (but
   * there is no guarantee, it depends on the actual values).
   */
  @Override
  public void repr(Printer printer) {
    boolean first = true;
    printer.append("struct(");
    // Sort by key to ensure deterministic output.
    for (String fieldName : Ordering.natural().sortedCopy(getFieldNames())) {
      if (!first) {
        printer.append(", ");
      }
      first = false;
      printer.append(fieldName);
      printer.append(" = ");
      printer.repr(getValueOrNull(fieldName));
    }
    printer.append(")");
  }

  private Object getValueOrNull(String name) {
    try {
      return getValue(name);
    } catch (EvalException e) {
      return null;
    }
  }

  @Override
  public String toProto() throws EvalException {
    StringBuilder sb = new StringBuilder();
    printProtoTextMessage(this, sb, 0);
    return sb.toString();
  }

  private static void printProtoTextMessage(ClassObject object, StringBuilder sb, int indent)
      throws EvalException {
    // For determinism sort the fields alphabetically.
    List<String> fields = new ArrayList<>(object.getFieldNames());
    Collections.sort(fields);
    for (String field : fields) {
      printProtoTextMessage(field, object.getValue(field), sb, indent);
    }
  }

  private static void printProtoTextMessage(
      String key, Object value, StringBuilder sb, int indent, String container)
      throws EvalException {
    if (value instanceof Map.Entry) {
      Map.Entry<?, ?> entry = (Map.Entry<?, ?>) value;
      print(sb, key + " {", indent);
      printProtoTextMessage("key", entry.getKey(), sb, indent + 1);
      printProtoTextMessage("value", entry.getValue(), sb, indent + 1);
      print(sb, "}", indent);
    } else if (value instanceof ClassObject) {
      print(sb, key + " {", indent);
      printProtoTextMessage((ClassObject) value, sb, indent + 1);
      print(sb, "}", indent);
    } else if (value instanceof String) {
      print(
          sb,
          key + ": \"" + escapeDoubleQuotesAndBackslashesAndNewlines((String) value) + "\"",
          indent);
    } else if (value instanceof Integer) {
      print(sb, key + ": " + value, indent);
    } else if (value instanceof Boolean) {
      // We're relying on the fact that Java converts Booleans to Strings in the same way
      // as the protocol buffers do.
      print(sb, key + ": " + value, indent);
    } else {
      throw Starlark.errorf(
          "Invalid text format, expected a struct, a dict, a string, a bool, or an int but got a"
              + " %s for %s '%s'",
          Starlark.type(value), container, key);
    }
  }

  private static void printProtoTextMessage(String key, Object value, StringBuilder sb, int indent)
      throws EvalException {
    if (value instanceof Sequence) {
      for (Object item : ((Sequence) value)) {
        // TODO(bazel-team): There should be some constraint on the fields of the structs
        // in the same list but we ignore that for now.
        printProtoTextMessage(key, item, sb, indent, "list element in struct field");
      }
    } else if (value instanceof Dict) {
      for (Map.Entry<?, ?> entry : ((Dict<?, ?>) value).entrySet()) {
        printProtoTextMessage(key, entry, sb, indent, "entry of dictionary");
      }
    } else {
      printProtoTextMessage(key, value, sb, indent, "struct field");
    }
  }

  private static void print(StringBuilder sb, String text, int indent) {
    for (int i = 0; i < indent; i++) {
      sb.append("  ");
    }
    sb.append(text);
    sb.append("\n");
  }

  /**
   * Escapes the given string for use in proto/JSON string.
   *
   * <p>This escapes double quotes, backslashes, and newlines.
   */
  private static String escapeDoubleQuotesAndBackslashesAndNewlines(String string) {
    return TextFormat.escapeDoubleQuotesAndBackslashes(string).replace("\n", "\\n");
  }

  @Override
  public String toJson() throws EvalException {
    StringBuilder sb = new StringBuilder();
    printJson(this, sb, "struct field", null);
    return sb.toString();
  }

  private static void printJson(Object value, StringBuilder sb, String container, String key)
      throws EvalException {
    if (value == Starlark.NONE) {
      sb.append("null");
    } else if (value instanceof ClassObject) {
      sb.append("{");

      String join = "";
      for (String field : ((ClassObject) value).getFieldNames()) {
        sb.append(join);
        join = ",";
        appendJSONStringLiteral(sb, field);
        sb.append(':');
        printJson(((ClassObject) value).getValue(field), sb, "struct field", field);
      }
      sb.append("}");
    } else if (value instanceof Dict) {
      sb.append("{");
      String join = "";
      for (Map.Entry<?, ?> entry : ((Dict<?, ?>) value).entrySet()) {
        sb.append(join);
        join = ",";
        if (!(entry.getKey() instanceof String)) {
          throw Starlark.errorf(
              "Keys must be a string but got a %s for %s%s",
              Starlark.type(entry.getKey()), container, key != null ? " '" + key + "'" : "");
        }
        appendJSONStringLiteral(sb, (String) entry.getKey());
        sb.append(':');
        printJson(entry.getValue(), sb, "dict value", String.valueOf(entry.getKey()));
      }
      sb.append("}");
    } else if (value instanceof List) {
      sb.append("[");
      String join = "";
      for (Object item : ((List) value)) {
        sb.append(join);
        join = ",";
        printJson(item, sb, "list element in struct field", key);
      }
      sb.append("]");
    } else if (value instanceof String) {
      appendJSONStringLiteral(sb, (String) value);
    } else if (value instanceof Integer || value instanceof Boolean) {
      sb.append(value);
    } else {
      throw Starlark.errorf(
          "Invalid text format, expected a struct, a string, a bool, or an int but got a %s for"
              + " %s%s",
          Starlark.type(value), container, key != null ? " '" + key + "'" : "");
    }
  }

  private static void appendJSONStringLiteral(StringBuilder out, String s) {
    out.append('"');
    out.append(
        escapeDoubleQuotesAndBackslashesAndNewlines(s).replace("\r", "\\r").replace("\t", "\\t"));
    out.append('"');
  }

  @Override
  public String toString() {
    return Starlark.repr(this);
  }
}
