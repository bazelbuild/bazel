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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Joiner;
import com.google.common.base.Objects;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableSortedMap;
import com.google.common.collect.Ordering;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.skylarkbuildapi.StructApi;
import com.google.devtools.build.lib.skylarkinterface.SkylarkPrinter;
import com.google.devtools.build.lib.syntax.ClassObject;
import com.google.devtools.build.lib.syntax.Dict;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.EvalUtils;
import com.google.devtools.build.lib.syntax.Sequence;
import com.google.devtools.build.lib.syntax.SkylarkType;
import com.google.devtools.build.lib.syntax.Starlark;
import com.google.protobuf.TextFormat;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * A generic skylark object with fields, constructable by calling {@code struct()} in skylark.
 */
public abstract class StructImpl extends Info
    implements ClassObject, StructApi, Serializable {

  /**
   * Constructs an {@link StructImpl}.
   *
   * @param provider the provider describing the type of this instance
   * @param location the Skylark location where this instance is created. If null, defaults to
   *     {@link Location#BUILTIN}.
   */
  protected StructImpl(Provider provider, @Nullable Location location) {
    super(provider, location);
  }

  /**
   * Preprocesses a map of field values to convert the field names and field values to
   * Skylark-acceptable names and types.
   *
   * <p>Entries are ordered by key.
   */
  static ImmutableSortedMap<String, Object> copyValues(Map<String, Object> values) {
    Preconditions.checkNotNull(values);
    ImmutableSortedMap.Builder<String, Object> builder = ImmutableSortedMap.naturalOrder();
    for (Map.Entry<String, Object> e : values.entrySet()) {
      builder.put(Attribute.getSkylarkName(e.getKey()), Starlark.fromJava(e.getValue(), null));
    }
    return builder.build();
  }

  /**
   * Returns whether the given field name exists.
   *
   * <p>This conceptually extends the API for {@link ClassObject}.
   */
  public abstract boolean hasField(String name);

  /**
   * <p>Wraps {@link ClassObject#getValue(String)}, returning null in cases where
   * {@link EvalException} would have been thrown.
   */
  @VisibleForTesting
  public Object getValueOrNull(String name) {
    try {
      return getValue(name);
    } catch (EvalException e) {
      return null;
    }
  }

  /**
   * Returns the result of {@link #getValue(String)}, cast as the given type, throwing {@link
   * EvalException} if the cast fails.
   */
  public <T> T getValue(String key, Class<T> type) throws EvalException {
    Object obj = getValue(key);
    if (obj == null) {
      return null;
    }
    SkylarkType.checkType(obj, type, key);
    return type.cast(obj);
  }

  /**
   * {@inheritDoc}
   *
   * <p>Overrides {@link ClassObject#getFieldNames()}, but does not allow {@link EvalException} to
   * be thrown.
   */
  @Override
  public abstract ImmutableCollection<String> getFieldNames();

  /**
   * Returns the error message format to use for unknown fields.
   *
   * <p>By default, it is the one specified by the provider.
   */
  protected String getErrorMessageFormatForUnknownField() {
    return provider.getErrorMessageFormatForUnknownField();
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
    if (!this.provider.equals(other.provider)) {
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
    objectsToHash.add(provider);
    for (String field : fields) {
      objectsToHash.add(field);
      objectsToHash.add(getValueOrNull(field));
    }
    return Objects.hashCode(objectsToHash.toArray());
  }

  /**
   * Convert the object to string using Skylark syntax. The output tries to be reversible (but there
   * is no guarantee, it depends on the actual values).
   */
  @Override
  public void repr(SkylarkPrinter printer) {
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

  @Override
  public String toProto(Location loc) throws EvalException {
    StringBuilder sb = new StringBuilder();
    printProtoTextMessage(this, sb, 0, loc);
    return sb.toString();
  }

  private void printProtoTextMessage(ClassObject object, StringBuilder sb, int indent, Location loc)
      throws EvalException {
    // For determinism sort the fields alphabetically.
    List<String> fields = new ArrayList<>(object.getFieldNames());
    Collections.sort(fields);
    for (String field : fields) {
      printProtoTextMessage(field, object.getValue(field), sb, indent, loc);
    }
  }

  private void printProtoTextMessage(
      String key, Object value, StringBuilder sb, int indent, Location loc, String container)
      throws EvalException {
    if (value instanceof Map.Entry) {
      Map.Entry<?, ?> entry = (Map.Entry<?, ?>) value;
      print(sb, key + " {", indent);
      printProtoTextMessage("key", entry.getKey(), sb, indent + 1, loc);
      printProtoTextMessage("value", entry.getValue(), sb, indent + 1, loc);
      print(sb, "}", indent);
    } else if (value instanceof ClassObject) {
      print(sb, key + " {", indent);
      printProtoTextMessage((ClassObject) value, sb, indent + 1, loc);
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
      throw new EvalException(
          loc,
          "Invalid text format, expected a struct, a dict, a string, a bool, or an int but got a "
              + EvalUtils.getDataTypeName(value)
              + " for "
              + container
              + " '"
              + key
              + "'");
    }
  }

  private void printProtoTextMessage(
      String key, Object value, StringBuilder sb, int indent, Location loc) throws EvalException {
    if (value instanceof Sequence) {
      for (Object item : ((Sequence) value)) {
        // TODO(bazel-team): There should be some constraint on the fields of the structs
        // in the same list but we ignore that for now.
        printProtoTextMessage(key, item, sb, indent, loc, "list element in struct field");
      }
    } else if (value instanceof Dict) {
      for (Map.Entry<?, ?> entry : ((Dict<?, ?>) value).entrySet()) {
        printProtoTextMessage(key, entry, sb, indent, loc, "entry of dictionary");
      }
    } else {
      printProtoTextMessage(key, value, sb, indent, loc, "struct field");
    }
  }

  private void print(StringBuilder sb, String text, int indent) {
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
  public String toJson(Location loc) throws EvalException {
    StringBuilder sb = new StringBuilder();
    printJson(this, sb, loc, "struct field", null);
    return sb.toString();
  }

  private void printJson(Object value, StringBuilder sb, Location loc, String container, String key)
      throws EvalException {
    if (value == Starlark.NONE) {
      sb.append("null");
    } else if (value instanceof ClassObject) {
      sb.append("{");

      String join = "";
      for (String field : ((ClassObject) value).getFieldNames()) {
        sb.append(join);
        join = ",";
        sb.append("\"");
        sb.append(field);
        sb.append("\":");
        printJson(((ClassObject) value).getValue(field), sb, loc, "struct field", field);
      }
      sb.append("}");
    } else if (value instanceof Dict) {
      sb.append("{");
      String join = "";
      for (Map.Entry<?, ?> entry : ((Dict<?, ?>) value).entrySet()) {
        sb.append(join);
        join = ",";
        if (!(entry.getKey() instanceof String)) {
          String errorMessage =
              "Keys must be a string but got a "
                  + EvalUtils.getDataTypeName(entry.getKey())
                  + " for "
                  + container;
          if (key != null) {
            errorMessage += " '" + key + "'";
          }
          throw new EvalException(loc, errorMessage);
        }
        sb.append("\"");
        sb.append(entry.getKey());
        sb.append("\":");
        printJson(entry.getValue(), sb, loc, "dict value", String.valueOf(entry.getKey()));
      }
      sb.append("}");
    } else if (value instanceof List) {
      sb.append("[");
      String join = "";
      for (Object item : ((List) value)) {
        sb.append(join);
        join = ",";
        printJson(item, sb, loc, "list element in struct field", key);
      }
      sb.append("]");
    } else if (value instanceof String) {
      sb.append("\"");
      sb.append(jsonEscapeString((String) value));
      sb.append("\"");
    } else if (value instanceof Integer || value instanceof Boolean) {
      sb.append(value);
    } else {
      String errorMessage =
          "Invalid text format, expected a struct, a string, a bool, or an int "
              + "but got a "
              + EvalUtils.getDataTypeName(value)
              + " for "
              + container;
      if (key != null) {
        errorMessage += " '" + key + "'";
      }
      throw new EvalException(loc, errorMessage);
    }
  }

  private String jsonEscapeString(String string) {
    return escapeDoubleQuotesAndBackslashesAndNewlines(string)
        .replace("\r", "\\r")
        .replace("\t", "\\t");
  }

  @Override
  public String toString() {
    return Starlark.repr(this);
  }
}
