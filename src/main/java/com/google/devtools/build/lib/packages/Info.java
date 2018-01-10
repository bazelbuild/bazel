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
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Ordering;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;
import com.google.devtools.build.lib.skylarkinterface.SkylarkPrinter;
import com.google.devtools.build.lib.skylarkinterface.SkylarkValue;
import com.google.devtools.build.lib.syntax.ClassObject;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.Printer;
import com.google.devtools.build.lib.syntax.SkylarkType;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import javax.annotation.Nullable;

/** An instance (in the Skylark sense, not Java) of a {@link Provider}. */
@SkylarkModule(
  name = "struct",
  category = SkylarkModuleCategory.BUILTIN,
  doc =
      "A generic object with fields. See the global <a href=\"globals.html#struct\"><code>struct"
          + "</code></a> function for more details."
          + "<p>Structs fields cannot be reassigned once the struct is created. Two structs are "
          + "equal if they have the same fields and if corresponding field values are equal."
)
public abstract class Info implements ClassObject, SkylarkValue, Serializable {

  /** The {@link Provider} that describes the type of this instance. */
  private final Provider provider;

  /**
   * The Skylark location where this provider instance was created.
   *
   * <p>Built-in provider instances may use {@link Location#BUILTIN}.
   */
  private final Location creationLoc;

  /**
   * Formattable string with one {@code '%s'} placeholder for the missing field name.
   */
  private final String errorMessageFormatForUnknownField;

  /**
   * Creates an empty struct with a given location.
   *
   * <p>If {@code location} is null, it defaults to {@link Location#BUILTIN}.
   */
  public Info(Provider provider, @Nullable Location location) {
    this.provider = provider;
    this.creationLoc = location == null ? Location.BUILTIN : location;
    this.errorMessageFormatForUnknownField = provider.getErrorMessageFormatForUnknownField();
  }

  /** Creates a built-in struct (i.e. without a Skylark creation location). */
  public Info(Provider provider) {
    this.provider = provider;
    this.creationLoc = Location.BUILTIN;
    this.errorMessageFormatForUnknownField = provider.getErrorMessageFormatForUnknownField();
  }

  /**
   * Creates a built-in struct (i.e. without creation location) that uses a specific error message
   * for missing fields.
   *
   * <p>Only used in {@link
   * com.google.devtools.build.lib.packages.NativeProvider.StructConstructor#create(Map, String)}.
   * If you need to override an error message, the preferred way is to create a new {@link
   * NativeProvider} subclass.
   */
  Info(Provider provider, String errorMessageFormatForUnknownField) {
    this.provider = provider;
    this.creationLoc = Location.BUILTIN;
    this.errorMessageFormatForUnknownField =
        Preconditions.checkNotNull(errorMessageFormatForUnknownField);
  }

  /**
   * Preprocesses a map of field values to convert the field names and field values to
   * Skylark-acceptable names and types.
   *
   * <p>This preserves the order of the map entries.
   */
  protected static ImmutableMap<String, Object> copyValues(Map<String, Object> values) {
    ImmutableMap.Builder<String, Object> builder = ImmutableMap.builder();
    for (Map.Entry<String, Object> e : values.entrySet()) {
      builder.put(
          Attribute.getSkylarkName(e.getKey()),
          SkylarkType.convertToSkylark(e.getValue(), /*env=*/ null));
    }
    return builder.build();
  }

  /**
   * Returns the Skylark location where this provider instance was created.
   *
   * <p>Builtin provider instances may return {@link Location#BUILTIN}.
   */
  public Location getCreationLoc() {
    return creationLoc;
  }

  public Provider getProvider() {
    return provider;
  }

  /**
   * Returns whether the given field name exists.
   *
   * <p>This conceptually extends the API for {@link ClassObject}.
   */
  public abstract boolean hasField(String name);

  /**
   * {@inheritDoc}
   *
   * <p>Overrides {@link ClassObject#getValue(String)}, but does not allow {@link EvalException} to
   * be thrown.
   */
  @Nullable
  @Override
  public abstract Object getValue(String name);

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

  @Override
  public String getErrorMessageForUnknownField(String name) {
    String suffix = "Available attributes: "
        + Joiner.on(", ").join(Ordering.natural().sortedCopy(getFieldNames()));
    return String.format(errorMessageFormatForUnknownField, name) + "\n" + suffix;
  }

  @Override
  public boolean equals(Object otherObject) {
    if (!(otherObject instanceof Info)) {
      return false;
    }
    Info other = (Info) otherObject;
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
      if (!this.getValue(field).equals(other.getValue(field))) {
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
      objectsToHash.add(getValue(field));
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
      printer.repr(getValue(fieldName));
    }
    printer.append(")");
  }

  @Override
  public String toString() {
    return Printer.repr(this);
  }
}
