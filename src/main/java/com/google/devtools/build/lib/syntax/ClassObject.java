// Copyright 2014 The Bazel Authors. All rights reserved.
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

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Ordering;
import com.google.common.collect.Sets;
import com.google.common.collect.Sets.SetView;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.util.Preconditions;

import java.io.Serializable;
import java.util.Map;

import javax.annotation.Nullable;

/**
 * An interface for objects behaving like Skylark structs.
 */
// TODO(bazel-team): type checks
public interface ClassObject {

  /**
   * Returns the value associated with the name field in this struct,
   * or null if the field does not exist.
   */
  @Nullable
  Object getValue(String name);

  /**
   * Returns the fields of this struct.
   */
  ImmutableCollection<String> getKeys();

  /**
   * Returns a customized error message to print if the name is not a valid struct field
   * of this struct, or returns null to use the default error message.
   */
  @Nullable String errorMessage(String name);

  /**
   * An implementation class of ClassObject for structs created in Skylark code.
   */
  // TODO(bazel-team): maybe move the SkylarkModule annotation to the ClassObject interface?
  @Immutable
  @SkylarkModule(name = "struct",
      doc = "A special language element to support structs (i.e. simple value objects). "
          + "See the global <a href=\"globals.html#struct\">struct</a> function "
          + "for more details.")
  public class SkylarkClassObject implements ClassObject, Serializable {
    /** Error message to use when errorMessage argument is null. */
    private static final String DEFAULT_ERROR_MESSAGE = "'struct' object has no attribute '%s'";

    private final ImmutableMap<String, Object> values;
    private final Location creationLoc;
    private final String errorMessage;

    /**
     * Creates a built-in struct (i.e. without creation loc). The errorMessage has to have
     * exactly one '%s' parameter to substitute the struct field name.
     */
    public SkylarkClassObject(Map<String, Object> values, String errorMessage) {
      this.values = copyValues(values);
      this.creationLoc = null;
      this.errorMessage = Preconditions.checkNotNull(errorMessage);
    }

    public SkylarkClassObject(Map<String, Object> values, Location creationLoc) {
      this.values = copyValues(values);
      this.creationLoc = Preconditions.checkNotNull(creationLoc);
      this.errorMessage = DEFAULT_ERROR_MESSAGE;
    }

    // Ensure that values are all acceptable to Skylark before to stuff them in a ClassObject
    private ImmutableMap<String, Object> copyValues(Map<String, Object> values) {
      ImmutableMap.Builder<String, Object> builder = ImmutableMap.builder();
      for (Map.Entry<String, Object> e : values.entrySet()) {
        builder.put(e.getKey(), SkylarkType.convertToSkylark(e.getValue(), null));
      }
      return builder.build();
    }

    @Override
    public Object getValue(String name) {
      return values.get(name);
    }

    /**
     *  Returns a value and try to cast it into specified type
     */
    public <TYPE> TYPE getValue(String key, Class<TYPE> type) throws EvalException {
      Object obj = values.get(key);
      if (obj == null) {
        return null;
      }
      SkylarkType.checkType(obj, type, key);
      return type.cast(obj);
    }

    @Override
    public ImmutableCollection<String> getKeys() {
      return values.keySet();
    }

    public Location getCreationLoc() {
      return Preconditions.checkNotNull(creationLoc,
          "This struct was not created in a Skylark code");
    }

    static SkylarkClassObject concat(
        SkylarkClassObject lval, SkylarkClassObject rval, Location loc) throws EvalException {
      SetView<String> commonFields = Sets.intersection(lval.values.keySet(), rval.values.keySet());
      if (!commonFields.isEmpty()) {
        throw new EvalException(loc, "Cannot concat structs with common field(s): "
            + Joiner.on(",").join(commonFields));
      }
      return new SkylarkClassObject(ImmutableMap.<String, Object>builder()
          .putAll(lval.values).putAll(rval.values).build(), loc);
    }

    @Override
    public String errorMessage(String name) {
      String suffix =
          "Available attributes: "
              + Joiner.on(", ").join(Ordering.natural().sortedCopy(values.keySet()));
      return String.format(errorMessage, name) + "\n" + suffix;
    }

    /**
     * Convert the object to string using Skylark syntax. The output tries to be
     * reversible (but there is no guarantee, it depends on the actual values).
     */
    @Override
    public String toString() {
      StringBuilder builder = new StringBuilder();
      boolean first = true;
      builder.append("struct(");
      // Sort by key to ensure deterministic output.
      for (String key : Ordering.natural().sortedCopy(values.keySet())) {
        if (!first) {
          builder.append(", ");
        }
        first = false;
        builder.append(key);
        builder.append(" = ");
        Printer.write(builder, values.get(key));
      }
      builder.append(")");
      return builder.toString();
    }
  }
}
