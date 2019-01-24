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

package com.google.devtools.build.lib.rules.python;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.StructImpl;
import com.google.devtools.build.lib.packages.StructProvider;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.EvalUtils;
import com.google.devtools.build.lib.syntax.SkylarkNestedSet;
import com.google.devtools.build.lib.syntax.SkylarkType;

/** Static helper class for creating and accessing instances of the "py" legacy struct provider. */
// TODO(#7010): Replace this with a real provider.
public class PyStructUtils {

  // Disable construction.
  private PyStructUtils() {}

  /** Name of the Python provider in Starlark code (as a field of a {@code Target}. */
  public static final String PROVIDER_NAME = "py";

  /**
   * Name of field holding a postorder-compatible depset of transitive sources (i.e., .py files in
   * {@code srcs} and in {@code srcs} of transitive {@code deps}).
   */
  public static final String TRANSITIVE_SOURCES = "transitive_sources";

  /**
   * Name of field holding a boolean indicating whether any transitive dep uses shared libraries.
   */
  public static final String USES_SHARED_LIBRARIES = "uses_shared_libraries";

  /**
   * Name of field holding a depset of import paths added by the transitive deps (including this
   * target).
   */
  // TODO(brandjon): Make this a pre-order depset, since higher-level targets should get precedence
  // on PYTHONPATH.
  // TODO(brandjon): Add assertions that this depset and transitive_sources have an order compatible
  // with the one expected by the rules.
  public static final String IMPORTS = "imports";

  /**
   * Name of field holding a boolean indicating whether there are any transitive sources that
   * require a Python 2 runtime.
   */
  public static final String HAS_PY2_ONLY_SOURCES = "has_py2_only_sources";

  /**
   * Name of field holding a boolean indicating whether there are any transitive sources that
   * require a Python 3 runtime.
   */
  public static final String HAS_PY3_ONLY_SOURCES = "has_py3_only_sources";

  private static final ImmutableMap<String, Object> DEFAULTS;

  static {
    ImmutableMap.Builder<String, Object> builder = ImmutableMap.builder();
    // TRANSITIVE_SOURCES is mandatory
    builder.put(USES_SHARED_LIBRARIES, false);
    builder.put(
        IMPORTS,
        SkylarkNestedSet.of(String.class, NestedSetBuilder.<String>compileOrder().build()));
    builder.put(HAS_PY2_ONLY_SOURCES, false);
    builder.put(HAS_PY3_ONLY_SOURCES, false);
    DEFAULTS = builder.build();
  }

  /** Returns whether a given dependency has the py provider. */
  public static boolean hasProvider(TransitiveInfoCollection dep) {
    return dep.get(PROVIDER_NAME) != null;
  }

  /**
   * Returns the struct representing the py provider, from the given target info.
   *
   * @throws EvalException if the provider does not exist or has the wrong type.
   */
  public static StructImpl getProvider(TransitiveInfoCollection dep) throws EvalException {
    Object info = dep.get(PROVIDER_NAME);
    if (info == null) {
      throw new EvalException(/*location=*/ null, "Target does not have 'py' provider");
    }
    return SkylarkType.cast(
        info, StructImpl.class, null, "'%s' provider should be a struct", PROVIDER_NAME);
  }

  private static Object getValue(StructImpl info, String fieldName) throws EvalException {
    Object fieldValue = info.getValue(fieldName);
    if (fieldValue == null) {
      fieldValue = DEFAULTS.get(fieldName);
      if (fieldValue == null) {
        throw new EvalException(
            /*location=*/ null, String.format("'py' provider missing '%s' field", fieldName));
      }
    }
    return fieldValue;
  }

  /**
   * Casts and returns the transitive sources field.
   *
   * @throws EvalException if the field does not exist or is not a depset of {@link Artifact}
   */
  public static NestedSet<Artifact> getTransitiveSources(StructImpl info) throws EvalException {
    Object fieldValue = getValue(info, TRANSITIVE_SOURCES);
    SkylarkNestedSet castValue =
        SkylarkType.cast(
            fieldValue,
            SkylarkNestedSet.class,
            Artifact.class,
            null,
            "'%s' provider's '%s' field should be a depset of Files (got a '%s')",
            PROVIDER_NAME,
            TRANSITIVE_SOURCES,
            EvalUtils.getDataTypeNameFromClass(fieldValue.getClass()));
    NestedSet<Artifact> unwrappedValue = castValue.getSet(Artifact.class);
    if (!unwrappedValue.getOrder().isCompatible(Order.COMPILE_ORDER)) {
      throw new EvalException(
          /*location=*/ null,
          String.format(
              "Incompatible depset order for 'transitive_sources': expected 'default' or "
                  + "'postorder', but got '%s'",
              unwrappedValue.getOrder().getSkylarkName()));
    }
    return unwrappedValue;
  }

  /**
   * Casts and returns the uses-shared-libraries field (or its default value).
   *
   * @throws EvalException if the field exists and is not a boolean
   */
  public static boolean getUsesSharedLibraries(StructImpl info) throws EvalException {
    Object fieldValue = getValue(info, USES_SHARED_LIBRARIES);
    return SkylarkType.cast(
        fieldValue,
        Boolean.class,
        null,
        "'%s' provider's '%s' field should be a boolean (got a '%s')",
        PROVIDER_NAME,
        USES_SHARED_LIBRARIES,
        EvalUtils.getDataTypeNameFromClass(fieldValue.getClass()));
  }

  /**
   * Casts and returns the imports field (or its default value).
   *
   * @throws EvalException if the field exists and is not a depset of strings
   */
  public static NestedSet<String> getImports(StructImpl info) throws EvalException {
    Object fieldValue = getValue(info, IMPORTS);
    SkylarkNestedSet castValue =
        SkylarkType.cast(
            fieldValue,
            SkylarkNestedSet.class,
            String.class,
            null,
            "'%s' provider's '%s' field should be a depset of strings (got a '%s')",
            PROVIDER_NAME,
            IMPORTS,
            EvalUtils.getDataTypeNameFromClass(fieldValue.getClass()));
    return castValue.getSet(String.class);
  }

  /**
   * Casts and returns the py2-only-sources field (or its default value).
   *
   * @throws EvalException if the field exists and is not a boolean
   */
  public static boolean getHasPy2OnlySources(StructImpl info) throws EvalException {
    Object fieldValue = getValue(info, HAS_PY2_ONLY_SOURCES);
    return SkylarkType.cast(
        fieldValue,
        Boolean.class,
        null,
        "'%s' provider's '%s' field should be a boolean (got a '%s')",
        PROVIDER_NAME,
        HAS_PY2_ONLY_SOURCES,
        EvalUtils.getDataTypeNameFromClass(fieldValue.getClass()));
  }

  /**
   * Casts and returns the py3-only-sources field (or its default value).
   *
   * @throws EvalException if the field exists and is not a boolean
   */
  public static boolean getHasPy3OnlySources(StructImpl info) throws EvalException {
    Object fieldValue = getValue(info, HAS_PY3_ONLY_SOURCES);
    return SkylarkType.cast(
        fieldValue,
        Boolean.class,
        null,
        "'%s' provider's '%s' field should be a boolean (got a '%s')",
        PROVIDER_NAME,
        HAS_PY3_ONLY_SOURCES,
        EvalUtils.getDataTypeNameFromClass(fieldValue.getClass()));
  }

  public static Builder builder() {
    return new Builder();
  }

  /** Builder for a py provider struct. */
  public static class Builder {
    SkylarkNestedSet transitiveSources = null;
    Boolean usesSharedLibraries = null;
    SkylarkNestedSet imports = null;
    Boolean hasPy2OnlySources = null;
    Boolean hasPy3OnlySources = null;

    // Use the static builder() method instead.
    private Builder() {}

    public Builder setTransitiveSources(NestedSet<Artifact> transitiveSources) {
      this.transitiveSources = SkylarkNestedSet.of(Artifact.class, transitiveSources);
      return this;
    }

    public Builder setUsesSharedLibraries(boolean usesSharedLibraries) {
      this.usesSharedLibraries = usesSharedLibraries;
      return this;
    }

    public Builder setImports(NestedSet<String> imports) {
      this.imports = SkylarkNestedSet.of(String.class, imports);
      return this;
    }

    public Builder setHasPy2OnlySources(boolean hasPy2OnlySources) {
      this.hasPy2OnlySources = hasPy2OnlySources;
      return this;
    }

    public Builder setHasPy3OnlySources(boolean hasPy3OnlySources) {
      this.hasPy3OnlySources = hasPy3OnlySources;
      return this;
    }

    private static void put(
        ImmutableMap.Builder<String, Object> fields, String fieldName, Object value) {
      fields.put(fieldName, value != null ? value : DEFAULTS.get(fieldName));
    }

    public StructImpl build() {
      ImmutableMap.Builder<String, Object> fields = ImmutableMap.builder();
      Preconditions.checkNotNull(transitiveSources, "setTransitiveSources is required");
      put(fields, TRANSITIVE_SOURCES, transitiveSources);
      put(fields, USES_SHARED_LIBRARIES, usesSharedLibraries);
      put(fields, IMPORTS, imports);
      put(fields, HAS_PY2_ONLY_SOURCES, hasPy2OnlySources);
      put(fields, HAS_PY3_ONLY_SOURCES, hasPy3OnlySources);
      return StructProvider.STRUCT.create(fields.build(), "No such attribute '%s'");
    }
  }
}
