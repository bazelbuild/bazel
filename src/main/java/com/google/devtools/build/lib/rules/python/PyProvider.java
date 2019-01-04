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

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.packages.StructImpl;
import com.google.devtools.build.lib.packages.StructProvider;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.EvalUtils;
import com.google.devtools.build.lib.syntax.SkylarkNestedSet;
import com.google.devtools.build.lib.syntax.SkylarkType;

/**
 * Static helper class for managing the "py" struct provider returned and consumed by Python rules.
 */
// TODO(brandjon): Replace this with a real provider.
public class PyProvider {

  // Disable construction.
  private PyProvider() {}

  /** Name of the Python provider in Starlark code (as a field of a {@code Target}. */
  public static final String PROVIDER_NAME = "py";

  /**
   * Name of field holding a depset of transitive sources (i.e., .py files in srcs and in srcs of
   * transitive deps).
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
  public static final String IMPORTS = "imports";

  /** Constructs a provider instance with the given field values. */
  public static StructImpl create(
      NestedSet<Artifact> transitiveSources,
      boolean usesSharedLibraries,
      NestedSet<String> imports) {
    return StructProvider.STRUCT.create(
        ImmutableMap.of(
            PyProvider.TRANSITIVE_SOURCES,
            SkylarkNestedSet.of(Artifact.class, transitiveSources),
            PyProvider.USES_SHARED_LIBRARIES,
            usesSharedLibraries,
            PyProvider.IMPORTS,
            SkylarkNestedSet.of(String.class, imports)),
        "No such attribute '%s'");
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
      throw new EvalException(
          /*location=*/ null, String.format("'py' provider missing '%s' field", fieldName));
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
    return castValue.getSet(Artifact.class);
  }

  /**
   * Casts and returns the uses-shared-libraries field.
   *
   * @throws EvalException if the field does not exist or is not a boolean
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
   * Casts and returns the imports field.
   *
   * @throws EvalException if the field does not exist or is not a depset of strings
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
}
