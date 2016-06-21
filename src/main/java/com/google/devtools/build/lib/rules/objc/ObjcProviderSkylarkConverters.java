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

package com.google.devtools.build.lib.rules.objc;

import static com.google.devtools.build.lib.rules.objc.AppleSkylarkCommon.BAD_SET_TYPE_ERROR;
import static com.google.devtools.build.lib.rules.objc.AppleSkylarkCommon.NOT_SET_ERROR;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.rules.objc.ObjcProvider.Key;
import com.google.devtools.build.lib.syntax.EvalUtils;
import com.google.devtools.build.lib.syntax.SkylarkNestedSet;
import com.google.devtools.build.lib.syntax.SkylarkType;
import com.google.devtools.build.lib.vfs.PathFragment;

/**
 * A utility class for converting ObjcProvider values between java and skylark representation.
 */
public class ObjcProviderSkylarkConverters {

  /**
   * A map of possible NestedSet types to the converters that should define their treatment
   * in translating between a java and skylark ObjcProvider.
   */
  private static final ImmutableMap<Class<?>, Converter> CONVERTERS =
      ImmutableMap.<Class<?>, Converter>builder()
          .put(Artifact.class, new DirectConverter())
          .put(String.class, new DirectConverter())
          .put(PathFragment.class, new PathFragmentToStringConverter())
          .build();

  /**
   * Returns a value for a skylark attribute given a java ObjcProvider key and value.
   */
  public static Object convertToSkylark(Key<?> javaKey, NestedSet<?> javaValue) {
    return CONVERTERS.get(javaKey.getType()).valueForSkylark(javaKey, javaValue);
  }

  /**
   * Returns a value for a java ObjcProvider given a key and a corresponding skylark value.
   */
  public static Iterable<?> convertToJava(Key<?> javaKey, Object skylarkValue) {
    return CONVERTERS.get(javaKey.getType()).valueForJava(javaKey, skylarkValue);
  }

  /**
   * A converter for ObjcProvider values.
   */
  private static interface Converter {
    /**
     * Translates a java ObjcProvider value to a skylark ObjcProvider value.
     */
    abstract Object valueForSkylark(Key<?> javaKey, NestedSet<?> javaValue);

    /**
     * Translates a skylark ObjcProvider value to a java ObjcProvider value.
     */
    abstract Iterable<?> valueForJava(Key<?> javaKey, Object skylarkValue);
  }

  /**
   * A converter that uses the same value for java and skylark.
   */
  private static class DirectConverter implements Converter {

    @Override
    public Object valueForSkylark(Key<?> javaKey, NestedSet<?> javaValue) {
      SkylarkType type = SkylarkType.of(javaKey.getType());
      return SkylarkNestedSet.of(type, javaValue);
    }

    @Override
    public Iterable<?> valueForJava(Key<?> javaKey, Object skylarkValue) {
      validateTypes(skylarkValue, javaKey.getType(), javaKey.getSkylarkKeyName());
      return (SkylarkNestedSet) skylarkValue;
    }
  }

  /**
   * A converter that that translates between a java PathFragment and a skylark string.
   */
  private static class PathFragmentToStringConverter implements Converter {

    @SuppressWarnings("unchecked")
    @Override
    public Object valueForSkylark(Key<?> javaKey, NestedSet<?> javaValue) {
      NestedSetBuilder<String> result = NestedSetBuilder.stableOrder();
      for (PathFragment path : (Iterable<PathFragment>) javaValue) {
        result.add(path.getSafePathString());
      }
      return SkylarkNestedSet.of(String.class, result.build());
    }

    @SuppressWarnings("unchecked")
    @Override
    public Iterable<?> valueForJava(Key<?> javaKey, Object skylarkValue) {
      validateTypes(skylarkValue, String.class, javaKey.getSkylarkKeyName());
      NestedSetBuilder<PathFragment> result = NestedSetBuilder.stableOrder();
      for (String path : (Iterable<String>) skylarkValue) {
        result.add(new PathFragment(path));
      }
      return result.build();
    }
  }

  /**
   * Throws an error if the given object is not a nested set of the given type.
   */
  private static void validateTypes(Object toCheck, Class<?> expectedSetType, String keyName) {
    if (!(toCheck instanceof SkylarkNestedSet)) {
      throw new IllegalArgumentException(
          String.format(NOT_SET_ERROR, keyName, EvalUtils.getDataTypeName(toCheck)));
    } else if (!((SkylarkNestedSet) toCheck).getContentType().canBeCastTo(expectedSetType)) {
      throw new IllegalArgumentException(
          String.format(
              BAD_SET_TYPE_ERROR,
              keyName,
              EvalUtils.getDataTypeNameFromClass(expectedSetType),
              EvalUtils.getDataTypeNameFromClass(
                  ((SkylarkNestedSet) toCheck).getContentType().getType())));
    }
  }
}
