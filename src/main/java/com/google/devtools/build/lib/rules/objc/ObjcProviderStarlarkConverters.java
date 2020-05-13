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

import static com.google.devtools.build.lib.rules.objc.AppleStarlarkCommon.NOT_SET_ERROR;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.rules.objc.ObjcProvider.Key;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.Starlark;
import com.google.devtools.build.lib.vfs.PathFragment;

/** A utility class for converting ObjcProvider values between java and Starlark representation. */
public class ObjcProviderStarlarkConverters {

  /**
   * A map of possible NestedSet types to the converters that should define their treatment in
   * translating between a java and Starlark ObjcProvider.
   */
  private static final ImmutableMap<Class<?>, Converter> CONVERTERS =
      ImmutableMap.<Class<?>, Converter>builder()
          .put(Artifact.class, new DirectConverter())
          .put(String.class, new DirectConverter())
          .put(PathFragment.class, new PathFragmentToStringConverter())
          .put(SdkFramework.class, new SdkFrameworkToStringConverter())
          .build();

  /** Returns a value for a Starlark attribute given a java ObjcProvider key and value. */
  public static Object convertToStarlark(Key<?> javaKey, NestedSet<?> javaValue) {
    return CONVERTERS.get(javaKey.getType()).valueForStarlark(javaKey, javaValue);
  }

  /** Returns a value for a java ObjcProvider given a key and a corresponding Starlark value. */
  public static NestedSet<?> convertToJava(Key<?> javaKey, Object starlarkValue)
      throws EvalException {
    return CONVERTERS.get(javaKey.getType()).valueForJava(javaKey, starlarkValue);
  }

  /** Converts {@link PathFragment}s into a Starlark-compatible nested set of path strings. */
  public static Depset convertPathFragmentsToStarlark(NestedSet<PathFragment> pathFragments) {
    NestedSetBuilder<String> result = NestedSetBuilder.stableOrder();
    for (PathFragment path : pathFragments.toList()) {
      result.add(path.getSafePathString());
    }
    return Depset.of(Depset.ElementType.STRING, result.build());
  }

  /** A converter for ObjcProvider values. */
  private interface Converter {
    /** Translates a java ObjcProvider value to a Starlark ObjcProvider value. */
    Object valueForStarlark(Key<?> javaKey, NestedSet<?> javaValue);

    /** Translates a Starlark ObjcProvider value to a java ObjcProvider value. */
    NestedSet<?> valueForJava(Key<?> javaKey, Object starlarkValue) throws EvalException;
  }

  /** A converter that uses the same value for java and Starlark. */
  private static class DirectConverter implements Converter {

    @Override
    public Object valueForStarlark(Key<?> javaKey, NestedSet<?> javaValue) {
      Depset.ElementType type = Depset.ElementType.of(javaKey.getType());
      return Depset.of(type, javaValue);
    }

    @Override
    public NestedSet<?> valueForJava(Key<?> javaKey, Object starlarkValue) throws EvalException {
      return nestedSetWithType(starlarkValue, javaKey.getType(), javaKey.getStarlarkKeyName());
    }
  }

  /** A converter that that translates between a java PathFragment and a Starlark string. */
  private static class PathFragmentToStringConverter implements Converter {

    @SuppressWarnings("unchecked")
    @Override
    public Object valueForStarlark(Key<?> javaKey, NestedSet<?> javaValue) {
      return convertPathFragmentsToStarlark((NestedSet<PathFragment>) javaValue);
    }

    @Override
    public NestedSet<?> valueForJava(Key<?> javaKey, Object starlarkValue) throws EvalException {
      NestedSet<String> nestedSet =
          nestedSetWithType(starlarkValue, String.class, javaKey.getStarlarkKeyName());
      NestedSetBuilder<PathFragment> result = NestedSetBuilder.stableOrder();
      for (String path : nestedSet.toList()) {
        result.add(PathFragment.create(path));
      }
      return result.build();
    }
  }

  /** A converter that that translates between a java {@link SdkFramework} and a Starlark string. */
  private static class SdkFrameworkToStringConverter implements Converter {

    @SuppressWarnings("unchecked")
    @Override
    public Object valueForStarlark(Key<?> javaKey, NestedSet<?> javaValue) {
      NestedSetBuilder<String> result = NestedSetBuilder.stableOrder();
      for (SdkFramework framework : ((NestedSet<SdkFramework>) javaValue).toList()) {
        result.add(framework.getName());
      }
      return Depset.of(Depset.ElementType.STRING, result.build());
    }

    @Override
    public NestedSet<?> valueForJava(Key<?> javaKey, Object starlarkValue) throws EvalException {
      NestedSet<String> nestedSet =
          nestedSetWithType(starlarkValue, String.class, javaKey.getStarlarkKeyName());
      NestedSetBuilder<SdkFramework> result = NestedSetBuilder.stableOrder();
      for (String path : nestedSet.toList()) {
        result.add(new SdkFramework(path));
      }
      return result.build();
    }
  }

  /** Throws EvalException if x is not a depset of the given type. */
  private static <T> NestedSet<T> nestedSetWithType(Object x, Class<T> elemType, String what)
      throws EvalException {
    if (x == null) {
      throw Starlark.errorf(NOT_SET_ERROR, what, Starlark.type(x));
    }
    return Depset.cast(x, elemType, what);
  }
}
