// Copyright 2024 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableList;
import java.lang.reflect.Field;
import java.lang.reflect.Modifier;

/** Utils for testing {@link Type}, {@link Types}, and {@link BuildType}. */
// Must live in the same package as BuildType to access non-public fields.
public final class BuildTypeTestHelper {

  /**
   * Returns all build types defined in {@link Type}, {@link Types}, and {@link BuildType}.
   *
   * @param publicOnly if true, only returns public types; otherwise, returns both public and
   *     internal package-private ones.
   */
  public static ImmutableList<Type<?>> getAllBuildTypes(boolean publicOnly)
      throws IllegalAccessException {
    ImmutableList.Builder<Type<?>> builder = ImmutableList.builder();
    collectBuildTypeStaticFields(builder, Type.class, publicOnly);
    collectBuildTypeStaticFields(builder, Types.class, publicOnly);
    collectBuildTypeStaticFields(builder, BuildType.class, publicOnly);
    return builder.build();
  }

  private static void collectBuildTypeStaticFields(
      ImmutableList.Builder<Type<?>> builder, Class<?> clazz, boolean publicOnly)
      throws IllegalAccessException {
    for (Field field : clazz.getDeclaredFields()) {
      if (Modifier.isStatic(field.getModifiers())
          && (Modifier.isPublic(field.getModifiers()) || !publicOnly)
          && Type.class.isAssignableFrom(field.getType())) {
        builder.add((Type<?>) field.get(null));
      }
    }
  }

  private BuildTypeTestHelper() {}
}
