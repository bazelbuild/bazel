// Copyright 2023 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe.serialization.autocodec;

import java.lang.reflect.Field;
import java.lang.reflect.Modifier;

/** Methods used by {@link AutoCodec} at runtime. */
public final class RuntimeHelpers {
  /** Returns the count of fields in {@code type} that are non-static and non-transient. */
  public static int getSerializableFieldCount(Class<?> type) {
    int count = 0;
    for (Class<?> next = type; next != null; next = next.getSuperclass()) {
      for (Field field : next.getDeclaredFields()) {
        if ((field.getModifiers() & (Modifier.STATIC | Modifier.TRANSIENT)) == 0) {
          count++;
        }
      }
    }
    return count;
  }

  private RuntimeHelpers() {}
}
