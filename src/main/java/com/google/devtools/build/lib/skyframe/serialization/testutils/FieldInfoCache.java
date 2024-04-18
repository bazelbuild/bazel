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
package com.google.devtools.build.lib.skyframe.serialization.testutils;

import static com.google.devtools.build.lib.unsafe.UnsafeProvider.getFieldOffset;
import static com.google.devtools.build.lib.unsafe.UnsafeProvider.unsafe;

import com.google.common.collect.ImmutableList;
import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.concurrent.ConcurrentHashMap;

/** A cache for {@link FieldInfo}. */
final class FieldInfoCache {
  private static final ConcurrentHashMap<Class<?>, ImmutableList<FieldInfo>> fieldInfoCache =
      new ConcurrentHashMap<>();

  static ImmutableList<FieldInfo> getFieldInfo(Class<?> type) {
    return fieldInfoCache.computeIfAbsent(type, FieldInfoCache::getFieldInfoUncached);
  }

  static sealed interface FieldInfo permits PrimitiveInfo, ObjectInfo {}

  private abstract static class AbstractFieldInfo {
    final String name;
    final long offset;

    private AbstractFieldInfo(Field field) {
      this.name = field.getName();
      try {
        this.offset = getFieldOffset(field.getDeclaringClass(), name);
      } catch (ReflectiveOperationException e) {
        throw new IllegalStateException(e);
      }
    }
  }

  static final class PrimitiveInfo extends AbstractFieldInfo implements FieldInfo {
    private final Class<?> type;

    private PrimitiveInfo(Field field) {
      super(field);
      this.type = field.getType();
    }

    void output(Object parent, StringBuilder out) {
      out.append(name).append('=');
      if (type.equals(boolean.class)) {
        out.append(unsafe().getBoolean(parent, offset));
      } else if (type.equals(byte.class)) {
        out.append(unsafe().getByte(parent, offset));
      } else if (type.equals(short.class)) {
        out.append(unsafe().getShort(parent, offset));
      } else if (type.equals(char.class)) {
        out.append(unsafe().getChar(parent, offset));
      } else if (type.equals(int.class)) {
        out.append(unsafe().getInt(parent, offset));
      } else if (type.equals(long.class)) {
        out.append(unsafe().getLong(parent, offset));
      } else if (type.equals(float.class)) {
        out.append(unsafe().getFloat(parent, offset));
      } else if (type.equals(double.class)) {
        out.append(unsafe().getDouble(parent, offset));
      } else {
        throw new UnsupportedOperationException("Unexpected primitive type: " + type);
      }
    }
  }

  static final class ObjectInfo extends AbstractFieldInfo implements FieldInfo {
    private ObjectInfo(Field field) {
      super(field);
    }

    String name() {
      return name;
    }

    Object getFieldValue(Object parent) {
      return unsafe().getObject(parent, offset);
    }
  }

  private static ImmutableList<FieldInfo> getFieldInfoUncached(Class<?> type) {
    var fieldInfo = ImmutableList.<FieldInfo>builder();
    for (Class<?> next = type; next != null; next = next.getSuperclass()) {
      Field[] declaredFields = next.getDeclaredFields();
      var classFields = new ArrayList<Field>(declaredFields.length);
      for (Field field : next.getDeclaredFields()) {
        if ((field.getModifiers() & (Modifier.STATIC | Modifier.TRANSIENT)) != 0) {
          continue; // Skips any static or transient fields.
        }
        classFields.add(field);
      }
      classFields.stream()
          // Sorts by name for determinism. Shadowed fields always have separate entries because
          // they occur at different levels in the inheritance hierarchy.
          //
          // Reverses the order here, then reverses it again below. This makes superclass fields
          // appear before subclass fields.
          .sorted(Comparator.comparing(Field::getName).reversed())
          .map(
              field -> {
                Class<?> fieldType = field.getType();
                if (fieldType.isPrimitive()) {
                  return new PrimitiveInfo(field);
                }
                return new ObjectInfo(field);
              })
          .forEach(fieldInfo::add);
    }
    return fieldInfo.build().reverse();
  }

  private FieldInfoCache() {}
}
