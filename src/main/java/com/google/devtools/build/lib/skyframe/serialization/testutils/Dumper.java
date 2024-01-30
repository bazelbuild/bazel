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
package com.google.devtools.build.lib.skyframe.serialization.testutils;

import static com.google.common.collect.Iterables.isEmpty;
import static com.google.devtools.build.lib.unsafe.UnsafeProvider.unsafe;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import java.lang.reflect.Array;
import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.IdentityHashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * A utility for creating high fidelity string dumps of arbitrary objects.
 *
 * <p>Uses reflection to perform depth-first traversal of arbitrary objects and formats them as an
 * indented, multiline string.
 *
 * <p>This class exists mainly to help test and debug serialization. Consequently, it skips {@code
 * transient} fields. It also performs reference-based memoization to handle cyclic structures or
 * structures that would have an exponential path structure, for example, {@code NestedSets}.
 */
public final class Dumper {
  /**
   * Stores an identifier for every object traversed.
   *
   * <p>When an object is encountered again, it is represented with just its type and previous
   * identifier instead of being fully expanded.
   */
  private final IdentityHashMap<Object, Integer> referenceIds = new IdentityHashMap<>();

  /** The current indentation level. */
  private int indent = 0;

  private final StringBuilder out = new StringBuilder();

  private Dumper() {}

  /**
   * Formats an arbitrary object into a string.
   *
   * <p>The format is verbose and suitable for tests and debugging.
   *
   * @return a multiline String representation of {@code obj} without a trailing newline.
   */
  public static String dumpStructure(Object obj) {
    var deep = new Dumper();
    deep.outputObject(obj);
    return deep.out.toString();
  }

  /** Formats an arbitrary object into {@link #out}. */
  private void outputObject(Object obj) {
    if (obj == null) {
      out.append("null");
      return;
    }

    var type = obj.getClass();
    if (shouldInline(type)) {
      out.append(obj);
      return;
    }

    Integer id = referenceIds.get(obj);
    if (id != null) {
      // This instance has been observed previously. Outputs only a backreference.
      outputIdentifier(type, id);
      return;
    }
    referenceIds.put(obj, id = referenceIds.size());

    // All non-inlined, non-backreference objects are represented like
    // "<type name>(<id>) [<contents>]".
    //
    // <contents> depends on the type, but is generally a sequence of recursively formatted
    // objects. For arrays and iterables, this is the sequence of elements, for maps,
    // it is an alternating sequence of keys and values and for any other type of object, it is
    // a sequence of its fields, like "<field name>=<object>".
    outputIdentifier(type, id);
    out.append(" [");
    indent++;

    boolean addedLine; // True if the <content> includes a newline.
    if (type.isArray()) {
      addedLine = outputArrayElements(obj);
    } else if (obj instanceof Map) {
      addedLine = outputMapEntries((Map<?, ?>) obj);
    } else if (obj instanceof Iterable) {
      addedLine = outputIterableElements((Iterable<?>) obj);
    } else {
      addedLine = outputObjectFields(obj);
    }
    indent--;

    if (addedLine) {
      // The <content> sequence typically emits a newline per-sequence element, like
      // \n<indent><e1>\n<indent><e2>\n<indent><e3>, which would look like the following.
      //
      //   <type name>(id) [
      //     <e1>
      //     <e2>
      //     <e3>▊
      //
      // The code below emits a newline and indents to the parent's indentation level before
      // emitting the closing bracket.
      //
      //   <type name>(id) [
      //     <e1>
      //     <e2>
      //     <e3>
      //   ]▊
      //
      // When the <content> sequence is empty, or one of the special cased arrays that do not
      // emit newlines, no trailing newline is needed before the closing bracket, as in the
      // following examples.
      //
      //   <type name>(id) []▊
      // or
      //   byte[](1234) [DEADBEEF]▊
      //
      // Note that the output always leaves the cursor at the end of the last written line. The
      // caller should add a trailing newline if needed.
      addNewlineAndIndent();
    }
    out.append(']');
  }

  /** Emits an object identifier like {@code "<type name>(<id>)"}. */
  private void outputIdentifier(Class<?> type, int id) {
    String name = type.getCanonicalName();
    if (name == null) {
      // According to the documentation for `Class.getCanonicalName`, not all classes have one.
      // Falls back on the name in such cases. (It's unclear if this code is reachable because
      // synthetic types are inlined).
      name = type.getName();
    }
    out.append(name).append('(').append(id).append(')');
  }

  private boolean outputArrayElements(Object arr) {
    var componentType = arr.getClass().getComponentType();
    if (componentType.equals(byte.class)) {
      // It's a byte array. Outputs as hex.
      for (byte b : (byte[]) arr) {
        out.append(String.format("%02X", b));
      }
      return false;
    }

    if (shouldInline(componentType)) {
      // It's a type that should be inlined. Outputs elements delimited by commas.
      boolean isFirst = true;
      for (int i = 0; i < Array.getLength(arr); i++) {
        if (isFirst) {
          isFirst = false;
        } else {
          out.append(", ");
        }
        out.append(Array.get(arr, i));
      }
      return false;
    }

    for (int i = 0; i < Array.getLength(arr); i++) {
      addNewlineAndIndent();
      outputObject(Array.get(arr, i));
    }
    return Array.getLength(arr) > 0;
  }

  private boolean outputMapEntries(Map<?, ?> map) {
    for (Map.Entry<?, ?> entry : map.entrySet()) {
      addNewlineAndIndent();
      out.append("key=");
      outputObject(entry.getKey());

      addNewlineAndIndent();
      out.append("value=");
      outputObject(entry.getValue());
    }
    return !map.isEmpty();
  }

  private boolean outputIterableElements(Iterable<?> iterable) {
    for (var next : iterable) {
      addNewlineAndIndent();
      outputObject(next);
    }
    return !isEmpty(iterable);
  }

  private boolean outputObjectFields(Object obj) {
    ImmutableList<FieldInfo> fieldInfo = getFieldInfo(obj.getClass());
    for (FieldInfo info : fieldInfo) {
      addNewlineAndIndent();
      outputField(obj, info);
    }
    return !fieldInfo.isEmpty();
  }

  private void outputField(Object parent, FieldInfo info) {
    out.append(info.name).append('=');

    Class<?> type = info.type;
    if (!type.isPrimitive()) {
      outputObject(unsafe().getObject(parent, info.offset));
      return;
    }

    if (type.equals(boolean.class)) {
      out.append(unsafe().getBoolean(parent, info.offset));
    } else if (type.equals(byte.class)) {
      out.append(unsafe().getByte(parent, info.offset));
    } else if (type.equals(short.class)) {
      out.append(unsafe().getShort(parent, info.offset));
    } else if (type.equals(char.class)) {
      out.append(unsafe().getChar(parent, info.offset));
    } else if (type.equals(int.class)) {
      out.append(unsafe().getInt(parent, info.offset));
    } else if (type.equals(long.class)) {
      out.append(unsafe().getLong(parent, info.offset));
    } else if (type.equals(float.class)) {
      out.append(unsafe().getFloat(parent, info.offset));
    } else if (type.equals(double.class)) {
      out.append(unsafe().getDouble(parent, info.offset));
    } else {
      throw new UnsupportedOperationException("Unexpected primitive type: " + type);
    }
  }

  private void addNewlineAndIndent() {
    out.append('\n');
    for (int i = 0; i < indent; i++) {
      out.append("  "); // Indentation is 2 spaces.
    }
  }

  private static final ConcurrentHashMap<Class<?>, ImmutableList<FieldInfo>> fieldInfoCache =
      new ConcurrentHashMap<>();

  private static ImmutableList<FieldInfo> getFieldInfo(Class<?> type) {
    return fieldInfoCache.computeIfAbsent(type, Dumper::getFieldInfoUncached);
  }

  private static class FieldInfo {
    private final String name;
    private final Class<?> type;
    private final long offset;

    private FieldInfo(Field field) {
      this.name = field.getName();
      this.type = field.getType();
      this.offset = unsafe().objectFieldOffset(field);
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
          .map(FieldInfo::new)
          .forEach(fieldInfo::add);
    }
    return fieldInfo.build().reverse();
  }

  private static boolean shouldInline(Class<?> type) {
    return type.isPrimitive() || DIRECT_INLINE_TYPES.contains(type) || type.isSynthetic();
  }

  private static final ImmutableSet<Class<?>> WRAPPER_TYPES =
      ImmutableSet.of(
          Byte.class,
          Short.class,
          Integer.class,
          Long.class,
          Float.class,
          Double.class,
          Boolean.class,
          Character.class);

  private static final ImmutableSet<Class<?>> DIRECT_INLINE_TYPES =
      ImmutableSet.<Class<?>>builder()
          .addAll(WRAPPER_TYPES)
          // Treats Strings as values for readability of the output. It might be good to make this
          // configurable later on.
          .add(String.class)
          // The string representation of a Class is sufficient to identify it.
          .add(Class.class)
          .build();
}
