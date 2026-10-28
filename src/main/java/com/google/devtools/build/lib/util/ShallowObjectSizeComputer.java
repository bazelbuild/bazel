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

package com.google.devtools.build.lib.util;

import com.sun.management.HotSpotDiagnosticMXBean;
import java.lang.management.ManagementFactory;
import java.lang.reflect.Array;
import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Utility class to calculate the shallow size of an object based on embedded knowledge about the
 * JVM.
 *
 * <p>"Shallow size" means that heap used by that given object, but not the ones it references. If
 * you want to know the "retained" size (i.e. the size of objects a given one transitively
 * references), you need to walk the object graph.
 */
public class ShallowObjectSizeComputer {
  private static final Layout LAYOUT = Layout.getCurrentLayout();

  private ShallowObjectSizeComputer() {}

  private static class Layout {
    private final long objectHeaderBytes;
    private final long objectAlignment;
    private final long referenceBytes;
    private final long superclassPaddingBytes;
    private final long arrayHeaderBytes;

    private Layout(
        long objectHeaderBytes,
        long objectAlignment,
        long referenceBytes,
        long superclassPaddingBytes,
        long arrayHeaderBytes) {
      this.objectHeaderBytes = objectHeaderBytes;
      this.objectAlignment = objectAlignment;
      this.referenceBytes = referenceBytes;
      this.superclassPaddingBytes = superclassPaddingBytes;
      this.arrayHeaderBytes = arrayHeaderBytes;
    }

    static Layout getCurrentLayout() {
      if (!System.getProperty("java.vm.name").startsWith("OpenJDK ")) {
        throw new IllegalStateException("Only OpenJDK is supported");
      }

      if (!System.getProperty("sun.arch.data.model").equals("64")) {
        throw new IllegalStateException("Only 64-bit JVMs are supported");
      }

      HotSpotDiagnosticMXBean diagnosticBean =
          ManagementFactory.getPlatformMXBean(HotSpotDiagnosticMXBean.class);
      boolean compressedOops =
          Boolean.parseBoolean(diagnosticBean.getVMOption("UseCompressedOops").getValue());

      boolean compactObjectHeaders;
      try {
        compactObjectHeaders =
            Boolean.parseBoolean(diagnosticBean.getVMOption("UseCompactObjectHeaders").getValue());
      } catch (IllegalArgumentException e) {
        // UseCompactObjectHeaders is available only in JDK 24 and later.
        compactObjectHeaders = false;
      }

      boolean compressedClassPointers;
      try {
        compressedClassPointers =
            Boolean.parseBoolean(
                diagnosticBean.getVMOption("UseCompressedClassPointers").getValue());
      } catch (IllegalArgumentException e) {
        // UseCompressedClassPointers was obsoleted in JDK 27, where class pointers are always
        // compressed on 64-bit JVMs.
        compressedClassPointers = Runtime.version().feature() >= 27;
      }

      long objectHeaderBytes = compactObjectHeaders ? 8 : compressedClassPointers ? 12 : 16;

      // Since JDK 23 (JDK-8139457), array elements smaller than 8 bytes pack immediately after the
      // 4-byte length field instead of aligning at 8 bytes.
      long arrayHeaderBytes =
          Runtime.version().feature() >= 23
              ? objectHeaderBytes + Integer.BYTES
              : roundUp(objectHeaderBytes + Integer.BYTES, Long.BYTES);

      return new Layout(
          /* objectHeaderBytes= */ objectHeaderBytes,
          /* objectAlignment= */ 8,
          /* referenceBytes= */ compressedOops ? 4 : 8,
          /* superclassPaddingBytes= */ compressedOops ? 4 : 8,
          /* arrayHeaderBytes= */ arrayHeaderBytes);
    }

    @Override
    public String toString() {
      return "Layout{"
          + "objectHeaderBytes="
          + objectHeaderBytes
          + ", objectAlignment="
          + objectAlignment
          + ", referenceBytes="
          + referenceBytes
          + ", superclassPaddingBytes="
          + superclassPaddingBytes
          + ", arrayHeaderBytes="
          + arrayHeaderBytes
          + '}';
    }
  }

  private static final class ClassSizes {
    private final long fieldsBytes;
    private final long objectBytes;

    private ClassSizes(long fieldsBytes, long objectBytes) {
      this.fieldsBytes = fieldsBytes;
      this.objectBytes = objectBytes;
    }
  }

  private static final ConcurrentHashMap<Class<?>, ClassSizes> classSizeCache =
      new ConcurrentHashMap<>();

  /** Returns the size of a field containing the given type. */
  private static long getStorageSize(Class<?> clazz) {
    if (!clazz.isPrimitive()) {
      return LAYOUT.referenceBytes;
    } else if (clazz == boolean.class) {
      return 1;
    } else if (clazz == byte.class) {
      return 1;
    } else if (clazz == char.class) {
      return Character.BYTES;
    } else if (clazz == short.class) {
      return Short.BYTES;
    } else if (clazz == int.class) {
      return Integer.BYTES;
    } else if (clazz == long.class) {
      return Long.BYTES;
    } else if (clazz == float.class) {
      return Float.BYTES;
    } else if (clazz == double.class) {
      return Double.BYTES;
    } else {
      throw new IllegalStateException();
    }
  }

  private static ClassSizes calculateClassSizes(Class<?> clazz) {
    long fieldsBytes = 0;
    for (Field f : clazz.getDeclaredFields()) {
      if (!Modifier.isStatic(f.getModifiers())) {
        fieldsBytes += getStorageSize(f.getType());
      }
    }

    Class<?> superClazz = clazz.getSuperclass();
    if (superClazz != null) {
      ClassSizes superClazzSizes = getClassSizes(superClazz);
      fieldsBytes += roundUp(superClazzSizes.fieldsBytes, LAYOUT.superclassPaddingBytes);
    }

    return new ClassSizes(
        fieldsBytes, roundUp(LAYOUT.objectHeaderBytes + fieldsBytes, LAYOUT.objectAlignment));
  }

  /** Returns the size of an array of a given length containing the given type. */
  public static long getArraySize(long length, Class<?> componentType) {
    long componentSize = getStorageSize(componentType);
    // Since JDK 23 (JDK-8139457), array elements smaller than 8 bytes pack immediately after the
    // array header (which ends in a 4-byte length field); while 8-byte elements (e.g. long, double,
    // uncompressed references) are aligned to an 8-byte boundary.
    // Meanwhile, for JDK < 23, where the first array element is always aligned at 8 bytes
    // regardless of the element type, Layout.getCurrentLayout() pre-rounds up arrayHeaderBytes to 8
    // bytes, and both branches below will produce the same (aligned) baseOffset value.
    long baseOffset =
        componentSize == Long.BYTES
            ? roundUp(LAYOUT.arrayHeaderBytes, Long.BYTES)
            : LAYOUT.arrayHeaderBytes;
    return roundUp(baseOffset + length * componentSize, LAYOUT.objectAlignment);
  }

  private static ClassSizes getClassSizes(Class<?> clazz) {
    // computeIfAbsent() doesn't work because that cannot be called recursively and
    // calculateClassSizes() needs to call getClassSizes(). There is a race condition here, but it
    // is benign, since the result of the computation will always be the same, so the worst thing
    // that can happen is that we calculate the size of a class twice.
    ClassSizes classSizes = classSizeCache.get(clazz);
    if (classSizes == null) {
      classSizes = calculateClassSizes(clazz);
      classSizeCache.putIfAbsent(clazz, classSizes);
    }

    return classSizes;
  }

  /**
   * Returns the shallow size of objects of a given class.
   *
   * <p>Does not include memory used by static fields, memory used in metaspace, etc., only the
   * amount of memory used by instances of the given class.
   */
  public static long getClassShallowSize(Class<?> clazz) {
    return getClassSizes(clazz).objectBytes;
  }

  /** Returns the shallow size of an object. */
  public static long getShallowSize(Object o) {
    Class<?> clazz = o.getClass();
    if (!clazz.isArray()) {
      return getClassShallowSize(clazz);
    } else {
      return getArraySize(Array.getLength(o), clazz.getComponentType());
    }
  }

  private static long roundUp(long x, long to) {
    long ceil = (x + to - 1) / to;
    return to * ceil;
  }
}
