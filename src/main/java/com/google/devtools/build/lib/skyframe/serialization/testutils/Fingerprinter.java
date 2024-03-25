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

import static com.google.common.base.Preconditions.checkState;
import static com.google.common.hash.Hashing.murmur3_128;
import static com.google.devtools.build.lib.skyframe.serialization.testutils.Dumper.getTypeName;
import static com.google.devtools.build.lib.skyframe.serialization.testutils.Dumper.shouldInline;
import static com.google.devtools.build.lib.skyframe.serialization.testutils.FieldInfoCache.getFieldInfo;
import static java.lang.Math.min;

import com.google.common.annotations.VisibleForTesting;
import com.google.devtools.build.lib.skyframe.serialization.testutils.FieldInfoCache.FieldInfo;
import com.google.devtools.build.lib.skyframe.serialization.testutils.FieldInfoCache.ObjectInfo;
import com.google.devtools.build.lib.skyframe.serialization.testutils.FieldInfoCache.PrimitiveInfo;
import java.lang.reflect.Array;
import java.util.IdentityHashMap;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * Fingerprints arbitrary objects for serialization testing.
 *
 * <p>Like serialization, it skips {@code transient} fields.
 */
final class Fingerprinter {
  private final IdentityHashMap<Object, String> fingerprints = new IdentityHashMap<>();

  /** Stack for detecting cyclic backreferences in depth-first traversal. */
  private final IdentityHashMap<Object, Integer> stack = new IdentityHashMap<>();

  private Fingerprinter() {}

  /**
   * Traverses {@code obj} and computes fingerprints for objects within.
   *
   * <p>Objects with cycles (or cyclic complexes) are handled in the following way. When a cycle is
   * encountered, logically, the first node encountered in the cycle becomes its owner. By
   * definition, every element of the cycle is reachable by a DFS traversal from the owner. The
   * cycle is fingerprinted with a DFS traversal using <em>relative</em> backreferences. Since these
   * backreferences only make sense locally, the fingerprints of the individual cycle elements are
   * not published. Only the owner's fingerprint is published.
   *
   * <p>This implies that if the same cycle is encountered, but the traversal starts at a different
   * node, it will receive a different fingerprint. A consequence of this is that deduplication of
   * cycles by fingerprint will be limited to only cycles that match on their owning nodes.
   *
   * <p>An alternative might be to fingerprint every cyclic complex starting from every node of the
   * complex and add those entries to the fingerprint map. That would make fingerprinting quadratic
   * in the size of the cyclic complex instead of the current, linear cost. As of 03/19/2024, there
   * is no evidence of that extra cost being necessary.
   */
  static IdentityHashMap<Object, String> computeFingerprints(Object obj) {
    StringBuilder fingerprintOut = new StringBuilder();
    Fingerprinter fingerprinter = new Fingerprinter();
    int unused = fingerprinter.outputFingerprintOrInlinedValue(obj, fingerprintOut);
    checkState(
        fingerprinter.stack.isEmpty(),
        "The stack should be empty after fingerprinting %s, but it was not. This means an"
            + " implementation bug.",
        obj);
    fingerprinter.fingerprints.put(obj, fingerprintOut.toString());
    return fingerprinter.fingerprints;
  }

  /**
   * Recursively computes the fingerprint of {@code obj} and outputs it to {@code fingerprintOut} or
   * when {@code obj} is an inlined type, outputs its value directly.
   *
   * @return the least backreferenced stack element index ({@code Integer.MAX_VALUE} if there were
   *     no backreferences). This corresponds to the highest cycle`s owner, if any.
   */
  private int outputFingerprintOrInlinedValue(@Nullable Object obj, StringBuilder fingerprintOut) {
    if (obj == null) {
      fingerprintOut.append("null");
      return Integer.MAX_VALUE;
    }

    { // Checks for an already computed fingerprint.
      String previousFingerprint = fingerprints.get(obj);
      if (previousFingerprint != null) {
        fingerprintOut.append(previousFingerprint);
        return Integer.MAX_VALUE;
      }
    }

    Class<?> type = obj.getClass();
    if (shouldInline(type)) {
      // Emits the type, even for inline values. This avoids a possible ambiguities. For example,
      // "-1" could be a backreference, String, Integer, or other things if there were no type
      // prefix.
      fingerprintOut.append(getTypeName(type)).append(':').append(obj);
      return Integer.MAX_VALUE;
    }

    {
      Integer cyclicBackReference = stack.get(obj);
      if (cyclicBackReference != null) {
        int currentIndex = stack.size() - 1;
        // Converts cyclic backreferences into a negative number, indicating how many levels up the
        // stack need to be traversed to reach the backreference.
        fingerprintOut.append(cyclicBackReference - currentIndex);
        return cyclicBackReference;
      }
    }

    StringBuilder textOut = new StringBuilder();
    int cycleOwnerIndex = outputObject(obj, type, textOut);

    String fingerprint = fingerprintString(textOut.toString());
    fingerprintOut.append(fingerprint);

    if (cycleOwnerIndex >= stack.size()) {
      // `obj` contains no backreferences to elements still on the stack so the fingerprint is
      // globally valid and can be published.
      fingerprints.put(obj, fingerprint);
    }
    return cycleOwnerIndex;
  }

  /**
   * Outputs a string representation of {@code obj} to {@code out}.
   *
   * <p>The string representation uses fingerprints anywhere a recursion occurs. The resulting
   * fingerprint should be unique for <em>equivalent</em> values.
   *
   * @return the least backreferenced stack element index
   */
  private int outputObject(Object obj, Class<?> type, StringBuilder out) {
    out.append(getTypeName(type)).append(": ["); // Emits a header.
    stack.put(obj, stack.size());
    try {
      if (type.isArray()) {
        return outputArrayElements(obj, out);
      }
      if (obj instanceof Map) {
        return outputMapEntries((Map<?, ?>) obj, out);
      }
      if (obj instanceof Iterable) {
        return outputIterableElements((Iterable<?>) obj, out);
      }
      return outputObjectFields(obj, out);
    } finally {
      stack.remove(obj);
      out.append(']');
    }
  }

  // The following output methods have the same return value contract as `outputObject`.

  private int outputArrayElements(Object arr, StringBuilder out) {
    var componentType = arr.getClass().getComponentType();
    if (componentType.equals(byte.class)) {
      // Byte arrays output as hex.
      for (byte b : (byte[]) arr) {
        out.append(String.format("%02X", b));
      }
      return Integer.MAX_VALUE;
    }

    if (shouldInline(componentType)) {
      // The component is an inlined type. Outputs elements delimited by commas.
      boolean isFirst = true;
      for (int i = 0; i < Array.getLength(arr); i++) {
        if (isFirst) {
          isFirst = false;
        } else {
          out.append(", ");
        }
        Object elt = Array.get(arr, i);
        if (elt != null) {
          out.append(getTypeName(elt.getClass())).append(':');
        }
        out.append(elt);
      }
      return Integer.MAX_VALUE;
    }

    // The component is some arbitrary type. Outputs the element fingerprints.
    int cycleOwnerIndex = Integer.MAX_VALUE;
    boolean isFirst = true;
    for (int i = 0; i < Array.getLength(arr); i++) {
      if (isFirst) {
        isFirst = false;
      } else {
        out.append(", ");
      }
      Object elt = Array.get(arr, i);
      cycleOwnerIndex = min(cycleOwnerIndex, outputFingerprintOrInlinedValue(elt, out));
    }
    return cycleOwnerIndex;
  }

  private int outputMapEntries(Map<?, ?> map, StringBuilder out) {
    int cycleOwnerIndex = Integer.MAX_VALUE;
    boolean isFirst = true;
    for (Map.Entry<?, ?> entry : map.entrySet()) {
      if (isFirst) {
        isFirst = false;
      } else {
        out.append(", ");
      }
      out.append("key=");
      Object key = entry.getKey();
      cycleOwnerIndex = min(cycleOwnerIndex, outputFingerprintOrInlinedValue(key, out));

      out.append(", value=");
      Object value = entry.getValue();
      cycleOwnerIndex = min(cycleOwnerIndex, outputFingerprintOrInlinedValue(value, out));
    }
    return cycleOwnerIndex;
  }

  private int outputIterableElements(Iterable<?> iterable, StringBuilder out) {
    int cycleOwnerIndex = Integer.MAX_VALUE;
    boolean isFirst = true;
    for (Object elt : iterable) {
      if (isFirst) {
        isFirst = false;
      } else {
        out.append(", ");
      }
      cycleOwnerIndex = min(cycleOwnerIndex, outputFingerprintOrInlinedValue(elt, out));
    }
    return cycleOwnerIndex;
  }

  private int outputObjectFields(Object obj, StringBuilder out) {
    int cycleOwnerIndex = Integer.MAX_VALUE;
    boolean isFirst = true;
    for (FieldInfo info : getFieldInfo(obj.getClass())) {
      if (isFirst) {
        isFirst = false;
      } else {
        out.append(", ");
      }
      if (info instanceof PrimitiveInfo primitiveInfo) {
        primitiveInfo.output(obj, out);
      } else if (info instanceof ObjectInfo objectInfo) {
        out.append(objectInfo.name()).append('=');
        cycleOwnerIndex =
            min(
                cycleOwnerIndex,
                outputFingerprintOrInlinedValue(objectInfo.getFieldValue(obj), out));
      } else {
        // TODO: b/297857068 - it should be possible to replace this with a pattern matching
        // switch which won't require this line, but that's not yet supported.
        throw new IllegalArgumentException("Unexpected FieldInfo type: " + info);
      }
    }
    return cycleOwnerIndex;
  }

  @VisibleForTesting // private
  static String fingerprintString(String text) {
    return murmur3_128().hashUnencodedChars(text).toString().intern();
  }
}
