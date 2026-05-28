// Copyright 2026 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.collect.nestedset;

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.common.base.Equivalence;
import com.google.common.math.IntMath;
import java.util.concurrent.atomic.AtomicReferenceArray;
import javax.annotation.Nullable;

/** Interner for {@link NestedSet} instances. */
public class NestedSetInterner {
  private final int size;
  private final int mask;
  private final Equivalence<NestedSet<?>> equivalence;
  private volatile AtomicReferenceArray<NestedSet<?>> array;

  private NestedSetInterner(int size, Equivalence<NestedSet<?>> equivalence) {
    int powerOfTwoSize = IntMath.ceilingPowerOfTwo(size);
    this.size = powerOfTwoSize;
    this.mask = powerOfTwoSize - 1;
    this.equivalence = equivalence;
    this.array = new AtomicReferenceArray<>(powerOfTwoSize);
  }

  private NestedSet<?> internImpl(NestedSet<?> sample) {
    checkNotNull(sample);
    int index = equivalence.hash(sample) & mask;
    AtomicReferenceArray<NestedSet<?>> currentArray = array;
    NestedSet<?> cached = currentArray.get(index);
    if (equivalence.equivalent(sample, cached)) {
      return cached;
    }
    currentArray.set(index, sample);
    return sample;
  }

  private void clearImpl() {
    this.array = new AtomicReferenceArray<>(size);
  }

  // Use a duo of strong interners to dedupe some NestedSet instances at low cost. For most builds
  // this saves 2-4% memory, and for some builds this saves CPU and wall time too (due to fewer
  // duplicate expansions, due to getting sharing via #cached),
  //
  // We have a separate interner for NestedSet<Artifact> for two reasons:
  //   * Artifacts get mutated via setGeneratingActionKey during ConfiguredTargetFunction. This is
  //     problematic for equality-based global interning of NestedSet<Artifact>. If a
  //     NestedSet<Artifact> instance from before a CTF restart for the node is used on the
  //     subsequent CTF computation for the node, then we're going to have an inconsistency due to
  //     setGeneratingActionKey being called on the new Artifact instance, but the original
  //     NestedSet<Artifact> instance actually getting used. There are various ways to work around
  //     this, but for the sake of simplicity and low cost, we went with the approach of
  //     identity-based interning for NestedSet<Artifact>, which means we want a separate interner.
  //   * There are more unique NestedSet<Artifact> instances than for other classes and also total
  //     waste from duplicates is higher, so it's worth spending more shallow heap in order to
  //     dedupe more of them.
  private static final NestedSetInterner artifactInterner =
      new NestedSetInterner(
          /* size= */ 2048 * 1024,
          new Equivalence<NestedSet<?>>() {
            @Override
            @SuppressWarnings("ReferenceEquality") // Cares about identity of children elements.
            protected boolean doEquivalent(NestedSet<?> nestedSetA, NestedSet<?> nestedSetB) {
              if (nestedSetA.getOrder() != nestedSetB.getOrder()) {
                return false;
              }
              if (nestedSetA.isSingleton() && nestedSetB.isSingleton()) {
                return nestedSetA.children == nestedSetB.children;
              }
              if (nestedSetA.isSingleton() || nestedSetB.isSingleton()) {
                return false;
              }
              if (nestedSetA.children instanceof Object[] arrayA
                  && nestedSetB.children instanceof Object[] arrayB) {
                if (arrayA.length != arrayB.length) {
                  return false;
                }
                for (int i = 0; i < arrayA.length; i++) {
                  if (arrayA[i] != arrayB[i]) {
                    return false;
                  }
                }
                return true;
              }
              return false;
            }

            @Override
            protected int doHash(NestedSet<?> nestedSet) {
              int result = nestedSet.getOrder().hashCode();
              if (nestedSet.isSingleton()) {
                return 31 * result + System.identityHashCode(nestedSet.children);
              }
              if (nestedSet.children instanceof Object[] array) {
                for (Object element : array) {
                  result = 31 * result + System.identityHashCode(element);
                }
                return result;
              }
              throw new IllegalStateException(
                  "Unexpected children type: " + nestedSet.children.getClass());
            }
          });
  private static final NestedSetInterner otherInterner =
      new NestedSetInterner(
          /* size= */ 1024 * 1024,
          new Equivalence<NestedSet<?>>() {
            @Override
            protected boolean doEquivalent(NestedSet<?> nestedSetA, NestedSet<?> nestedSetB) {
              return nestedSetA.shallowEquals(nestedSetB);
            }

            @Override
            protected int doHash(NestedSet<?> nestedSet) {
              return nestedSet.shallowHashCode();
            }
          });

  public static <T> NestedSet<T> intern(NestedSet<T> nestedSet) {
    NestedSetInterner interner = getInterner(nestedSet);
    if (interner == null) {
      return nestedSet;
    }
    @SuppressWarnings("unchecked") // Practically, a hit will always have the same element type.
    NestedSet<T> interned = (NestedSet<T>) interner.internImpl(nestedSet);
    return interned;
  }

  public static void clear() {
    artifactInterner.clearImpl();
    otherInterner.clearImpl();
  }

  @Nullable
  private static NestedSetInterner getInterner(NestedSet<?> nestedSet) {
    Object representative = nestedSet.children;
    // Pick the interner, if any, to use based on a representative element of the nested set.
    if (representative instanceof Object[] array) {
      // This loop doesn't show up in profiles. If it ever does, we can consider another approach to
      // picking the interner to use.
      while (array.length > 0 && array[0] instanceof Object[]) {
        array = (Object[]) array[0];
      }
      if (array.length == 0) {
        // Should never happen in practice via calls from NestedSetBuilder.build().
        return null;
      }
      representative = array[0];
    }

    if (representative instanceof IsArtifactForNestedSet) {
      return artifactInterner;
    }
    if (representative instanceof String
        || representative instanceof NestedSetsShouldBeInternedByEquality) {
      return otherInterner;
    }
    return null;
  }
}
