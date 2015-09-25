// Copyright 2014 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;

import java.util.List;
import java.util.Set;

/**
 * A NestedSet that keeps a memoized uniquifier so that it is faster to fill a set.
 *
 * <p>This class does not keep the memoized object itself so that we can take advantage of the
 * memory field alignment (Memory alignment does not put in the same structure the fields of a
 * class and its extensions).
 */
public abstract class MemoizedUniquefierNestedSet<E> extends NestedSet<E> {

  @Override
  public List<E> toList() {
    ImmutableList.Builder<E> builder = new ImmutableList.Builder<>();
    memoizedFill(builder);
    return builder.build();
  }

  @Override
  public Set<E> toSet() {
    ImmutableSet.Builder<E> builder = new ImmutableSet.Builder<>();
    memoizedFill(builder);
    return builder.build();
  }

  /**
   * It does not make sense to have a {@code MemoizedUniquefierNestedSet} if it is empty.
   */
  @Override
  public boolean isEmpty() { return false; }

  abstract Object getMemo();

  abstract void setMemo(Object object);

  /**
   * Fill a collection builder by using a memoized {@code Uniqueifier} for faster uniqueness check.
   */
  final void memoizedFill(ImmutableCollection.Builder<E> builder) {
    Uniqueifier memoed;
    synchronized (this) {
      Object memo = getMemo();
      if (memo == null) {
        RecordingUniqueifier uniqueifier = new RecordingUniqueifier();
        getOrder().<E>expander().expandInto(this, uniqueifier, builder);
        setMemo(uniqueifier.getMemo());
        return;
      } else {
        memoed = RecordingUniqueifier.createReplayUniqueifier(memo);
      }
    }
    getOrder().<E>expander().expandInto(this, memoed, builder);
  }
}
