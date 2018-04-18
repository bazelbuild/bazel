// Copyright 2018 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.collect;

import com.google.common.collect.AbstractIterator;
import java.util.HashSet;
import java.util.Iterator;

/**
 * An iterable implementation that removes duplicate elements (as determined by equals), using a
 * hash set.
 */
public final class DedupingIterable<T> implements Iterable<T> {
  private final Iterable<T> iterable;

  DedupingIterable(Iterable<T> iterable) {
    CollectionUtils.checkImmutable(iterable);
    this.iterable = iterable;
  }

  public static <T> Iterable<T> of(Iterable<T> iterable) {
    return new DedupingIterable<>(iterable);
  }

  @Override
  public Iterator<T> iterator() {
    return new DedupingIterator<>(iterable.iterator());
  }

  private static final class DedupingIterator<T> extends AbstractIterator<T> {
    private final HashSet<T> set = new HashSet<>();
    private final Iterator<T> it;

    DedupingIterator(Iterator<T> it) {
      this.it = it;
    }

    @Override
    protected T computeNext() {
      while (it.hasNext()) {
        T next = it.next();
        if (set.add(next)) {
          return next;
        }
      }
      return endOfData();
    }
  }
}
