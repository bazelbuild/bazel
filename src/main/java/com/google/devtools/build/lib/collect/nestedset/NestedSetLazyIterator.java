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

import java.util.Iterator;

/**
 * A NestedSet iterator that only expands the NestedSet when the first element is requested. This
 * allows code that calls unconditionally to {@code hasNext} to check if the iterator is empty
 * to not expand the nested set.
 */
final class NestedSetLazyIterator<E> implements Iterator<E> {

  private NestedSet<E> nestedSet;
  private Iterator<E> delegate = null;

  NestedSetLazyIterator(NestedSet<E> nestedSet) {
    this.nestedSet = nestedSet;
  }

  @Override
  public boolean hasNext() {
    if (delegate == null) {
      return !nestedSet.isEmpty();
    }
    return delegate.hasNext();
  }

  @Override
  public E next() {
    if (delegate == null) {
      delegate = nestedSet.toCollection().iterator();
      nestedSet = null;
    }
    return delegate.next();
  }

  @Override
  public void remove() {
    throw new UnsupportedOperationException();
  }
}
