// Copyright 2014 Google Inc. All rights reserved.
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

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;

/**
 * An implementation of the Index interface based on a HashMap.
 */
public class HashIndex<T> implements Index<T> {

  private final ArrayList<T> elements = new ArrayList<>();

  private final HashMap<T, Integer> indexMapping = new HashMap<>();

  /**
   *  Constructs an empty Index.
   */
  public HashIndex() {
  }

  /**
   *  Constructs an Index for the specified elements.
   */
  @SuppressWarnings("unchecked")
  public HashIndex(T... elements) {
    for (T element : elements) {
      put(element);
    }
  }

  /**
   *  Constructs an Index for the specified collection.
   */
  public HashIndex(Collection<T> elements) {
    for (T element : elements) {
      put(element);
    }
  }

  @Override
  public int indexOf(Object element) {
    Integer position = indexMapping.get(element);
    return (position != null) ? position : -1;
  }

  @Override
  public int size() {
    return elements.size();
  }

  @Override
  public int put(T element) {
    int index = indexOf(element);
    // if the element is not a duplicate
    // add it to our list of elements;
    if (index == -1) {
      index = size();
      elements.add(element);
      indexMapping.put(element, index);
    }
    return index;
  }

  @Override
  public T elementOf(int position) {
    return (position >= 0 && position < size()) ? elements.get(position) : null;
  }

  @Override
  public int hashCode() {
    return elements.hashCode();
  }

  @Override
  public boolean equals(Object o) {
    if (o == this) {
      return true;
    }
    if (o instanceof HashIndex<?>) {
      HashIndex<?> that = (HashIndex<?>) o;
      return this.elements.equals(that.elements);
    }
    return false;
  }

  @Override
  public String toString() {
    return elements.toString();
  }

}
