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

/**
 * An Index is a mapping from duplicate-free collection of <code>n</code>
 * elements to integers in the interval <code>(0,n]</code>.  The collection of
 * elements may grow, but not shrink.  An Index is the relational transpose of
 * a (duplicate-free) List.
 *
 * <p>Terminological note: the word "index" can refer to both the mapping (as
 * in the index of a book), and the integer values returned by the mapping (as
 * is typical when discussing arrays).  We will use the word "position" for the
 * latter meaning within the documentation of this class.
 */
public interface Index<T> {

  /**
   * Returns the position of the specified element or -1
   * if the element is not present in the index.
   */
  int indexOf(Object element);

  /**
   *  Gets the element that is mapped to the specified position
   *  or <code>null</code> if no element is mapped to the position.
   */
  T elementOf(int position);

  /**
   *  Returns the number of elements in the index.  This is one greater than
   *  the maximum value that may be returned by <code>getIndex()</code>.
   */
  int size();

  /**
   *  Adds the specified element to the index (iff not already present), and
   *  returns the position to which it is mapped. (optional method)
   */
  int put(T element);

}
