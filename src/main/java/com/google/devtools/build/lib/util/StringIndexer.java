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
package com.google.devtools.build.lib.util;

/**
 * An object that provides bidirectional String <-> unique integer mapping.
 */
public interface StringIndexer {

  /**
   * Removes all mappings.
   */
  public void clear();

  /**
   * @return some measure of the size of the index.
   */
  public int size();

  /**
   * Creates new mapping for the given string if necessary and returns
   * string index. Also, as a side effect, zero or more additional mappings
   * may be created for various prefixes of the given string.
   *
   * @return a unique index.
   */
  public int getOrCreateIndex(String s);

  /**
   * @return a unique index for the given string or -1 if string
   *         was not added.
   */
  public int getIndex(String s);

  /**
   * Creates mapping for the given string if necessary.
   * Also, as a side effect, zero or more additional mappings may be
   * created for various prefixes of the given string.
   *
   * @return true if new mapping was created, false if mapping already existed.
   */
  public boolean addString(String s);

  /**
   * @return string associated with the given index or null if
   *         mapping does not exist.
   */
  public String getStringForIndex(int i);

}
