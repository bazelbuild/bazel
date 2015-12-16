// Copyright 2015 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skylarkinterface;

/**
 * A Skylark value that is represented by {@code print()} differently than by {@code write()}.
 */
public interface SkylarkPrintableValue extends SkylarkValue {
  /**
   * Print an informal, human-readable representation of the value.
   *
   * @param buffer the buffer to append the representation to
   * @param quotationMark The quote style (" or ') to be used
   */
  void print(Appendable buffer, char quotationMark);
}
