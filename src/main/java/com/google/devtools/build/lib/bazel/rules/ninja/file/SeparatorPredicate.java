// Copyright 2019 The Bazel Authors. All rights reserved.
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
//

package com.google.devtools.build.lib.bazel.rules.ninja.file;

/** Interface for determining where the byte sequence should be split into parts. */
public interface SeparatorPredicate {

  /**
   * Returns true if the sequence should be split after <code>current</code> byte.
   *
   * @param previous previous byte (before current)
   * @param current current byte
   * @param next next byte (after current)
   */
  boolean test(byte previous, byte current, byte next);
}
