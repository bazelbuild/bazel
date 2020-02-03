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
public interface SeparatorFinder {

  /**
   * Returns the index of the end of the next separator (separator can be of two symbols, \r\n), or
   * -1 if the fragment does not contain any separators.
   *
   * @param fragment fragment to search in
   * @param startingFrom index to start search from
   * @param untilExcluded index to stop search before (excluded from search).
   * @throws IncorrectSeparatorException if the incorrect separator value (\r) is used
   */
  int findNextSeparator(ByteBufferFragment fragment, int startingFrom, int untilExcluded)
      throws IncorrectSeparatorException;
}
