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

package com.google.devtools.coverageoutputgenerator;

import com.google.auto.value.AutoValue;
import javax.annotation.Nullable;

/** Stores line execution coverage information. */
@AutoValue
abstract class LineCoverage {
  static LineCoverage create(int lineNumber, int executionCount, String checksum) {
    return new AutoValue_LineCoverage(lineNumber, executionCount, checksum);
  }

  static LineCoverage merge(LineCoverage first, LineCoverage second) {
    assert first.lineNumber() == second.lineNumber();
    assert (first.checksum() == null && second.checksum() == null)
        || first.checksum().equals(second.checksum());
    return create(
        first.lineNumber(), first.executionCount() + second.executionCount(), first.checksum());
  }

  abstract int lineNumber();

  abstract int executionCount();
  // The current geninfo implementation uses an MD5 hash as checksumming algorithm.
  @Nullable
  abstract String checksum(); // optional
}
