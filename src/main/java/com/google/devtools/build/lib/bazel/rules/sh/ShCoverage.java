// Copyright 2020 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.bazel.rules.sh;

import com.google.devtools.build.lib.analysis.test.InstrumentedFilesCollector.InstrumentationSpec;
import com.google.devtools.build.lib.util.FileTypeSet;

/** Common logic for coverage for sh_* rules. */
final class ShCoverage {

  private ShCoverage() {}

  public static final InstrumentationSpec INSTRUMENTATION_SPEC =
      new InstrumentationSpec(FileTypeSet.ANY_FILE)
          .withSourceAttributes("srcs")
          .withDependencyAttributes("deps", "data");
}
