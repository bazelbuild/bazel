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

package com.google.devtools.lcovmerger;

import java.util.Collection;
import java.util.TreeMap;

class Coverage {
  private final TreeMap<String, SourceFileCoverage> sourceFiles;

  Coverage() {
    sourceFiles = new TreeMap<>();
  }

  void add(SourceFileCoverage input) {
    String sourceFilename = input.sourceFileName();
    if (sourceFiles.containsKey(sourceFilename)) {
      SourceFileCoverage old = sourceFiles.get(sourceFilename);
      sourceFiles.put(sourceFilename, SourceFileCoverage.merge(old, input));
    } else {
      sourceFiles.put(sourceFilename, input);
    }
  }

  Collection<SourceFileCoverage> getAllSourceFiles() {
    return sourceFiles.values();
  }
}
