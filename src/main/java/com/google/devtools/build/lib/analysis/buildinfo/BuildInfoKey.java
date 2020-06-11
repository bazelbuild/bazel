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
package com.google.devtools.build.lib.analysis.buildinfo;

/**
 * Build-info key for lookup from the {@link
 * com.google.devtools.build.lib.analysis.AnalysisEnvironment}.
 */
public final class BuildInfoKey {
  private final String name;

  public BuildInfoKey(String name) {
    this.name = name;
  }

  @Override
  public String toString() {
    return name;
  }

  @Override
  public boolean equals(Object o) {
    if (!(o instanceof BuildInfoKey)) {
      return false;
    }
    return name.equals(((BuildInfoKey) o).name);
  }

  @Override
  public int hashCode() {
    return name.hashCode();
  }
}
