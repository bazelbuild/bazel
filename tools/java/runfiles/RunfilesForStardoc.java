// Copyright 2022 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.runfiles;

/** Additional runfiles functions only meant to be used by Stardoc. */
public final class RunfilesForStardoc {

  /**
   * Returns the canonical repository name.
   *
   * @param runfiles the {@link Runfiles} instance whose repo mapping should be used
   * @param apparentRepositoryName the apparent repository name to resolve to a canonical one
   */
  public static String getCanonicalRepositoryName(
      Runfiles runfiles, String apparentRepositoryName) {
    return runfiles.getCanonicalRepositoryName(apparentRepositoryName);
  }

  private RunfilesForStardoc() {}
}
