// Copyright 2025 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.skyframe;

import com.google.devtools.build.lib.packages.Packageoid;
import com.google.devtools.build.skyframe.NotComparableSkyValue;

/** A Skyframe value representing either a package or a package piece. */
public interface PackageoidValue extends NotComparableSkyValue {
  /**
   * Returns the package or package piece. This packageoid may contain errors, in which case the
   * caller should throw an appropriate subclass of {@link
   * com.google.devtools.build.lib.packages.NoSuchPackageException} or {@link
   * com.google.devtools.build.lib.packages.NoSuchPackagePieceException} if an error-free packageoid
   * is needed.
   */
  Packageoid getPackageoid();
}
