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
package com.google.devtools.build.lib.skyframe;

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.syntax.Environment.Extension;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

/**
 * A value that represents a Skylark import lookup result. The lookup value corresponds to
 * exactly one Skylark file, identified by an absolute {@link Label} {@link SkyKey} argument. The
 * Label should not reference the special {@code external} package.
 */
public class SkylarkImportLookupValue implements SkyValue {

  private final Extension environmentExtension;
  /**
   * The immediate Skylark file dependency descriptor class corresponding to this value.
   * Using this reference it's possible to reach the transitive closure of Skylark files
   * on which this Skylark file depends.
   */
  private final SkylarkFileDependency dependency;

  public SkylarkImportLookupValue(
      Extension environmentExtension, SkylarkFileDependency dependency) {
    this.environmentExtension = Preconditions.checkNotNull(environmentExtension);
    this.dependency = Preconditions.checkNotNull(dependency);
  }

  /**
   * Returns the Extension
   */
  public Extension getEnvironmentExtension() {
    return environmentExtension;
  }

  /**
   * Returns the immediate Skylark file dependency corresponding to this import lookup value.
   */
  public SkylarkFileDependency getDependency() {
    return dependency;
  }

  /**
   * Returns a SkyKey to look up {@link Label} {@code importLabel}, which must be an absolute
   * label.
   */
  static SkyKey key(Label importLabel) {
    return new SkyKey(SkyFunctions.SKYLARK_IMPORTS_LOOKUP, importLabel);  
  }
}
