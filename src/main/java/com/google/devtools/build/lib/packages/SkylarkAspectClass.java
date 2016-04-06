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

package com.google.devtools.build.lib.packages;

import com.google.devtools.build.lib.cmdline.Label;

import java.util.Objects;

/**
 * {@link AspectClass} for aspects defined in Skylark.
 */
public abstract class SkylarkAspectClass implements AspectClass {

  public abstract Label getExtensionLabel();

  public abstract String getExportedName();

  @Override
  public final String getName() {
    return getExtensionLabel() + "%" + getExportedName();
  }

  @Override
  public final boolean equals(Object o) {
    if (this == o) {
      return true;
    }

    if (!(o instanceof SkylarkAspectClass)) {
      return false;
    }

    SkylarkAspectClass that = (SkylarkAspectClass) o;

    return getExtensionLabel().equals(that.getExtensionLabel())
        && getExportedName().equals(that.getExportedName());
  }

  @Override
  public final int hashCode() {
    return Objects.hash(getExtensionLabel(), getExportedName());
  }

  @Deprecated
  public abstract AspectDefinition getDefinition();
}
