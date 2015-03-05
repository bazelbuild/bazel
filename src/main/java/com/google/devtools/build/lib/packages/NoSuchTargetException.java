// Copyright 2014 Google Inc. All rights reserved.
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

import com.google.devtools.build.lib.syntax.Label;

import javax.annotation.Nullable;

/**
 * Exception indicating an attempt to access a target which is not found or does
 * not exist.
 */
public class NoSuchTargetException extends NoSuchThingException {

  @Nullable private final Label label;
  // TODO(bazel-team): rename/refactor this class and NoSuchPackageException since it's confusing
  // that they embed Target/Package instances.
  @Nullable private final Target target;
  private final boolean packageLoadedSuccessfully;

  public NoSuchTargetException(String message) {
    this(null, message);
  }

  public NoSuchTargetException(@Nullable Label label, String message) {
    this((label != null ? "no such target '" + label + "': " : "") + message, label, null, null);
  }

  public NoSuchTargetException(Target targetInError, NoSuchPackageException nspe) {
    this(String.format("Target '%s' contains an error and its package is in error",
        targetInError.getLabel()), targetInError.getLabel(), targetInError, nspe);
  }

  private NoSuchTargetException(String message, @Nullable Label label, @Nullable Target target,
      @Nullable NoSuchPackageException nspe) {
    super(message, nspe);
    this.label = label;
    this.target = target;
    this.packageLoadedSuccessfully = nspe == null;
  }

  @Nullable
  public Label getLabel() {
    return label;
  }

  /**
   * Return the target (in error) if parsing completed enough to construct it. May return null.
   */
  @Nullable
  public Target getTarget() {
    return target;
  }

  public boolean getPackageLoadedSuccessfully() {
    return packageLoadedSuccessfully;
  }
}
