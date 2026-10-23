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

package com.google.devtools.build.lib.packages;

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.util.FileType;

/**
 * Common superclass for InputFile and OutputFile which provides implementation for the file
 * operations in common.
 */
@Immutable
public abstract class FileTarget implements Target, FileType.HasFileType {
  final Label label;

  /** Constructs a file with the given label, which must be in the given package. */
  FileTarget(Packageoid pkg, Label label) {
    Preconditions.checkArgument(
        label.getPackageFragment().equals(pkg.getPackageIdentifier().getPackageFragment()));
    this.label = label;
  }

  public String getFilename() {
    return label.getName();
  }

  @Override
  public Label getLabel() {
    return label;
  }

  @Override
  public String filePathForFileTypeMatcher() {
    return getFilename();
  }

  @Override
  public boolean isFile() {
    return true;
  }

  @Override
  public String toString() {
    return getTargetKind() + "(" + label + ")"; // Just for debugging
  }
}
