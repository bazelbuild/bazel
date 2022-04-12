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
import com.google.devtools.build.lib.packages.License.DistributionType;
import com.google.devtools.build.lib.util.FileType;
import java.util.Set;

/**
 * Common superclass for InputFile and OutputFile which provides implementation for the file
 * operations in common.
 */
@Immutable
public abstract class FileTarget implements Target, FileType.HasFileType {
  final Label label;

  /** Constructs a file with the given label, which must be in the given package. */
  FileTarget(Package pkg, Label label) {
    Preconditions.checkArgument(label.getPackageFragment().equals(pkg.getNameFragment()));
    this.label = label;
  }

  @Override
  public boolean isImmutable() {
    return true; // immutable and Starlark-hashable
  }

  public String getFilename() {
    return label.getName();
  }

  @Override
  public Label getLabel() {
    return label;
  }

  @Override
  public String getName() {
    return label.getName();
  }

  @Override
  public String filePathForFileTypeMatcher() {
    return getFilename();
  }

  @Override
  public String toString() {
    return getTargetKind() + "(" + label + ")"; // Just for debugging
  }

  @Override
  public Set<DistributionType> getDistributions() {
    return getPackage().getDefaultDistribs();
  }

  /**
   * {@inheritDoc}
   *
   * <p>File licenses are strange, and require some special handling. When
   * you ask "What license covers this file?" in a query, the answer should
   * be the license declared for the enclosing package. On the other hand,
   * if the file is a source for a rule target, and the rule's license declares
   * more exceptions than the default inherited by the file, the rule's
   * more liberal target should override the stricter license of the file. In
   * other words, the license of the rule always overrides the license of
   * the non-rule file targets that are inputs to that rule.
   */
  @Override
  public License getLicense() {
    return getPackage().getDefaultLicense();
  }
}
