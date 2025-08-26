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
package com.google.devtools.build.lib.query2.compat;

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.License;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.PackagePiece;
import com.google.devtools.build.lib.packages.Packageoid;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.RuleVisibility;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.packages.TargetData;
import java.util.Objects;
import net.starlark.java.syntax.Location;

/**
 * A fake Target - Use only so that "blaze query" can report Load files as Targets.
 */
public class FakeLoadTarget implements Target {

  private final Label label;
  private final Packageoid pkg;

  public FakeLoadTarget(Label label, Package pkg) {
    this.label = Preconditions.checkNotNull(label);
    this.pkg = Preconditions.checkNotNull(pkg);
  }

  // Fake load targets should be in the same package piece as the package's BUILD file.
  public FakeLoadTarget(Label label, PackagePiece.ForBuildFile pkgPiece) {
    this.label = Preconditions.checkNotNull(label);
    this.pkg = Preconditions.checkNotNull(pkgPiece);
  }

  @Override
  public Label getLabel() {
    return label;
  }

  @Override
  public Packageoid getPackageoid() {
    return pkg;
  }

  @Override
  public Package.Metadata getPackageMetadata() {
    return pkg.getMetadata();
  }

  @Override
  public Package.Declarations getPackageDeclarations() {
    return pkg.getDeclarations();
  }

  @Override
  public String getTargetKind() {
    return targetKind();
  }

  @Override
  public Rule getAssociatedRule() {
    return null;
  }

  @Override
  public License getLicense() {
    throw new UnsupportedOperationException();
  }

  @Override
  public Location getLocation() {
    return getPackageMetadata().getBuildFileLocation();
  }

  @Override
  public RuleVisibility getRawVisibility() {
    return RuleVisibility.PUBLIC;
  }

  @Override
  public boolean isConfigurable() {
    return true;
  }

  @Override
  public String toString() {
    return label.toString();
  }

  @Override
  public int hashCode() {
    return Objects.hash(label, pkg);
  }

  @Override
  public boolean equals(Object obj) {
    if (!(obj instanceof FakeLoadTarget other)) {
      return false;
    }
    return label.equals(other.label) && pkg.equals(other.pkg);
  }

  /** Returns the target kind for all fake sub-include targets. */
  public static String targetKind() {
    return "source file";
  }

  @Override
  public TargetData reduceForSerialization() {
    throw new UnsupportedOperationException();
  }
}
