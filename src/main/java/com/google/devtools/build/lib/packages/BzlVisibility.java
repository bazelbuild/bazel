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

package com.google.devtools.build.lib.packages;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import java.util.List;

/**
 * A visibility level governing the loading of a .bzl module.
 *
 * <p>This is just a container for a list of PackageSpecifications; the scope of visibility is the
 * union of all specifications.
 */
// TODO(brandjon): If we ever support negation then this class would become a bit fancier -- it'd
// have to assemble multiple distinct lists, a la PackageSpecificationProvider and
// PackageSpecification.PackageGroupContents. At the moment we don't anticipate that though.
//
// TODO(brandjon): If we have a large allowlist consumed by many .bzl files, we may end up with many
// copies of the allowlist embedded in different instances of this class. Consider more aggressive
// interning.
public abstract class BzlVisibility {

  private BzlVisibility() {}

  /**
   * Returns whether the given package's BUILD and .bzl files may load the .bzl according to this
   * visibility restriction. This does not include cases where {@code pkg} is the same package as
   * the one containing the .bzl (i.e. this method may return false in that case).
   */
  public abstract boolean allowsPackage(PackageIdentifier pkg);

  /**
   * Returns a (possibly interned) {@code BzlVisibility} that allows access as per the given package
   * specifications.
   */
  public static BzlVisibility of(List<PackageSpecification> specs) {
    if (specs.isEmpty()
        || (specs.size() == 1 && specs.get(0) instanceof PackageSpecification.NoPackages)) {
      return PRIVATE;
    }
    for (PackageSpecification spec : specs) {
      if (spec instanceof PackageSpecification.AllPackages) {
        return PUBLIC;
      }
    }
    return new ListBzlVisibility(specs);
  }

  /** A visibility indicating that everyone may load the .bzl */
  public static final BzlVisibility PUBLIC =
      new BzlVisibility() {
        @Override
        public boolean allowsPackage(PackageIdentifier pkg) {
          return true;
        }
      };

  /**
   * A visibility indicating that only BUILD and .bzl files within the same package (not including
   * subpackages) may load the .bzl.
   */
  public static final BzlVisibility PRIVATE =
      new BzlVisibility() {
        @Override
        public boolean allowsPackage(PackageIdentifier pkg) {
          return false;
        }
      };

  /**
   * A visibility that decides whether a package's BUILD and .bzl files may load the .bzl by
   * checking the package against a set of {@link PackageSpecifications}s.
   */
  private static class ListBzlVisibility extends BzlVisibility {
    private final ImmutableList<PackageSpecification> specs;

    public ListBzlVisibility(List<PackageSpecification> specs) {
      this.specs = ImmutableList.copyOf(specs);
    }

    @Override
    public boolean allowsPackage(PackageIdentifier pkg) {
      for (PackageSpecification spec : specs) {
        if (spec.containsPackage(pkg)) {
          return true;
        }
      }
      return false;
    }
  }
}
