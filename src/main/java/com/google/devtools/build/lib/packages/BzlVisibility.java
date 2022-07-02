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

/** A visibility level governing the loading of a .bzl module. */
public abstract class BzlVisibility {

  private BzlVisibility() {}

  /**
   * Returns whether the given package's BUILD and .bzl files may load the .bzl according to this
   * visibility restriction. This does not include cases where {@code pkg} is the same package as
   * the one containing the .bzl (i.e. this method may return false in that case).
   */
  public abstract boolean allowsPackage(PackageIdentifier pkg);

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
   * A visibility that enumerates the packages whose BUILD and .bzl files may load the .bzl.
   * Subpackages are not implicitly included.
   */
  public static class PackageListBzlVisibility extends BzlVisibility {
    private final ImmutableList<PackageIdentifier> packages;

    public PackageListBzlVisibility(List<PackageIdentifier> packages) {
      this.packages = ImmutableList.copyOf(packages);
    }

    @Override
    public boolean allowsPackage(PackageIdentifier pkg) {
      return packages.contains(pkg);
    }
  }
}
