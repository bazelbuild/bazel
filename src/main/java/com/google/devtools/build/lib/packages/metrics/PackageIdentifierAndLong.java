// Copyright 2020 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.packages.metrics;

import com.google.devtools.build.lib.cmdline.PackageIdentifier;

/** A pair of PackageIdentifier and a corresponding value. */
public class PackageIdentifierAndLong implements Comparable<PackageIdentifierAndLong> {
  private final PackageIdentifier pkgId;
  private final long val;

  private PackageIdentifierAndLong(PackageIdentifier pkgId, long val) {
    this.pkgId = pkgId;
    this.val = val;
  }

  public long getVal() {
    return val;
  }

  public PackageIdentifier getPkgId() {
    return pkgId;
  }

  @Override
  public int compareTo(PackageIdentifierAndLong other) {
    return Long.compare(val, other.val);
  }

  @Override
  public String toString() {
    return String.format("%s (%d)", pkgId, val);
  }
}
