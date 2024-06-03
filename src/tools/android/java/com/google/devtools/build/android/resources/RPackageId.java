// Copyright 2024 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.android.resources;

/**
 * Data for building and using packageId from RPackage class.
 *
 * <p>If RPackage class used during R.class generation then all ids will be generated as:
 *
 * <p>value = oldValue - packageId + GETSTATIC(rPackageClassName, "packageId").
 *
 * <p>Used by privacy sandbox for runtime resource remapping.
 */
public final class RPackageId {

  private static final int DEFAULT_PACKAGE_ID = 0x7f000000;

  private final String rPackageClassName;

  public static RPackageId createFor(String appPackageName) {
    final String rPackageClassName =
        appPackageName.isEmpty() ? "RPackage" : (appPackageName + ".RPackage");
    return new RPackageId(rPackageClassName);
  }

  private RPackageId(String rPackageClassName) {
    this.rPackageClassName = rPackageClassName;
  }

  public boolean owns(int resourceId) {
    return (resourceId & 0xff000000) == getPackageId();
  }

  public String getRPackageClassName() {
    return rPackageClassName;
  }

  public int getPackageId() {
    return DEFAULT_PACKAGE_ID;
  }
}
