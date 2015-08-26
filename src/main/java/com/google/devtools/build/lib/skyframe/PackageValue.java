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
package com.google.devtools.build.lib.skyframe;

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.PackageIdentifier;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

/**
 * A Skyframe value representing a package.
 */
@Immutable
@ThreadSafe
public class PackageValue implements SkyValue {

  private final Package pkg;

  public PackageValue(Package pkg) {
    this.pkg = Preconditions.checkNotNull(pkg);
  }

  public Package getPackage() {
    return pkg;
  }

  @Override
  public String toString() {
    return "<PackageValue name=" + pkg.getName() + ">";
  }

  @ThreadSafe
  public static SkyKey key(PathFragment pkgName) {
    return key(PackageIdentifier.createInDefaultRepo(pkgName));
  }

  public static SkyKey key(PackageIdentifier pkgIdentifier) {
    return new SkyKey(SkyFunctions.PACKAGE, pkgIdentifier);
  }

  /**
   * Returns a SkyKey to find the WORKSPACE file at the given path.
   */
  public static SkyKey workspaceKey(RootedPath workspacePath) {
    return new SkyKey(SkyFunctions.WORKSPACE_FILE, workspacePath);
  }
}
