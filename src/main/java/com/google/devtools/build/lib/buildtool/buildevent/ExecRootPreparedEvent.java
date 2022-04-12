// Copyright 2019 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.buildtool.buildevent;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.vfs.Root;
import java.util.Optional;
import javax.annotation.Nullable;

/**
 * Event fired after the execution root directory has been prepared for the execution phase of a
 * build. The event contains a package roots map for the symlink forest (i.e. the symlinks to source
 * directory roots) which was prepared in the execution root directory. The package roots map may be
 * empty (if {@link SkyframeExecutor#getForcedSingleSourceRootIfNoExecrootSymlinkCreation} is
 * non-null), indicating that a source symlink forest was not needed.
 */
public class ExecRootPreparedEvent {
  @Nullable private final ImmutableMap<PackageIdentifier, Root> packageRootsMap;
  public static final ExecRootPreparedEvent NO_PACKAGE_ROOTS_MAP = new ExecRootPreparedEvent();

  public ExecRootPreparedEvent(Optional<ImmutableMap<PackageIdentifier, Root>> packageRootsMap) {
    this.packageRootsMap = packageRootsMap.orElse(null);
  }

  /** Constructs an event indicating that the source symlink forest should not be planted. */
  private ExecRootPreparedEvent() {
    this.packageRootsMap = null;
  }

  public Optional<ImmutableMap<PackageIdentifier, Root>> getPackageRootsMap() {
    return Optional.ofNullable(packageRootsMap);
  }
}
