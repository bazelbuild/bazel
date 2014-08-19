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
package com.google.devtools.build.lib.pkgcache;

import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.util.concurrent.ThreadPoolExecutor;

import javax.annotation.Nullable;

/**
 * Support for resolving {@code package/...} target patterns.
 */
public interface RecursivePackageProvider extends PackageProvider {

  /**
   * <p>Visits the names of all packages beneath the given directory recursively and concurrently.
   *
   * <p>Note: This operation needs to stat directories recursively. It could be very expensive when
   * there is a big tree under the given directory.
   *
   * <p>Over a single iteration, package names are unique.
   *
   * <p>This method uses the given thread pool to call the observer method, possibly concurrently
   * (depending on the thread pool). When this method terminates, however, all such threads will
   * have completed.
   *
   * <p>To abort the traversal, call {@link Thread#interrupt()} on the calling thread.
   *
   * <p>This method guarantees that all BUILD files it returns correspond to valid package names
   * that are not marked as deleted within the current build.
   *
   * @param eventHandler an eventHandler which should be used to log any errors that occur while
   *    scanning directories for BUILD files
   * @param directory a relative, canonical path specifying the directory to search
   * @param useTopLevelExcludes whether to skip a pre-set list of top level directories
   * @param visitorPool the thread pool to use to visit packages in parallel
   * @param observer is called for each path fragment found; thread-safe if the thread pool supports
   *    multiple parallel threads
   * @throws InterruptedException if the calling thread was interrupted.
   */
  void visitPackageNamesRecursively(EventHandler eventHandler, PathFragment directory,
      boolean useTopLevelExcludes, @Nullable ThreadPoolExecutor visitorPool,
      PathPackageLocator.AcceptsPathFragment observer) throws InterruptedException;
}
