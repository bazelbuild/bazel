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

package com.google.devtools.build.lib.pkgcache;

import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.Target;

/**
 * API for retrieving targets.
 *
 * <p><b>Concurrency</b>: Implementations should be thread safe.
 */
public interface TargetProvider {
  /**
   * Returns the Target identified by "label", loading, parsing and evaluating the package if it is
   * not already loaded.
   *
   * @throws NoSuchPackageException if the package could not be found
   * @throws NoSuchTargetException if the package was loaded successfully, but the specified {@link
   *     Target} was not found in it
   * @throws InterruptedException if the package loading was interrupted
   */
  Target getTarget(ExtendedEventHandler eventHandler, Label label)
      throws NoSuchPackageException, NoSuchTargetException, InterruptedException;
}
