// Copyright 2018 The Bazel Authors. All rights reserved.
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

/** Partial implementation of RecursivePackageProvider to provide some common methods. */
public abstract class AbstractRecursivePackageProvider implements RecursivePackageProvider {

  protected AbstractRecursivePackageProvider() {
  }

  @Override
  public Target getTarget(ExtendedEventHandler eventHandler, Label label)
      throws NoSuchPackageException, NoSuchTargetException, InterruptedException {
    return getPackage(eventHandler, label.getPackageIdentifier()).getTarget(label.getName());
  }

  /**
   * Indicates that a missing dependency is needed before target parsing can proceed. Currently
   * used only in skyframe to notify the framework of missing dependencies. Caught by the compute
   * method in {@link com.google.devtools.build.lib.skyframe.TargetPatternFunction}, which then
   * returns null in accordance with the skyframe missing dependency policy.
   */
  public static class MissingDepException extends RuntimeException {
  }
}
