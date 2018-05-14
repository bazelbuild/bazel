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

package com.google.devtools.build.packageprinter;

import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.skyframe.packages.PackageLoader;
import java.util.logging.Logger;

/**
 * Lib prints the list of rules in a Bazel package.
 *
 * <p>That is, it prints the names of the rules in a BUILD file after macro expansion.
 */
public class Lib {
  private static final Logger logger = Logger.getLogger(Lib.class.getName());

  public static void printPackageContents(PackageLoader loader, String packageName)
      throws NoSuchPackageException, InterruptedException {
    logger.info("Start of 'load'");

    Package pkg = loader.loadPackage(PackageIdentifier.createInMainRepo(packageName));

    for (Target target : pkg.getTargets().values()) {
      if (target instanceof Rule) {
        System.out.println(((Rule) target).getLabel());
      }
    }

    logger.info("End of 'load'");
  }
}
