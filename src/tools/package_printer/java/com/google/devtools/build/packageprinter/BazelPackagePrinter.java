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

import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.events.PrintingEventHandler;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.skyframe.packages.BazelPackageLoader;
import com.google.devtools.build.lib.skyframe.packages.PackageLoader;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.JavaIoFileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.Root;
import java.nio.file.Paths;

/**
 * PackagePrinter prints the list of rules in a Bazel package.
 *
 * <p>That is, it prints the names of the rules in a BUILD file after macro expansion.
 */
public class BazelPackagePrinter {
  public static void main(String[] args) throws Exception {
    if (!verifyArgs(args)) {
      System.err.println(
          "Usage example: BazelPackagePrinter "
              + "--workspace_root=. "
              + "--install_base=$(bazel info install_base) "
              + "--output_base=$(bazel info output_base) "
              + "package/to/print");
      System.exit(1);
    }
    FileSystem fileSystem = new JavaIoFileSystem(DigestHashFunction.MD5);
    PackageLoader loader =
        newPackageLoader(
            Root.fromPath(fileSystem.getPath(getAbsPathFlag(args[0], "--workspace_root="))),
            fileSystem.getPath(getAbsPathFlag(args[1], "--install_base=")),
            fileSystem.getPath(getAbsPathFlag(args[2], "--output_base=")));

    Lib.printPackageContents(loader, args[3]);
  }

  /** newPackageLoader returns a new PackageLoader. */
  static PackageLoader newPackageLoader(Root workspaceDir, Path installBase, Path outputBase) {
    return BazelPackageLoader.builder(workspaceDir, installBase, outputBase)
        .useDefaultSkylarkSemantics()
        .setReporter(new Reporter(new EventBus(), PrintingEventHandler.ERRORS_TO_STDERR))
        .setLegacyGlobbingThreads(400)
        .setSkyframeThreads(300)
        .build();
  }

  static boolean verifyArgs(String[] args) {
    if (args.length != 4) {
      return false;
    }
    if (!args[0].startsWith("--workspace_root=")) {
      return false;
    }
    if (!args[1].startsWith("--install_base=")) {
      return false;
    }
    if (!args[2].startsWith("--output_base=")) {
      return false;
    }
    return true;
  }

  static String getAbsPathFlag(String flagName, String arg) {
    return Paths.get(flagName.substring(arg.length())).toAbsolutePath().toString();
  }
}
