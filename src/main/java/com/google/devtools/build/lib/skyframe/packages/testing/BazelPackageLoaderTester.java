// Copyright 2023 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe.packages.testing;

import com.google.common.base.StandardSystemProperty;
import com.google.common.collect.ImmutableList;
import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.events.PrintingEventHandler;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.packages.semantics.BuildLanguageOptions;
import com.google.devtools.build.lib.skyframe.packages.BazelPackageLoader;
import com.google.devtools.build.lib.skyframe.packages.PackageLoader;
import com.google.devtools.build.lib.unix.UnixFileSystem;
import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.Root;
import net.starlark.java.eval.StarlarkSemantics;

/**
 * Simple main class for end-to-end testing of {@link BazelPackageLoader}. Prints target labels in
 * packages specified as command line arguments.
 *
 * <p>See usage in src/test/shell/bazel/bazel_package_loader_test.sh.
 */
public final class BazelPackageLoaderTester {

  public static void main(String[] args) throws Exception {
    String installBase = args[0];
    try (PackageLoader packageLoader = createPackageLoader(installBase)) {
      for (int i = 1; i < args.length; i++) {
        Package pkg = packageLoader.loadPackage(PackageIdentifier.createInMainRepo(args[i]));
        pkg.getTargets().values().stream().map(Target::getLabel).forEach(System.out::println);
      }
    }
  }

  private static PackageLoader createPackageLoader(String installBase) {
    FileSystem fs = new UnixFileSystem(DigestHashFunction.SHA256, /* hashAttributeName= */ "");
    Root workspaceDir = Root.fromPath(fs.getPath(StandardSystemProperty.USER_DIR.value()));
    Path installBasePath = fs.getPath(installBase);
    return BazelPackageLoader.builder(workspaceDir, installBasePath, installBasePath)
        .setFetchForTesting()
        .setStarlarkSemantics(
            StarlarkSemantics.builder()
                .set(BuildLanguageOptions.INCOMPATIBLE_AUTOLOAD_EXTERNALLY, ImmutableList.of())
                .build())
        .setCommonReporter(new Reporter(new EventBus(), PrintingEventHandler.ERRORS_TO_STDERR))
        .build();
  }

  private BazelPackageLoaderTester() {}
}
