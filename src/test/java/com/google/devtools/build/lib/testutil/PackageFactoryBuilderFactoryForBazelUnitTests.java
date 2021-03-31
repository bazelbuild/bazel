// Copyright 2016 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.testutil;

import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.packages.BuilderFactoryForTesting;
import com.google.devtools.build.lib.packages.Package.Builder.DefaultPackageSettings;
import com.google.devtools.build.lib.packages.PackageFactory;
import com.google.devtools.build.lib.packages.PackageLoadingListener;
import com.google.devtools.build.lib.packages.RuleClassProvider;
import com.google.devtools.build.lib.skyframe.packages.PackageFactoryBuilderWithSkyframeForTesting;
import com.google.devtools.build.lib.vfs.FileSystem;

/**
 * A {@link BuilderFactoryForTesting} implementation that injects a {@link
 * BazelPackageLoadingListenerForTesting}.
 */
class PackageFactoryBuilderFactoryForBazelUnitTests implements BuilderFactoryForTesting {
  static final PackageFactoryBuilderFactoryForBazelUnitTests INSTANCE =
      new PackageFactoryBuilderFactoryForBazelUnitTests();

  private PackageFactoryBuilderFactoryForBazelUnitTests() {
  }

  @Override
  public PackageFactoryBuilderWithSkyframeForTesting builder(BlazeDirectories directories) {
    return new PackageFactoryBuilderForBazelUnitTests(directories);
  }

  private static class PackageFactoryBuilderForBazelUnitTests
      extends PackageFactoryBuilderWithSkyframeForTesting {

    private final BlazeDirectories directories;

    public PackageFactoryBuilderForBazelUnitTests(BlazeDirectories directories) {
      this.directories = directories;
    }

    @Override
    public PackageFactory build(RuleClassProvider ruleClassProvider, FileSystem fs) {
      return new PackageFactory(
          ruleClassProvider,
          PackageFactory.makeDefaultSizedForkJoinPoolForGlobbing(),
          environmentExtensions,
          version,
          DefaultPackageSettings.INSTANCE,
          packageValidator,
          packageOverheadEstimator,
          doChecksForTesting
              ? new BazelPackageLoadingListenerForTesting(
                  (ConfiguredRuleClassProvider) ruleClassProvider, directories)
              : PackageLoadingListener.NOOP_LISTENER);
    }
  }
}
