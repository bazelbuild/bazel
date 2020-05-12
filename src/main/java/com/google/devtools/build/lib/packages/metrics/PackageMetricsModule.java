// Copyright 2020 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.packages.metrics;

import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.PackageLoadingListener;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.vfs.FileSystem;
import javax.annotation.Nullable;

/** Provides logging for extreme package-loading events. */
public class PackageMetricsModule extends BlazeModule {

  @Nullable
  @Override
  public PackageLoadingListener getPackageLoadingListener(
      Package.Builder.Helper packageBuilderHelper,
      ConfiguredRuleClassProvider ruleClassProvider,
      FileSystem fs) {
    return ExtremaPackageLoadingListener.getInstance();
  }

  @Override
  public void afterCommand() {
    ExtremaPackageLoadingListener.getInstance().logAndResetExtrema();
  }
}
