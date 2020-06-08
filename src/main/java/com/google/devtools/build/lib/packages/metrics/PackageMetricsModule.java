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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.PackageLoadingListener;
import com.google.devtools.build.lib.packages.metrics.ExtremaPackageLoadingListener.PackageIdentifierAndLong;
import com.google.devtools.build.lib.packages.metrics.ExtremaPackageLoadingListener.TopPackages;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionsBase;
import java.util.List;
import javax.annotation.Nullable;

/** Provides logging for extreme package-loading events. */
public class PackageMetricsModule extends BlazeModule {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  /** Options for {@link PackageMetricsModule}. */
  public static class Options extends OptionsBase {
    @Option(
        name = "log_top_n_packages",
        defaultValue = "10",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.BAZEL_MONITORING},
        help = "Configures number of packages included in top-package INFO logging, <= 0 disables.")
    public int numberOfPackagesToTrack;
  }

  private final ExtremaPackageLoadingListener packageLoadingListener;

  public PackageMetricsModule() {
    this(ExtremaPackageLoadingListener.getInstance());
  }

  @VisibleForTesting
  PackageMetricsModule(ExtremaPackageLoadingListener packageLoadingListener) {
    this.packageLoadingListener = packageLoadingListener;
  }

  @Nullable
  @Override
  public PackageLoadingListener getPackageLoadingListener(
      Package.Builder.Helper packageBuilderHelper,
      ConfiguredRuleClassProvider ruleClassProvider,
      FileSystem fs) {
    return packageLoadingListener;
  }

  @Override
  public Iterable<Class<? extends OptionsBase>> getCommonCommandOptions() {
    return ImmutableList.of(Options.class);
  }

  @Override
  public void beforeCommand(CommandEnvironment commandEnvironment) {
    Options options = commandEnvironment.getOptions().getOptions(Options.class);
    packageLoadingListener.setNumPackagesToTrack(Math.max(options.numberOfPackagesToTrack, 0));
  }

  @Override
  public void afterCommand() {
    TopPackages topPackages = packageLoadingListener.getAndResetTopPackages();
    logIfNonEmpty("Slowest packages (ms)", topPackages.getSlowestPackages());
    logIfNonEmpty("Largest packages (num targets)", topPackages.getLargestPackages());
    logIfNonEmpty(
        "Packages with most computation steps", topPackages.getPackagesWithMostComputationSteps());
    logIfNonEmpty(
        "Packages with most transitive loads (num bzl files)",
        topPackages.getPackagesWithMostTransitiveLoads());
  }

  private static void logIfNonEmpty(
      String logLinePrefix, List<PackageIdentifierAndLong> extremeElements) {
    if (!extremeElements.isEmpty()) {
      logger.atInfo().log("%s: %s", logLinePrefix, Joiner.on(", ").join(extremeElements));
    }
  }
}
