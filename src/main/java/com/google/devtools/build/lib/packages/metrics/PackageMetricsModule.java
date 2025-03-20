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
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.packages.Package.Builder.PackageSettings;
import com.google.devtools.build.lib.packages.PackageLoadingListener;
import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionsBase;
import javax.annotation.Nullable;

/** Provides logging for extreme package-loading events. */
public class PackageMetricsModule extends BlazeModule {
  /** Options for {@link PackageMetricsModule}. */
  public static class Options extends OptionsBase {
    @Option(
        name = "log_top_n_packages",
        defaultValue = "10",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.BAZEL_MONITORING},
        help = "Configures number of packages included in top-package INFO logging, <= 0 disables.")
    public int numberOfPackagesToTrack;

    @Option(
        name = "record_metrics_for_all_packages",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.BAZEL_MONITORING},
        help =
            "Configures PackageMetrics to record all metrics for all packages. Disables Top-n INFO"
                + " logging.")
    public boolean enableAllMetrics;

    @Option(
        name = "experimental_publish_package_metrics_in_bep",
        defaultValue = "false",
        documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
        effectTags = {OptionEffectTag.BAZEL_MONITORING},
        help = "Whether to publish package metrics in the BEP.")
    public boolean publishPackageMetricsInBep;
  }

  private final PackageMetricsPackageLoadingListener packageLoadingListener;

  public PackageMetricsModule() {
    this(PackageMetricsPackageLoadingListener.getInstance());
  }

  @VisibleForTesting
  PackageMetricsModule(PackageMetricsPackageLoadingListener packageLoadingListener) {
    this.packageLoadingListener = packageLoadingListener;
  }

  @Nullable
  @Override
  public PackageLoadingListener getPackageLoadingListener(
      PackageSettings packageSettings,
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
    PackageMetricsRecorder recorder =
        options.enableAllMetrics
            ? new CompletePackageMetricsRecorder()
            : new ExtremaPackageMetricsRecorder(Math.max(options.numberOfPackagesToTrack, 0));
    packageLoadingListener.setPackageMetricsRecorder(recorder);
    packageLoadingListener.setPublishPackageMetricsInBep(options.publishPackageMetricsInBep);
  }

  @Override
  public void afterCommand() {
    if (packageLoadingListener.getPackageMetricsRecorder() != null) {
      packageLoadingListener.getPackageMetricsRecorder().loadingFinished();
    }
  }
}
