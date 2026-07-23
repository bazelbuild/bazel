// Copyright 2026 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe.serialization.analysis;

import com.google.common.annotations.VisibleForTesting;
import com.google.devtools.build.lib.skybridge.ScOnly;
import com.google.devtools.common.options.Converters.DurationConverter;
import com.google.devtools.common.options.Converters.RangeConverter;
import com.google.devtools.common.options.EnumConverter;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsClass;
import java.time.Duration;

/**
 * Options for caching analysis results remotely (Service Component).
 *
 * <p>If a flag is needed in both the LC and the SC, a judgement call is needed.
 *
 * <p>A flag should be in the LC if it fundamentally dictates the LC's behavior even before {@link
 * RemoteAnalysisCachingServicesSupplier#configure} is called, and supplying it to the SC e.g. via
 * #configure() seems like a reasonably stable interface that we can commit to. An example is
 * --experimental_remote_analysis_cache_mode.
 *
 * <p>A flag should be in the SC otherwise, as it offers simpler backwards compatibility.
 */
@OptionsClass
@ScOnly
public abstract class RemoteAnalysisCachingServicesOptions extends OptionsBase {

  /** A converter for integers that must be at least 1. */
  public static final class PositiveIntegerConverter extends RangeConverter {
    public PositiveIntegerConverter() {
      super(1, Integer.MAX_VALUE);
    }
  }

  @Option(
      name = "experimental_remote_analysis_cache_max_batch_size",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.BAZEL_INTERNAL_CONFIGURATION},
      defaultValue = "4095",
      converter = PositiveIntegerConverter.class,
      help = "Batch size limit for remote analysis caching RPCs.")
  public abstract int getMaxBatchSize();

  @Option(
      name = "experimental_remote_analysis_cache_concurrency",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.BAZEL_INTERNAL_CONFIGURATION},
      defaultValue = "4",
      converter = PositiveIntegerConverter.class,
      help = "Target concurrency for remote analysis caching RPCs.")
  public abstract int getConcurrency();

  @Option(
      name = "experimental_remote_analysis_cache_max_write_concurrency",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.BAZEL_INTERNAL_CONFIGURATION},
      defaultValue = "16",
      converter = PositiveIntegerConverter.class,
      help = "Max write concurrency for remote analysis caching RPCs.")
  public abstract int getMaxWriteConcurrency();

  @Option(
      name = "experimental_remote_analysis_cache_target_write_concurrency",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.BAZEL_INTERNAL_CONFIGURATION},
      defaultValue = "1",
      converter = PositiveIntegerConverter.class,
      help = "Target write concurrency for remote analysis caching RPCs.")
  public abstract int getTargetWriteConcurrency();

  @Option(
      name = "experimental_remote_analysis_cache_deadline",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.BAZEL_INTERNAL_CONFIGURATION},
      defaultValue = "45s",
      converter = DurationConverter.class,
      help = "Deadline to use for remote analysis cache operations.")
  public abstract Duration getDeadline();

  public abstract void setDeadline(Duration value);

  @Option(
      name = "experimental_analysis_cache_service",
      defaultValue = "",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.BAZEL_INTERNAL_CONFIGURATION},
      help = "Locator for the AnalysisCacheService instance.")
  public abstract String getAnalysisCacheService();

  public abstract void setAnalysisCacheService(String value);

  @Option(
      name = "experimental_remote_analysis_cache_storage",
      defaultValue = "RAM",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.BAZEL_INTERNAL_CONFIGURATION},
      converter = RemoteAnalysisCacheStorageTypeConverter.class,
      help = "The storage type for the remote analysis cache.")
  public abstract RemoteAnalysisCacheStorageType getStorageType();

  /** Enum converter for {@link RemoteAnalysisCacheStorageType}. */
  public static class RemoteAnalysisCacheStorageTypeConverter
      extends EnumConverter<RemoteAnalysisCacheStorageType> {
    public RemoteAnalysisCacheStorageTypeConverter() {
      super(RemoteAnalysisCacheStorageType.class, "Remote analysis cache storage type");
    }
  }

  // Configuration Modes:
  // 1. Write Proxy: If --experimental_remote_analysis_write_proxy is set, all uploads go through
  //    the write proxy. --experimental_remote_analysis_cache_mode must be UPLOAD.
  //    --experimental_analysis_cache_service is ignored.
  //
  // 2. Read Proxy: If --experimental_analysis_cache_service is set, downloads are proxied through
  // the
  //    AnalysisCacheService. --experimental_remote_analysis_cache_mode must be DOWNLOAD.

  @Option(
      name = "experimental_remote_analysis_write_proxy",
      defaultValue = "",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.BAZEL_INTERNAL_CONFIGURATION},
      help =
          "The address of the SkycacheStorageWriteProxyService. If set, this service will be used "
              + "for uploading analysis cache data.")
  public abstract String getRemoteAnalysisWriteProxy();

  public abstract void setRemoteAnalysisWriteProxy(String value);

  @Option(
      name = "experimental_analysis_cache_enable_metadata_queries",
      defaultValue = "true",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.BAZEL_INTERNAL_CONFIGURATION},
      help = "A flag to switch on/off inserting and querying the metadata db (b/425247333).")
  public abstract boolean getAnalysisCacheEnableMetadataQueries();

  public abstract void setAnalysisCacheEnableMetadataQueries(boolean value);

  @Option(
      name = "remote_analysis_debug_entries",
      defaultValue = "",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.BAZEL_INTERNAL_CONFIGURATION},
      help = "Path to a local file containing remote analysis cache entries for debugging.")
  public abstract String getRemoteAnalysisDebugEntries();

  @VisibleForTesting
  public abstract void setRemoteAnalysisDebugEntries(String value);
}
