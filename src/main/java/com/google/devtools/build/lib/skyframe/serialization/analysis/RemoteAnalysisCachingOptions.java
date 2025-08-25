// Copyright 2024 The Bazel Authors. All rights reserved.
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

import com.google.devtools.common.options.Converters.DurationConverter;
import com.google.devtools.common.options.EnumConverter;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionsBase;
import java.time.Duration;

/** Options for caching analysis results remotely. */
public class RemoteAnalysisCachingOptions extends OptionsBase {

  @Option(
      name = "serialized_frontier_profile",
      defaultValue = "",
      documentationCategory = OptionDocumentationCategory.OUTPUT_SELECTION,
      effectTags = {OptionEffectTag.BAZEL_MONITORING},
      help = "Dump a profile of serialized frontier bytes. Specifies the output path.")
  public String serializedFrontierProfile;

  @Option(
      name = "experimental_remote_analysis_cache_mode",
      defaultValue = "off",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.BAZEL_INTERNAL_CONFIGURATION},
      converter = RemoteAnalysisCacheModeConverter.class,
      help = "The transport direction for the remote analysis cache.")
  public RemoteAnalysisCacheMode mode;

  /** * The transport direction for the remote analysis cache. */
  public enum RemoteAnalysisCacheMode {
    /** Serializes and uploads Skyframe analysis nodes after the build command finishes. */
    UPLOAD,

    /**
     * Dumps the manifest of SkyKeys computed in the frontier and the active set. This mode does not
     * serialize and upload the keys.
     */
    DUMP_UPLOAD_MANIFEST_ONLY,

    /** Fetches and deserializes the Skyframe analysis nodes during the build. */
    DOWNLOAD,

    /** Disabled. */
    OFF;

    /** Returns true if the selected mode needs to connect to a backend. */
    public boolean requiresBackendConnectivity() {
      return switch (this) {
        case UPLOAD, DOWNLOAD -> true;
        case DUMP_UPLOAD_MANIFEST_ONLY, OFF -> false;
      };
    }

    /**
     * Returns true if the mode serializes <i>values</i>.
     *
     * <p>{@link DOWNLOAD} serializes keys, but not values.
     */
    public boolean serializesValues() {
      return switch (this) {
        case UPLOAD, DUMP_UPLOAD_MANIFEST_ONLY -> true;
        case DOWNLOAD, OFF -> false;
      };
    }
  }

  /** Enum converter for {@link RemoteAnalysisCacheMode}. */
  private static class RemoteAnalysisCacheModeConverter
      extends EnumConverter<RemoteAnalysisCacheMode> {
    public RemoteAnalysisCacheModeConverter() {
      super(RemoteAnalysisCacheMode.class, "Frontier serialization mode");
    }
  }

  @Option(
      name = "experimental_remote_analysis_cache",
      defaultValue = "",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.BAZEL_INTERNAL_CONFIGURATION},
      help = "The URL for the remote analysis caching backend.")
  public String remoteAnalysisCache;

  @Option(
      name = "experimental_remote_analysis_cache_max_batch_size",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.BAZEL_INTERNAL_CONFIGURATION},
      defaultValue = "4095",
      help = "Batch size limit for remote analysis caching RPCs.")
  public int maxBatchSize;

  @Option(
      name = "experimental_remote_analysis_cache_concurrency",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.BAZEL_INTERNAL_CONFIGURATION},
      defaultValue = "4",
      help = "Target concurrency for remote analysis caching RPCs.")
  public int concurrency;

  @Option(
      name = "experimental_remote_analysis_cache_deadline",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.BAZEL_INTERNAL_CONFIGURATION},
      defaultValue = "120s",
      converter = DurationConverter.class,
      help = "Deadline to use for remote analysis cache operations.")
  public Duration deadline;

  @Option(
      name = "experimental_analysis_cache_service",
      defaultValue = "",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.BAZEL_INTERNAL_CONFIGURATION},
      help = "Locator for the AnalysisCacheService instance.")
  public String analysisCacheService;

  // Configuration Modes:
  // 1. Write Proxy: If --experimental_remote_analysis_write_proxy is set, all uploads go through
  //    the write proxy. --experimental_remote_analysis_cache_mode must be UPLOAD.
  //    --experimental_analysis_cache_service and --experimental_remote_analysis_cache are ignored.
  //
  // 2. Read Proxy: If --experimental_analysis_cache_service is set but
  //    --experimental_remote_analysis_cache is NOT set, downloads are proxied through the
  //    AnalysisCacheService. --experimental_remote_analysis_cache_mode must be DOWNLOAD.
  //
  // 3. Legacy Direct: Otherwise, connections are made directly to storage (specified by
  //    --experimental_remote_analysis_cache) and AnalysisCacheService (specified by
  //    --experimental_analysis_cache_service, if provided).

  @Option(
      name = "experimental_remote_analysis_write_proxy",
      defaultValue = "",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.BAZEL_INTERNAL_CONFIGURATION},
      help =
          "The address of the SkycacheStorageWriteProxyService. If set, this service will be used "
              + "for uploading analysis cache data.")
  public String remoteAnalysisWriteProxy;

  @Option(
      name = "experimental_analysis_cache_key_distinguisher_for_testing",
      defaultValue = "null",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.BAZEL_INTERNAL_CONFIGURATION},
      help = "An opaque string used as part of the cache key. Should only be used for testing.")
  public String analysisCacheKeyDistinguisherForTesting;

  @Option(
      name = "experimental_analysis_cache_enable_metadata_queries",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.BAZEL_INTERNAL_CONFIGURATION},
      help =
          "A flag to switch on/off inserting and querying the metadata db (b/425247333). The idea"
              + " is for this flag to only exist temporarily for a careful rollout of the feature"
              + " then be deleted later. For writers it requires passing an analysis cache service"
              + " address.")
  public boolean analysisCacheEnableMetadataQueries;
}
