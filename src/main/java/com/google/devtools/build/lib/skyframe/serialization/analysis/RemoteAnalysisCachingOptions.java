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

import com.google.common.base.Strings;
import com.google.common.hash.HashCode;
import com.google.common.hash.Hashing;
import com.google.devtools.common.options.Converter;
import com.google.devtools.common.options.Converters.DurationConverter;
import com.google.devtools.common.options.EnumConverter;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParsingException;
import java.time.Duration;
import javax.annotation.Nullable;

/** Options for caching analysis results remotely. */
public class RemoteAnalysisCachingOptions extends OptionsBase {

  /** A converter for MD5 checksums. */
  public static final class Md5Converter implements Converter<HashCode> {
    @Override
    public HashCode convert(String input, @Nullable Object conversionContext)
        throws OptionsParsingException {
      if (Strings.isNullOrEmpty(input)) {
        return null;
      }

      HashCode result = null;
      try {
        result = HashCode.fromString(input);
      } catch (IllegalArgumentException e) {
        // Handled just below in the if (result == null) branch
      }

      if (result == null || result.bits() != Hashing.md5().bits()) {
        throw new OptionsParsingException("Blaze checksum must be exactly 32 hex characters");
      }

      return result;
    }

    @Override
    public String getTypeDescription() {
      return "";
    }
  }

  @Option(
      name = "serialized_frontier_profile",
      defaultValue = "",
      documentationCategory = OptionDocumentationCategory.OUTPUT_SELECTION,
      effectTags = {OptionEffectTag.BAZEL_MONITORING},
      help = "Dump a profile of serialized frontier bytes. Specifies the output path.")
  public String serializedFrontierProfile;

  @Option(
      name = "remote_analysis_json_log",
      defaultValue = "null",
      documentationCategory = OptionDocumentationCategory.LOGGING,
      effectTags = {OptionEffectTag.BAZEL_MONITORING},
      help =
          "If set, a JSON file is written to this location that contains a detailed log of "
              + "the behavior of remote analysis caching. It's interpreted as a path relative "
              + "to the current working directory.")
  public String jsonLog;

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

    public boolean isRetrievalEnabled() {
      return this == DOWNLOAD;
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

  // TODO: b/443947033 - add a way to disable retries
  @Option(
      name = "experimental_remote_analysis_unreachable_cache_retry_interval",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.BAZEL_INTERNAL_CONFIGURATION},
      defaultValue = "250ms",
      converter = DurationConverter.class,
      help =
          "How long to wait before retrying a cache get request that failed due to an UNREACHABLE"
              + " channel. This is a workaround for the client library reporting 'ready' "
              + "prematurely.")
  public Duration unreachableCacheRetryInterval;

  @Option(
      name = "experimental_analysis_cache_service",
      defaultValue = "",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.BAZEL_INTERNAL_CONFIGURATION},
      help = "Locator for the AnalysisCacheService instance.")
  public String analysisCacheService;

  @Option(
      name = "experimental_remote_analysis_cache_storage",
      defaultValue = "RAM",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.BAZEL_INTERNAL_CONFIGURATION},
      converter = RemoteAnalysisCacheStorageTypeConverter.class,
      help = "The storage type for the remote analysis cache.")
  public RemoteAnalysisCacheStorageType storageType;

  /** The storage type for the remote analysis cache. */
  public enum RemoteAnalysisCacheStorageType {
    /** Write to RAM. */
    RAM,

    /** Write to HDD. */
    HDD,

    /** Write to both RAM and HDD. */
    BOTH
  }

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
      defaultValue = "true",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.BAZEL_INTERNAL_CONFIGURATION},
      help = "A flag to switch on/off inserting and querying the metadata db (b/425247333).")
  public boolean analysisCacheEnableMetadataQueries;

  @Option(
      name = "experimental_analysis_cache_server_checksum_override",
      converter = Md5Converter.class,
      defaultValue = "null",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.BAZEL_INTERNAL_CONFIGURATION},
      help =
          "If set, Blaze will use this checksum to look up entries in the remote analysis cache"
              + " and not its own. WARNING: this might result in incorrect behavior. Only for"
              + " debugging. It's best if the difference between the writer and the reader is only"
              + " additional logging. In particular, the data structures that are being serialized "
              + " and the observable behavior of the serialization machinery must not change.")
  public HashCode serverChecksumOverride;

  @Option(
      name = "experimental_skycache_minimize_memory",
      defaultValue = "false",
      oldName = "experimental_discard_package_values_post_analysis",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.BAZEL_INTERNAL_CONFIGURATION},
      help =
          "DO NOT USE: This flag is currently in development and does not work with every target."
              + " If enabled, Blaze will discard values after the analysis phase is"
              + " complete to provide Skycache writers with more headroom.")
  public boolean skycacheMinimizeMemory;

  @Option(
      name = "experimental_analysis_cache_bail_on_missing_fingerprint",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.BAZEL_INTERNAL_CONFIGURATION},
      help =
          "If true, bails out from remote analysis cache retrieval if a single fingerprint is"
              + " missing.")
  public boolean analysisCacheBailOnMissingFingerprint;
}
