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

import com.google.common.base.Ascii;
import com.google.common.base.Joiner;
import com.google.common.base.Strings;
import com.google.common.hash.HashCode;
import com.google.common.hash.Hashing;
import com.google.devtools.common.options.Converter;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsClass;
import com.google.devtools.common.options.OptionsParsingException;
import javax.annotation.Nullable;

/** Options for caching analysis results remotely. */
@OptionsClass
public abstract class RemoteAnalysisCachingOptions extends OptionsBase {

  /** A converter for MD5 checksums. */
  public static final class Md5Converter implements Converter<HashCode> {
    @Nullable
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
  public abstract String getSerializedFrontierProfile();

  @Option(
      name = "experimental_remote_analysis_cache_mode",
      defaultValue = "off",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.BAZEL_INTERNAL_CONFIGURATION},
      converter = RemoteAnalysisCacheModeConverter.class,
      help = "The transport direction for the remote analysis cache.")
  public abstract RemoteAnalysisCacheMode getMode();

  public abstract void setMode(RemoteAnalysisCacheMode value);

  /** Converter for {@link RemoteAnalysisCacheMode}. */
  private static class RemoteAnalysisCacheModeConverter
      extends Converter.Contextless<RemoteAnalysisCacheMode> {
    @Override
    public RemoteAnalysisCacheMode convert(String input) throws OptionsParsingException {
      for (RemoteAnalysisCacheMode value : RemoteAnalysisCacheMode.values()) {
        if (Ascii.equalsIgnoreCase(value.toString(), input)) {
          return value;
        }
      }
      throw new OptionsParsingException(
          "Not a valid remote analysis cache mode: '"
              + input
              + "' (should be "
              + getTypeDescription()
              + ")");
    }

    @Override
    public String getTypeDescription() {
      return Joiner.on(", ").join(RemoteAnalysisCacheMode.values());
    }
  }

  @Option(
      name = "experimental_analysis_cache_key_distinguisher_for_testing",
      defaultValue = "null",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.BAZEL_INTERNAL_CONFIGURATION},
      help = "An opaque string used as part of the cache key. Should only be used for testing.")
  public abstract String getAnalysisCacheKeyDistinguisherForTesting();

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
  public abstract HashCode getServerChecksumOverride();

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
  public abstract boolean getSkycacheMinimizeMemory();

  @Option(
      name = "experimental_analysis_cache_bail_on_missing_fingerprint",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.BAZEL_INTERNAL_CONFIGURATION},
      help =
          "If true, bails out from remote analysis cache retrieval if a single fingerprint is"
              + " missing.")
  public abstract boolean getAnalysisCacheBailOnMissingFingerprint();

  @Option(
      name = "experimental_skycache_analysis_only",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.BAZEL_INTERNAL_CONFIGURATION},
      help = "If true, Skycache will only be used for analysis phase.")
  public abstract boolean getSkycacheAnalysisOnly();

  @Option(
      name = "remote_analysis_cache_emit_bep_upload_events",
      defaultValue = "false",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.BAZEL_INTERNAL_CONFIGURATION},
      help = "If true, Blaze will emit debug events for remote analysis caching.")
  public abstract boolean getEmitBepUploadEvents();
}
