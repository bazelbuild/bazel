// Copyright 2017 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.buildeventservice;

import com.google.devtools.common.options.Converters;
import com.google.devtools.common.options.EnumConverter;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionsBase;
import java.time.Duration;
import java.util.List;

/** Options used by {@link BuildEventServiceModule}. */
public class BuildEventServiceOptions extends OptionsBase {

  @Option(
      name = "bes_backend",
      defaultValue = "",
      documentationCategory = OptionDocumentationCategory.LOGGING,
      effectTags = {OptionEffectTag.AFFECTS_OUTPUTS},
      help =
          "Specifies the build event service (BES) backend endpoint as HOST or HOST:PORT. Disabled"
              + " by default.The supported schemas are grpc and grpcs (grpc with TLS enabled). If"
              + " no schema is provided bazel'll default to grpcs. Specify grpc:// schema to"
              + " disable TLS.")
  public String besBackend;

  @Option(
    name = "bes_timeout",
    defaultValue = "0s",
    documentationCategory = OptionDocumentationCategory.LOGGING,
    effectTags = {OptionEffectTag.AFFECTS_OUTPUTS},
    help =
        "Specifies how long bazel should wait for the BES/BEP upload to complete after the "
            + "build and tests have finished. A valid timeout is a natural number followed by a "
            + "unit: Days (d), hours (h), minutes (m), seconds (s), and milliseconds (ms). The "
            + "default value is '0' which means that there is no timeout."
  )
  public Duration besTimeout;

  @Option(
      name = "bes_best_effort",
      defaultValue = "false",
      deprecationWarning =
          "BES best effort upload has been removed. The flag has no more "
              + "functionality attached to it and will be removed in a future release.",
      documentationCategory = OptionDocumentationCategory.LOGGING,
      effectTags = {OptionEffectTag.AFFECTS_OUTPUTS},
      help =
          "BES best effort upload has been removed. The flag has no more "
              + "functionality attached to it and will be removed in a future release.")
  public boolean besBestEffort;

  @Option(
    name = "bes_lifecycle_events",
    defaultValue = "true",
    documentationCategory = OptionDocumentationCategory.LOGGING,
    effectTags = {OptionEffectTag.AFFECTS_OUTPUTS},
    help = "Specifies whether to publish BES lifecycle events. (defaults to 'true')."
  )
  public boolean besLifecycleEvents;

  @Option(
    name = "project_id",
    defaultValue = "null",
    documentationCategory = OptionDocumentationCategory.LOGGING,
    effectTags = {OptionEffectTag.AFFECTS_OUTPUTS},
    help = "Specifies the BES project identifier. Defaults to null."
  )
  public String projectId;

  @Option(
      name = "bes_keywords",
      defaultValue = "null",
      converter = Converters.CommaSeparatedOptionListConverter.class,
      documentationCategory = OptionDocumentationCategory.LOGGING,
      effectTags = {OptionEffectTag.AFFECTS_OUTPUTS},
      allowMultiple = true,
      help =
          "Specifies a list of notification keywords to be added the default set of keywords "
              + "published to BES (\"command_name=<command_name> \", \"protocol_name=BEP\"). "
              + "Defaults to none.")
  public List<String> besKeywords;

  @Option(
      name = "bes_outerr_buffer_size",
      defaultValue = "10240",
      documentationCategory = OptionDocumentationCategory.LOGGING,
      effectTags = {OptionEffectTag.AFFECTS_OUTPUTS},
      help =
          "Specifies the maximal size of stdout or stderr to be buffered in BEP, before it is "
              + "reported as a progress event. Individual writes are still reported in a single "
              + "event, even if larger than the specified value up to --bes_outerr_chunk_size.")
  public int besOuterrBufferSize;

  @Option(
      name = "bes_outerr_chunk_size",
      defaultValue = "1048576", // 2^20 = 1MB
      documentationCategory = OptionDocumentationCategory.LOGGING,
      effectTags = {OptionEffectTag.AFFECTS_OUTPUTS},
      help =
          "Specifies the maximal size of stdout or stderr to be sent to BEP in a single message.")
  public int besOuterrChunkSize;

  @Option(
      name = "bes_results_url",
      defaultValue = "",
      documentationCategory = OptionDocumentationCategory.LOGGING,
      effectTags = {OptionEffectTag.TERMINAL_OUTPUT},
      help =
          "Specifies the base URL where a user can view the information streamed to the BES"
              + " backend. Bazel will output the URL appended by the invocation id to the"
              + " terminal.")
  public String besResultsUrl;

  @Option(
      name = "bes_upload_mode",
      defaultValue = "wait_for_upload_complete",
      converter = BesUploadModeConverter.class,
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.EAGERNESS_TO_EXIT},
      help =
          "Specifies whether the Build Event Service upload should block the build completion "
              + "or should end the invocation immediately and finish the upload in the background.")
  public BesUploadMode besUploadMode;

  @Option(
      name = "bes_proxy",
      defaultValue = "null",
      documentationCategory = OptionDocumentationCategory.LOGGING,
      effectTags = {OptionEffectTag.UNKNOWN},
      help =
          "Connect to the Build Event Service through a proxy. Currently this flag can only be"
              + " used to configure a Unix domain socket (unix:/path/to/socket).")
  public String besProxy;

  /** Determines the mode that will be used to upload data to the Build Event Service. */
  public enum BesUploadMode {
    /** Block at the end of the build waiting for the upload to complete */
    WAIT_FOR_UPLOAD_COMPLETE,
    /** Block at the beginning of the next build waiting for upload completion */
    NOWAIT_FOR_UPLOAD_COMPLETE,
    /**
     * Block at the beginning of the next build waiting for the client to finish uploading the data,
     * but possibly not blocking on the server acknowledgement.
     */
    FULLY_ASYNC,
  }

  /** Converter for {@link BesUploadMode} */
  public static class BesUploadModeConverter extends EnumConverter<BesUploadMode> {
    public BesUploadModeConverter() {
      super(BesUploadMode.class, "Mode for uploading to the Build Event Service");
    }
  }
}
