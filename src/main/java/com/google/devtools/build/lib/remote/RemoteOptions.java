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

package com.google.devtools.build.lib.remote;

import com.google.devtools.build.lib.util.OptionsUtils;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionMetadataTag;
import com.google.devtools.common.options.OptionsBase;

/** Options for remote execution and distributed caching. */
public final class RemoteOptions extends OptionsBase {
  @Option(
    name = "remote_http_cache",
    oldName = "remote_rest_cache",
    defaultValue = "null",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help =
        "A base URL of a HTTP caching service. Both http:// https:// are supported. BLOBs are "
            + "stored with PUT and retrieved with GET. See remote/README.md for more information."
  )
  public String remoteHttpCache;

  @Option(
      name = "remote_cache_proxy",
      defaultValue = "null",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help =
          "Connect to the remote cache through a proxy. Currently this flag can only be used to "
              + "configure a Unix domain socket (unix:/path/to/socket) for the HTTP cache."
  )
  public String remoteCacheProxy;

    @Option(
      name = "remote_s3_region",
      defaultValue = "null",
      category = "remote",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help =
          "The specific region for an S3 bucket, used as a HTTP REST cache"
              + ". See remote/README.md for more information."
  )
  public String awsS3Region;

  @Option(
      name = "remote_max_connections",
      defaultValue = "100",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.HOST_MACHINE_RESOURCE_OPTIMIZATIONS},
      help =
          "The max. number of concurrent network connections to the remote cache/executor. By "
              + "default Bazel limits the number of TCP connections to 100. Setting this flag to "
              + "0 will make Bazel choose the number of connections automatically."
  )
  public int remoteMaxConnections;

  @Option(
    name = "remote_executor",
    defaultValue = "null",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help = "HOST or HOST:PORT of a remote execution endpoint."
  )
  public String remoteExecutor;

  @Option(
    name = "remote_cache",
    defaultValue = "null",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help = "HOST or HOST:PORT of a remote caching endpoint."
  )
  public String remoteCache;

  @Option(
    name = "remote_timeout",
    defaultValue = "60",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help = "The maximum number of seconds to wait for remote execution and cache calls."
  )
  public int remoteTimeout;

  @Option(
    name = "remote_accept_cached",
    defaultValue = "true",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help = "Whether to accept remotely cached action results."
  )
  public boolean remoteAcceptCached;

  @Option(
    name = "remote_local_fallback",
    defaultValue = "false",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help = "Whether to fall back to standalone local execution strategy if remote execution fails."
  )
  public boolean remoteLocalFallback;

  @Option(
      name = "remote_local_fallback_strategy",
      defaultValue = "local",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help = "The strategy to use when remote execution has to fallback to local execution."
  )
  public String remoteLocalFallbackStrategy;

  @Option(
    name = "remote_upload_local_results",
    defaultValue = "true",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help = "Whether to upload locally executed action results to the remote cache."
  )
  public boolean remoteUploadLocalResults;

  @Option(
    name = "remote_instance_name",
    defaultValue = "",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help = "Value to pass as instance_name in the remote execution API."
  )
  public String remoteInstanceName;

  @Option(
    name = "experimental_remote_retry",
    defaultValue = "true",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help = "Whether to retry transient remote execution/cache errors."
  )
  public boolean experimentalRemoteRetry;

  @Option(
    name = "experimental_remote_retry_start_delay_millis",
    defaultValue = "100",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help = "The initial delay before retrying a transient error."
  )
  public long experimentalRemoteRetryStartDelayMillis;

  @Option(
    name = "experimental_remote_retry_max_delay_millis",
    defaultValue = "5000",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help = "The maximum delay before retrying a transient error."
  )
  public long experimentalRemoteRetryMaxDelayMillis;

  @Option(
    name = "experimental_remote_retry_max_attempts",
    defaultValue = "5",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help = "The maximum number of attempts to retry a transient error."
  )
  public int experimentalRemoteRetryMaxAttempts;

  @Option(
    name = "experimental_remote_retry_multiplier",
    defaultValue = "2",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help = "The multiplier by which to increase the retry delay on transient errors."
  )
  public double experimentalRemoteRetryMultiplier;

  @Option(
    name = "experimental_remote_retry_jitter",
    defaultValue = "0.1",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help = "The random factor to apply to retry delays on transient errors."
  )
  public double experimentalRemoteRetryJitter;

  @Deprecated
  @Option(
    name = "experimental_remote_spawn_cache",
    defaultValue = "false",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.NO_OP},
    help =
        "Whether to use the experimental spawn cache infrastructure for remote caching. "
            + "Enabling this flag makes Bazel ignore any setting for remote_executor."
  )
  public boolean experimentalRemoteSpawnCache;

  @Option(
    name = "disk_cache",
    defaultValue = "null",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    converter = OptionsUtils.PathFragmentConverter.class,
    help =
        "A path to a directory where Bazel can read and write actions and action outputs. "
            + "If the directory does not exist, it will be created."
  )
  public PathFragment diskCache;

  @Option(
    name = "experimental_guard_against_concurrent_changes",
    defaultValue = "false",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help =
        "Turn this off to disable checking the ctime of input files of an action before "
            + "uploading it to a remote cache. There may be cases where the Linux kernel delays "
            + "writing of files, which could cause false positives."
  )
  public boolean experimentalGuardAgainstConcurrentChanges;

  @Option(
    name = "experimental_remote_grpc_log",
    defaultValue = "",
    category = "remote",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help =
        "If specified, a path to a file to log gRPC call related details. This log consists "
            + "of a sequence of serialized "
            + "com.google.devtools.build.lib.remote.logging.RemoteExecutionLog.LogEntry "
            + "protobufs with each message prefixed by a varint denoting the size of the following "
            + "serialized protobuf message, as performed by the method "
            + "LogEntry.writeDelimitedTo(OutputStream)."
  )
  public String experimentalRemoteGrpcLog;

  @Option(
      name = "incompatible_remote_symlinks",
      defaultValue = "false",
      category = "remote",
      documentationCategory = OptionDocumentationCategory.EXECUTION_STRATEGY,
      effectTags = {OptionEffectTag.EXECUTION},
      metadataTags = {
        OptionMetadataTag.INCOMPATIBLE_CHANGE,
        OptionMetadataTag.TRIGGERED_BY_ALL_INCOMPATIBLE_CHANGES
      },
      help =
          "If set to true, Bazel will represent symlinks in action outputs "
              + "in the remote caching/execution protocol as such. The "
              + "current behavior is for remote caches/executors to follow "
              + "symlinks and represent them as files. See #6631 for details.")
  public boolean incompatibleRemoteSymlinks;

  @Option(
      name = "build_event_upload_max_threads",
      defaultValue = "100",
      documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help = "The number of threads used to do build event uploads. Capped at 1000.")
  public int buildEventUploadMaxThreads;

  @Deprecated
  @Option(
      name = "remote_allow_symlink_upload",
      defaultValue = "true",
      category = "remote",
      documentationCategory = OptionDocumentationCategory.EXECUTION_STRATEGY,
      effectTags = {OptionEffectTag.EXECUTION},
      help =
          "If true, upload action symlink outputs to the remote cache. "
              + "If this option is not enabled, "
              + "cachable actions that output symlinks will fail.")
  public boolean allowSymlinkUpload;

  // The below options are not configurable by users, only tests.
  // This is part of the effort to reduce the overall number of flags.

  /** The maximum size of an outbound message sent via a gRPC channel. */
  public int maxOutboundMessageSize = 1024 * 1024;
}
