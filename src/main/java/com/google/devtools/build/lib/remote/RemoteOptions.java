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

import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.proto.OptionFilters.OptionEffectTag;

/** Options for remote execution and distributed caching. */
public final class RemoteOptions extends OptionsBase {
  @Option(
    name = "remote_rest_cache",
    defaultValue = "null",
    category = "remote",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help =
        "A base URL for a RESTful cache server for storing build artifacts."
            + "It has to support PUT, GET, and HEAD requests."
  )
  public String remoteRestCache;

  @Option(
    name = "remote_rest_cache_pool_size",
    defaultValue = "20",
    category = "remote",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help = "Size of the HTTP pool for making requests to the REST cache."
  )
  public int restCachePoolSize;

  @Option(
    name = "hazelcast_node",
    defaultValue = "null",
    category = "remote",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help = "A comma separated list of hostnames of hazelcast nodes."
  )
  public String hazelcastNode;

  @Option(
    name = "hazelcast_client_config",
    defaultValue = "null",
    category = "remote",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help = "A file path to a hazelcast client config XML file."
  )
  public String hazelcastClientConfig;

  @Option(
    name = "hazelcast_standalone_listen_port",
    defaultValue = "0",
    category = "build_worker",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help =
        "Runs an embedded hazelcast server that listens to this port. The server does not join"
            + " any cluster. This is useful for testing."
  )
  public int hazelcastStandaloneListenPort;

  @Option(
    name = "remote_executor",
    defaultValue = "null",
    category = "remote",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help = "HOST or HOST:PORT of a remote execution endpoint."
  )
  public String remoteExecutor;

  @Option(
    name = "remote_cache",
    defaultValue = "null",
    category = "remote",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help = "HOST or HOST:PORT of a remote caching endpoint."
  )
  public String remoteCache;

  @Option(
    name = "remote_timeout",
    defaultValue = "60",
    category = "remote",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help = "The maximum number of seconds to wait for remote execution and cache calls."
  )
  public int remoteTimeout;

  @Option(
    name = "remote_accept_cached",
    defaultValue = "true",
    category = "remote",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help = "Whether to accept remotely cached action results."
  )
  public boolean remoteAcceptCached;

  @Option(
    name = "remote_local_fallback",
    defaultValue = "false",
    category = "remote",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help = "Whether to fall back to standalone local execution strategy if remote execution fails."
  )
  public boolean remoteLocalFallback;

  @Option(
    name = "remote_upload_local_results",
    defaultValue = "true",
    category = "remote",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help = "Whether to upload locally executed action results to the remote cache."
  )
  public boolean remoteUploadLocalResults;

  @Option(
    name = "experimental_remote_platform_override",
    defaultValue = "null",
    category = "remote",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help = "Temporary, for testing only. Manually set a Platform to pass to remote execution."
  )
  public String experimentalRemotePlatformOverride;

  @Option(
    name = "remote_instance_name",
    defaultValue = "",
    category = "remote",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help = "Value to pass as instance_name in the remote execution API."
  )
  public String remoteInstanceName;

  @Option(
    name = "experimental_remote_retry",
    defaultValue = "true",
    category = "remote",
    help = "Whether to retry transient remote execution/cache errors."
  )
  public boolean experimentalRemoteRetry;

  @Option(
    name = "experimental_remote_retry_start_delay_millis",
    defaultValue = "100",
    category = "remote",
    help = "The initial delay before retrying a transient error."
  )
  public long experimentalRemoteRetryStartDelayMillis;

  @Option(
    name = "experimental_remote_retry_max_delay_millis",
    defaultValue = "5000",
    category = "remote",
    help = "The maximum delay before retrying a transient error."
  )
  public long experimentalRemoteRetryMaxDelayMillis;

  @Option(
    name = "experimental_remote_retry_max_attempts",
    defaultValue = "5",
    category = "remote",
    help = "The maximum number of attempts to retry a transient error."
  )
  public int experimentalRemoteRetryMaxAttempts;

  @Option(
    name = "experimental_remote_retry_multiplier",
    defaultValue = "2",
    category = "remote",
    help = "The multiplier by which to increase the retry delay on transient errors."
  )
  public double experimentalRemoteRetryMultiplier;

  @Option(
    name = "experimental_remote_retry_jitter",
    defaultValue = "0.1",
    category = "remote",
    help = "The random factor to apply to retry delays on transient errors."
  )
  public double experimentalRemoteRetryJitter;
}
