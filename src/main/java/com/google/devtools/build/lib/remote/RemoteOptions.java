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
import com.google.devtools.common.options.OptionsBase;

/** Options for remote execution and distributed caching. */
public final class RemoteOptions extends OptionsBase {
  @Option(
    name = "rest_cache_url",
    defaultValue = "null",
    category = "remote",
    help =
        "A base URL for a RESTful cache server for storing build artifacts."
            + "It has to support PUT, GET, and HEAD requests."
  )
  public String restCacheUrl;

  @Option(
    name = "hazelcast_node",
    defaultValue = "null",
    category = "remote",
    help = "A comma separated list of hostnames of hazelcast nodes. For client mode only."
  )
  public String hazelcastNode;

  @Option(
    name = "hazelcast_client_config",
    defaultValue = "null",
    category = "remote",
    help = "A file path to a hazelcast client config XML file. For client mode only."
  )
  public String hazelcastClientConfig;

  @Option(
    name = "hazelcast_standalone_listen_port",
    defaultValue = "0",
    category = "build_worker",
    help =
        "Runs an embedded hazelcast server that listens to this port. The server does not join"
            + " any cluster. This is useful for testing."
  )
  public int hazelcastStandaloneListenPort;

  @Option(
    name = "remote_worker",
    defaultValue = "null",
    category = "remote",
    help =
        "Hostname and port number of remote worker in the form of host:port. "
            + "For client mode only."
  )
  public String remoteWorker;

  @Option(
    name = "remote_cache",
    defaultValue = "null",
    category = "remote",
    help =
        "Hostname and port number of remote gRPC cache in the form of host:port. "
            + "For client mode only."
  )
  public String remoteCache;

  @Option(
    name = "grpc_max_chunk_size_bytes",
    defaultValue = "400000", // <4MB. Bounded by the gRPC size limit on the overall message.
    category = "remote",
    help = "The maximal number of bytes to be sent in a single message. For client mode only."
  )
  public int grpcMaxChunkSizeBytes;

  @Option(
    name = "grpc_max_batch_inputs",
    defaultValue = "100",
    category = "remote",
    help = "The maximal number of input file to be sent in a single batch. For client mode only."
  )
  public int grpcMaxBatchInputs;

  @Option(
    name = "grpc_max_batch_size_bytes",
    defaultValue = "10485760", // 10MB
    category = "remote",
    help = "The maximal number of input bytes to be sent in a single batch. For client mode only."
  )
  public int grpcMaxBatchSizeBytes;

  @Option(
    name = "grpc_timeout_seconds",
    defaultValue = "60",
    category = "remote",
    help = "The maximal number of seconds to wait for remote calls. For client mode only."
  )
  public int grpcTimeoutSeconds;

  @Option(
    name = "remote_accept_cached",
    defaultValue = "true",
    category = "remote",
    help = "Whether to accept remotely cached action results."
  )
  public boolean remoteAcceptCached;

  @Option(
    name = "remote_allow_local_fallback",
    defaultValue = "true",
    category = "remote",
    help = "Whether to fall back to standalone strategy if remote fails."
  )
  public boolean remoteAllowLocalFallback;

  @Option(
    name = "remote_local_exec_upload_results",
    defaultValue = "true",
    category = "remote",
    help = "Whether to upload action results to the remote cache after executing locally."
  )
  public boolean remoteLocalExecUploadResults;
}
