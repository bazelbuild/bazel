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

/**
 * Options for remote execution and distributed caching.
 */
public final class RemoteOptions extends OptionsBase {
  @Option(
    name = "hazelcast_node",
    defaultValue = "null",
    category = "remote",
    help = "A comma separated list of hostnames of hazelcast nodes. For client mode only."
  )
  public String hazelcastNode;

  @Option(
    name = "rest_worker_url",
    defaultValue = "null",
    category = "remote",
    help = "URL for the REST worker."
  )
  public String restWorkerUrl;
}
