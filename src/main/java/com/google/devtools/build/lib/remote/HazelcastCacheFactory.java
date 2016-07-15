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

import com.hazelcast.client.HazelcastClient;
import com.hazelcast.client.config.ClientConfig;
import com.hazelcast.client.config.ClientNetworkConfig;
import com.hazelcast.client.config.XmlClientConfigBuilder;
import com.hazelcast.config.Config;
import com.hazelcast.core.Hazelcast;
import com.hazelcast.core.HazelcastInstance;

import java.io.IOException;
import java.util.concurrent.ConcurrentMap;

/**
 * A factory class for providing a {@link ConcurrentMap} object implemented by Hazelcast.
 * Hazelcast will work as a distributed memory cache.
 */
public final class HazelcastCacheFactory {

  private static final String CACHE_NAME = "hazelcast-build-cache";

  public static ConcurrentMap<String, byte[]> create(RemoteOptions options) {
    HazelcastInstance instance;
    if (options.hazelcastClientConfig != null) {
      try {
        ClientConfig config = new XmlClientConfigBuilder(options.hazelcastClientConfig).build();
        instance = HazelcastClient.newHazelcastClient(config);
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
    } else if (options.hazelcastNode != null) {
      // If --hazelcast_node is then create a client instance.
      ClientConfig config = new ClientConfig();
      ClientNetworkConfig net = config.getNetworkConfig();
      net.addAddress(options.hazelcastNode.split(","));
      instance = HazelcastClient.newHazelcastClient(config);
    } else if (options.hazelcastStandaloneListenPort != 0) {
      Config config = new Config();
      config
          .getNetworkConfig()
          .setPort(options.hazelcastStandaloneListenPort)
          .getJoin()
          .getMulticastConfig()
          .setEnabled(false);
      instance = Hazelcast.newHazelcastInstance(config);
    } else {
      // Otherwise create a default instance. This is going to look at
      // -Dhazelcast.config=some-hazelcast.xml for configuration.
      instance = Hazelcast.newHazelcastInstance();
    }
    return instance.getMap(CACHE_NAME);
  }
}
