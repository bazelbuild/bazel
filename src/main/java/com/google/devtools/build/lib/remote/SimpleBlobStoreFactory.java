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

import com.google.devtools.build.lib.remote.blobstore.ConcurrentMapBlobStore;
import com.google.devtools.build.lib.remote.blobstore.RestBlobStore;
import com.google.devtools.build.lib.remote.blobstore.SimpleBlobStore;
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
 * A factory class for providing a {@link SimpleBlobStore} to be used with {@link
 * SimpleBlobStoreActionCache}. Currently implemented with Hazelcast or REST.
 */
public final class SimpleBlobStoreFactory {

  private static final String HAZELCAST_CACHE_NAME = "hazelcast-build-cache";

  private SimpleBlobStoreFactory() {}

  /** Construct a {@link SimpleBlobStore} using Hazelcast's version of {@link ConcurrentMap} */
  public static SimpleBlobStore createHazelcast(RemoteOptions options) {
    HazelcastInstance instance;
    if (options.hazelcastClientConfig != null) {
      try {
        ClientConfig config = new XmlClientConfigBuilder(options.hazelcastClientConfig).build();
        instance = HazelcastClient.newHazelcastClient(config);
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
    } else if (options.hazelcastNode != null) {
      // If --hazelcast_node is specified then create a client instance.
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
    return new ConcurrentMapBlobStore(instance.<String, byte[]>getMap(HAZELCAST_CACHE_NAME));
  }

  public static SimpleBlobStore createRest(RemoteOptions options) throws IOException {
    return new RestBlobStore(options.remoteRestCache, options.restCachePoolSize);
  }

  public static SimpleBlobStore create(RemoteOptions options) throws IOException {
    if (isHazelcastOptions(options)) {
      return createHazelcast(options);
    }
    if (isRestUrlOptions(options)) {
      return createRest(options);
    }
    throw new IllegalArgumentException(
        "Unrecognized concurrent map RemoteOptions: must specify "
            + "either Hazelcast or Rest URL options.");
  }

  public static boolean isRemoteCacheOptions(RemoteOptions options) {
    return isHazelcastOptions(options) || isRestUrlOptions(options);
  }

  private static boolean isHazelcastOptions(RemoteOptions options) {
    return options.hazelcastNode != null
        || options.hazelcastClientConfig != null
        || options.hazelcastStandaloneListenPort != 0;
  }

  private static boolean isRestUrlOptions(RemoteOptions options) {
    return options.remoteRestCache != null;
  }
}
