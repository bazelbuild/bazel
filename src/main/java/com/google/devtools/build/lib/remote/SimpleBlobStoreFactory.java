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
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.concurrent.ConcurrentMap;
import org.apache.http.HttpEntity;
import org.apache.http.HttpResponse;
import org.apache.http.HttpStatus;
import org.apache.http.client.HttpClient;
import org.apache.http.client.methods.HttpGet;
import org.apache.http.client.methods.HttpHead;
import org.apache.http.client.methods.HttpPut;
import org.apache.http.entity.ByteArrayEntity;
import org.apache.http.impl.client.HttpClientBuilder;
import org.apache.http.util.EntityUtils;

/**
 * A factory class for providing a {@link SimpleBlobStore} to be used with {@link
 * SimpleBlobStoreActionCache}. Currently implemented with Hazelcast or REST.
 */
public final class SimpleBlobStoreFactory {

  private static final String HAZELCAST_CACHE_NAME = "hazelcast-build-cache";

  private SimpleBlobStoreFactory() {}

  /** A {@link SimpleBlobStore} implementation using a {@link ConcurrentMap}. */
  public static class ConcurrentMapBlobStore implements SimpleBlobStore {
    private final ConcurrentMap<String, byte[]> map;

    public ConcurrentMapBlobStore(ConcurrentMap<String, byte[]> map) {
      this.map = map;
    }

    @Override
    public boolean containsKey(String key) {
      return map.containsKey(key);
    }

    @Override
    public byte[] get(String key) {
      return map.get(key);
    }

    @Override
    public void put(String key, byte[] value) {
      map.put(key, value);
    }
  }

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

  /**
   * Implementation of {@link SimpleBlobStore} with a REST service. The REST service needs to
   * support the following HTTP methods.
   *
   * <p>PUT /cache/1234 HTTP/1.1 PUT method is used to upload a blob with a base16 key. In this
   * example the key is 1234. Valid status codes are 200, 201, 202 and 204.
   *
   * <p>GET /cache/1234 HTTP/1.1 GET method fetches a blob with the specified key. In this example
   * the key is 1234. A status code of 200 should be followed by the content of blob. Status code of
   * 404 or 204 means the key cannot be found.
   *
   * <p>HEAD /cache/1234 HTTP/1.1 HEAD method checks to see if the specified key exists in the blob
   * store. A status code of 200 indicates the key is found in the blob store. A status code of 404
   * indicates the key is not found in the blob store.
   */
  private static class RestBlobStore implements SimpleBlobStore {

    private final String baseUrl;

    RestBlobStore(String baseUrl) {
      this.baseUrl = baseUrl;
    }

    @Override
    public boolean containsKey(String key) {
      try {
        HttpClient client = HttpClientBuilder.create().build();
        HttpHead head = new HttpHead(baseUrl + "/" + key);
        HttpResponse response = client.execute(head);
        int statusCode = response.getStatusLine().getStatusCode();
        return HttpStatus.SC_OK == statusCode;
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
    }

    @Override
    public byte[] get(String key) {
      try {
        HttpClient client = HttpClientBuilder.create().build();
        HttpGet get = new HttpGet(baseUrl + "/" + key);
        HttpResponse response = client.execute(get);
        int statusCode = response.getStatusLine().getStatusCode();
        if (HttpStatus.SC_NOT_FOUND == statusCode || HttpStatus.SC_NO_CONTENT == statusCode) {
          return null;
        }
        if (HttpStatus.SC_OK != statusCode) {
          throw new RuntimeException("GET failed with status code " + statusCode);
        }
        ByteArrayOutputStream buffer = new ByteArrayOutputStream();
        HttpEntity entity = response.getEntity();
        entity.writeTo(buffer);
        buffer.flush();
        EntityUtils.consume(entity);

        return buffer.toByteArray();

      } catch (IOException e) {
        throw new RuntimeException(e);
      }
    }

    @Override
    public void put(String key, byte[] value) {
      try {
        HttpClient client = HttpClientBuilder.create().build();
        HttpPut put = new HttpPut(baseUrl + "/" + key);
        put.setEntity(new ByteArrayEntity(value));
        put.setHeader("Content-Type", "application/octet-stream");
        HttpResponse response = client.execute(put);
        int statusCode = response.getStatusLine().getStatusCode();

        // Accept more than SC_OK to be compatible with Nginx WebDav module.
        if (HttpStatus.SC_OK != statusCode
            && HttpStatus.SC_ACCEPTED != statusCode
            && HttpStatus.SC_CREATED != statusCode
            && HttpStatus.SC_NO_CONTENT != statusCode) {
          throw new RuntimeException("PUT failed with status code " + statusCode);
        }
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
    }
  }

  public static SimpleBlobStore createRest(RemoteOptions options) {
    return new RestBlobStore(options.remoteRestCache);
  }

  public static SimpleBlobStore create(RemoteOptions options) {
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
