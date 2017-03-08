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
import java.util.Collection;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentMap;
import org.apache.http.HttpEntity;
import org.apache.http.HttpResponse;
import org.apache.http.HttpStatus;
import org.apache.http.client.HttpClient;
import org.apache.http.client.methods.HttpGet;
import org.apache.http.client.methods.HttpHead;
import org.apache.http.client.methods.HttpPut;
import org.apache.http.entity.ByteArrayEntity;
import org.apache.http.impl.client.DefaultHttpClient;
import org.apache.http.util.EntityUtils;

/**
 * A factory class for providing a {@link ConcurrentMap} objects to be used with {@link
 * ConcurrentMapActionCache} objects. The underlying maps can be Hazelcast or RestUrl based.
 */
public final class ConcurrentMapFactory {

  private static final String HAZELCAST_CACHE_NAME = "hazelcast-build-cache";

  private ConcurrentMapFactory() {}

  public static ConcurrentMap<String, byte[]> createHazelcast(RemoteOptions options) {
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
    return instance.getMap(HAZELCAST_CACHE_NAME);
  }

  private static class RestUrlCache implements ConcurrentMap<String, byte[]> {

    final String baseUrl;

    RestUrlCache(String baseUrl) {
      this.baseUrl = baseUrl;
    }

    @Override
    public boolean containsKey(Object key) {
      try {
        HttpClient client = new DefaultHttpClient();
        HttpHead head = new HttpHead(baseUrl + "/" + key);
        HttpResponse response = client.execute(head);
        int statusCode = response.getStatusLine().getStatusCode();
        return HttpStatus.SC_OK == statusCode;
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
    }

    @Override
    public byte[] get(Object key) {
      try {
        HttpClient client = new DefaultHttpClient();
        HttpGet get = new HttpGet(baseUrl + "/" + key);
        HttpResponse response = client.execute(get);
        int statusCode = response.getStatusLine().getStatusCode();
        if (HttpStatus.SC_NOT_FOUND == statusCode) {
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
    public byte[] put(String key, byte[] value) {
      try {
        HttpClient client = new DefaultHttpClient();
        HttpPut put = new HttpPut(baseUrl + "/" + key);
        put.setEntity(new ByteArrayEntity(value));
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
      return null;
    }

    //UnsupportedOperationExceptions from here down
    @Override
    public int size() {
      throw new UnsupportedOperationException();
    }

    @Override
    public boolean isEmpty() {
      throw new UnsupportedOperationException();
    }

    @Override
    public boolean containsValue(Object value) {
      throw new UnsupportedOperationException();
    }

    @Override
    public byte[] remove(Object key) {
      throw new UnsupportedOperationException();
    }

    @Override
    public void putAll(Map<? extends String, ? extends byte[]> m) {
      throw new UnsupportedOperationException();
    }

    @Override
    public void clear() {
      throw new UnsupportedOperationException();
    }

    @Override
    public Set<String> keySet() {
      throw new UnsupportedOperationException();
    }

    @Override
    public Collection<byte[]> values() {
      throw new UnsupportedOperationException();
    }

    @Override
    public Set<Entry<String, byte[]>> entrySet() {
      throw new UnsupportedOperationException();
    }

    @Override
    public byte[] putIfAbsent(String key, byte[] value) {
      throw new UnsupportedOperationException();
    }

    @Override
    public boolean remove(Object key, Object value) {
      throw new UnsupportedOperationException();
    }

    @Override
    public boolean replace(String key, byte[] oldValue, byte[] newValue) {
      throw new UnsupportedOperationException();
    }

    @Override
    public byte[] replace(String key, byte[] value) {
      throw new UnsupportedOperationException();
    }
  }

  public static ConcurrentMap<String, byte[]> createRestUrl(RemoteOptions options) {
    return new RestUrlCache(options.restCacheUrl);
  }

  public static ConcurrentMap<String, byte[]> create(RemoteOptions options) {
    if (isHazelcastOptions(options)) {
      return createHazelcast(options);
    }
    if (isRestUrlOptions(options)) {
      return createRestUrl(options);
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
    return options.restCacheUrl != null;
  }
}
