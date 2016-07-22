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
 * A factory class for providing a {@link ConcurrentMap} object implemented by a REST service. The
 * URL has to support PUT, GET, and HEAD operations
 */
public final class RestUrlCacheFactory {

  public static ConcurrentMap<String, byte[]> create(RemoteOptions options) {
    return new RestUrlCache(options.restCacheUrl);
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

        if (HttpStatus.SC_OK != statusCode) {
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
}
