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
package com.google.devtools.build.lib.remote.blobstore;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.URI;
import java.net.URISyntaxException;
import org.apache.http.HttpEntity;
import org.apache.http.HttpStatus;
import org.apache.http.client.HttpClient;
import org.apache.http.client.config.RequestConfig;
import org.apache.http.client.methods.HttpGet;
import org.apache.http.client.methods.HttpHead;
import org.apache.http.client.methods.HttpPut;
import org.apache.http.entity.ByteArrayEntity;
import org.apache.http.entity.ContentType;
import org.apache.http.entity.InputStreamEntity;
import org.apache.http.impl.client.HttpClientBuilder;
import org.apache.http.impl.conn.PoolingHttpClientConnectionManager;

/**
 * Implementation of {@link SimpleBlobStore} with a REST service. The REST service needs to
 * support the following HTTP methods.
 *
 * <p>PUT /{actioncache,cas}/1234 HTTP/1.1 PUT method is used to upload a blob with a base16 key.
 * In this example the key is 1234. Valid status codes are 200, 201, 202 and 204.
 *
 * <p>GET /{actioncache,cas}/1234 HTTP/1.1 GET method fetches a blob with the specified key. In this
 * example the key is 1234. A status code of 200 should be followed by the content of blob. Status
 * code of 404 or 204 means the key cannot be found.
 *
 * <p>HEAD /{actioncache,cas}/1234 HTTP/1.1 HEAD method checks to see if the specified key exists in
 * the blob store. A status code of 200 indicates the key is found in the blob store. A status code
 * of 404 indicates the key is not found in the blob store.
 */
public final class RestBlobStore implements SimpleBlobStore {

  private static final String ACTION_CACHE_PREFIX = "ac";
  private static final String CAS_PREFIX = "cas";

  private final String baseUrl;
  private final PoolingHttpClientConnectionManager connMan;
  private final HttpClientBuilder clientFactory;

  /**
   * Creates a new instance.
   *
   * @param baseUrl base URL for the remote cache
   * @param poolSize maximum number of simultaneous connections
   */
  public RestBlobStore(String baseUrl, int poolSize, int timeoutMillis) throws IOException {
    validateUrl(baseUrl);
    this.baseUrl = baseUrl;
    connMan = new PoolingHttpClientConnectionManager();
    connMan.setDefaultMaxPerRoute(poolSize);
    connMan.setMaxTotal(poolSize);
    clientFactory = HttpClientBuilder.create();
    clientFactory.setConnectionManager(connMan);
    clientFactory.setConnectionManagerShared(true);
    clientFactory.setDefaultRequestConfig(RequestConfig.custom()
        // Timeout to establish a connection.
        .setConnectTimeout(timeoutMillis)
        // Timeout between reading data.
        .setSocketTimeout(timeoutMillis)
        .build());
  }

  @Override
  public void close() {
    connMan.close();
  }

  @Override
  public boolean containsKey(String key) throws IOException {
    HttpClient client = clientFactory.build();
    HttpHead head = new HttpHead(baseUrl + "/" + CAS_PREFIX + "/" + key);
    return client.execute(
        head,
        response -> {
          int statusCode = response.getStatusLine().getStatusCode();
          return HttpStatus.SC_OK == statusCode;
        });
  }

  @Override
  public boolean get(String key, OutputStream out) throws IOException {
    return get(CAS_PREFIX, key, out);
  }

  @Override
  public boolean getActionResult(String key, OutputStream out)
      throws IOException, InterruptedException {
    return get(ACTION_CACHE_PREFIX, key, out);
  }

  private boolean get(String urlPrefix, String key, OutputStream out) throws IOException {
    HttpClient client = clientFactory.build();
    HttpGet get = new HttpGet(baseUrl + "/" + urlPrefix + "/" + key);
    return client.execute(
        get,
        response -> {
          int statusCode = response.getStatusLine().getStatusCode();
          if (HttpStatus.SC_NOT_FOUND == statusCode
              || HttpStatus.SC_NO_CONTENT == statusCode) {
            return false;
          }
          if (HttpStatus.SC_OK != statusCode) {
            throw new IOException("GET failed with status code " + statusCode);
          }
          response.getEntity().writeTo(out);
          return true;
        });
  }

  @Override
  public void put(String key, long length, InputStream in) throws IOException {
    put(CAS_PREFIX, key, new InputStreamEntity(in, length, ContentType.APPLICATION_OCTET_STREAM));
  }

  @Override
  public void putActionResult(String key, byte[] in) throws IOException, InterruptedException {
    put(ACTION_CACHE_PREFIX, key, new ByteArrayEntity(in, ContentType.APPLICATION_OCTET_STREAM));
  }

  private void put(String urlPrefix, String key, HttpEntity entity) throws IOException {
    HttpClient client = clientFactory.build();
    HttpPut put = new HttpPut(baseUrl + "/" + urlPrefix + "/" + key);
    put.setEntity(entity);
    client.execute(
        put,
        (response) -> {
          int statusCode = response.getStatusLine().getStatusCode();
          // Accept more than SC_OK to be compatible with Nginx WebDav module.
          if (HttpStatus.SC_OK != statusCode
              && HttpStatus.SC_ACCEPTED != statusCode
              && HttpStatus.SC_CREATED != statusCode
              && HttpStatus.SC_NO_CONTENT != statusCode) {
            throw new IOException("PUT failed with status code " + statusCode);
          }
          return null;
        });
  }

  private void validateUrl(String url) throws IOException {
    try {
      new URI(url);
    } catch (URISyntaxException e) {
      throw new IOException("Failed to parse remote REST cache URL: " + baseUrl, e);
    }
  }
}
