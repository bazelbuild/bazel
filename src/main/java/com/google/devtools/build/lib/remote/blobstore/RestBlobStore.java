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

import com.google.api.client.http.ByteArrayContent;
import com.google.api.client.http.GenericUrl;
import com.google.api.client.http.HttpContent;
import com.google.api.client.http.HttpRequestFactory;
import com.google.api.client.http.HttpResponse;
import com.google.api.client.http.InputStreamContent;
import com.google.api.client.http.apache.ApacheHttpTransport;
import com.google.auth.Credentials;
import com.google.auth.http.HttpCredentialsAdapter;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.URI;
import java.net.URISyntaxException;
import javax.annotation.Nullable;
import org.apache.http.HttpStatus;
import org.apache.http.client.config.RequestConfig;
import org.apache.http.impl.client.HttpClientBuilder;
import org.apache.http.impl.conn.PoolingHttpClientConnectionManager;

/**
 * Implementation of {@link SimpleBlobStore} with a REST service. The REST service needs to support
 * the following HTTP methods.
 *
 * <p>PUT /{ac,cas}/1234 HTTP/1.1 PUT method is used to upload a blob with a base16 key. In this
 * example the key is 1234. Valid status codes are 200, 201, 202 and 204.
 *
 * <p>GET /{ac,cas}/1234 HTTP/1.1 GET method fetches a blob with the specified key. In this example
 * the key is 1234. A status code of 200 should be followed by the content of blob. Status code of
 * 404 or 204 means the key cannot be found.
 */
public final class RestBlobStore implements SimpleBlobStore {

  private static final String ACTION_CACHE_PREFIX = "ac";
  private static final String CAS_PREFIX = "cas";

  private final String baseUrl;
  private final HttpClientBuilder clientFactory;
  private final ApacheHttpTransport transport;
  private final HttpRequestFactory requestFactory;

  /**
   * Creates a new instance.
   *
   * @param baseUrl base URL for the remote cache
   * @param poolSize maximum number of simultaneous connections
   */
  public RestBlobStore(String baseUrl, int timeoutMillis, @Nullable Credentials creds)
      throws IOException {
    validateUrl(baseUrl);
    this.baseUrl = baseUrl;
    PoolingHttpClientConnectionManager connMan = new PoolingHttpClientConnectionManager();
    // We'll use as many connections as necessary. The connection pool tries to re-use open
    // connections before creating new ones, so in practice we should have as many connections
    // as concurrent actions.
    connMan.setDefaultMaxPerRoute(Integer.MAX_VALUE);
    connMan.setMaxTotal(Integer.MAX_VALUE);
    clientFactory = HttpClientBuilder.create();
    clientFactory.setConnectionManager(connMan);
    clientFactory.setConnectionManagerShared(true);
    clientFactory.setDefaultRequestConfig(RequestConfig.custom()
        // Timeout to establish a connection.
        .setConnectTimeout(timeoutMillis)
        // Timeout between reading data.
        .setSocketTimeout(timeoutMillis)
        .build());
    transport = new ApacheHttpTransport(clientFactory.build());
    if (creds != null) {
      requestFactory = transport.createRequestFactory(new HttpCredentialsAdapter(creds));
    } else {
      requestFactory = transport.createRequestFactory();
    }
  }

  @Override
  public void close() {
    transport.shutdown();
  }

  @Override
  public boolean containsKey(String key) throws IOException {
    throw new UnsupportedOperationException("HTTP Caching does not use this method.");
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
    HttpResponse response = null;
    try {
      response =
          requestFactory
              .buildGetRequest(new GenericUrl(baseUrl + "/" + urlPrefix + "/" + key))
              .setThrowExceptionOnExecuteError(false)
              .execute();
      int statusCode = response.getStatusCode();
      if (HttpStatus.SC_NOT_FOUND == statusCode || HttpStatus.SC_NO_CONTENT == statusCode) {
        return false;
      }
      if (HttpStatus.SC_OK != statusCode) {
        throw new IOException("GET failed with status code " + statusCode);
      }
      response.download(out);
      return true;
    } finally {
      if (response != null) {
        response.disconnect();
      }
    }
  }

  @Override
  public void put(String key, long length, InputStream in) throws IOException {
    put(CAS_PREFIX, key, new InputStreamContent("application/octext-stream", in));
  }

  @Override
  public void putActionResult(String key, byte[] in) throws IOException, InterruptedException {
    put(ACTION_CACHE_PREFIX, key, new ByteArrayContent("application/octet-stream", in));
  }

  private void put(String urlPrefix, String key, HttpContent content) throws IOException {
    HttpResponse response = null;
    try {
      response =
          requestFactory
              .buildPutRequest(new GenericUrl(baseUrl + "/" + urlPrefix + "/" + key), content)
              .setThrowExceptionOnExecuteError(false)
              .execute();
      int statusCode = response.getStatusCode();
      // Accept more than SC_OK to be compatible with Nginx WebDav module.
      if (HttpStatus.SC_OK != statusCode
          && HttpStatus.SC_ACCEPTED != statusCode
          && HttpStatus.SC_CREATED != statusCode
          && HttpStatus.SC_NO_CONTENT != statusCode) {
        throw new IOException("PUT failed with status code " + statusCode);
      }
    } finally {
      if (response != null) {
        response.disconnect();
      }
    }
  }
  
  private void validateUrl(String url) throws IOException {
    try {
      new URI(url);
    } catch (URISyntaxException e) {
      throw new IOException("Failed to parse remote REST cache URL: " + baseUrl, e);
    }
  }
}
