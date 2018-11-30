// Copyright 2018 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.remote.blobstore.http;

import com.amazonaws.auth.AWSCredentialsProvider;
import com.google.auth.Credentials;
import com.google.errorprone.annotations.concurrent.GuardedBy;
import io.netty.handler.codec.http.HttpRequest;

import java.io.IOException;
import java.net.URI;
import java.util.*;
import java.util.concurrent.TimeUnit;

/**
 * Base class for credentials adapters.
 */
public abstract class HttpCredentialsAdapter {

  private final Object credentialsLock = new Object();

  @GuardedBy("credentialsLock")
  private long lastRefreshTime;

  /**
   * Force refresh the underlying credentials for this blob store
   */
  public void refreshCredentials() throws IOException {
    synchronized (credentialsLock) {
      long now = System.currentTimeMillis();
      // Call creds.refresh() at most once per second. The one second was arbitrarily chosen, as
      // a small enough value that we don't expect to interfere with actual token lifetimes, but
      // it should just make sure that potentially hundreds of threads don't call this method
      // at the same time.
      if ((now - lastRefreshTime) > TimeUnit.SECONDS.toMillis(1)) {
        lastRefreshTime = now;
        refresh();
      }
    }
  }

  /**
   * @return true if the underlying credential store alters request metadata (i.e. adds Http headers)
   */
  public boolean hasRequestHeaders() {
    return true;
  }

  /**
   * Gets any credential data that would need to be associated with the request
   * @return auth headers used for associating credentials to a request
   */
  public abstract void setRequestHeaders(final HttpRequest request) throws IOException;

  /**
   * Refresh credentials on the underlying credential delegate
   */
  protected abstract void refresh() throws IOException;

  /**
   * Factory method to create credentials from AWS implementations
   */
  public static HttpCredentialsAdapter fromAwsCredentails(final String awsS3Region, final String remoteHttpCache,
                                                          final String service,
                                                          final AWSCredentialsProvider awsCredsProvider) {
    return new AwsHttpCredentialsAdapter(awsS3Region, remoteHttpCache, service, awsCredsProvider, true);
  }

  /**
   * Factory method to create credentials from gauth implementations
   */
  public static HttpCredentialsAdapter fromGoogleCredentials(final Credentials creds) {
    return new HttpCredentialsAdapter() {
      @Override
      public void refresh() throws IOException {
        creds.refresh();
      }

      @Override
      public void setRequestHeaders(final HttpRequest request) throws IOException {
        final Map<String, List<String>> headers = creds.getRequestMetadata(URI.create(request.uri()));
        headers.forEach((key, values) -> request.headers().add(key, values));
      }
    };
  }
}

