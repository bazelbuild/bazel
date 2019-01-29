// Copyright 2019 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.runtime;

import java.io.IOException;
import java.net.URI;
import java.util.List;
import java.util.Map;
import javax.annotation.concurrent.ThreadSafe;

/**
 * Generic interface to provide authentication headers for http/grpc requests to bazel modules.
 *
 * <p>Implementations need to be thread-safe.
 */
@ThreadSafe
public interface AuthHeadersProvider {

  /** Returns the type of authentication mechanism used i.e. oauth. */
  String getType();

  /**
   * Returns request headers necessary for authentication to be added to the http/grpc request.
   *
   * <p>For auth tokens with a TTL attached an implementation is expected to transparently refresh
   * the tokens and to always return headers with sufficient TTL left to complete the request.
   */
  Map<String, List<String>> getRequestHeaders(URI uri) throws IOException;

  /**
   * Refreshes the authentication credentials i.e. if an access token expired.
   *
   * <p>This method should be called by clients if a request fails due to expired authentication
   * credentials. Clients may then retry the request by retrieving new headers via {@link
   * #getRequestHeaders(URI)}.
   */
  void refresh() throws IOException;

  /**
   * Returns {@code true} if this provider is enabled and can provide auth headers.
   *
   * <p>This method is a necessity due to the way blaze modules work.
   */
  boolean isEnabled();
}
