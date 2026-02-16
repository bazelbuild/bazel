// Copyright 2025 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.bazel.repository.downloader;

import java.io.InputStream;
import java.net.URI;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import javax.annotation.Nullable;

/**
 * A lightweight value class representing an HTTP response, replacing {@link
 * java.net.URLConnection} in inter-component signatures.
 *
 * <p>This decouples stream-processing from transport and simplifies testing.
 */
final class DownloadResponse {

  private final URI uri;
  private final Map<String, List<String>> headers;
  private final InputStream body;

  DownloadResponse(URI uri, Map<String, List<String>> headers, InputStream body) {
    this.uri = uri;
    // Store headers in a case-insensitive map.
    TreeMap<String, List<String>> caseInsensitive =
        new TreeMap<>(String.CASE_INSENSITIVE_ORDER);
    caseInsensitive.putAll(headers);
    this.headers = Collections.unmodifiableMap(caseInsensitive);
    this.body = body;
  }

  /** Returns the final URI (after redirects). */
  URI uri() {
    return uri;
  }

  /** Returns the response body stream. */
  InputStream body() {
    return body;
  }

  /** Returns the first value for the given header name (case-insensitive), or null. */
  @Nullable
  String headerValue(String name) {
    List<String> values = headers.get(name);
    if (values == null || values.isEmpty()) {
      return null;
    }
    return values.get(0);
  }

  /** Shorthand for the Content-Encoding header value. */
  @Nullable
  String contentEncoding() {
    return headerValue("Content-Encoding");
  }
}
