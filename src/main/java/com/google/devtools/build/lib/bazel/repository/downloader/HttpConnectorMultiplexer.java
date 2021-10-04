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

package com.google.devtools.build.lib.bazel.repository.downloader;

import com.google.common.base.Function;
import com.google.common.base.Optional;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.BlazeVersionInfo;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import java.io.IOException;
import java.io.InputStream;
import java.io.InterruptedIOException;
import java.net.URI;
import java.net.URISyntaxException;
import java.net.URL;
import java.net.URLConnection;
import java.util.HashMap;
import java.util.Map;

/**
 * Class for establishing HTTP connections.
 *
 * <p>This is the most amazing way to download files ever. It makes Bazel builds as reliable as
 * Blaze builds in Google's internal hermetically sealed repository. But this class isn't just
 * reliable. It's also fast. It even works on the worst Internet connections in the farthest corners
 * of the Earth. You are just not going to believe how fast and reliable this design is. It's
 * incredible. Your builds are never going to break again due to downloads. You're going to be so
 * happy. Your developer community is going to be happy. Mr. Jenkins will be happy too. Everyone is
 * going to have such a magnificent developer experience due to the product excellence of this
 * class.
 */
@ThreadSafe
final class HttpConnectorMultiplexer {

  private static final ImmutableMap<String, String> REQUEST_HEADERS =
      ImmutableMap.of(
          "Accept-Encoding",
          "gzip",
          "User-Agent",
          "Bazel/" + BlazeVersionInfo.instance().getReleaseName());

  private final EventHandler eventHandler;
  private final HttpConnector connector;
  private final HttpStream.Factory httpStreamFactory;

  /**
   * Creates a new instance.
   *
   * <p>Instances are thread safe and can be reused.
   */
  HttpConnectorMultiplexer(
      EventHandler eventHandler, HttpConnector connector, HttpStream.Factory httpStreamFactory) {
    this.eventHandler = eventHandler;
    this.connector = connector;
    this.httpStreamFactory = httpStreamFactory;
  }

  public HttpStream connect(URL url, Optional<Checksum> checksum) throws IOException {
    return connect(url, checksum, ImmutableMap.of(), Optional.absent());
  }

  /**
   * Establishes reliable HTTP connection to a URL.
   *
   * <p>This routine supports HTTP redirects in an RFC compliant manner. It requests gzip content
   * encoding when appropriate in order to minimize bandwidth consumption when downloading
   * uncompressed files. It reports download progress. It enforces a SHA-256 checksum which
   * continues to be enforced even after this method returns.
   *
   * @param url the URL to conenct to. can be: file, http, or https
   * @param checksum checksum lazily checked on entire payload, or empty to disable
   * @return an {@link InputStream} of response payload
   * @param type extension, e.g. "tar.gz" to force on downloaded filename, or empty to not do this
   * @throws IOException if all mirrors are down and contains suppressed exception of each attempt
   * @throws InterruptedIOException if current thread is being cast into oblivion
   * @throws IllegalArgumentException if {@code urls} is empty or has an unsupported protocol
   */
  public HttpStream connect(
      URL url,
      Optional<Checksum> checksum,
      Map<URI, Map<String, String>> authHeaders,
      Optional<String> type)
      throws IOException {
    Preconditions.checkArgument(HttpUtils.isUrlSupportedByDownloader(url));
    if (Thread.interrupted()) {
      throw new InterruptedIOException();
    }
    Function<URL, ImmutableMap<String, String>> headerFunction =
        getHeaderFunction(REQUEST_HEADERS, authHeaders);
    URLConnection connection = connector.connect(url, headerFunction);
    return httpStreamFactory.create(
        connection,
        url,
        checksum,
        (Throwable cause, ImmutableMap<String, String> extraHeaders) -> {
          eventHandler.handle(
              Event.progress(String.format("Lost connection for %s due to %s", url, cause)));
          return connector.connect(
              connection.getURL(),
              newUrl ->
                  new ImmutableMap.Builder<String, String>()
                      .putAll(headerFunction.apply(newUrl))
                      .putAll(extraHeaders)
                      .build());
        },
        type);
  }

  public static Function<URL, ImmutableMap<String, String>> getHeaderFunction(
      Map<String, String> baseHeaders, Map<URI, Map<String, String>> additionalHeaders) {
    return url -> {
      ImmutableMap<String, String> headers = ImmutableMap.copyOf(baseHeaders);
      try {
        if (additionalHeaders.containsKey(url.toURI())) {
          Map<String, String> newHeaders = new HashMap<>(headers);
          newHeaders.putAll(additionalHeaders.get(url.toURI()));
          headers = ImmutableMap.copyOf(newHeaders);
        }
      } catch (URISyntaxException e) {
        // If we can't convert the URL to a URI (because it is syntactically malformed), still try
        // to do the connection, not adding authentication information as we cannot look it up.
      }
      return headers;
    };
  }
}
