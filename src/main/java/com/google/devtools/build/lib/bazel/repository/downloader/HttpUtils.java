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

import static com.google.devtools.build.lib.util.StringEncoding.internalToPlatform;
import static com.google.devtools.build.lib.util.StringEncoding.internalToUnicode;
import static com.google.devtools.build.lib.util.StringEncoding.unicodeToInternal;

import com.google.common.base.Ascii;
import com.google.common.base.MoreObjects;
import com.google.common.base.Preconditions;
import java.io.File;
import java.io.IOException;
import java.net.HttpURLConnection;
import java.net.URI;
import java.net.URISyntaxException;
import java.net.URLConnection;
import java.util.Collection;
import java.util.Objects;

/** HTTP utilities. */
public final class HttpUtils {

  /** Returns {@code true} if {@code uri} is supported by {@link HttpDownloader}. */
  public static boolean isUrlSupportedByDownloader(URI uri) {
    return isHttp(uri) || isProtocol(uri, "file");
  }

  static boolean isHttp(URI uri) {
    return isProtocol(uri, "http") || isProtocol(uri, "https");
  }

  static boolean isProtocol(URI uri, String protocol) {
    // An implementation should accept uppercase letters as equivalent to lowercase in scheme names
    // (e.g., allow "HTTP" as well as "http") for the sake of robustness. Quoth RFC3986 § 3.1
    return Ascii.equalsIgnoreCase(protocol, uri.getScheme());
  }

  /** Parses a URL given as an internal string into an ASCII-only {@link URI}. */
  public static URI parseUrl(String url) throws URISyntaxException {
    return toAsciiUri(parseUri(url));
  }

  /** Parses a URI given as an internal string. */
  public static URI parseUri(String uri) throws URISyntaxException {
    try {
      return new URI(internalToUnicode(uri));
    } catch (URISyntaxException e) {
      throw new URISyntaxException(uri, e.getReason());
    }
  }

  /** Percent-encodes non-ASCII characters, which the JDK would send as ISO-8859-1. */
  static URI toAsciiUri(URI uri) {
    return URI.create(uri.toASCIIString());
  }

  /** Opens a {@code file:} URL, whose path the JDK decodes as Unicode. */
  static URLConnection openFileConnection(URI url) throws IOException {
    String path = url.getPath();
    if (path != null
        && (url.getHost() == null || Ascii.equalsIgnoreCase(url.getHost(), "localhost"))) {
      String platformPath = internalToPlatform(unicodeToInternal(path));
      if (!platformPath.equals(path)) {
        return new File(platformPath).toURI().toURL().openConnection();
      }
    }
    return url.toURL().openConnection();
  }

  static void checkUrlsArgument(Collection<URI> uris) {
    Preconditions.checkArgument(!uris.isEmpty(), "urls list empty");
    for (URI uri : uris) {
      Preconditions.checkArgument(isUrlSupportedByDownloader(uri), "unsupported protocol: %s", uri);
    }
  }

  static String getExtension(String path) {
    int index = path.lastIndexOf('.');
    if (index == -1) {
      return "";
    }
    return Ascii.toLowerCase(path.substring(index + 1));
  }

  static URI getLocation(HttpURLConnection connection) throws IOException {
    String newLocation = connection.getHeaderField("Location");
    if (newLocation == null) {
      throw new IOException("Remote redirect missing Location.");
    }
    // The JDK decodes response headers as ISO-8859-1, which results in an internal string.
    URI result = mergeUrls(URI.create(internalToUnicode(newLocation)), toUri(connection));
    if (!isHttp(result)) {
      throw new IOException("Bad Location: " + newLocation);
    }
    return toAsciiUri(result);
  }

  private static URI mergeUrls(URI preferred, URI original) throws IOException {
    // Try to short cut to preferred to preserve the original presentation of the
    // quoting (as a call to the structured URI constructor puts quoting into a canonical form).
    // This is necessary as some sites rely on the precise presentation for the authentication
    // scheme of their redirect URLs.
    if (preferred.getHost() != null
        && preferred.getScheme() != null
        && (preferred.getFragment() != null || original.getFragment() == null)
        // Forward user info to the same origin.
        && (preferred.getUserInfo() != null
            || original.getUserInfo() == null
            || !(Objects.equals(preferred.getHost(), original.getHost())
                && preferred.getPort() == original.getPort()))) {
      // In this case we obviously do not inherit anything from the original URL, as all inheritable
      // fields are either set explicitly or not present in the original either. Therefore, it is
      // safe to short cut.
      return preferred;
    }

    // If the Location value provided in a 3xx (Redirection) response does not have a fragment
    // component, a user agent MUST process the redirection as if the value inherits the fragment
    // component of the URI reference used to generate the request target (i.e., the redirection
    // inherits the original reference's fragment, if any). Quoth RFC7231 § 7.1.2
    String protocol = MoreObjects.firstNonNull(preferred.getScheme(), original.getScheme());
    String userInfo = preferred.getUserInfo();
    String host = preferred.getHost();
    int port;
    if (host == null) {
      host = original.getHost();
      port = original.getPort();
      userInfo = original.getUserInfo();
    } else {
      port = preferred.getPort();
      if (userInfo == null && host.equals(original.getHost()) && port == original.getPort()) {
        userInfo = original.getUserInfo();
      }
    }
    String path = preferred.getPath();
    String query = preferred.getQuery();
    String fragment = preferred.getFragment();
    if (fragment == null) {
      fragment = original.getFragment();
    }
    URI result;
    try {
      result = new URI(protocol, userInfo, host, port, path, query, fragment);
    } catch (URISyntaxException e) {
      throw new IOException("Could not merge " + preferred + " into " + original, e);
    }
    return result;
  }

  /**
   * Converts a {@link URLConnection}'s URL to a {@link URI}. Since the URL comes from an active
   * connection, it should always be a valid URI.
   */
  static URI toUri(URLConnection connection) {
    try {
      return connection.getURL().toURI();
    } catch (URISyntaxException e) {
      throw new IllegalStateException("Invalid URI from connection URL: " + connection.getURL(), e);
    }
  }

  private HttpUtils() {}
}
