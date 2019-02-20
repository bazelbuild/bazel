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

import com.google.common.base.Ascii;
import com.google.common.base.MoreObjects;
import com.google.common.base.Preconditions;
import java.io.IOException;
import java.net.HttpURLConnection;
import java.net.MalformedURLException;
import java.net.URI;
import java.net.URISyntaxException;
import java.net.URL;
import java.util.Collection;

/** HTTP utilities. */
public final class HttpUtils {

  /** Returns {@code true} if {@code url} is supported by {@link HttpDownloader}. */
  public static boolean isUrlSupportedByDownloader(URL url) {
    return isHttp(url) || isProtocol(url, "file");
  }

  static boolean isHttp(URL url) {
    return isProtocol(url, "http") || isProtocol(url, "https");
  }

  static boolean isProtocol(URL url, String protocol) {
    // An implementation should accept uppercase letters as equivalent to lowercase in scheme names
    // (e.g., allow "HTTP" as well as "http") for the sake of robustness. Quoth RFC3986 § 3.1
    return Ascii.equalsIgnoreCase(protocol, url.getProtocol());
  }

  static void checkUrlsArgument(Collection<URL> urls) {
    Preconditions.checkArgument(!urls.isEmpty(), "urls list empty");
    for (URL url : urls) {
      Preconditions.checkArgument(isUrlSupportedByDownloader(url), "unsupported protocol: %s", url);
    }
  }

  static String getExtension(String path) {
    int index = path.lastIndexOf('.');
    if (index == -1) {
      return "";
    }
    return Ascii.toLowerCase(path.substring(index + 1));
  }

  static URL getLocation(HttpURLConnection connection) throws IOException {
    String newLocation = connection.getHeaderField("Location");
    if (newLocation == null) {
      throw new IOException("Remote redirect missing Location.");
    }
    URL result = mergeUrls(URI.create(newLocation), connection.getURL());
    if (!isHttp(result)) {
      throw new IOException("Bad Location: " + newLocation);
    }
    return result;
  }

  private static URL mergeUrls(URI preferred, URL original) throws IOException {
    // Try to short cut to preferred.toURL() to preserve the original presentation of the
    // quoting (as a call to the structed URI constructor puts quoting into a canocial form).
    // This is necessary as some sites rely on the precise presentation for the authentication
    // scheme of their redirect URLs.
    if (preferred.getHost() != null
        && preferred.getScheme() != null
        && (preferred.getUserInfo() != null || original.getUserInfo() == null)
        && (preferred.getFragment() != null || original.getRef() == null)) {
      // In this case we obviously do not inherit anything from the original URL, as all inheritable
      // fields are either set explicitly or not present in the original either. Therefore, it is
      // safe to short cut.
      return preferred.toURL();
    }

    // If the Location value provided in a 3xx (Redirection) response does not have a fragment
    // component, a user agent MUST process the redirection as if the value inherits the fragment
    // component of the URI reference used to generate the request target (i.e., the redirection
    // inherits the original reference's fragment, if any). Quoth RFC7231 § 7.1.2
    String protocol = MoreObjects.firstNonNull(preferred.getScheme(), original.getProtocol());
    String userInfo = preferred.getUserInfo();
    String host = preferred.getHost();
    int port;
    if (host == null) {
      host = original.getHost();
      port = original.getPort();
      userInfo = original.getUserInfo();
    } else {
      port = preferred.getPort();
      if (userInfo == null
          && host.equals(original.getHost())
          && port == original.getPort()) {
        userInfo = original.getUserInfo();
      }
    }
    String path = preferred.getPath();
    String query = preferred.getQuery();
    String fragment = preferred.getFragment();
    if (fragment == null) {
      fragment = original.getRef();
    }
    URL result;
    try {
      result = new URI(protocol, userInfo, host, port, path, query, fragment).toURL();
    } catch (URISyntaxException | MalformedURLException e) {
      throw new IOException("Could not merge " + preferred + " into " + original, e);
    }
    return result;
  }

  private HttpUtils() {}
}
