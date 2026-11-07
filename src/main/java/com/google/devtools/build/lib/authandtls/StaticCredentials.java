// Copyright 2022 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.authandtls;

import com.google.auth.Credentials;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableMap;
import java.net.URI;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import javax.annotation.Nullable;

/**
 * Implementation of {@link Credentials} which provides a static set of credentials.
 *
 * <p>Credentials are looked up by exact {@link URI} first. If there is no exact match, the
 * credentials registered for the same origin (scheme, host and port) are used, provided they are
 * unambiguous. This keeps requests authenticated when the server redirects to another path on the
 * same host, and never leaks credentials to a different host, port or scheme.
 */
public final class StaticCredentials extends Credentials {
  public static final StaticCredentials EMPTY = new StaticCredentials(ImmutableMap.of());

  private final ImmutableMap<URI, Map<String, List<String>>> credentials;

  /**
   * Credentials by origin, only for origins whose URIs all map to the same credentials. Origins
   * with conflicting credentials are left out so that no guess is made between them.
   */
  private final ImmutableMap<String, Map<String, List<String>>> credentialsByOrigin;

  public StaticCredentials(Map<URI, Map<String, List<String>>> credentials) {
    Preconditions.checkNotNull(credentials);

    this.credentials = ImmutableMap.copyOf(credentials);
    this.credentialsByOrigin = indexByOrigin(this.credentials);
  }

  @Override
  public String getAuthenticationType() {
    return "static";
  }

  @Override
  public Map<String, List<String>> getRequestMetadata(URI uri) {
    Preconditions.checkNotNull(uri);

    Map<String, List<String>> exact = credentials.get(uri);
    if (exact != null) {
      return exact;
    }

    String origin = originOf(uri);
    if (origin != null) {
      Map<String, List<String>> sameOrigin = credentialsByOrigin.get(origin);
      if (sameOrigin != null) {
        return sameOrigin;
      }
    }
    return ImmutableMap.of();
  }

  @Override
  public boolean hasRequestMetadata() {
    return true;
  }

  @Override
  public boolean hasRequestMetadataOnly() {
    return true;
  }

  @Override
  public void refresh() {
    // Can't refresh static credentials.
  }

  private static ImmutableMap<String, Map<String, List<String>>> indexByOrigin(
      Map<URI, Map<String, List<String>>> credentials) {
    Map<String, Map<String, List<String>>> byOrigin = new HashMap<>();
    Set<String> ambiguous = new HashSet<>();
    for (Map.Entry<URI, Map<String, List<String>>> entry : credentials.entrySet()) {
      String origin = originOf(entry.getKey());
      if (origin == null) {
        continue;
      }
      Map<String, List<String>> previous = byOrigin.putIfAbsent(origin, entry.getValue());
      if (previous != null && !previous.equals(entry.getValue())) {
        ambiguous.add(origin);
      }
    }
    byOrigin.keySet().removeAll(ambiguous);
    return ImmutableMap.copyOf(byOrigin);
  }

  /** Returns "scheme://host:port" (normalized), or null if the URI has no scheme or host. */
  @Nullable
  private static String originOf(URI uri) {
    String scheme = uri.getScheme();
    String host = uri.getHost();
    if (scheme == null || host == null) {
      return null;
    }
    scheme = scheme.toLowerCase(Locale.ROOT);
    int port = uri.getPort();
    if (port == -1) {
      port =
          switch (scheme) {
            case "http" -> 80;
            case "https" -> 443;
            default -> -1;
          };
    }
    return scheme + "://" + host.toLowerCase(Locale.ROOT) + ":" + port;
  }
}