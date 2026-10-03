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
import com.google.common.base.Ascii;
import com.google.common.base.Preconditions;
import com.google.common.base.Strings;
import com.google.common.collect.ImmutableMap;
import java.net.URI;
import java.util.List;
import java.util.Map;

/** Implementation of {@link Credentials} which provides a static set of credentials. */
public final class StaticCredentials extends Credentials {
  public static final StaticCredentials EMPTY = new StaticCredentials(ImmutableMap.of());

  private final ImmutableMap<URI, Map<String, List<String>>> credentials;

  public StaticCredentials(Map<URI, Map<String, List<String>>> credentials) {
    Preconditions.checkNotNull(credentials);

    this.credentials = ImmutableMap.copyOf(credentials);
  }

  @Override
  public String getAuthenticationType() {
    return "static";
  }

  @Override
  public Map<String, List<String>> getRequestMetadata(URI uri) {
    Preconditions.checkNotNull(uri);

    Map<String, List<String>> metadata = credentials.get(uri);
    if (metadata != null) {
      return metadata;
    }
    // On a same-host, same-scheme redirect to a different path the request URI is not an exact key.
    // Fall back to credentials registered for the same endpoint, mirroring the per-host behavior of
    // `curl --netrc` and `wget`. Credentials are never reused across hosts, schemes or ports.
    for (Map.Entry<URI, Map<String, List<String>>> entry : credentials.entrySet()) {
      if (sameEndpoint(uri, entry.getKey())) {
        return entry.getValue();
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

  /**
   * Returns whether {@code a} and {@code b} address the same endpoint, i.e. have an equal port and a
   * case-insensitively equal scheme and host (per RFC 3986).
   */
  private static boolean sameEndpoint(URI a, URI b) {
    return a.getPort() == b.getPort()
        && Ascii.equalsIgnoreCase(
            Strings.nullToEmpty(a.getScheme()), Strings.nullToEmpty(b.getScheme()))
        && Ascii.equalsIgnoreCase(Strings.nullToEmpty(a.getHost()), Strings.nullToEmpty(b.getHost()));
  }
}
