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

package com.google.devtools.build.lib.authandtls.credentialhelper;

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Strings;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.vfs.Path;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import com.google.errorprone.annotations.Immutable;
import java.io.IOException;
import java.net.URI;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;

/**
 * A provider for {@link CredentialHelper}s.
 *
 * <p>This class is used to find the right {@link CredentialHelper} for a {@link URI}, using the
 * most specific match.
 */
@Immutable
public final class CredentialHelperProvider {
  // `Path` is immutable, but not annotated.
  @SuppressWarnings("Immutable")
  private final Optional<Path> defaultHelper;

  @SuppressWarnings("Immutable")
  private final ImmutableMap<String, Path> hostToHelper;

  @SuppressWarnings("Immutable")
  private final ImmutableMap<String, Path> suffixToHelper;

  private CredentialHelperProvider(
      Optional<Path> defaultHelper,
      ImmutableMap<String, Path> hostToHelper,
      ImmutableMap<String, Path> suffixToHelper) {
    this.defaultHelper = checkNotNull(defaultHelper);
    this.hostToHelper = checkNotNull(hostToHelper);
    this.suffixToHelper = checkNotNull(suffixToHelper);
  }

  /**
   * Returns {@link CredentialHelper} to use for getting credentials for connection to the provided
   * {@link URI}.
   *
   * @param uri The {@link URI} to get a credential helper for.
   * @return The {@link CredentialHelper}, or nothing if no {@link CredentialHelper} is configured
   *     for the provided {@link URI}.
   */
  public Optional<CredentialHelper> findCredentialHelper(URI uri) {
    checkNotNull(uri);

    String host = uri.getHost();
    if (Strings.isNullOrEmpty(host)) {
      // Some URIs (e.g. unix://) legitimately have no host component.
      // Use the default helper if one is provided.
      return defaultHelper.map(CredentialHelper::new);
    }

    Optional<Path> credentialHelper =
        findHostCredentialHelper(host)
            .or(() -> findWildcardCredentialHelper(host))
            .or(() -> defaultHelper);
    return credentialHelper.map(CredentialHelper::new);
  }

  private Optional<Path> findHostCredentialHelper(String host) {
    checkNotNull(host);

    return Optional.ofNullable(hostToHelper.get(host));
  }

  private Optional<Path> findWildcardCredentialHelper(String host) {
    checkNotNull(host);

    return Optional.ofNullable(suffixToHelper.get(host))
        .or(
            () -> {
              Optional<String> subdomain = parentDomain(host);
              if (subdomain.isEmpty()) {
                return Optional.empty();
              }
              return findWildcardCredentialHelper(subdomain.get());
            });
  }

  /**
   * Returns the parent domain of the provided domain (e.g., {@code foo.example.com} for {@code
   * bar.foo.example.com}).
   */
  @VisibleForTesting
  static Optional<String> parentDomain(String domain) {
    int dot = domain.indexOf('.');
    if (dot < 0) {
      // We reached the last segment, end.
      return Optional.empty();
    }

    return Optional.of(domain.substring(dot + 1));
  }

  /** Returns a new builder for a {@link CredentialHelperProvider}. */
  public static Builder builder() {
    return new Builder();
  }

  /** Builder for {@link CredentialHelperProvider}. */
  public static final class Builder {
    private Optional<Path> defaultHelper = Optional.empty();
    private final Map<String, Path> hostToHelper = new HashMap<>();
    private final Map<String, Path> suffixToHelper = new HashMap<>();

    /**
     * Adds a default credential helper to use for all {@link URI}s that don't specify a more
     * specific credential helper.
     */
    @CanIgnoreReturnValue
    public Builder add(Path helper) throws IOException {
      checkNotNull(helper);
      defaultHelper = Optional.of(helper);
      return this;
    }

    /**
     * Adds a credential helper to use for all {@link URI}s matching the provided pattern.
     *
     * <p>If {@code pattern} starts with a {@code *.} wildcard, it matches every subdomain in
     * addition to the domain itself. For example {@code *.example.com} would match {@code
     * example.com}, {@code foo.example.com}, {@code bar.example.com}, {@code baz.bar.example.com}
     * and so on, but not anything that isn't a subdomain of {@code example.com}.
     *
     * <p>More complex wildcard patterns are not supported.
     */
    @CanIgnoreReturnValue
    public Builder add(String pattern, Path helper) throws IOException {
      checkNotNull(pattern);
      checkNotNull(helper);

      // The pattern has already been normalized during options parsing.
      if (pattern.startsWith("*.")) {
        suffixToHelper.put(pattern.substring(2), helper);
      } else {
        hostToHelper.put(pattern, helper);
      }

      return this;
    }

    public CredentialHelperProvider build() {
      return new CredentialHelperProvider(
          defaultHelper, ImmutableMap.copyOf(hostToHelper), ImmutableMap.copyOf(suffixToHelper));
    }
  }
}
