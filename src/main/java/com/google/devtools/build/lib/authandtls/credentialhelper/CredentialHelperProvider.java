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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.base.Strings;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.vfs.Path;
import com.google.errorprone.annotations.Immutable;
import java.io.IOException;
import java.net.IDN;
import java.net.URI;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.regex.Pattern;

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
    this.defaultHelper = Preconditions.checkNotNull(defaultHelper);
    this.hostToHelper = Preconditions.checkNotNull(hostToHelper);
    this.suffixToHelper = Preconditions.checkNotNull(suffixToHelper);
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
    Preconditions.checkNotNull(uri);

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
    Preconditions.checkNotNull(host);

    return Optional.ofNullable(hostToHelper.get(host));
  }

  private Optional<Path> findWildcardCredentialHelper(String host) {
    Preconditions.checkNotNull(host);

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
    private static final Pattern DOMAIN_PATTERN =
        Pattern.compile("(\\*|[-a-zA-Z0-9]+)(\\.[-a-zA-Z0-9]+)+");

    private Optional<Path> defaultHelper = Optional.empty();
    private final Map<String, Path> hostToHelper = new HashMap<>();
    private final Map<String, Path> suffixToHelper = new HashMap<>();

    private void checkHelper(Path path) throws IOException {
      Preconditions.checkNotNull(path);
      Preconditions.checkArgument(
          path.isExecutable(), "Credential helper %s is not executable", path);
    }

    /**
     * Adds a default credential helper to use for all {@link URI}s that don't specify a more
     * specific credential helper.
     */
    public Builder add(Path helper) throws IOException {
      checkHelper(helper);

      defaultHelper = Optional.of(helper);
      return this;
    }

    /**
     * Adds a credential helper to use for all {@link URI}s matching the provided pattern.
     *
     * <p>As of 2022-06-20, only matching based on (wildcard) domain name is supported.
     *
     * <p>If {@code pattern} starts with {@code *.}, it is considered a wildcard pattern matching
     * all subdomains in addition to the domain itself. For example {@code *.example.com} would
     * match {@code example.com}, {@code foo.example.com}, {@code bar.example.com}, {@code
     * baz.bar.example.com} and so on, but not anything that isn't a subdomain of {@code
     * example.com}.
     */
    public Builder add(String pattern, Path helper) throws IOException {
      Preconditions.checkNotNull(pattern);
      checkHelper(helper);

      String punycodePattern = toPunycodePattern(pattern);
      Preconditions.checkArgument(
          DOMAIN_PATTERN.matcher(punycodePattern).matches(),
          "Pattern '%s' is not a valid (wildcard) DNS name",
          pattern);

      if (pattern.startsWith("*.")) {
        suffixToHelper.put(punycodePattern.substring(2), helper);
      } else {
        hostToHelper.put(punycodePattern, helper);
      }

      return this;
    }

    /** Converts a pattern to Punycode (see https://en.wikipedia.org/wiki/Punycode). */
    private final String toPunycodePattern(String pattern) {
      Preconditions.checkNotNull(pattern);

      try {
        return IDN.toASCII(pattern);
      } catch (IllegalArgumentException e) {
        throw new IllegalArgumentException(
            String.format(Locale.US, "Could not convert '%s' to punycode", pattern), e);
      }
    }

    public CredentialHelperProvider build() {
      return new CredentialHelperProvider(
          defaultHelper, ImmutableMap.copyOf(hostToHelper), ImmutableMap.copyOf(suffixToHelper));
    }
  }
}
