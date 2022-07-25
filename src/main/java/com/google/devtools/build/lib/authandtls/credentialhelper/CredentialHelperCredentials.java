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

import com.github.benmanes.caffeine.cache.CacheLoader;
import com.github.benmanes.caffeine.cache.Caffeine;
import com.github.benmanes.caffeine.cache.LoadingCache;
import com.google.auth.Credentials;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableMap;
import java.io.IOException;
import java.net.URI;
import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import javax.annotation.Nullable;

/**
 * Implementation of {@link Credentials} which fetches credentials by invoking a {@code credential
 * helper} as subprocess, falling back to another {@link Credentials} if no suitable helper exists.
 */
public class CredentialHelperCredentials extends Credentials {
  private final Optional<Credentials> fallbackCredentials;

  private final LoadingCache<URI, GetCredentialsResponse> credentialCache;

  public CredentialHelperCredentials(
      CredentialHelperProvider credentialHelperProvider,
      CredentialHelperEnvironment credentialHelperEnvironment,
      Optional<Credentials> fallbackCredentials,
      Duration cacheTimeout) {
    Preconditions.checkNotNull(credentialHelperProvider);
    Preconditions.checkNotNull(credentialHelperEnvironment);
    this.fallbackCredentials = Preconditions.checkNotNull(fallbackCredentials);
    Preconditions.checkNotNull(cacheTimeout);
    Preconditions.checkArgument(
        !cacheTimeout.isNegative() && !cacheTimeout.isZero(),
        "Cache timeout must be greater than 0");

    credentialCache =
        Caffeine.newBuilder()
            .expireAfterWrite(cacheTimeout)
            .build(
                new CredentialHelperCacheLoader(
                    credentialHelperProvider, credentialHelperEnvironment));
  }

  @Override
  public String getAuthenticationType() {
    if (fallbackCredentials.isPresent()) {
      return "credential-helper-with-fallback-" + fallbackCredentials.get().getAuthenticationType();
    }

    return "credential-helper";
  }

  @Override
  public Map<String, List<String>> getRequestMetadata(URI uri) throws IOException {
    Preconditions.checkNotNull(uri);

    Optional<Map<String, List<String>>> credentials = getRequestMetadataFromCredentialHelper(uri);
    if (credentials.isPresent()) {
      return credentials.get();
    }

    if (fallbackCredentials.isPresent()) {
      return fallbackCredentials.get().getRequestMetadata(uri);
    }

    return ImmutableMap.of();
  }

  @SuppressWarnings("unchecked") // Map<String, ImmutableList<String>> to Map<String<List<String>>
  private Optional<Map<String, List<String>>> getRequestMetadataFromCredentialHelper(URI uri) {
    Preconditions.checkNotNull(uri);

    GetCredentialsResponse response = credentialCache.get(uri);

    return Optional.ofNullable(response).map(value -> (Map) value.getHeaders());
  }

  @Override
  public boolean hasRequestMetadata() {
    return true;
  }

  @Override
  public boolean hasRequestMetadataOnly() {
    return false;
  }

  @Override
  public void refresh() throws IOException {
    if (fallbackCredentials.isPresent()) {
      fallbackCredentials.get().refresh();
    }

    credentialCache.invalidateAll();
  }

  private static final class CredentialHelperCacheLoader
      implements CacheLoader<URI, GetCredentialsResponse> {
    private final CredentialHelperProvider credentialHelperProvider;
    private final CredentialHelperEnvironment credentialHelperEnvironment;

    public CredentialHelperCacheLoader(
        CredentialHelperProvider credentialHelperProvider,
        CredentialHelperEnvironment credentialHelperEnvironment) {
      this.credentialHelperProvider = Preconditions.checkNotNull(credentialHelperProvider);
      this.credentialHelperEnvironment = Preconditions.checkNotNull(credentialHelperEnvironment);
    }

    @Nullable
    @Override
    public GetCredentialsResponse load(URI uri) throws IOException, InterruptedException {
      Preconditions.checkNotNull(uri);

      Optional<CredentialHelper> maybeCredentialHelper =
          credentialHelperProvider.findCredentialHelper(uri);
      if (maybeCredentialHelper.isEmpty()) {
        return null;
      }
      CredentialHelper credentialHelper = maybeCredentialHelper.get();

      return credentialHelper.getCredentials(credentialHelperEnvironment, uri);
    }
  }
}
