package com.google.devtools.build.lib.authandtls.credentialhelper;

import com.github.benmanes.caffeine.cache.Cache;
import com.github.benmanes.caffeine.cache.CacheLoader;
import com.github.benmanes.caffeine.cache.Caffeine;
import com.github.benmanes.caffeine.cache.LoadingCache;
import com.google.auth.Credentials;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.events.Event;
import java.io.IOException;
import java.net.URI;
import java.time.Duration;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;

/**
 * Implementation of {@link Credentials} which fetches credentials by invoking a {@code credential
 * helper} as subprocess.
 */
public class CredentialHelperCredentials extends Credentials {
  private final CredentialHelperProvider credentialHelperProvider;
  private final CredentialHelperEnvironment credentialHelperEnvironment;
  private final Optional<Credentials> fallbackCredentials;

  private final LoadingCache<URI, GetCredentialsResponse> cache =
      Caffeine.newBuilder()
          .expireAfterWrite(Duration.ofMinutes(5))
          .build(new CacheLoader<URI, GetCredentialsResponse>() {
    @Override
    public GetCredentialsResponse load(URI uri) throws IOException, InterruptedException {
      Preconditions.checkNotNull(uri);

      Optional<GetCredentialsResponse> response = loadInternal(uri);
      if (response.isPresent()) {
        return response.get();
      }

      return null;
    }

    private Optional<GetCredentialsResponse> loadInternal(URI uri) throws IOException, InterruptedException {
      Preconditions.checkNotNull(uri);

      credentialHelperEnvironment.getEventReporter().handle(Event.debug(String.format(Locale.US, "Invoking credential helper for %s", uri)));

      Optional<CredentialHelper> maybeCredentialHelper = credentialHelperProvider.findCredentialHelper(uri);
      if (!maybeCredentialHelper.isPresent()) {
        return Optional.empty();
      }
      CredentialHelper credentialHelper = maybeCredentialHelper.get();

      return Optional.of(credentialHelper.getCredentials(credentialHelperEnvironment, uri));
    }
  });

  public CredentialHelperCredentials(
      CredentialHelperProvider credentialHelperProvider,
      CredentialHelperEnvironment credentialHelperEnvironment,
      Optional<Credentials> fallbackCredentials) {
    this.credentialHelperProvider = Preconditions.checkNotNull(credentialHelperProvider);
    this.credentialHelperEnvironment = Preconditions.checkNotNull(credentialHelperEnvironment);
    this.fallbackCredentials = Preconditions.checkNotNull(fallbackCredentials);
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

    Optional<Map<String, List<String>>> credentials =
        getRequestMetadataFromCredentialHelper(uri);
    if (credentials.isPresent()) {
      return credentials.get();
    }

    if (fallbackCredentials.isPresent()) {
      return fallbackCredentials.get().getRequestMetadata(uri);
    }

    return ImmutableMap.of();
  }

  private Optional<Map<String, List<String>>> getRequestMetadataFromCredentialHelper(
      URI uri) throws IOException {
    Preconditions.checkNotNull(uri);

    GetCredentialsResponse response = cache.get(uri);
    if (response == null) {
      return Optional.empty();
    }

    // The cast is needed to convert value type of map from `ImmutableList` to `List`.
    return Optional.of((Map)response.getHeaders());
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
    credentialHelperEnvironment.getEventReporter().handle(Event.debug("Invalidating all cached credentials"));

    if (fallbackCredentials.isPresent()) {
      fallbackCredentials.get().refresh();
    }

    cache.invalidateAll();
  }
}
