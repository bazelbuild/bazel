package com.google.devtools.build.lib.authandtls.credentialhelper;

import com.google.auth.Credentials;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableMap;
import java.io.IOException;
import java.net.URI;
import java.util.List;
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

    Optional<CredentialHelper> maybeCredentialHelper = credentialHelperProvider.findCredentialHelper(uri);
    if (!maybeCredentialHelper.isPresent()) {
      return Optional.empty();
    }
    CredentialHelper credentialHelper = maybeCredentialHelper.get();

    GetCredentialsResponse response;
    try {
      response = credentialHelper.getCredentials(credentialHelperEnvironment, uri);
    } catch (InterruptedException e) {
      throw new RuntimeException(e);
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
    if (fallbackCredentials.isPresent()) {
      fallbackCredentials.get().refresh();
    }
  }
}
