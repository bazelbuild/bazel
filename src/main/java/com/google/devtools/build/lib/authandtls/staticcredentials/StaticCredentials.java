package com.google.devtools.build.lib.authandtls.staticcredentials;

import com.google.auth.Credentials;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableMap;
import java.io.IOException;
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
  public Map<String, List<String>> getRequestMetadata(URI uri) throws IOException {
    Preconditions.checkNotNull(uri);

    return credentials.getOrDefault(uri, ImmutableMap.of());
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
}
