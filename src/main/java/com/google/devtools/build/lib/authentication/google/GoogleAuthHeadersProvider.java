package com.google.devtools.build.lib.authentication.google;

import com.google.auth.Credentials;
import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.runtime.AuthHeadersProvider;
import java.io.IOException;
import java.net.URI;
import java.util.List;
import java.util.Map;

class GoogleAuthHeadersProvider implements AuthHeadersProvider {

  private final Credentials credentials;

  public GoogleAuthHeadersProvider(Credentials credentials) {
    this.credentials = Preconditions.checkNotNull(credentials, "credentials");
  }

  @Override
  public String getType() {
    return credentials.getAuthenticationType();
  }

  @Override
  public Map<String, List<String>> getRequestHeaders(URI uri) throws IOException {
    return credentials.getRequestMetadata(uri);
  }

  @Override
  public void refresh() throws IOException {
    credentials.refresh();
  }

  @Override
  public boolean isEnabled() {
    return true;
  }
}
