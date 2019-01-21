package com.google.devtools.build.lib.grpc;

import com.google.auth.Credentials;
import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.runtime.AuthHeadersProvider;
import java.io.IOException;
import java.net.URI;
import java.util.List;
import java.util.Map;

class CredentialsAdapter extends Credentials  {

  private final AuthHeadersProvider authHeadersProvider;

  public CredentialsAdapter(AuthHeadersProvider authHeadersProvider) {
    this.authHeadersProvider =
        Preconditions.checkNotNull(authHeadersProvider, "authHeadersProvider");
  }

  @Override
  public String getAuthenticationType() {
    return authHeadersProvider.getType();
  }

  @Override
  public Map<String, List<String>> getRequestMetadata(URI uri) throws IOException {
    return authHeadersProvider.getRequestHeaders(uri);
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
  public void refresh() throws IOException {
    authHeadersProvider.refresh();
  }
}
