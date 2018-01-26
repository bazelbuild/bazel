package com.google.devtools.build.lib.remote.blobstore;

import com.google.auth.Credentials;
import java.io.IOException;
import java.net.URI;
import java.util.Base64;
import java.util.Collections;
import java.util.List;
import java.util.Map;

/** Credentials that support basic auth that comes from a userInfo string */
class BasicAuthCredentials extends Credentials {
  private final String base64EndodedUserInfo;

  BasicAuthCredentials(String userInfo) {
    base64EndodedUserInfo = Base64.getEncoder().encodeToString(userInfo.getBytes());
  }

  @Override public String getAuthenticationType() {
    return "Basic";
  }

  @Override public Map<String, List<String>> getRequestMetadata(URI uri)
      throws IOException {
    return Collections.singletonMap("Authorization",
        Collections.singletonList("Basic " + base64EndodedUserInfo));
  }

  @Override public boolean hasRequestMetadata() {
    return true;
  }

  @Override public boolean hasRequestMetadataOnly() {
    return true;
  }

  @Override public void refresh() throws IOException {
  }
}
