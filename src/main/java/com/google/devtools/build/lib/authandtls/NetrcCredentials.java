package com.google.devtools.build.lib.authandtls;

import com.google.auth.Credentials;
import com.google.devtools.build.lib.authandtls.Netrc.Credential;
import java.io.IOException;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.util.Base64;
import java.util.Collections;
import java.util.List;
import java.util.Map;

public class NetrcCredentials extends Credentials {

  private final Netrc netrc;

  public NetrcCredentials(Netrc netrc) {
    this.netrc = netrc;
  }

  @Override
  public String getAuthenticationType() {
    return "netrc";
  }

  @Override
  public Map<String, List<String>> getRequestMetadata(URI uri) throws IOException {
    Credential credential = netrc.getCredential(uri.getHost());
    if (credential != null) {
      String credentialString = credential.login() + ":" + credential.password();
      String token = "Basic " + Base64.getEncoder()
          .encodeToString(credentialString.getBytes(StandardCharsets.UTF_8));
      return Collections.singletonMap(
          "Authorization",
          Collections.singletonList(token)
      );
    } else {
      return Collections.emptyMap();
    }
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

  }
}
