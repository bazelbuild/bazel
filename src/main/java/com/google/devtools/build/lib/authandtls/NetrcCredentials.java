// Copyright 2020 The Bazel Authors. All rights reserved.
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

/**
 * Subclass of {@link Credentials} which uses username and password from {@link Netrc} to provide
 * request metadata.
 */
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
