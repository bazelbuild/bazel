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

import static com.google.common.truth.Truth.assertThat;

import com.google.common.base.Strings;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.authandtls.Netrc.Credential;
import java.io.IOException;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.util.Base64;
import java.util.List;
import java.util.Map;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for {@link NetrcCredentials}.
 */
@RunWith(JUnit4.class)
public class NetrcCredentialsTest {

  private final static String fooMachine = "foo.example.org";
  private final static Credential fooCredential = Credential.builder(fooMachine)
      .setLogin("foouser")
      .setPassword("foopass").build();
  private final static String barMachine = "bar.example.org";
  private final static Credential defaultCredential = Credential.builder("default")
      .setLogin("defaultuser")
      .setPassword("defaultpass").build();

  @Test
  public void shouldWorkWithEmptyNetrc() throws IOException {
    Netrc netrc = new Netrc(null, ImmutableMap.of());
    NetrcCredentials netrcCredentials = new NetrcCredentials(netrc);

    Map<String, List<String>> requestMetadata = netrcCredentials
        .getRequestMetadata(URI.create("https://example.org"));

    assertThat(requestMetadata).isEmpty();
  }

  @Test
  public void shouldReturnMatchedAuthorizationHeader() throws IOException {
    Netrc netrc = new Netrc(null, ImmutableMap.of(fooMachine, fooCredential));
    NetrcCredentials netrcCredentials = new NetrcCredentials(netrc);

    Map<String, List<String>> fooRequestMetadata = netrcCredentials
        .getRequestMetadata(URI.create("https://" + fooMachine));
    Map<String, List<String>> barRequestMetadata = netrcCredentials
        .getRequestMetadata(URI.create("https://" + barMachine));

    assertThat(fooRequestMetadata)
        .isEqualTo(requestMetadata(fooCredential.login(), fooCredential.password()));
    assertThat(barRequestMetadata).isEmpty();
  }

  @Test
  public void shouldReturnDefaultAuthorizationHeaderForNonMatched() throws IOException {
    Netrc netrc = new Netrc(defaultCredential, ImmutableMap.of(fooMachine, fooCredential));
    NetrcCredentials netrcCredentials = new NetrcCredentials(netrc);

    Map<String, List<String>> fooRequestMetadata = netrcCredentials
        .getRequestMetadata(URI.create("https://" + fooMachine));
    Map<String, List<String>> barRequestMetadata = netrcCredentials
        .getRequestMetadata(URI.create("https://" + barMachine));

    assertThat(fooRequestMetadata)
        .isEqualTo(requestMetadata(fooCredential.login(), fooCredential.password()));
    assertThat(barRequestMetadata)
        .isEqualTo(requestMetadata(defaultCredential.login(), defaultCredential.password()));
  }

  @Test
  public void shouldWorkWithEmptyLogin() throws IOException {
    Netrc netrc = new Netrc(null, ImmutableMap.of(fooMachine,
        Credential.builder(fooMachine).setPassword(fooCredential.password()).build()));
    NetrcCredentials netrcCredentials = new NetrcCredentials(netrc);

    Map<String, List<String>> fooRequestMetadata = netrcCredentials
        .getRequestMetadata(URI.create("https://" + fooMachine));

    assertThat(fooRequestMetadata)
        .isEqualTo(requestMetadata("", fooCredential.password()));
  }

  @Test
  public void shouldWorkWithEmptyPassword() throws IOException {
    Netrc netrc = new Netrc(null, ImmutableMap
        .of(fooMachine, Credential.builder(fooMachine).setLogin(fooCredential.login()).build()));
    NetrcCredentials netrcCredentials = new NetrcCredentials(netrc);

    Map<String, List<String>> fooRequestMetadata = netrcCredentials
        .getRequestMetadata(URI.create("https://" + fooMachine));

    assertThat(fooRequestMetadata)
        .isEqualTo(requestMetadata(fooCredential.login(), ""));
  }

  @Test
  public void shouldWorkWithEmptyLoginAndPassword() throws IOException {
    Netrc netrc = new Netrc(null, ImmutableMap
        .of(fooMachine, Credential.builder(fooMachine).build()));
    NetrcCredentials netrcCredentials = new NetrcCredentials(netrc);

    Map<String, List<String>> fooRequestMetadata = netrcCredentials
        .getRequestMetadata(URI.create("https://" + fooMachine));

    assertThat(fooRequestMetadata).isEqualTo(requestMetadata("", ""));
  }

  private static String basicAuthenticationToken(String username, String password) {
    StringBuilder sb = new StringBuilder();
    if (!Strings.isNullOrEmpty(username)) {
      sb.append(username);
    }
    sb.append(":");
    if (!Strings.isNullOrEmpty(password)) {
      sb.append(password);
    }
    return "Basic " + Base64.getEncoder()
        .encodeToString(sb.toString().getBytes(StandardCharsets.UTF_8));
  }

  private static Map<String, List<String>> requestMetadata(String username, String password) {
    return ImmutableMap.of(
        "Authorization",
        ImmutableList.of(basicAuthenticationToken(username, password))
    );
  }
}
