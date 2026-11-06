// Copyright 2026 The Bazel Authors. All rights reserved.
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

import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsParsingException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link AuthAndTLSOptions}. */
@RunWith(JUnit4.class)
public class AuthAndTlsOptionsTest {

  private static AuthAndTLSOptions parseOptions(String... args) throws OptionsParsingException {
    OptionsParser parser = OptionsParser.builder().optionsClasses(AuthAndTLSOptions.class).build();
    parser.parse(args);
    return parser.getOptions(AuthAndTLSOptions.class);
  }

  @Test
  public void googleCredentials_emptyValue_resetsToNull() throws Exception {
    assertThat(
            parseOptions("--google_credentials=/path/to/key", "--google_credentials=")
                .getGoogleCredentials())
        .isNull();
  }

  @Test
  public void tlsAuthorityOverride_emptyValue_resetsToNull() throws Exception {
    assertThat(
            parseOptions("--tls_authority_override=some-auth", "--tls_authority_override=")
                .getTlsAuthorityOverride())
        .isNull();
  }

  @Test
  public void tlsCertificate_emptyValue_resetsToNull() throws Exception {
    assertThat(
            parseOptions("--tls_certificate=/path/to/cert.pem", "--tls_certificate=")
                .getTlsCertificate())
        .isNull();
  }

  @Test
  public void tlsClientCertificate_emptyValue_resetsToNull() throws Exception {
    assertThat(
            parseOptions(
                    "--tls_client_certificate=/path/to/client.pem", "--tls_client_certificate=")
                .getTlsClientCertificate())
        .isNull();
  }

  @Test
  public void tlsClientKey_emptyValue_resetsToNull() throws Exception {
    assertThat(
            parseOptions("--tls_client_key=/path/to/key.pem", "--tls_client_key=")
                .getTlsClientKey())
        .isNull();
  }
}
