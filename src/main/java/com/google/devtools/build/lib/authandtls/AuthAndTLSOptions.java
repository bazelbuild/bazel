// Copyright 2017 The Bazel Authors. All rights reserved.
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

import com.google.devtools.common.options.Converters.CommaSeparatedOptionListConverter;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionMetadataTag;
import com.google.devtools.common.options.OptionsBase;
import java.util.List;

/**
 * Common options for authentication and TLS.
 */
public class AuthAndTLSOptions extends OptionsBase {
  @Option(
    name = "google_default_credentials",
    oldName = "auth_enabled",
    defaultValue = "false",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help =
        "Whether to use 'Google Application Default Credentials' for authentication."
            + " See https://cloud.google.com/docs/authentication for details. Disabled by default."
  )
  public boolean useGoogleDefaultCredentials;

  @Option(
    name = "google_auth_scopes",
    oldName = "auth_scope",
    defaultValue = "https://www.googleapis.com/auth/cloud-platform",
    converter = CommaSeparatedOptionListConverter.class,
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help = "A comma-separated list of Google Cloud authentication scopes."
  )
  public List<String> googleAuthScopes;

  @Option(
    name = "google_credentials",
    oldName = "auth_credentials",
    defaultValue = "null",
    documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help =
        "Specifies the file to get authentication credentials from. See "
            + "https://cloud.google.com/docs/authentication for details."
  )
  public String googleCredentials;

  @Option(
      name = "tls_certificate",
      defaultValue = "null",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help = "Specify a path to a TLS certificate that is trusted to sign server certificates.")
  public String tlsCertificate;

  @Option(
      name = "tls_client_certificate",
      defaultValue = "null",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help =
          "Specify the TLS client certificate to use; you also need to provide a client key to "
              + "enable client authentication.")
  public String tlsClientCertificate;

  @Option(
      name = "tls_client_key",
      defaultValue = "null",
      documentationCategory = OptionDocumentationCategory.UNCATEGORIZED,
      effectTags = {OptionEffectTag.UNKNOWN},
      help =
          "Specify the TLS client key to use; you also need to provide a client certificate to "
              + "enable client authentication.")
  public String tlsClientKey;

  @Option(
    name = "tls_authority_override",
    defaultValue = "null",
    metadataTags = {OptionMetadataTag.HIDDEN},
    documentationCategory = OptionDocumentationCategory.UNDOCUMENTED,
    effectTags = {OptionEffectTag.UNKNOWN},
    help =
        "TESTING ONLY! Can be used with a self-signed certificate to consider the specified "
            + "value a valid TLS authority."
  )
  public String tlsAuthorityOverride;
}
