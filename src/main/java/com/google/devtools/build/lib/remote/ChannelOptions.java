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

package com.google.devtools.build.lib.remote;

import com.google.auth.oauth2.GoogleCredentials;
import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.authandtls.AuthAndTLSOptions;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import io.grpc.CallCredentials;
import io.grpc.auth.MoreCallCredentials;
import io.grpc.netty.GrpcSslContexts;
import io.netty.handler.ssl.SslContext;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import javax.annotation.Nullable;

/** Instantiate all authentication helpers from build options. */
@ThreadSafe
public final class ChannelOptions {
  private final boolean tlsEnabled;
  private final SslContext sslContext;
  private final String tlsAuthorityOverride;
  private final CallCredentials credentials;

  private ChannelOptions(
      boolean tlsEnabled,
      SslContext sslContext,
      String tlsAuthorityOverride,
      CallCredentials credentials) {
    this.tlsEnabled = tlsEnabled;
    this.sslContext = sslContext;
    this.tlsAuthorityOverride = tlsAuthorityOverride;
    this.credentials = credentials;
  }

  public boolean tlsEnabled() {
    return tlsEnabled;
  }

  public CallCredentials getCallCredentials() {
    return credentials;
  }

  public String getTlsAuthorityOverride() {
    return tlsAuthorityOverride;
  }

  public SslContext getSslContext() {
    return sslContext;
  }

  public static ChannelOptions create(AuthAndTLSOptions options) throws IOException {
    if (options.authCredentials != null) {
      try (InputStream authFile = new FileInputStream(options.authCredentials)) {
        return create(options, authFile);
      } catch (FileNotFoundException e) {
        String message = String.format("Could not open auth credentials file '%s': %s",
            options.authCredentials, e.getMessage());
        throw new IOException(message, e);
      }
    } else {
      return create(options, null);
    }
  }

  @VisibleForTesting
  static ChannelOptions create(
      AuthAndTLSOptions options,
      @Nullable InputStream credentialsFile) throws IOException {
    final SslContext sslContext =
        options.tlsEnabled ? createSSlContext(options.tlsCertificate) : null;

    final CallCredentials callCredentials =
        options.authEnabled ? createCallCredentials(credentialsFile, options.authScope) : null;

    return new ChannelOptions(
        sslContext != null, sslContext, options.tlsAuthorityOverride, callCredentials);
  }

  private static CallCredentials createCallCredentials(@Nullable InputStream credentialsFile,
      @Nullable String authScope) throws IOException {
    try {
      GoogleCredentials creds =
          credentialsFile == null
              ? GoogleCredentials.getApplicationDefault()
              : GoogleCredentials.fromStream(credentialsFile);
      if (authScope != null) {
        creds = creds.createScoped(ImmutableList.of(authScope));
      }
      return MoreCallCredentials.from(creds);
    } catch (IOException e) {
      String message = "Failed to init auth credentials for remote caching/execution: "
          + e.getMessage();
      throw new IOException(message, e);
    }
  }

  private static SslContext createSSlContext(@Nullable String rootCert) throws IOException {
    if (rootCert == null) {
      try {
        return GrpcSslContexts.forClient().build();
      } catch (Exception e) {
        String message = "Failed to init TLS infrastructure for remote caching/execution: "
            + e.getMessage();
        throw new IOException(message, e);
      }
    } else {
      try {
        return GrpcSslContexts.forClient().trustManager(new File(rootCert)).build();
      } catch (Exception e) {
        String message = "Failed to init TLS infrastructure for remote caching/execution using "
            + "'%s' as root certificate: %s";
        message = String.format(message, rootCert, e.getMessage());
        throw new IOException(message, e);
      }
    }
  }
}
