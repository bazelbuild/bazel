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
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import io.grpc.CallCredentials;
import io.grpc.auth.MoreCallCredentials;
import io.grpc.netty.GrpcSslContexts;
import io.netty.handler.ssl.SslContext;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import javax.annotation.Nullable;
import javax.net.ssl.SSLException;

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

  public static ChannelOptions create(RemoteOptions options) {
    try {
      return create(
          options,
          options.authCredentialsJson != null
              ? new FileInputStream(options.authCredentialsJson)
              : null);
    } catch (IOException e) {
      throw new IllegalArgumentException(
          "Failed initializing auth credentials for remote cache/execution " + e);
    }
  }

  @VisibleForTesting
  public static ChannelOptions create(
      RemoteOptions options, @Nullable InputStream credentialsInputStream) {
    boolean tlsEnabled = options.tlsEnabled;
    SslContext sslContext = null;
    String tlsAuthorityOverride = options.tlsAuthorityOverride;
    CallCredentials credentials = null;
    if (options.tlsEnabled && options.tlsCert != null) {
      try {
        sslContext = GrpcSslContexts.forClient().trustManager(new File(options.tlsCert)).build();
      } catch (SSLException e) {
        throw new IllegalArgumentException(
            "SSL error initializing cert " + options.tlsCert + " : " + e);
      }
    }
    if (options.authEnabled) {
      try {
        GoogleCredentials creds =
            credentialsInputStream == null
                ? GoogleCredentials.getApplicationDefault()
                : GoogleCredentials.fromStream(credentialsInputStream);
        if (options.authScope != null) {
          creds = creds.createScoped(ImmutableList.of(options.authScope));
        }
        credentials = MoreCallCredentials.from(creds);
      } catch (IOException e) {
        throw new IllegalArgumentException(
            "Failed initializing auth credentials for remote cache/execution " + e);
      }
    }
    return new ChannelOptions(tlsEnabled, sslContext, tlsAuthorityOverride, credentials);
  }
}
