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

import com.google.auth.Credentials;
import com.google.auth.oauth2.GoogleCredentials;
import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.base.Strings;
import io.grpc.CallCredentials;
import io.grpc.ClientInterceptor;
import io.grpc.ManagedChannel;
import io.grpc.auth.MoreCallCredentials;
import io.grpc.netty.GrpcSslContexts;
import io.grpc.netty.NegotiationType;
import io.grpc.netty.NettyChannelBuilder;
import io.netty.channel.epoll.Epoll;
import io.netty.channel.epoll.EpollDomainSocketChannel;
import io.netty.channel.epoll.EpollEventLoopGroup;
import io.netty.channel.kqueue.KQueue;
import io.netty.channel.kqueue.KQueueDomainSocketChannel;
import io.netty.channel.kqueue.KQueueEventLoopGroup;
import io.netty.channel.unix.DomainSocketAddress;
import io.netty.handler.ssl.SslContext;
import io.netty.handler.ssl.SslContextBuilder;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.util.List;
import java.util.concurrent.TimeUnit;
import javax.annotation.Nullable;

/** Utility methods for using {@link AuthAndTLSOptions} with Google Cloud. */
public final class GoogleAuthUtils {

  /**
   * Create a new gRPC {@link ManagedChannel}.
   *
   * @throws IOException in case the channel can't be constructed.
   */
  public static ManagedChannel newChannel(
      String target,
      String proxy,
      AuthAndTLSOptions options,
      @Nullable List<ClientInterceptor> interceptors)
      throws IOException {
    Preconditions.checkNotNull(target);
    Preconditions.checkNotNull(options);

    SslContext sslContext =
        isTlsEnabled(target)
            ? createSSlContext(
                options.tlsCertificate, options.tlsClientCertificate, options.tlsClientKey)
            : null;

    String targetUrl = convertTargetScheme(target);
    try {
      NettyChannelBuilder builder =
          newNettyChannelBuilder(targetUrl, proxy)
              .negotiationType(
                  isTlsEnabled(target) ? NegotiationType.TLS : NegotiationType.PLAINTEXT);
      if (options.grpcKeepaliveTime != null) {
        builder.keepAliveTime(options.grpcKeepaliveTime.getSeconds(), TimeUnit.SECONDS);
        builder.keepAliveTimeout(options.grpcKeepaliveTimeout.getSeconds(), TimeUnit.SECONDS);
      }
      if (interceptors != null) {
        builder.intercept(interceptors);
      }
      if (sslContext != null) {
        builder.sslContext(sslContext);
        if (options.tlsAuthorityOverride != null) {
          builder.overrideAuthority(options.tlsAuthorityOverride);
        }
      }
      return builder.build();
    } catch (RuntimeException e) {
      // gRPC might throw all kinds of RuntimeExceptions: StatusRuntimeException,
      // IllegalStateException, NullPointerException, ...
      String message = "Failed to connect to '%s': %s";
      throw new IOException(String.format(message, targetUrl, e.getMessage()));
    }
  }

  /**
   * Converts 'grpc(s)' into an empty protocol, because 'grpc(s)' is not a widely supported scheme
   * and is interpreted as 'dns' under the hood.
   *
   * @return target URL with converted scheme
   */
  private static String convertTargetScheme(String target) {
    return target.replace("grpcs://", "").replace("grpc://", "");
  }

  private static boolean isTlsEnabled(String target) {
    // 'grpcs://' or empty prefix => TLS-enabled
    // when no schema prefix is provided in URL, bazel will treat it as a gRPC request with TLS
    // enabled
    return !target.startsWith("grpc://") && !target.startsWith("unix:");
  }

  private static SslContext createSSlContext(
      @Nullable String rootCert, @Nullable String clientCert, @Nullable String clientKey)
      throws IOException {
    SslContextBuilder sslContextBuilder;
    try {
      sslContextBuilder = GrpcSslContexts.forClient();
    } catch (Exception e) {
      String message = "Failed to init TLS infrastructure: " + e.getMessage();
      throw new IOException(message, e);
    }
    if (rootCert != null) {
      try {
        sslContextBuilder.trustManager(new File(rootCert));
      } catch (Exception e) {
        String message = "Failed to init TLS infrastructure using '%s' as root certificate: %s";
        message = String.format(message, rootCert, e.getMessage());
        throw new IOException(message, e);
      }
    }
    if (clientCert != null && clientKey != null) {
      try {
        sslContextBuilder.keyManager(new File(clientCert), new File(clientKey));
      } catch (Exception e) {
        String message = "Failed to init TLS infrastructure using '%s' as client certificate: %s";
        message = String.format(message, clientCert, e.getMessage());
        throw new IOException(message, e);
      }
    }
    try {
      return sslContextBuilder.build();
    } catch (Exception e) {
      String message = "Failed to init TLS infrastructure: " + e.getMessage();
      throw new IOException(message, e);
    }
  }

  private static NettyChannelBuilder newUnixNettyChannelBuilder(String target) throws IOException {
    DomainSocketAddress address = new DomainSocketAddress(target.replaceFirst("^unix:", ""));
    NettyChannelBuilder builder = NettyChannelBuilder.forAddress(address);
    if (KQueue.isAvailable()) {
      return builder
          .channelType(KQueueDomainSocketChannel.class)
          .eventLoopGroup(new KQueueEventLoopGroup());
    }
    if (Epoll.isAvailable()) {
      return builder
          .channelType(EpollDomainSocketChannel.class)
          .eventLoopGroup(new EpollEventLoopGroup());
    }

    throw new IOException("Unix domain sockets are unsupported on this platform");
  }

  private static NettyChannelBuilder newNettyChannelBuilder(String targetUrl, String proxy)
      throws IOException {
    if (targetUrl.startsWith("unix:")) {
      return newUnixNettyChannelBuilder(targetUrl);
    }

    if (Strings.isNullOrEmpty(proxy)) {
      return NettyChannelBuilder.forTarget(targetUrl).defaultLoadBalancingPolicy("round_robin");
    }

    if (!proxy.startsWith("unix:")) {
      throw new IOException("Remote proxy unsupported: " + proxy);
    }

    return newUnixNettyChannelBuilder(proxy).overrideAuthority(targetUrl);
  }

  /**
   * Create a new {@link CallCredentials} object.
   *
   * @throws IOException in case the call credentials can't be constructed.
   */
  public static CallCredentials newCallCredentials(AuthAndTLSOptions options) throws IOException {
    Credentials creds = newCredentials(options);
    if (creds != null) {
      return MoreCallCredentials.from(creds);
    }
    return null;
  }

  @VisibleForTesting
  public static CallCredentials newCallCredentials(
      @Nullable InputStream credentialsFile, List<String> authScope) throws IOException {
    Credentials creds = newCredentials(credentialsFile, authScope);
    if (creds != null) {
      return MoreCallCredentials.from(creds);
    }
    return null;
  }

  /**
   * Create a new {@link Credentials} object, or {@code null} if no options are provided.
   *
   * @throws IOException in case the credentials can't be constructed.
   */
  @Nullable
  public static Credentials newCredentials(@Nullable AuthAndTLSOptions options) throws IOException {
    if (options == null) {
      return null;
    } else if (options.googleCredentials != null) {
      // Credentials from file
      try (InputStream authFile = new FileInputStream(options.googleCredentials)) {
        return newCredentials(authFile, options.googleAuthScopes);
      } catch (FileNotFoundException e) {
        String message =
            String.format(
                "Could not open auth credentials file '%s': %s",
                options.googleCredentials, e.getMessage());
        throw new IOException(message, e);
      }
    } else if (options.useGoogleDefaultCredentials) {
      return newCredentials(
          null /* Google Application Default Credentials */, options.googleAuthScopes);
    }
    return null;
  }

  private static Credentials newCredentials(
      @Nullable InputStream credentialsFile, List<String> authScopes) throws IOException {
    try {
      GoogleCredentials creds =
          credentialsFile == null
              ? GoogleCredentials.getApplicationDefault()
              : GoogleCredentials.fromStream(credentialsFile);
      if (!authScopes.isEmpty()) {
        creds = creds.createScoped(authScopes);
      }
      return creds;
    } catch (IOException e) {
      String message = "Failed to init auth credentials: " + e.getMessage();
      throw new IOException(message, e);
    }
  }
}
