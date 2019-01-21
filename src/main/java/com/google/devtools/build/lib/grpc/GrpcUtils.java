package com.google.devtools.build.lib.grpc;

import com.google.auth.Credentials;
import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.runtime.AuthHeadersProvider;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.build.lib.vfs.PathFragment;
import io.grpc.CallCredentials;
import io.grpc.ClientInterceptor;
import io.grpc.ManagedChannel;
import io.grpc.StatusRuntimeException;
import io.grpc.auth.MoreCallCredentials;
import io.grpc.netty.GrpcSslContexts;
import io.grpc.netty.NegotiationType;
import io.grpc.netty.NettyChannelBuilder;
import io.grpc.util.RoundRobinLoadBalancerFactory;
import io.netty.handler.ssl.SslContext;
import java.io.File;
import java.util.List;
import javax.annotation.Nullable;
import javax.net.ssl.SSLException;

public final class GrpcUtils {

  private GrpcUtils() {
  }

  @Nullable
  public static CallCredentials newCallCredentials(AuthHeadersProvider authHeadersProvider) {
    if (authHeadersProvider == null) {
      return null;
    }
    Preconditions.checkState(authHeadersProvider.isEnabled());
    Credentials credentials = new CredentialsAdapter(authHeadersProvider);
    return MoreCallCredentials.from(credentials);
  }

  public static ManagedChannel newManagedChannel(String target,
      List<ClientInterceptor> interceptors,
      boolean enableTls,
      @Nullable String tlsAuthorityOverride,
      PathFragment tlsCertificate) throws AbruptExitException {
    Preconditions.checkNotNull(target, "target");
    Preconditions.checkNotNull(interceptors, "interceptors");
    Preconditions.checkNotNull(tlsCertificate, "tlsCertificate");

    final SslContext sslContext = enableTls
        ? createSSlContext(tlsCertificate)
        : null;
    try {
      NettyChannelBuilder builder = NettyChannelBuilder.forTarget(target)
          .negotiationType(enableTls ? NegotiationType.TLS : NegotiationType.PLAINTEXT)
          .loadBalancerFactory(RoundRobinLoadBalancerFactory.getInstance())
          .intercept(interceptors);
      if (sslContext != null) {
        builder.sslContext(sslContext);
        if (tlsAuthorityOverride != null) {
          builder.overrideAuthority(tlsAuthorityOverride);
        }
      }
      return builder.build();
    } catch (StatusRuntimeException e) {
      String message = String.format("Failed to created gRPC channel for target '%s'", target);
      throw new AbruptExitException(message, ExitCode.COMMAND_LINE_ERROR, e);
    }
  }

  private static SslContext createSSlContext(PathFragment rootCert)
      throws AbruptExitException {
    if (rootCert.equals(PathFragment.EMPTY_FRAGMENT)) {
      try {
        return GrpcSslContexts.forClient().build();
      } catch (SSLException e) {
        throw new AbruptExitException("Failed to initialize TLS infrastructure",
            ExitCode.BLAZE_INTERNAL_ERROR, e);
      }
    } else {
      try {
        return GrpcSslContexts.forClient().trustManager(new File(rootCert.getPathString())).build();
      } catch (SSLException e) {
        String message = String.format("Failed to initialize TLS infrastructure using "
            + "'%s' as the root certificate", rootCert);
        throw new AbruptExitException(message, ExitCode.COMMAND_LINE_ERROR, e);
      }
    }
  }

}
