package com.google.devtools.build.lib.authandtls;

import com.google.auth.Credentials;
import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import io.grpc.CallCredentials;
import io.grpc.auth.MoreCallCredentials;
import java.io.IOException;
import java.util.concurrent.TimeUnit;

@ThreadSafe
public class GoogleAuthCallCredentialsProvider implements CallCredentialsProvider {

  private final Credentials credentials;

  private CallCredentials callCredentials;
  private long lastRefreshTime;

  public GoogleAuthCallCredentialsProvider(Credentials credentials) {
    Preconditions.checkNotNull(credentials, "credentials");
    this.credentials = credentials;
    callCredentials = MoreCallCredentials.from(credentials);
  }

  @Override
  public CallCredentials getCallCredentials() {
    return callCredentials;
  }

  @Override
  public void refresh() throws IOException {
    synchronized (this) {
      long now = System.currentTimeMillis();
      // Call credentials.refresh() at most once per second. The one second was arbitrarily chosen,
      // as a small enough value that we don't expect to interfere with actual token lifetimes, but
      // it should just make sure that potentially hundreds of threads don't call this method
      // at the same time.
      if ((now - lastRefreshTime) > TimeUnit.SECONDS.toMillis(1)) {
        lastRefreshTime = now;
        credentials.refresh();
      }
    }
  }
}
