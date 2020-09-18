package com.google.devtools.build.lib.authandtls;

import com.google.auth.Credentials;
import com.google.common.base.Preconditions;
import com.google.common.base.Stopwatch;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import io.grpc.CallCredentials;
import io.grpc.auth.MoreCallCredentials;
import java.io.IOException;
import java.time.Duration;

@ThreadSafe
public class GoogleAuthCallCredentialsProvider implements CallCredentialsProvider {

  private final Credentials credentials;
  private final CallCredentials callCredentials;
  private final Stopwatch refreshStopwatch;

  public GoogleAuthCallCredentialsProvider(Credentials credentials) {
    Preconditions.checkNotNull(credentials, "credentials");
    this.credentials = credentials;

    callCredentials = MoreCallCredentials.from(credentials);
    refreshStopwatch = Stopwatch.createStarted();
  }

  @Override
  public CallCredentials getCallCredentials() {
    return callCredentials;
  }

  @Override
  public void refresh() throws IOException {
    synchronized (this) {
      // Call credentials.refresh() at most once per second. The one second was arbitrarily chosen,
      // as a small enough value that we don't expect to interfere with actual token lifetimes, but
      // it should just make sure that potentially hundreds of threads don't call this method
      // at the same time.
      if (refreshStopwatch.elapsed().compareTo(Duration.ofSeconds(1)) > 0) {
        credentials.refresh();
        refreshStopwatch.reset().start();
      }
    }
  }
}
