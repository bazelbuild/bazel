package com.google.devtools.build.lib.remote;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.remote.util.Utils;
import io.grpc.Status;
import java.io.IOException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for remote utility methods */
@RunWith(JUnit4.class)
public class UtilsTest {

  @Test
  public void testGrpcAwareErrorMessages() {
    IOException ioError = new IOException("io error");
    IOException wrappedGrpcError = new IOException("wrapped error",
        Status.ABORTED.withDescription("grpc error").asRuntimeException());

    assertThat(Utils.grpcAwareErrorMessage(ioError)).isEqualTo("io error");
    assertThat(Utils.grpcAwareErrorMessage(wrappedGrpcError)).isEqualTo("ABORTED: grpc error");
  }
}
