// Copyright 2019 The Bazel Authors. All rights reserved.
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

import static org.junit.Assume.assumeTrue;

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.util.OS;
import io.netty.handler.ssl.SslContextBuilder;
import io.netty.handler.ssl.SslProvider;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests that we use OpenSSL instead of a Java implementation. */
@RunWith(JUnit4.class)
public class NativeSslTest {
  private static final ImmutableSet<OS> OS_WITH_NATIVE_SSL =
      ImmutableSet.of(OS.LINUX, OS.DARWIN, OS.WINDOWS);

  @Test
  public void nativeSslPresent() throws Exception {
    // Skip the test on platforms where native SSL is not available.
    assumeTrue(OS_WITH_NATIVE_SSL.contains(OS.getCurrent()));

    SslContextBuilder.forClient().sslProvider(SslProvider.OPENSSL).build();
  }
}
