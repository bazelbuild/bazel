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
package com.google.devtools.build.lib.remote;

import static com.google.common.truth.extensions.proto.ProtoTruth.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import build.bazel.remote.execution.v2.ServerCapabilities;
import build.bazel.semver.SemVer;
import com.google.devtools.build.lib.remote.ChannelConnectionWithServerCapabilitiesFactory.ChannelConnectionWithServerCapabilities;
import io.grpc.ManagedChannel;
import io.reactivex.rxjava3.core.Single;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link ReferenceCountedChannel}. */
@RunWith(JUnit4.class)
public class ReferenceCountedChannelTest {

  @Test
  public void getServerCapabilities_memoizesCapabilitiesAcrossCalls() throws Exception {
    ChannelConnectionWithServerCapabilitiesFactory factory =
        mock(ChannelConnectionWithServerCapabilitiesFactory.class);
    when(factory.maxConcurrency()).thenReturn(100);

    ManagedChannel channel = mock(ManagedChannel.class);
    ChannelConnectionWithServerCapabilities connection =
        mock(ChannelConnectionWithServerCapabilities.class);
    when(connection.getChannel()).thenReturn(channel);

    ServerCapabilities expectedCaps =
        ServerCapabilities.newBuilder()
            .setHighApiVersion(SemVer.newBuilder().setMajor(2).build())
            .build();
    when(connection.getServerCapabilities()).thenReturn(Single.just(expectedCaps));
    when(factory.create()).thenAnswer(_ -> Single.just(connection));

    ReferenceCountedChannel refCntChannel = new ReferenceCountedChannel(factory);

    ServerCapabilities caps1 = refCntChannel.getServerCapabilities();
    ServerCapabilities caps2 = refCntChannel.getServerCapabilities();
    ServerCapabilities caps3 = refCntChannel.getServerCapabilities();

    assertThat(caps1).isEqualTo(expectedCaps);
    assertThat(caps2).isEqualTo(expectedCaps);
    assertThat(caps3).isEqualTo(expectedCaps);

    // Verify factory only created connection once for capabilities check
    verify(factory).create();
  }
}
