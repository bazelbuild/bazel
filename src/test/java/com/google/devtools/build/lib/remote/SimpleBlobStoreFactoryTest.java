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

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.common.options.Options;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link SimpleBlobStoreFactory}. */
@RunWith(JUnit4.class)
public class SimpleBlobStoreFactoryTest {
  private RemoteOptions remoteOptions;

  @Before
  public final void setUp() throws Exception {
    remoteOptions = Options.getDefaults(RemoteOptions.class);
  }

  @Test
  public void testIsRemoteCacheOptions() throws Exception {
    assertThat(SimpleBlobStoreFactory.isRemoteCacheOptions(remoteOptions)).isFalse();

    remoteOptions.remoteHttpCache = "";
    assertThat(SimpleBlobStoreFactory.isRemoteCacheOptions(remoteOptions)).isFalse();

    remoteOptions.remoteHttpCache = "http://127.0.0.1";
    assertThat(SimpleBlobStoreFactory.isRemoteCacheOptions(remoteOptions)).isTrue();

    remoteOptions.remoteHttpCache = null;
    assertThat(SimpleBlobStoreFactory.isRemoteCacheOptions(remoteOptions)).isFalse();
  }
}
