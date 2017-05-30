// Copyright 2010 The Bazel Authors. All Rights Reserved.
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

package com.google.testing.junit.runner.sharding;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.testutil.TestUtils;
import java.io.File;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests interactions with the test environment related to sharding. */
@RunWith(JUnit4.class)
public class ShardingEnvironmentTest {

  @SuppressWarnings({"ResultOfMethodCallIgnored"})
  @Test
  public void testTouchShardingFile() {
    File shardFile = new File(TestUtils.tmpDirFile(), "shard_file_123");
    assertThat(shardFile.exists()).isFalse();
    try {
      ShardingEnvironment.touchShardFile(shardFile);
      assertThat(shardFile.exists()).isTrue();
    } finally {
      shardFile.delete();
    }
  }

}
