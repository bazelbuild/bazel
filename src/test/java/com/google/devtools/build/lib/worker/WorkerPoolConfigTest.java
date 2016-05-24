// Copyright 2016 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.worker;

import com.google.common.testing.EqualsTester;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Test WorkerPoolConfig.
 */
@RunWith(JUnit4.class)
public class WorkerPoolConfigTest {

  @Test
  public void testEquals() throws Exception {
    WorkerPoolConfig config1a = new WorkerPoolConfig();
    config1a.setLifo(true);
    config1a.setMaxIdlePerKey(4);
    config1a.setMaxTotalPerKey(4);
    config1a.setMinIdlePerKey(4);
    config1a.setMaxTotal(-1);
    config1a.setBlockWhenExhausted(true);
    config1a.setTestOnBorrow(true);
    config1a.setTestOnCreate(false);
    config1a.setTestOnReturn(true);
    config1a.setTimeBetweenEvictionRunsMillis(-1);

    WorkerPoolConfig config1b = new WorkerPoolConfig();
    config1b.setLifo(true);
    config1b.setMaxIdlePerKey(4);
    config1b.setMaxTotalPerKey(4);
    config1b.setMinIdlePerKey(4);
    config1b.setMaxTotal(-1);
    config1b.setBlockWhenExhausted(true);
    config1b.setTestOnBorrow(true);
    config1b.setTestOnCreate(false);
    config1b.setTestOnReturn(true);
    config1b.setTimeBetweenEvictionRunsMillis(-1);

    WorkerPoolConfig config2a = new WorkerPoolConfig();
    config2a.setLifo(true);
    config2a.setMaxIdlePerKey(1);
    config2a.setMaxTotalPerKey(1);
    config2a.setMinIdlePerKey(1);
    config2a.setMaxTotal(-1);
    config2a.setBlockWhenExhausted(true);
    config2a.setTestOnBorrow(true);
    config2a.setTestOnCreate(false);
    config2a.setTestOnReturn(true);
    config2a.setTimeBetweenEvictionRunsMillis(-1);

    WorkerPoolConfig config2b = new WorkerPoolConfig();
    config2b.setLifo(true);
    config2b.setMaxIdlePerKey(1);
    config2b.setMaxTotalPerKey(1);
    config2b.setMinIdlePerKey(1);
    config2b.setMaxTotal(-1);
    config2b.setBlockWhenExhausted(true);
    config2b.setTestOnBorrow(true);
    config2b.setTestOnCreate(false);
    config2b.setTestOnReturn(true);
    config2b.setTimeBetweenEvictionRunsMillis(-1);

    new EqualsTester()
        .addEqualityGroup(config1a, config1b)
        .addEqualityGroup(config2a, config2b)
        .testEquals();
  }
}
