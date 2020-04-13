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

import java.util.Objects;
import org.apache.commons.pool2.impl.GenericKeyedObjectPoolConfig;

/**
 * Our own configuration class for the {@code WorkerPool} that correctly implements {@code equals()}
 * and {@code hashCode()}.
 */
final class WorkerPoolConfig extends GenericKeyedObjectPoolConfig<Worker> {
  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (o == null || getClass() != o.getClass()) {
      return false;
    }
    WorkerPoolConfig that = (WorkerPoolConfig) o;
    return getBlockWhenExhausted() == that.getBlockWhenExhausted()
        && getFairness() == that.getFairness()
        && getJmxEnabled() == that.getJmxEnabled()
        && getLifo() == that.getLifo()
        && getMaxWaitMillis() == that.getMaxWaitMillis()
        && getMinEvictableIdleTimeMillis() == that.getMinEvictableIdleTimeMillis()
        && getNumTestsPerEvictionRun() == that.getNumTestsPerEvictionRun()
        && getSoftMinEvictableIdleTimeMillis() == that.getSoftMinEvictableIdleTimeMillis()
        && getTestOnBorrow() == that.getTestOnBorrow()
        && getTestOnCreate() == that.getTestOnCreate()
        && getTestOnReturn() == that.getTestOnReturn()
        && getTestWhileIdle() == that.getTestWhileIdle()
        && getTimeBetweenEvictionRunsMillis() == that.getTimeBetweenEvictionRunsMillis()
        && getMaxIdlePerKey() == that.getMaxIdlePerKey()
        && getMaxTotal() == that.getMaxTotal()
        && getMaxTotalPerKey() == that.getMaxTotalPerKey()
        && getMinIdlePerKey() == that.getMinIdlePerKey()
        && Objects.equals(getEvictionPolicyClassName(), that.getEvictionPolicyClassName())
        && Objects.equals(getJmxNameBase(), that.getJmxNameBase())
        && Objects.equals(getJmxNamePrefix(), that.getJmxNamePrefix());
  }

  @Override
  public int hashCode() {
    return Objects.hash(
        getBlockWhenExhausted(),
        getFairness(),
        getJmxEnabled(),
        getLifo(),
        getMaxWaitMillis(),
        getMinEvictableIdleTimeMillis(),
        getNumTestsPerEvictionRun(),
        getSoftMinEvictableIdleTimeMillis(),
        getTestOnBorrow(),
        getTestOnCreate(),
        getTestOnReturn(),
        getTestWhileIdle(),
        getTimeBetweenEvictionRunsMillis(),
        getMaxIdlePerKey(),
        getMaxTotal(),
        getMaxTotalPerKey(),
        getMinIdlePerKey(),
        getEvictionPolicyClassName(),
        getJmxNameBase(),
        getJmxNamePrefix());
  }
}
