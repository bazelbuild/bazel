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
import static org.mockito.Mockito.when;

import com.google.testing.junit.runner.sharding.api.ShardingFilterFactory;
import com.google.testing.junit.runner.sharding.testing.ShardingFilterTestCase;
import java.util.Collection;
import java.util.List;
import org.junit.Test;
import org.junit.runner.Description;
import org.junit.runner.RunWith;
import org.junit.runner.manipulation.Filter;
import org.mockito.Mock;
import org.mockito.runners.MockitoJUnitRunner;

/**
 * Tests for {@link ShardingFilters}.
 */
@RunWith(MockitoJUnitRunner.class)
public class ShardingFiltersTest {
  @Mock ShardingEnvironment mockShardingEnvironment;

  @Test
  public void testCreateShardingFilter_defaultStrategy() {
    List<Description> descriptions = ShardingFilterTestCase.createGenericTestCaseDescriptions(6);
    RoundRobinShardingFilter expectedFilter = new RoundRobinShardingFilter(descriptions, 0, 5);

    when(mockShardingEnvironment.getShardIndex()).thenReturn(0);
    when(mockShardingEnvironment.getTotalShards()).thenReturn(5);
    when(mockShardingEnvironment.getTestShardingStrategy()).thenReturn(null);

    ShardingFilters shardingFilters = new ShardingFilters(mockShardingEnvironment,
        ShardingFilters.ShardingStrategy.ROUND_ROBIN);
    Filter filter = shardingFilters.createShardingFilter(descriptions);

    assertThat(filter).isInstanceOf(RoundRobinShardingFilter.class);
    RoundRobinShardingFilter shardingFilter = (RoundRobinShardingFilter) filter;
    assertThat(shardingFilter.testToShardMap).isEqualTo(expectedFilter.testToShardMap);
    assertThat(shardingFilter.shardIndex).isEqualTo(expectedFilter.shardIndex);
    assertThat(shardingFilter.totalShards).isEqualTo(expectedFilter.totalShards);
  }

  @Test
  public void testCreateShardingFilter_customStrategy() {
    List<Description> descriptions = ShardingFilterTestCase.createGenericTestCaseDescriptions(6);

    when(mockShardingEnvironment.getShardIndex()).thenReturn(0);
    when(mockShardingEnvironment.getTotalShards()).thenReturn(5);
    when(mockShardingEnvironment.getTestShardingStrategy()).thenReturn(
        "com.google.testing.junit.runner.sharding.ShardingFiltersTest$TestFilterFactory");

    ShardingFilters shardingFilters = new ShardingFilters(mockShardingEnvironment);
    Filter filter = shardingFilters.createShardingFilter(descriptions);

    assertThat(filter.getClass().getCanonicalName())
        .isEqualTo("com.google.testing.junit.runner.sharding.ShardingFiltersTest.TestFilter");
  }

  public static class TestFilterFactory implements ShardingFilterFactory {
    @Override
    public Filter createFilter(
        Collection<Description> testDescriptions, int shardIndex, int totalShards) {
      return new TestFilter();
    }
  }

  static class TestFilter extends Filter {
    @Override
    public boolean shouldRun(Description description) {
      return false;
    }

    @Override
    public String describe() {
      return "test filter factory";
    }
  }
}
