// Copyright 2015 The Bazel Authors. All Rights Reserved.
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

package com.google.testing.junit.runner.sharding.api;

import org.junit.runner.Description;
import org.junit.runner.manipulation.Filter;

import java.util.Collection;

/**
 * Creates custom test sharding filters. Classes that implement this interface must have a public
 * no-argument constructor.
 */
public interface ShardingFilterFactory {

  /**
   * Creates a test sharding filter.
   *  
   * @param testDescriptions collection of descriptions of the tests to be run 
   * @param shardIndex 0-indexed test shard number, where 0 <= shard index < totalShards
   * @param totalShards the total number of test shards
   */
  Filter createFilter(Collection<Description> testDescriptions, int shardIndex, int totalShards);
}
