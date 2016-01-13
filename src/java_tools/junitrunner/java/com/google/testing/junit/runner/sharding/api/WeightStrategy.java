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

/**
 * Extracts the weight associated with a test for use by sharding filters.
 */
public interface WeightStrategy {
  
  /**
   * Returns the weight of a test extracted from its description.
   * 
   * @param description the description that contains the associated weight for a test
   */
  int getDescriptionWeight(Description description);
}
