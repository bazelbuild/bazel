// Copyright 2014 Google Inc. All rights reserved.
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

package com.google.devtools.build.xcode.xcodegen;

import com.google.common.base.Optional;
import com.google.devtools.build.xcode.util.Value;

import java.nio.file.Path;

/**
 * Data used to as a key when assigning items to aggregate references
 * (see {@link AggregateReferenceType}). All items sharing a single key value should belong to the
 * same group.
 * <p>
 * Note that the information in the grouping key is also the information that is used to create a
 * new group, minus the actual list of files in the group. See
 * {@link AggregateReferenceType#create(AggregateKey, Iterable)}.
 */
public class AggregateKey extends Value<AggregateKey> {
  private final Optional<String> name;
  private final Optional<Path> path;

  public AggregateKey(Optional<String> name, Optional<Path> path) {
    super(name, path);
    this.name = name;
    this.path = path;
  }

  public Optional<String> name() {
    return name;
  }

  public Optional<Path> path() {
    return path;
  }

  /**
   * Returns an instance that indicates any item assigned to it should not belong to a group.
   */
  public static AggregateKey standalone() {
    return new AggregateKey(Optional.<String>absent(), Optional.<Path>absent());
  }

  /**
   * Indicates that this grouping key means an item should not belong to any group.
   */
  public boolean isStandalone() {
    return !path().isPresent() && !name().isPresent();
  }
}
