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

package com.google.devtools.build.xcode.plmerge;

import com.google.common.collect.ImmutableList;

import java.util.Iterator;

/**
 * A glorified {@link Iterable} which contains keys which should be automatically removed from the
 * final plist if they are empty strings.
 */
public final class KeysToRemoveIfEmptyString implements Iterable<String> {
  private final Iterable<String> keyNames;

  public KeysToRemoveIfEmptyString(String... keyNames) {
    this.keyNames = ImmutableList.copyOf(keyNames);
  }

  @Override
  public Iterator<String> iterator() {
    return keyNames.iterator();
  }
}

