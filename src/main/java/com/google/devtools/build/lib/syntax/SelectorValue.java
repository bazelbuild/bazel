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
package com.google.devtools.build.lib.syntax;

import java.util.Map;

/**
 * The value passed to a select({...}) statement, e.g.:
 *
 * <pre>
 *   rule(
 *       name = 'myrule',
 *       deps = select({
 *           'a': [':adep'],
 *           'b': [':bdep'],
 *       })
 * </pre>
 */
public final class SelectorValue {
  Map<?, ?> dictionary;

  public SelectorValue(Map<?, ?> dictionary) {
    this.dictionary = dictionary;
  }

  public Map<?, ?> getDictionary() {
    return dictionary;
  }

  @Override
  public String toString() {
    return "selector({...})";
  }
}
