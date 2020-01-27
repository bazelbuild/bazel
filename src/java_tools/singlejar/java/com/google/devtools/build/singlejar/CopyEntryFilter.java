// Copyright 2014 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.singlejar;

import java.io.IOException;

import javax.annotation.concurrent.Immutable;

/**
 * A filter which invokes {@link StrategyCallback#copy} for every entry. As a
 * result, the first entry for every given name is copied and further entries
 * with the same name are skipped.
 */
@Immutable
public final class CopyEntryFilter implements ZipEntryFilter {

  @Override
  public void accept(String filename, StrategyCallback callback) throws IOException {
    callback.copy(null);
  }
}
