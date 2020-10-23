// Copyright 2019 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.supplier;

/**
 * An {@link InterruptibleSupplier} that can be prompted to fetch its value.
 *
 * <p>Callers that know they will later need the value, but do not yet need it, may call {@link
 * #startGet} to start the get asynchronously, so that when they need it, it may already be present.
 */
public interface InterruptiblePromptableSupplier<T> extends InterruptibleSupplier<T> {

  /**
   * Signals to this supplier that it should asynchronously begin work necessary to provide the
   * return value of {@link #get}.
   */
  void startGet();
}
