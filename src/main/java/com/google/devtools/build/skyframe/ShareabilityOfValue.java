// Copyright 2018 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.skyframe;

/**
 * When the {@link NodeEntry#getValue} corresponding to a given {@link SkyFunctionName} is
 * shareable: always, sometimes (depending on the specific key argument and/or value), or never.
 *
 * <p>Values may be unshareable because they are just not serializable, or because they contain data
 * that cannot safely be re-used as-is by another invocation.
 *
 * <p>Unshareable data should not be serialized, since it will never be re-used. Attempts to fetch
 * serialized data will check this value and only perform the fetch if the value is not {@link
 * #NEVER}.
 */
public enum ShareabilityOfValue {
  /**
   * Indicates that values produced by the function are shareable unless they override {@link
   * SkyValue#dataIsShareable}.
   */
  SOMETIMES,
  /** Indicates that values produced by the function are not shareable. */
  NEVER
}
