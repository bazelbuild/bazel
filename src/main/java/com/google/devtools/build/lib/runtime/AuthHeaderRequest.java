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
package com.google.devtools.build.lib.runtime;

import com.google.common.collect.ImmutableMultimap;
import java.net.URI;
import java.util.Optional;

/**
 * Value type encapsulating http details relating to a request we are asking an implementation of
 * {@see AuthHeadersProvider} to generate authentication parameters for
 *
 * This allows blaze modules to be independent from the underlying implementations (e.g. Netty)
 * while still providing a read-only view of the protocol details
 */
public interface AuthHeaderRequest {

  /**
   * @return The URI this outgoing request should be authenticated for
   */
  URI uri();

  /**
   * Is this call made on a pure HTTP interface, or at some other abstraction such as GRPC?
   */
  default boolean isHttp() {
    return false;
  }

  /**
   * @return the Http method name used for this request (i.e. GET/POST/HEAD) if it exists
   */
  default Optional<String> method() {
    return Optional.empty();
  }

  /**
   * @return Any existing Http headers used for this request, if they exist
   */
  default Optional<ImmutableMultimap<String, String>> headers() {
    return Optional.empty();
  }
}
