// Copyright 2020 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.authandtls;

import io.grpc.CallCredentials;
import java.io.IOException;
import javax.annotation.Nullable;

/** Interface for providing {@link CallCredentials}. Implementations must be thread-safe. */
public interface CallCredentialsProvider {
  /**
   * Returns the current {@link CallCredentials}. May be {@code null}, in which case no
   * authentication is required.
   */
  @Nullable
  CallCredentials getCallCredentials();

  /**
   * Refresh the authorization data, discarding any cached state.
   *
   * <p>For use by the transport to allow retry after getting an error indicating there may be
   * invalid tokens or other cached state.
   *
   * @throws IOException if there was an error getting up-to-date access.
   */
  void refresh() throws IOException;

  /** A no-op implementation that has no credentials and performs no refreshes. */
  public static class NoCredentials implements CallCredentialsProvider {

    @Nullable
    @Override
    public CallCredentials getCallCredentials() {
      return null;
    }

    @Override
    public void refresh() throws IOException {}
  }

  public static CallCredentialsProvider NO_CREDENTIALS = new NoCredentials();
}
