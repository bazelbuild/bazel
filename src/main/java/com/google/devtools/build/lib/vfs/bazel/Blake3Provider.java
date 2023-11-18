// Copyright 2023 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.vfs.bazel;

import java.security.Provider;

/** A {@link Provider} for BLAKE3. */
public final class Blake3Provider extends Provider {
  public Blake3Provider() {
    super("BLAKE3Provider", "1.0", "A BLAKE3 digest provider");
    put("MessageDigest.BLAKE3", Blake3MessageDigest.class.getName());
  }
}
