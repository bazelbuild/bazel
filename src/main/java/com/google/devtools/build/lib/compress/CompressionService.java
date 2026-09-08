// Copyright 2026 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.compress;

import com.google.devtools.build.lib.runtime.BlazeService;
import com.google.devtools.build.lib.skybridge.SkybridgeInterface;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

/** A {@link BlazeService} providing access to compression libraries. */
@SkybridgeInterface
public interface CompressionService extends BlazeService {

  /** Returns an {@link InputStream} that decompresses the given {@link InputStream} using zstd. */
  InputStream newZstdInputStream(InputStream inputStream) throws IOException;

  /** Returns an {@link OutputStream} that compresses the given {@link OutputStream} using zstd. */
  OutputStream newZstdOutputStream(OutputStream outputStream) throws IOException;
}
