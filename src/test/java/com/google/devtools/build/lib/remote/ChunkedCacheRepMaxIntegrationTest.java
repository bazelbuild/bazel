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
package com.google.devtools.build.lib.remote;

import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Integration tests for chunked remote cache using SplitBlob/SpliceBlob APIs with the RepMaxCDC
 * chunking function.
 *
 * <p>Runs all tests of {@link ChunkedCacheIntegrationTest} with
 * {@code --experimental_remote_cache_chunking_function=rep_max_cdc}.
 */
@RunWith(JUnit4.class)
public class ChunkedCacheRepMaxIntegrationTest extends ChunkedCacheIntegrationTest {

  @Override
  protected void setupOptions() throws Exception {
    super.setupOptions();
    addOptions("--experimental_remote_cache_chunking_function=rep_max_cdc");
  }
}
