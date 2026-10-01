// Copyright 2024 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe.serialization;

import com.google.devtools.build.lib.util.Bucket;
import java.util.List;

/** Concrete record implementation of {@link FingerprintValueStore.Stats}. */
public record StatsImpl(
    long valueBytesReceived,
    long valueBytesSent,
    long keyBytesSent,
    long entriesWritten,
    long entriesFound,
    long entriesNotFound,
    long getBatches,
    long setBatches,
    List<Bucket> getLatencyMicros,
    List<Bucket> setLatencyMicros,
    List<Bucket> getBatchLatencyMicros,
    List<Bucket> setBatchLatencyMicros)
    implements FingerprintValueStore.Stats {}
