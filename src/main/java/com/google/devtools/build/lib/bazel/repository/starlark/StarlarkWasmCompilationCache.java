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

package com.google.devtools.build.lib.bazel.repository.starlark;

import com.github.benmanes.caffeine.cache.Cache;
import com.github.benmanes.caffeine.cache.Caffeine;
import com.github.benmanes.caffeine.cache.Scheduler;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import java.io.IOException;
import java.time.Duration;
import javax.annotation.Nullable;

@ThreadSafe
final class StarlarkWasmCompilationCache implements com.dylibso.chicory.compiler.Cache {
    private static final int CACHE_MAX_SIZE = 1000;
    private static final Duration CACHE_DURATION = Duration.ofMinutes(15);

    private final Cache<String, byte[]> cache;

    public StarlarkWasmCompilationCache() {
        this.cache = Caffeine.newBuilder()
            .maximumSize(CACHE_MAX_SIZE)
            .expireAfterAccess(CACHE_DURATION)
            .scheduler(Scheduler.systemScheduler())
            .build();
    }

    @Override
    @Nullable
    public byte[] get(String key) throws IOException {
        return cache.getIfPresent(key);
    }

    @Override
    public void putIfAbsent(String key, byte[] data) throws IOException {
        cache.asMap().putIfAbsent(key, data);
    }  
}
