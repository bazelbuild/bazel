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
package com.google.devtools.build.skyframe;

/**
 * An empty marker interface for {@link SkyKey}s evaluated after target analysis completes.
 *
 * <p>When a separate post-analysis thread pool ({@link
 * com.google.devtools.build.lib.concurrent.MultiThreadPoolsQuiescingExecutor.ThreadPoolType#POST_ANALYSIS})
 * is present, {@link NodeEntryVisitor} routes these keys to it unless the key also implements
 * {@link CPUHeavySkyKey} or {@link ExecutionPhaseSkyKey} (which take precedence).
 */
public interface PostAnalysisPhaseSkyKey extends SkyKey {}
