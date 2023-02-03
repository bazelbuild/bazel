// Copyright 2021 The Bazel Authors. All rights reserved.
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
 * A {@link SkyKey} for a {@link SkyFunction} that causes heavy resource consumption.
 *
 * <p>This applies to both {@link SkyKey}s that have a high resource footprint and ancestors of
 * those {@link SkyKey}s that depend on them, transitively.
 *
 * <p>This is currently only applicable to the loading/analysis phase of Skyframe.
 */
public abstract class CPUHeavySkyKey implements SkyKey {}
