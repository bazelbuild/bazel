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
package com.google.devtools.build.lib.skyframe;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Multiset;
import com.google.devtools.build.skyframe.SkyFunctionName;

/**
 * Container for Stats we want to generate for BEP, `blaze dump --rules` and `blaze dump
 * --skyframe=count` which extracts information from a SkyframeExecutor.
 *
 * <p>ruleStats and aspectStats are expected to be sorted.
 */
public final record SkyframeStats(
    ImmutableList<SkyKeyStats> ruleStats,
    ImmutableList<SkyKeyStats> aspectStats,
    Multiset<SkyFunctionName> functionNameStats) {}
