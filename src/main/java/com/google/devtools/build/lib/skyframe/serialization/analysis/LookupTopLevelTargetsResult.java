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
package com.google.devtools.build.lib.skyframe.serialization.analysis;

import com.google.devtools.build.lib.skybridge.SkybridgeInterface;

/**
 * The result of a top-level targets lookup.
 *
 * @param status Corresponds to
 *     com.google.devtools.build.lib.skyframe.serialization.analysis.proto.TopLevelTargetsMatchStatus.
 *     We use an int instead of the proto to keep the SkybridgeInterface simple. Since older LCs may
 *     not know about the new enum values, consumers must check for possible version skews and map
 *     the value to MATCH_STATUS_UNSPECIFIED.
 * @param statusMessage A human-readable message explaining the status.
 */
@SkybridgeInterface
public record LookupTopLevelTargetsResult(int status, String statusMessage) {}
