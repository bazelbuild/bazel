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

import com.google.devtools.build.lib.events.ExtendedEventHandler.Postable;
import com.google.devtools.build.lib.skyframe.serialization.analysis.proto.TopLevelTargetsMatchStatus;

/**
 * Posted by {@link RemoteAnalysisCacheManager} with the result of the Skycache metadata query.
 *
 * <p>Subscribers are responsible for reporting the result to the user. This keeps the user-facing
 * text, which may reference tooling specific to the deployment, out of this package.
 *
 * @param status the match status returned by the metadata service. Unknown values are mapped to
 *     {@link TopLevelTargetsMatchStatus#MATCH_STATUS_UNSPECIFIED}.
 * @param serverMessage the human-readable message returned by the metadata service. Carries
 *     diagnostic detail for {@link TopLevelTargetsMatchStatus#MATCH_STATUS_FAILURE}; may be empty
 *     for other statuses.
 */
public record SkycacheMetadataQueryResultEvent(
    TopLevelTargetsMatchStatus status, String serverMessage) implements Postable {}
