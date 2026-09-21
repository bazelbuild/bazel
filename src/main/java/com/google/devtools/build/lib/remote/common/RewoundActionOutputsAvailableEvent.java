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
package com.google.devtools.build.lib.remote.common;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.ActionInput;
import com.google.devtools.build.lib.actions.FileArtifactValue;
import com.google.devtools.build.lib.events.ExtendedEventHandler.Postable;

/**
 * An event sent once a rewound action has regenerated its outputs. They are then available
 * locally and/or remotely, depending on where the action ran and whether the build downloads
 * outputs.
 *
 * <p>The metadata covers the children of a tree artifact rather than the tree itself.
 *
 * <p>Not suppressed during rewinding. Subscribers must be idempotent, since an action may be
 * rewound more than once.
 */
public record RewoundActionOutputsAvailableEvent(
    ImmutableMap<ActionInput, FileArtifactValue> outputFileMetadata) implements Postable {}
