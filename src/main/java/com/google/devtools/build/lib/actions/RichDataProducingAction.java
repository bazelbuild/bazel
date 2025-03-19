// Copyright 2025 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.actions;

import com.google.devtools.build.lib.vfs.Path;
import javax.annotation.Nullable;

/**
 * An action that emits rich data.
 *
 * <p>This needs to be a concept because when such an action is an action cache hit, the rich data
 * is currently not reconstructed from the action cache (it's theoretically possible, it's just that
 * it's not done). So the action must reconstruct this data itself.
 */
// TODO(lberki): Maybe merge this with NotifyOnActionCacheHit?
public interface RichDataProducingAction {
  @Nullable
  RichArtifactData reconstructRichDataOnActionCacheHit(
      Path execRoot, InputMetadataProvider inputMetadataProvider) throws ActionExecutionException;
}
