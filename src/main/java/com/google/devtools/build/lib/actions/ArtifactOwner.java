// Copyright 2014 The Bazel Authors. All rights reserved.
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

import com.google.common.annotations.VisibleForTesting;
import com.google.devtools.build.lib.cmdline.Label;

/**
 * An interface for {@code ActionLookupKey}, or at least for a {@link Label}. Only tests and
 * internal {@link Artifact}-generators should implement this interface -- otherwise, {@code
 * ActionLookupKey} and its subclasses should be the only implementation.
 */
public interface ArtifactOwner {
  Label getLabel();

  @VisibleForTesting
  ArtifactOwner NULL_OWNER =
      new ArtifactOwner() {
        @Override
        public Label getLabel() {
          return null;
        }

        @Override
        public String toString() {
          return "NULL_OWNER";
        }
      };
}
