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
package com.google.devtools.build.lib.skyframe;

import static com.google.common.base.Preconditions.checkArgument;

import com.google.common.collect.Interner;
import com.google.devtools.build.lib.actions.ActionLookupValue;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.concurrent.BlazeInterners;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.skyframe.AbstractSkyKey;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyValue;

/**
 * A marker for an {@link ActionLookupValue} which is known to be transitively error-free from
 * action conflict issues.
 */
public class ActionLookupConflictFindingValue implements SkyValue {
  @AutoCodec
  static final ActionLookupConflictFindingValue INSTANCE = new ActionLookupConflictFindingValue();

  private ActionLookupConflictFindingValue() {}

  public static Key key(ActionLookupValue.ActionLookupKey lookupKey) {
    return Key.create(lookupKey);
  }

  public static Key key(Artifact artifact) {
    checkArgument(artifact instanceof Artifact.DerivedArtifact, artifact);
    return ActionLookupConflictFindingValue.key(
        ((Artifact.DerivedArtifact) artifact).getGeneratingActionKey().getActionLookupKey());
  }

  @AutoCodec.VisibleForSerialization
  @AutoCodec
  static class Key extends AbstractSkyKey<ActionLookupValue.ActionLookupKey> {
    private static final Interner<Key> interner = BlazeInterners.newWeakInterner();

    private Key(ActionLookupValue.ActionLookupKey arg) {
      super(arg);
    }

    @AutoCodec.VisibleForSerialization
    @AutoCodec.Instantiator
    static Key create(ActionLookupValue.ActionLookupKey arg) {
      return interner.intern(new Key(arg));
    }

    @Override
    public SkyFunctionName functionName() {
      return SkyFunctions.ACTION_LOOKUP_CONFLICT_FINDING;
    }
  }
}
