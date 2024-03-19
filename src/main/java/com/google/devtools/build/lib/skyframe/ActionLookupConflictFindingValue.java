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

import com.google.devtools.build.lib.actions.ActionLookupKey;
import com.google.devtools.build.lib.actions.ActionLookupValue;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.SerializationConstant;
import com.google.devtools.build.skyframe.AbstractSkyKey;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

/**
 * A marker for an {@link ActionLookupValue} which is known to be transitively error-free from
 * action conflict issues.
 */
public class ActionLookupConflictFindingValue implements SkyValue {
  @SerializationConstant
  static final ActionLookupConflictFindingValue INSTANCE = new ActionLookupConflictFindingValue();

  private ActionLookupConflictFindingValue() {}

  public static Key key(ActionLookupKey lookupKey) {
    return Key.create(lookupKey);
  }

  public static Key key(Artifact artifact) {
    checkArgument(artifact instanceof Artifact.DerivedArtifact, artifact);
    return ActionLookupConflictFindingValue.key(
        ((Artifact.DerivedArtifact) artifact).getGeneratingActionKey().getActionLookupKey());
  }

  @AutoCodec.VisibleForSerialization
  @AutoCodec
  static class Key extends AbstractSkyKey<ActionLookupKey> {
    private static final SkyKeyInterner<Key> interner = SkyKey.newInterner();

    private Key(ActionLookupKey arg) {
      super(arg);
    }

    @AutoCodec.VisibleForSerialization
    @AutoCodec.Instantiator
    static Key create(ActionLookupKey arg) {
      return interner.intern(new Key(arg));
    }

    @Override
    public SkyFunctionName functionName() {
      return SkyFunctions.ACTION_LOOKUP_CONFLICT_FINDING;
    }

    @Override
    public SkyKeyInterner<Key> getSkyKeyInterner() {
      return interner;
    }
  }
}
