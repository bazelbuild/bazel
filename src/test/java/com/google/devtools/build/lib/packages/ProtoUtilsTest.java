// Copyright 2015 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.packages;

import static org.junit.Assert.assertTrue;

import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.query2.proto.proto2api.Build.Attribute.Discriminator;
import com.google.devtools.build.lib.syntax.Type;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.util.Collection;
import java.util.Map.Entry;

/** Tests for values and functions in ProtoUtils. */
@RunWith(JUnit4.class)
public class ProtoUtilsTest {

  @Test
  public void testTypeMap() throws Exception {
    // The ProtoUtils TYPE_MAP (and its inverse, INVERSE_TYPE_MAP) are used to translate between
    // rule attribute types and Discriminator values used to encode them in GPB messages. For each
    // discriminator value there must be exactly one type, or there must be exactly two types, one
    // which is a nodep type and the other which is not.
    ImmutableSet<Entry<Discriminator, Collection<Type<?>>>> inverseMapEntries =
        ProtoUtils.INVERSE_TYPE_MAP.asMap().entrySet();
    for (Entry<Discriminator, Collection<Type<?>>> entry : inverseMapEntries) {
      ImmutableSet<Type<?>> types = ImmutableSet.copyOf(entry.getValue());
      String assertionMessage =
          String.format(
              "Cannot map from discriminator \"%s\" to exactly one Type.",
              entry.getKey().toString());
      boolean exactlyOneType = types.size() == 1;
      boolean twoTypesDistinguishableUsingNodepHint =
          types.size() == 2 && Sets.difference(types, ProtoUtils.NODEP_TYPES).size() == 1;
      assertTrue(assertionMessage, exactlyOneType || twoTypesDistinguishableUsingNodepHint);
    }
  }
}
