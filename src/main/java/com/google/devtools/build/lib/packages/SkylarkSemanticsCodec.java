// Copyright 2017 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.skyframe.serialization.ObjectCodec;
import com.google.devtools.build.lib.skyframe.serialization.SerializationException;
import com.google.devtools.build.lib.syntax.SkylarkSemantics;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;

/**
 * Codec for a {@link SkylarkSemantics} object.
 *
 * <p>Since the core Skylark interpreter should not depend on the protobuf library, this codec
 * cannot be nested alongside the {@code SkylarkSemantics} class definition.
 *
 * <p>Tests for this codec are in {@link SkylarkSemanticsConsistencyTest}.
 */
public final class SkylarkSemanticsCodec implements ObjectCodec<SkylarkSemantics> {

  @Override
  public Class<SkylarkSemantics> getEncodedClass() {
    return SkylarkSemantics.class;
  }

  @Override
  public void serialize(SkylarkSemantics semantics, CodedOutputStream codedOut)
      throws SerializationException, IOException {
    // <== Add new options here in alphabetic order ==>
    codedOut.writeBoolNoTag(semantics.incompatibleBzlDisallowLoadAfterStatement());
    codedOut.writeBoolNoTag(semantics.incompatibleCheckedArithmetic());
    codedOut.writeBoolNoTag(semantics.incompatibleComprehensionVariablesDoNotLeak());
    codedOut.writeBoolNoTag(semantics.incompatibleDepsetIsNotIterable());
    codedOut.writeBoolNoTag(semantics.incompatibleDepsetUnion());
    codedOut.writeBoolNoTag(semantics.incompatibleDictLiteralHasNoDuplicates());
    codedOut.writeBoolNoTag(semantics.incompatibleDisableGlobTracking());
    codedOut.writeBoolNoTag(semantics.incompatibleDisallowDictPlus());
    codedOut.writeBoolNoTag(semantics.incompatibleDisallowKeywordOnlyArgs());
    codedOut.writeBoolNoTag(semantics.incompatibleDisallowToplevelIfStatement());
    codedOut.writeBoolNoTag(semantics.incompatibleDisallowUncalledSetConstructor());
    codedOut.writeBoolNoTag(semantics.incompatibleLoadArgumentIsLabel());
    codedOut.writeBoolNoTag(semantics.incompatibleNewActionsApi());
    codedOut.writeBoolNoTag(semantics.incompatibleShowAllPrintMessages());
    codedOut.writeBoolNoTag(semantics.incompatibleStringIsNotIterable());
    codedOut.writeBoolNoTag(semantics.internalSkylarkFlagTestCanary());
  }

  @Override
  public SkylarkSemantics deserialize(CodedInputStream codedIn)
      throws SerializationException, IOException {
    SkylarkSemantics.Builder builder = SkylarkSemantics.builder();

    // <== Add new options here in alphabetic order ==>
    builder.incompatibleBzlDisallowLoadAfterStatement(codedIn.readBool());
    builder.incompatibleCheckedArithmetic(codedIn.readBool());
    builder.incompatibleComprehensionVariablesDoNotLeak(codedIn.readBool());
    builder.incompatibleDepsetIsNotIterable(codedIn.readBool());
    builder.incompatibleDepsetUnion(codedIn.readBool());
    builder.incompatibleDictLiteralHasNoDuplicates(codedIn.readBool());
    builder.incompatibleDisableGlobTracking(codedIn.readBool());
    builder.incompatibleDisallowDictPlus(codedIn.readBool());
    builder.incompatibleDisallowKeywordOnlyArgs(codedIn.readBool());
    builder.incompatibleDisallowToplevelIfStatement(codedIn.readBool());
    builder.incompatibleDisallowUncalledSetConstructor(codedIn.readBool());
    builder.incompatibleLoadArgumentIsLabel(codedIn.readBool());
    builder.incompatibleNewActionsApi(codedIn.readBool());
    builder.incompatibleShowAllPrintMessages(codedIn.readBool());
    builder.incompatibleStringIsNotIterable(codedIn.readBool());
    builder.internalSkylarkFlagTestCanary(codedIn.readBool());

    return builder.build();
  }
}
