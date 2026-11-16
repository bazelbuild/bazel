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

package com.google.devtools.build.lib.query2.query.output;

import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.AttributeFormatter;
import com.google.devtools.build.lib.packages.LabelPrinter;
import com.google.devtools.build.lib.query2.proto.proto2api.Build;
import java.util.concurrent.atomic.AtomicReferenceArray;
import javax.annotation.Nullable;

/** Lazily caches literal defaults at their declaration indices for one rule class and query. */
final class DefaultAttributeProtos {
  private final AtomicReferenceArray<Build.Attribute> outputProtos;
  private final AtomicReferenceArray<byte[]> hashBytes;
  private final LabelPrinter labelPrinter;
  private final boolean includeAttributeSourceAspects;

  DefaultAttributeProtos(
      int attributeCount, LabelPrinter labelPrinter, boolean includeAttributeSourceAspects) {
    this.outputProtos = new AtomicReferenceArray<>(attributeCount);
    this.hashBytes = new AtomicReferenceArray<>(attributeCount);
    this.labelPrinter = labelPrinter;
    this.includeAttributeSourceAspects = includeAttributeSourceAspects;
  }

  Build.Attribute getOutputProto(int index, Attribute attr, @Nullable Object value) {
    Build.Attribute proto = outputProtos.get(index);
    if (proto == null) {
      proto =
          AttributeFormatter.getAttributeProto(
              attr,
              value,
              /* explicitlySpecified= */ false,
              /* encodeBooleanAndTriStateAsIntegerAndString= */ true,
              /* sourceAspect= */ null,
              includeAttributeSourceAspects,
              labelPrinter);
      outputProtos.set(index, proto);
    }
    return proto;
  }

  byte[] getHashBytes(int index, Attribute attr, @Nullable Object value) {
    byte[] bytes = hashBytes.get(index);
    if (bytes == null) {
      bytes =
          AttributeFormatter.getAttributeProto(
                  attr,
                  value,
                  /* explicitlySpecified= */ false,
                  /* encodeBooleanAndTriStateAsIntegerAndString= */ false,
                  /* sourceAspect= */ null,
                  includeAttributeSourceAspects,
                  LabelPrinter.legacy())
              .toByteArray();
      hashBytes.set(index, bytes);
    }
    return bytes;
  }
}
