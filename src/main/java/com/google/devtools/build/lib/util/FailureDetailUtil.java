// Copyright 2020 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.util;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkNotNull;

import com.google.devtools.build.lib.server.FailureDetails;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.server.FailureDetails.Interrupted;
import com.google.protobuf.Descriptors.EnumValueDescriptor;
import com.google.protobuf.Descriptors.FieldDescriptor;
import com.google.protobuf.MessageOrBuilder;
import java.util.Map;

/** Utilities for constructing and managing {@link FailureDetail}s. */
public class FailureDetailUtil {

  private FailureDetailUtil() {}

  /**
   * Returns a {@link FailureDetail} specifying {@code code} in its {@link Interrupted} submessage.
   */
  public static FailureDetail interrupted(Interrupted.Code code) {
    return FailureDetail.newBuilder()
        .setInterrupted(Interrupted.newBuilder().setCode(code))
        .build();
  }

  /** Returns the registered {@link ExitCode} associated with a {@link FailureDetail} message. */
  public static ExitCode getExitCode(FailureDetail failureDetail) {
    // TODO(mschaller): Consider specializing for unregistered exit codes here, if absolutely
    //  necessary.
    int numericExitCode = FailureDetailUtil.getNumericExitCode(failureDetail);
    return checkNotNull(
        ExitCode.forCode(numericExitCode), "No ExitCode for numericExitCode %s", numericExitCode);
  }

  /** Returns the numeric exit code associated with a {@link FailureDetail} message. */
  private static int getNumericExitCode(FailureDetail failureDetail) {
    MessageOrBuilder categoryMsg = getCategorySubmessage(failureDetail);
    EnumValueDescriptor subcategoryDescriptor =
        getSubcategoryDescriptor(failureDetail, categoryMsg);
    return getNumericExitCode(subcategoryDescriptor);
  }

  /**
   * Returns the numeric exit code associated with a {@link FailureDetail} submessage's subcategory
   * enum value.
   */
  private static int getNumericExitCode(EnumValueDescriptor subcategoryDescriptor) {
    checkArgument(
        subcategoryDescriptor.getOptions().hasExtension(FailureDetails.metadata),
        "Enum value %s has no FailureDetails.metadata",
        subcategoryDescriptor);
    return subcategoryDescriptor.getOptions().getExtension(FailureDetails.metadata).getExitCode();
  }

  /**
   * Returns the category submessage, i.e. the message in {@link FailureDetail}'s oneof. Throws if
   * none of those fields are set.
   */
  private static MessageOrBuilder getCategorySubmessage(FailureDetail failureDetail) {
    MessageOrBuilder categoryMsg = null;
    for (Map.Entry<FieldDescriptor, Object> entry : failureDetail.getAllFields().entrySet()) {
      FieldDescriptor fieldDescriptor = entry.getKey();
      if (isCategoryField(fieldDescriptor)) {
        categoryMsg = (MessageOrBuilder) entry.getValue();
        break;
      }
    }
    return checkNotNull(
        categoryMsg, "FailureDetail missing category submessage: %s", failureDetail);
  }

  /**
   * Returns whether the {@link FieldDescriptor} describes a field in {@link FailureDetail}'s oneof.
   *
   * <p>Uses the field number criteria described in failure_details.proto.
   */
  private static boolean isCategoryField(FieldDescriptor fieldDescriptor) {
    int fieldNum = fieldDescriptor.getNumber();
    return 100 < fieldNum && fieldNum <= 10_000;
  }

  /**
   * Returns the enum value descriptor for the enum field with field number 1 in the {@link
   * FailureDetail}'s category submessage.
   */
  private static EnumValueDescriptor getSubcategoryDescriptor(
      FailureDetail failureDetail, MessageOrBuilder categoryMsg) {
    FieldDescriptor fieldNumberOne = categoryMsg.getDescriptorForType().findFieldByNumber(1);
    checkNotNull(
        fieldNumberOne, "FailureDetail category submessage has no field #1: %s", failureDetail);
    Object fieldNumberOneVal = categoryMsg.getField(fieldNumberOne);
    checkArgument(
        fieldNumberOneVal instanceof EnumValueDescriptor,
        "FailureDetail category submessage has non-enum field #1: %s",
        failureDetail);
    return (EnumValueDescriptor) fieldNumberOneVal;
  }
}
