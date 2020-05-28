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
import com.google.protobuf.Descriptors.EnumValueDescriptor;
import com.google.protobuf.Descriptors.FieldDescriptor;
import com.google.protobuf.MessageOrBuilder;
import java.util.Comparator;
import java.util.Map;
import java.util.Objects;
import javax.annotation.Nullable;

/** An {@link ExitCode} and an optional {@link FailureDetail}. */
public class DetailedExitCode {
  private final ExitCode exitCode;
  @Nullable private final FailureDetail failureDetail;

  private DetailedExitCode(ExitCode exitCode, @Nullable FailureDetail failureDetail) {
    this.exitCode = exitCode;
    this.failureDetail = failureDetail;
  }

  public ExitCode getExitCode() {
    return exitCode;
  }

  /** Returns the registered {@link ExitCode} associated with a {@link FailureDetail} message. */
  private static ExitCode getExitCode(FailureDetail failureDetail) {
    // TODO(mschaller): Consider specializing for unregistered exit codes here, if absolutely
    //  necessary.
    int numericExitCode = getNumericExitCode(failureDetail);
    return checkNotNull(
        ExitCode.forCode(numericExitCode), "No ExitCode for numericExitCode %s", numericExitCode);
  }

  @Nullable
  public FailureDetail getFailureDetail() {
    return failureDetail;
  }

  public boolean isSuccess() {
    return exitCode.equals(ExitCode.SUCCESS);
  }

  /** Returns a {@link DetailedExitCode} specifying success (i.e. exit code 0). */
  public static DetailedExitCode success() {
    return new DetailedExitCode(ExitCode.SUCCESS, null);
  }

  /**
   * Returns a {@link DetailedExitCode} specifying {@link ExitCode} but no {@link FailureDetail}.
   *
   * <p>This method exists in order to allow for code which has not yet been wired for {@link
   * FailureDetail) support to interact with {@link FailureDetail}-handling code infrastructure.
   *
   * <p>Callsites should migrate to using either:
   *
   * <ul>
   *   <li>{@link #of(ExitCode, FailureDetail)}, when they're wired for {@link FailureDetail}
   *   support but not yet ready to have {@link FailureDetail} metadata determine exit code behavior
   *   <li>{@link #of(FailureDetail)}, when changing exit code behavior is desired.
   * </ul>
   *
   */
  public static DetailedExitCode justExitCode(ExitCode exitCode) {
    return new DetailedExitCode(checkNotNull(exitCode), null);
  }

  /**
   * Returns a {@link DetailedExitCode} combining the provided {@link FailureDetail} and {@link
   * ExitCode}.
   *
   * <p>This method exists in order to allow for the introduction of new {@link
   * FailureDetail)-handling code infrastructure without requiring any simultaneous change in exit
   * code behavior.
   *
   * <p>Callsites should migrate to using {@link #of(FailureDetail)} instead.
   */
  // TODO(b/138456686): consider controlling this behavior by flag if migration appears risky.
  public static DetailedExitCode of(ExitCode exitCode, FailureDetail failureDetail) {
    return new DetailedExitCode(checkNotNull(exitCode), checkNotNull(failureDetail));
  }

  /**
   * Returns a {@link DetailedExitCode} whose {@link ExitCode} is chosen referencing {@link
   * FailureDetail}'s metadata.
   */
  public static DetailedExitCode of(FailureDetail failureDetail) {
    return new DetailedExitCode(getExitCode(failureDetail), failureDetail);
  }

  @Override
  public int hashCode() {
    return Objects.hash(exitCode, failureDetail);
  }

  @Override
  public boolean equals(Object obj) {
    if (obj == this) {
      return true;
    }
    if (!(obj instanceof DetailedExitCode)) {
      return false;
    }
    DetailedExitCode that = (DetailedExitCode) obj;
    return this.exitCode.equals(that.exitCode)
        && Objects.equals(this.failureDetail, that.failureDetail);
  }

  @Override
  public String toString() {
    return String.format(
        "DetailedExitCode{exitCode=%s, failureDetail=%s}", exitCode, failureDetail);
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

  /**
   * A comparator to determine the reporting priority of {@link DetailedExitCode}.
   *
   * <p>Priority: infrastructure exit codes > non-infrastructure exit codes > null exit codes, with
   * exit codes that contain failure details taking priority within each class.
   */
  public static class DetailedExitCodeComparator implements Comparator<DetailedExitCode> {
    public static final DetailedExitCodeComparator INSTANCE = new DetailedExitCodeComparator();

    private DetailedExitCodeComparator() {}

    @Nullable
    public static DetailedExitCode chooseMoreImportantWithFirstIfTie(
        @Nullable DetailedExitCode first, @Nullable DetailedExitCode second) {
      return INSTANCE.compare(first, second) >= 0 ? first : second;
    }

    @Override
    public int compare(DetailedExitCode c1, DetailedExitCode c2) {
      // returns POSITIVE result when the priority of c1 is HIGHER than the priority of c2
      return getPriority(c1) - getPriority(c2);
    }

    private static int getPriority(DetailedExitCode code) {
      if (code == null) {
        return 0;
      } else {
        int codeClass = code.getExitCode().isInfrastructureFailure() ? 4 : 2;
        return codeClass + (code.getFailureDetail() != null ? 1 : 0);
      }
    }
  }
}
