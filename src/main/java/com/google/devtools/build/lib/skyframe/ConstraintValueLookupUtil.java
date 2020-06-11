// Copyright 2018 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.platform.ConstraintValueInfo;
import com.google.devtools.build.lib.analysis.platform.PlatformProviderUtils;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.NoSuchThingException;
import com.google.devtools.build.skyframe.SkyFunction.Environment;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.ValueOrException3;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import javax.annotation.Nullable;

/** Helper class that looks up {@link ConstraintValueInfo} data. */
public class ConstraintValueLookupUtil {

  @Nullable
  public static List<ConstraintValueInfo> getConstraintValueInfo(
      Iterable<ConfiguredTargetKey> constraintValueKeys, Environment env)
      throws InterruptedException, InvalidConstraintValueException {

    Map<
            SkyKey,
            ValueOrException3<
                ConfiguredValueCreationException, NoSuchThingException, ActionConflictException>>
        values =
            env.getValuesOrThrow(
                constraintValueKeys,
                ConfiguredValueCreationException.class,
                NoSuchThingException.class,
                ActionConflictException.class);
    boolean valuesMissing = env.valuesMissing();
    List<ConstraintValueInfo> constraintValues = valuesMissing ? null : new ArrayList<>();
    for (ConfiguredTargetKey key : constraintValueKeys) {
      ConstraintValueInfo constraintValueInfo = findConstraintValueInfo(key, values.get(key));
      if (!valuesMissing && constraintValueInfo != null) {
        constraintValues.add(constraintValueInfo);
      }
    }
    if (valuesMissing) {
      return null;
    }
    return constraintValues;
  }

  /**
   * Returns the {@link ConstraintValueInfo} provider from the {@link ConfiguredTarget} in the
   * {@link ValueOrException3}, or {@code null} if the {@link ConfiguredTarget} is not present. If
   * the {@link ConfiguredTarget} does not have a {@link ConstraintValueInfo} provider, a {@link
   * InvalidConstraintValueException} is thrown.
   */
  @Nullable
  private static ConstraintValueInfo findConstraintValueInfo(
      ConfiguredTargetKey key,
      ValueOrException3<
              ConfiguredValueCreationException, NoSuchThingException, ActionConflictException>
          valueOrException)
      throws InvalidConstraintValueException {

    try {
      ConfiguredTargetValue ctv = (ConfiguredTargetValue) valueOrException.get();
      if (ctv == null) {
        return null;
      }

      ConfiguredTarget configuredTarget = ctv.getConfiguredTarget();
      ConstraintValueInfo constraintValueInfo =
          PlatformProviderUtils.constraintValue(configuredTarget);
      if (constraintValueInfo == null) {
        throw new InvalidConstraintValueException(configuredTarget.getLabel());
      }

      return constraintValueInfo;
    } catch (ConfiguredValueCreationException e) {
      throw new InvalidConstraintValueException(key.getLabel(), e);
    } catch (NoSuchThingException e) {
      throw new InvalidConstraintValueException(key.getLabel(), e);
    } catch (ActionConflictException e) {
      throw new InvalidConstraintValueException(key.getLabel(), e);
    }
  }

  /** Exception used when a constraint value label is not a valid constraint value. */
  public static final class InvalidConstraintValueException extends ToolchainException {
    InvalidConstraintValueException(Label label) {
      super(formatError(label));
    }

    InvalidConstraintValueException(Label label, ConfiguredValueCreationException e) {
      super(formatError(label), e);
    }

    public InvalidConstraintValueException(Label label, NoSuchThingException e) {
      // Just propagate the inner exception, because it's directly actionable.
      super(e);
    }

    public InvalidConstraintValueException(Label label, ActionConflictException e) {
      super(formatError(label), e);
    }

    private static String formatError(Label label) {
      return String.format(
          "Target %s was referenced as a constraint_value, "
              + "but does not provide ConstraintValueInfo",
          label);
    }
  }

  /**
   * Used to declare all the exception types that can be wrapped in the exception thrown by {@link
   * #compute}.
   */
  public static class ConstraintValueLookupFunctionException extends SkyFunctionException {

    public ConstraintValueLookupFunctionException(
        InvalidConstraintValueException cause, Transience transience) {
      super(cause, transience);
    }
  }
}
