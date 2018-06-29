package com.google.devtools.build.lib.skyframe;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.platform.PlatformInfo;
import com.google.devtools.build.lib.analysis.platform.PlatformProviderUtils;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.NoSuchThingException;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetFunction.ConfiguredValueCreationException;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.ValueOrException;
import com.google.devtools.build.skyframe.ValueOrException3;
import java.util.HashMap;
import java.util.Map;
import javax.annotation.Nullable;

/** {@link SkyFunction} that looks up {@link PlatformInfo} data. */
public class PlatformLookupFunction implements SkyFunction {

  @Nullable
  @Override
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws SkyFunctionException, InterruptedException {
    ImmutableList<ConfiguredTargetKey> platformKeys =
        ((PlatformLookupValue.Key) skyKey).platformKeys();

    try {
      Map<
              SkyKey,
              ValueOrException3<
                  ConfiguredValueCreationException, NoSuchThingException, ActionConflictException>>
          values =
              env.getValuesOrThrow(
                  platformKeys,
                  ConfiguredValueCreationException.class,
                  NoSuchThingException.class,
                  ActionConflictException.class);
      boolean valuesMissing = env.valuesMissing();
      Map<ConfiguredTargetKey, PlatformInfo> platforms = valuesMissing ? null : new HashMap<>();
      for (ConfiguredTargetKey key : platformKeys) {
        PlatformInfo platformInfo = findPlatformInfo(key, values.get(key));
        if (!valuesMissing && platformInfo != null) {
          platforms.put(key, platformInfo);
        }
      }
      if (valuesMissing) {
        return null;
      }
      return PlatformLookupValue.create(platforms);
    } catch (InvalidPlatformException e) {
      throw new PlatformLookupFunctionException(e, Transience.PERSISTENT);
    }
  }

  /**
   * Returns the {@link PlatformInfo} provider from the {@link ConfiguredTarget} in the {@link
   * ValueOrException}, or {@code null} if the {@link ConfiguredTarget} is not present. If the
   * {@link ConfiguredTarget} does not have a {@link PlatformInfo} provider, a {@link
   * InvalidPlatformException} is thrown.
   */
  @Nullable
  private PlatformInfo findPlatformInfo(
      ConfiguredTargetKey key,
      ValueOrException3<
              ConfiguredValueCreationException, NoSuchThingException, ActionConflictException>
          valueOrException)
      throws InvalidPlatformException {

    try {
      ConfiguredTargetValue ctv = (ConfiguredTargetValue) valueOrException.get();
      if (ctv == null) {
        return null;
      }

      ConfiguredTarget configuredTarget = ctv.getConfiguredTarget();
      PlatformInfo platformInfo = PlatformProviderUtils.platform(configuredTarget);
      if (platformInfo == null) {
        throw new InvalidPlatformException(configuredTarget.getLabel());
      }

      return platformInfo;
    } catch (ConfiguredValueCreationException e) {
      throw new InvalidPlatformException(key.getLabel(), e);
    } catch (NoSuchThingException e) {
      throw new InvalidPlatformException(key.getLabel(), e);
    } catch (ActionConflictException e) {
      throw new InvalidPlatformException(key.getLabel(), e);
    }
  }

  @Nullable
  @Override
  public String extractTag(SkyKey skyKey) {
    return null;
  }

  /** Exception used when a platform label is not a valid platform. */
  public static final class InvalidPlatformException extends ToolchainException {
    InvalidPlatformException(Label label) {
      super(formatError(label));
    }

    InvalidPlatformException(Label label, ConfiguredValueCreationException e) {
      super(formatError(label), e);
    }

    public InvalidPlatformException(Label label, NoSuchThingException e) {
      super(formatError(label), e);
    }

    public InvalidPlatformException(Label label, ActionConflictException e) {
      super(formatError(label), e);
    }

    private static String formatError(Label label) {
      return String.format(
          "Target %s was referenced as a platform, but does not provide PlatformInfo", label);
    }
  }

  /**
   * Used to declare all the exception types that can be wrapped in the exception thrown by {@link
   * #compute}.
   */
  public static class PlatformLookupFunctionException extends SkyFunctionException {

    public PlatformLookupFunctionException(InvalidPlatformException cause, Transience transience) {
      super(cause, transience);
    }
  }
}
