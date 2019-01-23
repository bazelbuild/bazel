package com.google.devtools.build.lib.skyframe;

import com.google.common.base.Function;
import com.google.common.base.Predicate;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.skyframe.CycleInfo;
import com.google.devtools.build.skyframe.CyclesReporter;
import com.google.devtools.build.skyframe.SkyKey;
import java.util.Optional;

public class RegisteredToolchainsCycleReporter implements CyclesReporter.SingleCycleReporter {

  private static final Predicate<SkyKey> IS_REGISTERED_TOOLCHAINS_SKY_KEY =
      SkyFunctions.isSkyFunction(SkyFunctions.REGISTERED_TOOLCHAINS);

  private static final Predicate<SkyKey> IS_CONFIGURED_TARGET_SKY_KEY =
      SkyFunctions.isSkyFunction(SkyFunctions.CONFIGURED_TARGET);

  private static final Predicate<SkyKey> IS_TOOLCHAIN_RESOLUTION_SKY_KEY =
      SkyFunctions.isSkyFunction(SkyFunctions.TOOLCHAIN_RESOLUTION);

  @Override
  public boolean maybeReportCycle(
      SkyKey topLevelKey,
      CycleInfo cycleInfo,
      boolean alreadyReported,
      ExtendedEventHandler eventHandler) {
    ImmutableList<SkyKey> pathToCycle = cycleInfo.getPathToCycle();
    ImmutableList<SkyKey> cycle = cycleInfo.getCycle();
    if (pathToCycle.isEmpty()) {
      return false;
    } else if (alreadyReported) {
      return true;
    } else if (!Iterables.any(cycle, IS_REGISTERED_TOOLCHAINS_SKY_KEY)
        || !Iterables.any(cycle, IS_CONFIGURED_TARGET_SKY_KEY)
        || !Iterables.any(cycle, IS_TOOLCHAIN_RESOLUTION_SKY_KEY)) {
      return false;
    }

    // Find the ConfiguredTargetKey, this should tell the problem.
    Optional<SkyKey> configuredTargetKey =
        cycle.stream().filter(IS_CONFIGURED_TARGET_SKY_KEY).findFirst();
    if (!configuredTargetKey.isPresent()) {
      return false;
    }

    Function<SkyKey, String> printer =
        new Function<SkyKey, String>() {
          @Override
          public String apply(SkyKey input) {
            if (input.argument() instanceof ConfiguredTargetKey) {
              Label label = ((ConfiguredTargetKey) input.argument()).getLabel();
              return label.toString();
            }
            if (input.argument() instanceof RegisteredToolchainsValue.Key) {
              return "All registered toolchains";
            }
            if (input.argument() instanceof ToolchainResolutionValue.Key) {
              Label toolchainType =
                  ((ToolchainResolutionValue.Key) input.argument()).toolchainTypeLabel();
              return String.format("toolchain type %s", toolchainType.toString());
            } else {
              throw new UnsupportedOperationException();
            }
          }
        };

    StringBuilder cycleMessage =
        new StringBuilder()
            .append("Misconfigured toolchains: ")
            .append(printer.apply(configuredTargetKey.get()))
            .append(" is declared as a toolchain but has inappropriate dependencies.")
            .append(" Declared toolchains should be created with the 'toolchain' rule")
            .append(" and should not have dependencies that themselves require toolchains.");

    AbstractLabelCycleReporter.printCycle(cycleInfo.getCycle(), cycleMessage, printer);
    eventHandler.handle(Event.error(null, cycleMessage.toString()));
    return true;
  }
}
