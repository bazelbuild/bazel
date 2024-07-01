package com.google.devtools.build.lib.analysis.constraints;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.IncompatiblePlatformProvider;
import com.google.devtools.build.lib.analysis.platform.ConstraintValueInfo;
import com.google.devtools.build.lib.cmdline.Label;

/** Utility functions for handling incompatible targets. */
public class IncompatibleTargetUtils {

  /**
   * Assembles the explanation for a platform incompatibility.
   *
   * <p>The goal is to print out the dependency chain and the constraint that wasn't satisfied so
   * that the user can immediately figure out why a certain target is incompatible.
   *
   * @param label the label of the incompatible target
   * @param configurationChecksum the configuration checksum of the incompatible target
   * @param provider the {@link IncompatiblePlatformProvider} of the incompatible target
   * @return the verbose error message to show to the user.
   */
  static String reportOnIncompatibility(
      Label label, String configurationChecksum, IncompatiblePlatformProvider provider) {
    StringBuilder message =
        new StringBuilder(
            String.format(
                "\nDependency chain:\n    %s (%s)", label, configurationChecksum.substring(0, 6)));

    // TODO(austinschuh): While the first error is helpful, reporting all the errors at once would
    // save the user bazel round trips.
    ConfiguredTarget target;
    while (true) {
      ImmutableList<ConfiguredTarget> targetList = provider.targetsResponsibleForIncompatibility();
      if (targetList == null) {
        target = null;
      } else {
        target = targetList.getFirst();
      }
      if (target == null) {
        break;
      }
      message.append(
          String.format(
              "\n    %s (%s)",
              target.getLabel(), target.getConfigurationChecksum().substring(0, 6)));
      provider = target.get(IncompatiblePlatformProvider.PROVIDER);
    }

    message.append(
        String.format(
            "   <-- target platform (%s) didn't satisfy constraint", provider.targetPlatform()));
    if (provider.constraintsResponsibleForIncompatibility().size() == 1) {
      message
          .append(" ")
          .append(provider.constraintsResponsibleForIncompatibility().getFirst().label());
      return message.toString();
    }

    message.append("s [");

    boolean first = true;
    for (ConstraintValueInfo constraintValueInfo :
        provider.constraintsResponsibleForIncompatibility()) {
      if (first) {
        first = false;
      } else {
        message.append(", ");
      }
      message.append(constraintValueInfo.label());
    }

    message.append("]");

    return message.toString();
  }

  private IncompatibleTargetUtils() {}
}
