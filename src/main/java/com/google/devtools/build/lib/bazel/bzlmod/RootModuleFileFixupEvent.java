package com.google.devtools.build.lib.bazel.bzlmod;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import java.util.List;

/**
 * Generated when incorrect use_repo calls are detected in the root module file according to {@link
 * ModuleExtensionMetadata}, this event contains the buildozer commands required to bring the root
 * module file into the expected state.
 */
public final class RootModuleFileFixupEvent implements ExtendedEventHandler.Postable {
  private final ImmutableList<String> buildozerCommands;
  private final ModuleExtensionUsage usage;

  public RootModuleFileFixupEvent(List<String> buildozerCommands, ModuleExtensionUsage usage) {
    this.buildozerCommands = ImmutableList.copyOf(buildozerCommands);
    this.usage = usage;
  }

  /** The buildozer commands required to bring the root module file into the expected state. */
  public ImmutableList<String> getBuildozerCommands() {
    return buildozerCommands;
  }

  /** A human-readable message describing the fixup after it has been applied. */
  public String getSuccessMessage() {
    String extensionId = usage.getExtensionBzlFile() + "%" + usage.getExtensionName();
    return usage
        .getIsolationKey()
        .map(
            key ->
                String.format(
                    "Updated use_repo calls for isolated usage '%s' of %s",
                    key.getUsageExportedName(), extensionId))
        .orElseGet(() -> String.format("Updated use_repo calls for %s", extensionId));
  }

  @Override
  public boolean storeForReplay() {
    return true;
  }
}
