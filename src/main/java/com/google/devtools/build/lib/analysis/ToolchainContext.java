package com.google.devtools.build.lib.analysis;

import static java.util.stream.Collectors.joining;

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableBiMap;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.platform.PlatformInfo;
import com.google.devtools.build.lib.analysis.platform.ToolchainInfo;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.skylarkbuildapi.ToolchainContextApi;
import com.google.devtools.build.lib.skylarkinterface.SkylarkPrinter;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.EvalUtils;

/** Represents the data needed for a specific target's use of toolchains and platforms. */
@AutoValue
@Immutable
@ThreadSafe
public abstract class ToolchainContext implements ToolchainContextApi {

  /**
   * Create a new {@link ToolchainContext} using the supplied data about required toolchain types
   * and the resolved toolchain providers.
   *
   * @param targetDescription a context description of the target, used for error messaging
   * @param executionPlatform the selected execution platform that these toolchains use
   * @param targetPlatform the target platform that these toolchains generate output for
   * @param requiredToolchainTypes the toolchain types that were requested
   * @param toolchainLabels the labels of the specific toolchains being used
   * @param toolchains the actual {@link ToolchainInfo toolchain providers} to be used
   * @return information about the toolchains to be used by the requesting configured target
   */
  protected static ToolchainContext create(
      String targetDescription,
      PlatformInfo executionPlatform,
      PlatformInfo targetPlatform,
      ImmutableSet<Label> requiredToolchainTypes,
      ImmutableBiMap<Label, Label> toolchainLabels,
      ImmutableMap<Label, ToolchainInfo> toolchains) {

    return new AutoValue_ToolchainContext(
        targetDescription,
        executionPlatform,
        targetPlatform,
        requiredToolchainTypes,
        toolchainLabels,
        toolchains);
  }

  /** Returns a description of the target being used, for error messaging. */
  abstract String targetDescription();

  /** Returns the selected execution platform that these toolchains use. */
  public abstract PlatformInfo executionPlatform();

  /** Returns the target platform that these toolchains generate output for. */
  public abstract PlatformInfo targetPlatform();

  /** Returns the toolchain types that were requested. */
  public abstract ImmutableSet<Label> requiredToolchainTypes();

  // DO NOT USE: Internal only.
  abstract ImmutableBiMap<Label, Label> toolchainTypeToResolved();

  // DO NOT USE: Internal only.
  abstract ImmutableMap<Label, ToolchainInfo> toolchains();

  /** Returns the labels of the specific toolchains being used. */
  public ImmutableSet<Label> resolvedToolchainLabels() {
    return ImmutableSet.copyOf(toolchainTypeToResolved().values());
  }

  /** Returns the toolchain for the given type */
  public ToolchainInfo forToolchainType(Label toolchainType) {
    return toolchains().get(toolchainType);
  }

  // Implement SkylarkValue and SkylarkIndexable.

  @Override
  public boolean isImmutable() {
    return true;
  }

  @Override
  public void repr(SkylarkPrinter printer) {
    printer.append("<toolchain_context.resolved_labels: ");
    printer.append(toolchains().keySet().stream().map(Label::toString).collect(joining(", ")));
    printer.append(">");
  }

  private Label transformKey(Object key, Location loc) throws EvalException {
    if (key instanceof Label) {
      return (Label) key;
    } else if (key instanceof String) {
      Label toolchainType;
      String rawLabel = (String) key;
      try {
        toolchainType = Label.parseAbsolute(rawLabel, ImmutableMap.of());
      } catch (LabelSyntaxException e) {
        throw new EvalException(
            loc, String.format("Unable to parse toolchain %s: %s", rawLabel, e.getMessage()), e);
      }
      return toolchainType;
    } else {
      throw new EvalException(
          loc,
          String.format(
              "Toolchains only supports indexing by toolchain type, got %s instead",
              EvalUtils.getDataTypeName(key)));
    }
  }

  @Override
  public ToolchainInfo getIndex(Object key, Location loc) throws EvalException {
    Label toolchainType = transformKey(key, loc);

    if (!requiredToolchainTypes().contains(toolchainType)) {
      throw new EvalException(
          loc,
          String.format(
              "In %s, toolchain type %s was requested but only types [%s] are configured",
              targetDescription(),
              toolchainType,
              requiredToolchainTypes().stream().map(Label::toString).collect(joining())));
    }
    return forToolchainType(toolchainType);
  }

  @Override
  public boolean containsKey(Object key, Location loc) throws EvalException {
    Label toolchainType = transformKey(key, loc);
    return toolchains().containsKey(toolchainType);
  }
}
