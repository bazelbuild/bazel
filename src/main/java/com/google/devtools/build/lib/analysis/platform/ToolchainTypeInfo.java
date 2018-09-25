package com.google.devtools.build.lib.analysis.platform;

import com.google.common.base.Strings;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.packages.NativeInfo;
import com.google.devtools.build.lib.packages.NativeProvider;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec.VisibleForSerialization;
import com.google.devtools.build.lib.skylarkbuildapi.platform.ToolchainTypeInfoApi;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;
import com.google.devtools.build.lib.skylarkinterface.SkylarkPrinter;
import java.util.Objects;

/** A provider that supplies information about a specific toolchain type. */
@Immutable
@AutoCodec
public class ToolchainTypeInfo extends NativeInfo implements ToolchainTypeInfoApi {
  /** Name used in Skylark for accessing this provider. */
  public static final String SKYLARK_NAME = "ToolchainTypeInfo";

  /** Skylark constructor and identifier for this provider. */
  @AutoCodec
  public static final NativeProvider<ToolchainTypeInfo> PROVIDER =
      new NativeProvider<ToolchainTypeInfo>(ToolchainTypeInfo.class, SKYLARK_NAME) {};

  private final Label typeLabel;

  public static ToolchainTypeInfo create(Label typeLabel, Location location) {
    return new ToolchainTypeInfo(typeLabel, location);
  }

  public static ToolchainTypeInfo create(Label typeLabel) {
    return create(typeLabel, Location.BUILTIN);
  }

  @VisibleForSerialization
  ToolchainTypeInfo(Label typeLabel, Location location) {
    super(PROVIDER, location);
    this.typeLabel = typeLabel;
  }

  @Override
  public Label typeLabel() {
    return typeLabel;
  }

  @Override
  public void repr(SkylarkPrinter printer) {
    printer.format("ToolchainTypeInfo(%s)", typeLabel);
  }

  @Override
  public int hashCode() {
    return Objects.hash(typeLabel);
  }

  @Override
  public boolean equals(Object other) {
    if (other == null || !(other instanceof ToolchainTypeInfo)) {
      return false;
    }

    ToolchainTypeInfo otherToolchainTypeInfo = (ToolchainTypeInfo) other;
    return Objects.equals(typeLabel, otherToolchainTypeInfo.typeLabel);
  }
}
