package com.google.devtools.build.lib.analysis.platform;

import com.google.auto.value.AutoValue;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.BuiltinProvider;
import com.google.devtools.build.lib.packages.NativeInfo;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.starlarkbuildapi.platform.PlatformInfoApi;
import java.util.List;

@Immutable
@AutoCodec
public class FatPlatformInfo extends NativeInfo {

  /**
   * Name used in Starlark for accessing this provider.
   */
  public static final String STARLARK_NAME = "PlatformInfo";

  /**
   * Provider singleton constant.
   */
  public static final BuiltinProvider<FatPlatformInfo> PROVIDER = new FatPlatformInfo.Provider();

  /**
   * Provider for {@link FatPlatformInfo} objects.
   */
  private static class Provider extends BuiltinProvider<FatPlatformInfo> {
    private Provider() {
      super(STARLARK_NAME, FatPlatformInfo.class);
    }
  }

  private final Label label;
  private final ImmutableList<PlatformInfo> platforms;

  public FatPlatformInfo(Label label, List<PlatformInfo> platforms) {
    super();
    Preconditions.checkArgument(platforms.size() >= 1, "Fat platforms require at least one platform");
    this.label = label;
    this.platforms = ImmutableList.copyOf(platforms);
  }

  @Override
  public BuiltinProvider<FatPlatformInfo> getProvider() {
    return PROVIDER;
  }

  public Label label() {
    return label;
  }
  public ImmutableList<PlatformInfo> platforms() {
    return platforms;
  }
  public PlatformInfo getDefaultPlatform() {
    return platforms().get(0);
  }
}
