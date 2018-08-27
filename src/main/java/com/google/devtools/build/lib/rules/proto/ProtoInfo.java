
package com.google.devtools.build.lib.rules.proto;

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec.VisibleForSerialization;
import com.google.devtools.build.lib.skylarkbuildapi.proto.ProtoInfoApi;
import com.google.devtools.build.lib.syntax.Environment;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.packages.BuiltinProvider;
import com.google.devtools.build.lib.packages.NativeInfo;

@Immutable
@AutoCodec
public final class ProtoInfo extends NativeInfo implements ProtoInfoApi<Artifact> {
  public static final String SKYLARK_NAME = "ProtoInfo";

  public static final ProtoInfoProvider PROVIDER = new ProtoInfoProvider();

  public static final ProtoInfo EMPTY = ProtoInfo.Builder.create().build();

  @VisibleForSerialization
  @AutoCodec.Instantiator
  ProtoInfo(
      Location location) {
    super(PROVIDER, location);
  }

  /** Provider class for {@link ProtoInfo} objects. */
  public static class ProtoInfoProvider extends BuiltinProvider<ProtoInfo>
      implements ProtoInfoProviderApi {
    private ProtoInfoProvider() {
      super(SKYLARK_NAME, ProtoInfo.class);
    }

    @Override
    @SuppressWarnings({"unchecked"})
    public ProtoInfo protoInfo(
        Location loc,
        Environment env) throws EvalException {
      return new ProtoInfo(loc);
        }
  }

  /**
   * A Builder for {@link ProtoInfo}.
   */
  public static class Builder {
    private Location location = Location.BUILTIN;

    private Builder() {
    }

    public static Builder create() {
      return new Builder();
    }

    public ProtoInfo build() {
      return new ProtoInfo(location);
    }
  }
}
