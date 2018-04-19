package com.google.devtools.build.lib.skyframe;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Interner;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.concurrent.BlazeInterners;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.skyframe.AbstractSkyKey;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyValue;

import java.util.Objects;

public class WorkspaceMappingsValue implements SkyValue {

    private final ImmutableMap<RepositoryName, RepositoryName> workspaceMappings;

    private WorkspaceMappingsValue(ImmutableMap<RepositoryName, RepositoryName> workspaceMappings) {
        this.workspaceMappings = workspaceMappings;
    }

    /** Returns the workspace mappings. */
    public ImmutableMap<RepositoryName, RepositoryName> getWorkspaceMappings() {
        return workspaceMappings;
    }

    /** Returns the {@link Key} for {@link WorkspaceMappingsValue}s. */
    public static Key key(RepositoryName repositoryName) {
        return WorkspaceMappingsValue.Key.create(repositoryName);
    }

    /** Returns a {@link WorkspaceMappingsValue} for a workspace with the given name. */
    public static WorkspaceMappingsValue withMappings(ImmutableMap<RepositoryName, RepositoryName> workspaceMappings) {
        return new WorkspaceMappingsValue(Preconditions.checkNotNull(workspaceMappings));
    }

  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (o == null || getClass() != o.getClass()) return false;
    WorkspaceMappingsValue that = (WorkspaceMappingsValue) o;
    return Objects.equals(workspaceMappings, that.workspaceMappings);
  }

  @Override
  public int hashCode() {
    return Objects.hash(workspaceMappings);
  }

    @AutoCodec.VisibleForSerialization
    @AutoCodec
    public static class Key extends AbstractSkyKey<RepositoryName> {

        private static final Interner<Key> interner = BlazeInterners.newWeakInterner();

        private Key(RepositoryName arg) {
            super(arg);
        }

        @AutoCodec.VisibleForSerialization
        @AutoCodec.Instantiator
        static Key create(RepositoryName arg) {
            return interner.intern(new Key(arg));
        }

        @Override
        public SkyFunctionName functionName() {
            return SkyFunctions.WORKSPACE_MAPPINGS;
        }
    }
}
