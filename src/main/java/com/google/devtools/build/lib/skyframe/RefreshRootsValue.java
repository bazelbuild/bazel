package com.google.devtools.build.lib.skyframe;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.skyframe.ExternalFilesHelper.FileType;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunction.Environment;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.Objects;
import java.util.Set;
import javax.annotation.Nullable;

@AutoCodec
public class RefreshRootsValue implements SkyValue {
  private final ImmutableMap<PathFragment, RepositoryName> roots;

  @AutoCodec.VisibleForSerialization @AutoCodec
  static final SkyKey REFRESH_ROOTS_KEY = () -> SkyFunctions.REFRESH_ROOTS;

  public RefreshRootsValue(
      ImmutableMap<PathFragment, RepositoryName> roots) {
    this.roots = roots;
  }

  public static SkyKey key() {
    return REFRESH_ROOTS_KEY;
  }

  public ImmutableMap<PathFragment, RepositoryName> getRoots() {
    return roots;
  }

  public boolean isEmpty() {
    return roots.isEmpty();
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (o == null || getClass() != o.getClass()) {
      return false;
    }
    RefreshRootsValue that = (RefreshRootsValue) o;
    return roots.equals(that.roots);
  }

  @Override
  public int hashCode() {
    return Objects.hash(roots);
  }

  @Nullable
  public static RepositoryName getRepositoryForRefreshRoot(
      final BlacklistedPackagePrefixesValue blackListed,
      final RefreshRootsValue refreshRootsValue,
      final FileType fileType,
      final RootedPath rootedPath) {

    // todo seems it is already checked before; remove this
    if (fileType == FileType.INTERNAL) {
      // Only black listed files and directories may be refreshed by external repositories
      if (!Boolean.TRUE.equals(blackListed.isUnderBlacklisted(rootedPath))) {
        return null;
      }
    }

    ImmutableMap<PathFragment, RepositoryName> roots = refreshRootsValue.getRoots();
    // todo how can we improve here???
    PathFragment relativePath = rootedPath.getRootRelativePath();
    for (PathFragment refreshRoot : roots.keySet()) {
      if (relativePath.startsWith(refreshRoot)) {
        return roots.get(refreshRoot);
      }
    }
    return null;
  }
}
