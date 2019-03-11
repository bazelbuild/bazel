package com.google.devtools.build.lib.skyframe;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.WorkspaceFileValue;
import com.google.devtools.build.lib.repository.ExternalPackageException;
import com.google.devtools.build.lib.repository.ExternalPackageUtil;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.Map;
import java.util.stream.Collectors;
import javax.annotation.Nullable;

public class RefreshRootsFunction implements SkyFunction {

  @Nullable
  @Override
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws SkyFunctionException, InterruptedException {

    RootedPath workspacePath = ExternalPackageUtil.getWorkspacePath(env);
    if (env.valuesMissing()) {
      return null;
    }
    SkyKey workspaceKey = WorkspaceFileValue.key(workspacePath);
    WorkspaceFileValue value = (WorkspaceFileValue) env.getValue(workspaceKey);
    if (value == null) {
      return null;
    }
    Package externalPackage = value.getPackage();
    // if (externalPackage.containsErrors()) {
    //   // todo (ichern, prototype) exception kind
    //   throw new IllegalStateException("Could not load //external package");
    // }
    Map<String, RepositoryName> map = externalPackage.getRefreshRootsToRepository();
    ImmutableMap<PathFragment, RepositoryName> asRootsMap = ImmutableMap.copyOf(map.keySet().stream()
        .collect(Collectors.toMap(PathFragment::create, map::get)));

    return new RefreshRootsValue(asRootsMap);
  }

  @Nullable
  @Override
  public String extractTag(SkyKey skyKey) {
    return null;
  }
}
