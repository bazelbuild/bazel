package com.google.devtools.build.lib.skyframe;

import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.cmdline.RepositoryMapping;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.packages.LabelConverter;
import com.google.devtools.build.lib.packages.PackageArgs;
import com.google.devtools.build.lib.skyframe.RepoFileFunction.BadRepoFileException;
import com.google.devtools.build.lib.skyframe.RepoFileFunction.RepoFileFunctionException;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.AbstractSkyKey;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.Map;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;

public class PackageArgsFunction implements SkyFunction  {
  public static final PackageArgsFunction INSTANCE = new PackageArgsFunction();

  public static final class PackageArgsValue implements SkyValue {
    public static final PackageArgsValue EMPTY = new PackageArgsValue(PackageArgs.EMPTY);

    private final PackageArgs packageArgs;

    public PackageArgsValue(PackageArgs packageArgs) {
      this.packageArgs = packageArgs;
    }

    public PackageArgs getPackageArgs() {
      return packageArgs;
    }

    @Override
    public int hashCode() {
      return packageArgs.hashCode();
    }

    @Override
    public boolean equals(Object other) {
      if (other instanceof PackageArgsValue that) {
        return that.packageArgs.equals(packageArgs);
      } else {
        return false;
      }
    }
  }

  public static Key key(RepositoryName repoName) {
    return new Key(repoName);
  }

  /** Key type for {@link RepoFileValue}. */
  public static class Key extends AbstractSkyKey<RepositoryName> {

    private Key(RepositoryName repoName) {
      super(repoName);
    }

    @Override
    public SkyFunctionName functionName() {
      return SkyFunctions.PACKAGE_ARGS;
    }
  }

  private PackageArgsFunction() {}

  @Nullable
  @Override
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws SkyFunctionException, InterruptedException {
    RepositoryName repositoryName = (RepositoryName) skyKey.argument();

    RepoFileValue repoFileValue = (RepoFileValue) env.getValue(RepoFileValue.key(repositoryName));
    RepositoryMappingValue repositoryMappingValue = (RepositoryMappingValue) env.getValue(
        RepositoryMappingValue.key(repositoryName));

    if (env.valuesMissing()) {
      return null;
    }
    String repoDisplayName = RepoFileFunction.getDisplayNameForRepo(repositoryName, null);

    PackageArgs.Builder pkgArgsBuilder = PackageArgs.builder();
    LabelConverter labelConverter = new LabelConverter(
        PackageIdentifier.create(repositoryName, PathFragment.EMPTY_FRAGMENT),
        repositoryMappingValue.getRepositoryMapping());
    try {
      for (Map.Entry<String, Object> kwarg : repoFileValue.packageArgsMap().entrySet()) {
        PackageArgs.processParam(
            kwarg.getKey(),
            kwarg.getValue(),
            "repo() argument '" + kwarg.getKey() + "'",
            labelConverter,
            pkgArgsBuilder);
      }
    } catch (EvalException e) {
      env.getListener().handle(Event.error(e.getMessageWithStack()));
      throw new RepoFileFunctionException(
          new BadRepoFileException("error evaluating REPO.bazel file for " + repoDisplayName, e));
    }

    return new PackageArgsValue(pkgArgsBuilder.build());
  }
}
