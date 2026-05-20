package com.google.devtools.build.lib.skyframe;

import com.google.devtools.build.lib.vfs.RootedPath;
import java.util.List;
import java.util.Map;
import com.google.devtools.build.lib.vfs.UnixGlob;
import java.util.regex.Pattern;
import com.google.devtools.build.lib.vfs.PathFragment;
import javax.annotation.Nonnull;

/**
 * A {@link RootedPath} along with metadata indicating what files should be filtered away under it.
 *
 * <p>The {@code globBase} path is the root path of the glob {@code excludes} patterns. They are
 * joined together to create the paths to be filtered out. For example, if the given parameters are:
 *
 * <pre>
 * path = "/tmp"
 * globBase = "/tmp/path"
 * excludes = [".git/**", "cache/ignoreMe"]
 * </pre>
 *
 * Then the glob patterns that will be filtered/excluded from under <code>patch</code> would be:
 *
 * <pre>
 * /tmp/path/.git/**
 * /tmp/path/cache/ignoreMe
 * </pre>
 *
 * <p>This record is used as the return value for {@link SkyKey#argument()} when computing the
 * digest in {@link com.google.devtools.build.lib.skyframe.DirectoryTreeDigestFunction}</p>
 */
public record FilteredRootedPath(RootedPath path, RootedPath globBase, @Nonnull List<String> excludes) {

  /** Returns if the given {@code rootedPath} would be filtered/excluded out. */
  public boolean excludes(RootedPath rootedPath, Map<String, Pattern> patternCache) {
    // Are we comparing the same roots?
    if (!rootedPath.getRoot().equals(globBase.getRoot())) {
      return false;
    }
    String path = rootedPath.getRootRelativePath().toString();
    return excludes(path, patternCache);
  }

  /** Returns if the given {@code path} would be filtered/excluded out. */
  public boolean excludes(String path, Map<String, Pattern> patternCache) {
    PathFragment baseExclude = globBase.getRootRelativePath();
    for (String exclude : excludes) {
      String excludePattern = baseExclude.getRelative(exclude).toString();
      if (UnixGlob.matches(excludePattern, path, patternCache)) {
        return true;
      }
    }
    return false;
  }
}
