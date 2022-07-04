package com.google.devtools.build.lib.runtime;

import com.google.common.base.Preconditions;
import com.google.common.base.Strings;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.Locale;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Factory for creating {@link PathFragment}s from command-line options.
 *
 * <p>The difference between this and using {@link PathFragment#create(String)} directly is that
 * this factory replaces values starting with {@code %<name>%} with the corresponding (named) roots
 * (e.g., {@code %workspace%/foo} becomes {@code </path/to/workspace>/foo}).
 */
public final class CommandLinePathFactory {
  private static final Pattern REPLACEMENT_PATTERN = Pattern.compile("^(%([a-z_]+)%/)?([^%].*)$");

  private final ImmutableMap<String, PathFragment> roots;

  public CommandLinePathFactory(ImmutableMap<String, PathFragment> roots) {
    this.roots = Preconditions.checkNotNull(roots);
  }

  /** Creates a {@link Path}. */
  public PathFragment create(String value) {
    Preconditions.checkNotNull(value);

    Matcher matcher = REPLACEMENT_PATTERN.matcher(value);
    Preconditions.checkArgument(matcher.matches());

    String rootName = matcher.group(2);
    String path = matcher.group(3);
    if (!Strings.isNullOrEmpty(rootName)) {
      PathFragment root = roots.get(rootName);
      if (root == null) {
        throw new IllegalArgumentException(String.format(Locale.US, "Unknown root %s", rootName));
      }
      return root.getRelative(path);
    }
    return PathFragment.create(path);
  }
}
