// Copyright 2014 Google Inc. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.lib.rules.cpp;

import static java.nio.charset.StandardCharsets.ISO_8859_1;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.base.Predicate;
import com.google.common.base.Supplier;
import com.google.common.base.Suppliers;
import com.google.common.cache.CacheBuilder;
import com.google.common.cache.CacheLoader;
import com.google.common.cache.LoadingCache;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Sets;
import com.google.common.io.CharStreams;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactFactory;
import com.google.devtools.build.lib.actions.Root;
import com.google.devtools.build.lib.rules.cpp.IncludeParser.Inclusion.Kind;
import com.google.devtools.build.lib.rules.cpp.RemoteIncludeExtractor.RemoteParseData;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.SkyValue;

import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Set;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.regex.PatternSyntaxException;

import javax.annotation.Nullable;

/**
 * Scans a source file and extracts the literal inclusions it specifies. Does not store results --
 * repeated requests to the same file will result in repeated scans. Clients should implement a
 * caching layer in order to avoid unnecessary disk access when requesting an already scanned file.
 */
public class IncludeParser implements SkyValue {
  private static final Logger LOG = Logger.getLogger(IncludeParser.class.getName());
  private static final boolean LOG_FINE = LOG.isLoggable(Level.FINE);
  private static final boolean LOG_FINER = LOG.isLoggable(Level.FINER);

  /**
   * Immutable object representation of the four columns making up a single Rule
   * in a Hints set. See {@link Hints} for more details.
   */
  private static class Rule {
    private enum Type { PATH, FILE, INCLUDE_QUOTE, INCLUDE_ANGLE; }
    final Type type;
    final Pattern pattern;
    final String findRoot;
    final Pattern findFilter;

    private Rule(String type, String pattern, String findRoot, Pattern findFilter) {
      this.type = Type.valueOf(type.trim().toUpperCase());
      this.pattern = Pattern.compile("^" + pattern + "$");
      this.findRoot = findRoot;
      this.findFilter = findFilter;
    }

    /**
     * @throws PatternSyntaxException, IllegalArgumentException if bad values
     *         are provided
     */
    public Rule(String type, String pattern, String findRoot, String findFilter) {
      this(type, pattern, findRoot.replace('\\', '$'), Pattern.compile(findFilter));
      Preconditions.checkArgument((this.type == Type.PATH) || (this.type == Type.FILE));
    }

    public Rule(String type, String pattern, String findRoot) {
      this(type, pattern, findRoot, (Pattern) null);
      Preconditions.checkArgument((this.type == Type.INCLUDE_QUOTE)
          || (this.type == Type.INCLUDE_ANGLE));
    }

    @Override public String toString() {
      return "" + type + " " + pattern + " " + findRoot + " " + findFilter;
    }
  }

  /**
   * This class is a representation of the INCLUDE_HINTS file maintained and
   * delivered with the remote client. The hints file contains regexp-based rules
   * to help this simple include scanner cope with computed includes, which
   * would otherwise require a full preprocessor with symbol support. Instead of
   * actually processing symbols to evaluate the computed includes, we instead
   * apply rules to gather inclusions for matching paths.
   * <p>
   * The hints file is read, line by line, into a list of rules each of which
   * encapsulates a line of four columns. Each non-blank, non-comment line has
   * the format:
   *
   * <pre>
   *   &quot;file&quot;|&quot;path&quot;  match-pattern  find-root  find-filter
   * </pre>
   *
   * <p>
   * The first column specifies whether the line is a rule based on matching
   * source <em>files</em> (passed directly to gcc as inputs, or transitively
   * #included by other inputs) or include <em>paths</em> (passed to gcc as
   * -I, -iquote, or -isystem flags).
   * <p>
   * The second column is a regexp for files or paths. Whenever a compiler
   * argument of the specified type matches that regexp, the rule is taken. (All
   * matching rules for every path and file on a compiler command line are
   * followed, and the results are combined.)
   * <p>
   * The third column is a point in the local filesystem from which to extract a
   * recursive listing. (This follows symlinks) Backrefs may be used to refer to
   * the regexp or its capturing groups. (This is mostly necessary because
   * --package_path can cause input paths to carry arbitrary prefixes.)
   * <p>
   * The fourth column is a regexp applied to each file found by the recursive
   * listing. All matching files are treated as dependencies.
   */
  public static class Hints implements SkyValue {

    private static final Pattern WS_PAT = Pattern.compile("\\s+");

    private final Path workingDir;
    private final List<Rule> rules = new ArrayList<>();
    private final ArtifactFactory artifactFactory;

    private final LoadingCache<Artifact, Collection<Artifact>> fileLevelHintsCache =
        CacheBuilder.newBuilder().build(
            new CacheLoader<Artifact, Collection<Artifact>>() {
              @Override
              public Collection<Artifact> load(Artifact path) {
                return getHintedInclusions(Rule.Type.FILE, path.getPath(), path.getRoot());
              }
            });

    private final LoadingCache<Path, Collection<Artifact>> pathLevelHintsCache =
        CacheBuilder.newBuilder().build(
            new CacheLoader<Path, Collection<Artifact>>() {
              @Override
              public Collection<Artifact> load(Path path) {
                return getHintedInclusions(Rule.Type.PATH, path, null);
              }
            });

    /**
     * Constructs a hint set for a given working/exec directory and INCLUDE_HINTS file to read.
     *
     * @param workingDir the working/exec directory that processed paths are relative to
     * @param hintsFile  the hints file to read
     * @throws IOException if the hints file can't be read or parsed
     */
    public Hints(Path workingDir, Path hintsFile, ArtifactFactory artifactFactory)
        throws IOException {
      this.workingDir = workingDir;
      this.artifactFactory = artifactFactory;
      try (InputStream is = hintsFile.getInputStream()) {
        for (String line : CharStreams.readLines(new InputStreamReader(is, "UTF-8"))) {
          line = line.trim();
          if (line.length() == 0 || line.startsWith("#")) {
            continue;
          }
          String[] tokens = WS_PAT.split(line);
          try {
            if (tokens.length == 3) {
              rules.add(new Rule(tokens[0], tokens[1], tokens[2]));
            } else if (tokens.length == 4) {
              rules.add(new Rule(tokens[0], tokens[1], tokens[2], tokens[3]));
            } else {
              throw new IOException("Malformed hint line: " + line);
            }
          } catch (PatternSyntaxException e) {
            throw new IOException("Malformed hint regex on: " + line + "\n  " + e.getMessage());
          } catch (IllegalArgumentException e) {
            throw new IOException("Invalid type on: " + line + "\n  " + e.getMessage());
          }
        }
      }
    }

    /**
     * Returns the "file" type hinted inclusions for a given path, caching results by path.
     */
    public Collection<Artifact> getFileLevelHintedInclusions(Artifact path) {
      return fileLevelHintsCache.getUnchecked(path);
    }

    public Collection<Artifact> getPathLevelHintedInclusions(Path path) {
      return pathLevelHintsCache.getUnchecked(path);
    }

    /**
     * Performs the work of matching a given file/path of a specified file/path type against the
     * hints and returns the expanded paths.
     */
    private Collection<Artifact> getHintedInclusions(Rule.Type type, Path path,
        @Nullable Root sourceRoot) {
      String pathString = path.getPathString();
      // Delay creation until we know we need one. Use a TreeSet to make sure that the results are
      // sorted with a stable order and unique.
      Set<Path> hints = null;
      for (final Rule rule : rules) {
        if (type != rule.type) {
          continue;
        }
        Matcher m = rule.pattern.matcher(pathString);
        if (!m.matches()) {
          continue;
        }
        if (hints == null) { hints = Sets.newTreeSet(); }
        Path root = workingDir.getRelative(m.replaceFirst(rule.findRoot));
        if (LOG_FINE) {
          LOG.fine("hint for " + rule.type + " " + pathString + " root: " + root);
        }
        try {
          // The assumption is made here that all files specified by this hint are under the same
          // package path as the original file -- this filesystem tree traversal is completely
          // ignorant of package paths. This could be violated if there were a hint that resolved to
          // foo/**/*.h, there was a package foo/bar, and the packages foo and foo/bar were in
          // different package paths. In that case, this traversal would fail to pick up
          // foo/bar/**/*.h. No examples of this currently exist in the INCLUDE_HINTS
          // file.
          FileSystemUtils.traverseTree(hints, root, new Predicate<Path>() {
            @Override
            public boolean apply(Path p) {
              boolean take = p.isFile() && rule.findFilter.matcher(p.getPathString()).matches();
              if (LOG_FINER && take) {
                LOG.finer("hinted include: " + p);
              }
              return take;
            }
          });
        } catch (IOException e) {
          LOG.warning("Error in hint expansion: " + e);
        }
      }
      if (hints != null && !hints.isEmpty()) {
        // Transform paths into source artifacts (all hints must be to source artifacts).
        List<Artifact> result = new ArrayList<>(hints.size());
        for (Path hint : hints) {
          if (hint.startsWith(workingDir)) {
            // Paths that are under the execRoot can be resolved as source artifacts as usual. All
            // include directories are specified relative to the execRoot, and so fall here.
            result.add(Preconditions.checkNotNull(
                artifactFactory.resolveSourceArtifact(hint.relativeTo(workingDir)), hint));
          } else {
            // The file passed in might not have been under the execRoot, for instance
            // <workspace>/foo/foo.cc.
            Preconditions.checkNotNull(sourceRoot, "%s %s", path, hint);
            Path sourcePath = sourceRoot.getPath();
            Preconditions.checkState(hint.startsWith(sourcePath),
                "%s %s %s", hint, path, sourceRoot);
            result.add(Preconditions.checkNotNull(
                artifactFactory.getSourceArtifact(hint.relativeTo(sourcePath), sourceRoot)));
          }
        }
        return result;
      } else {
        return ImmutableList.of();
      }
    }

    private Collection<Inclusion> getHintedInclusions(Artifact path) {
      String pathString = path.getPath().getPathString();
      // Delay creation until we know we need one. Use a LinkedHashSet to make sure that the results
      // are sorted with a stable order and unique.
      Set<Inclusion> hints = null;
      for (final Rule rule : rules) {
        if ((rule.type != Rule.Type.INCLUDE_ANGLE) && (rule.type != Rule.Type.INCLUDE_QUOTE)) {
          continue;
        }
        Matcher m = rule.pattern.matcher(pathString);
        if (!m.matches()) {
          continue;
        }
        if (hints == null) { hints = Sets.newLinkedHashSet(); }
        Inclusion inclusion = new Inclusion(rule.findRoot, rule.type == Rule.Type.INCLUDE_QUOTE
            ? Kind.QUOTE : Kind.ANGLE);
        hints.add(inclusion);
        if (LOG_FINE) {
          LOG.fine("hint for " + rule.type + " " + pathString + " root: " + inclusion);
        }
      }
      if (hints != null && !hints.isEmpty()) {
        return ImmutableList.copyOf(hints);
      } else {
        return ImmutableList.of();
      }
    }
  }

  public Hints getHints() {
    return hints;
  }

  /**
   * An immutable inclusion tuple. This models an {@code #include} or {@code
   * #include_next} line in a file without the context how this file got
   * included.
   */
  public static class Inclusion {
    /** The format of the #include in the source file -- quoted, angle bracket, etc. */
    public enum Kind {
      /** Quote includes: {@code #include "name"}. */
      QUOTE,

      /** Angle bracket includes: {@code #include <name>}. */
      ANGLE,

      /** Quote next includes: {@code #include_next "name"}. */
      NEXT_QUOTE,

      /** Angle next includes: {@code #include_next <name>}. */
      NEXT_ANGLE,

      /** Computed or other unhandlable includes: {@code #include HEADER}. */
      OTHER;

      /**
       * Returns true if this is an {@code #include_next} inclusion,
       */
      public boolean isNext() {
        return this == NEXT_ANGLE || this == NEXT_QUOTE;
      }
    }

    /** The kind of inclusion. */
    public final Kind kind;
    /** The relative path of the inclusion. */
    public final PathFragment pathFragment;

    public Inclusion(String includeTarget, Kind kind) {
      this.kind = kind;
      this.pathFragment = new PathFragment(includeTarget);
    }

    public Inclusion(PathFragment pathFragment, Kind kind) {
      this.kind = kind;
      this.pathFragment = Preconditions.checkNotNull(pathFragment);
    }

    public String getPathString() {
      return pathFragment.getPathString();
    }

    @Override
    public String toString() {
      return kind + ":" + pathFragment.getPathString();
    }

    @Override
    public boolean equals(Object o) {
      if (o == this) {
        return true;
      }
      if (!(o instanceof Inclusion)) {
        return false;
      }
      Inclusion that = (Inclusion) o;
      return kind == that.kind && pathFragment.equals(that.pathFragment);
    }

    @Override
    public int hashCode() {
      return pathFragment.hashCode() * 37 + kind.hashCode();
    }
  }

  /**
   * The externally-scoped immutable hints helper that is shared by all scanners.
   */
  private final Hints hints;

  /**
   * A scanner that extracts includes from an individual files remotely, used when scanning files
   * generated remotely.
   */
  private final Supplier<? extends RemoteIncludeExtractor> remoteExtractor;

  /**
   * Constructs a new FileParser.
   * @param remoteExtractor a processor that extracts includes from an individual file remotely.
   * @param hints regexps for converting computed includes into simple strings
   */
  public IncludeParser(@Nullable RemoteIncludeExtractor remoteExtractor, Hints hints) {
    this.hints = hints;
    this.remoteExtractor = Suppliers.ofInstance(remoteExtractor);
  }

  /**
   * Constructs a new FileParser.
   * @param remoteExtractorSupplier a supplier of a processor that extracts includes from an
   *        individual file remotely.
   * @param hints regexps for converting computed includes into simple strings
   */
  public IncludeParser(Supplier<? extends RemoteIncludeExtractor> remoteExtractorSupplier,
      Hints hints) {
    this.hints = hints;
    this.remoteExtractor = remoteExtractorSupplier;
  }

  /**
   * Skips whitespace, \+NL pairs, and block-style / * * / comments. Assumes
   * line comments are handled outside. Does not handle digraphs, trigraphs or
   * decahexagraphs.
   *
   * @param chars characters to scan
   * @param pos the starting position
   * @return the resulting position after skipping whitespace and comments.
   */
  protected static int skipWhitespace(char[] chars, int pos, int end) {
    while (pos < end) {
      if (Character.isWhitespace(chars[pos])) {
        pos++;
      } else if (chars[pos] == '\\' && pos + 1 < end && chars[pos + 1] == '\n') {
        pos++;
      } else if (chars[pos] == '/' && pos + 1 < end && chars[pos + 1] == '*') {
        pos += 2;
        while (pos < end - 1) {
          if (chars[pos++] == '*') {
            if (chars[pos] == '/') {
              pos++;
              break;  // proper comment end
            }
          }
        }
      } else {  // not whitespace
        return pos;
      }
    }
    return pos;  // pos == len, meaning we fell off the end.
  }

  /**
   * Checks for and skips a given token.
   *
   * @param chars characters to scan
   * @param pos the starting position
   * @param expected the expected token
   * @return the resulting position if found, otherwise -1
   */
  protected static int expect(char[] chars, int pos, int end, String expected) {
    int si = 0;
    int expectedLen = expected.length();
    while (pos < end) {
      if (si == expectedLen) {
        return pos;
      }
      if (chars[pos++] != expected.charAt(si++)) {
        return -1;
      }
    }
    return -1;
  }

  /**
   * Finds the index of a given character token from a starting pos.
   *
   * @param chars characters to scan
   * @param pos the starting position
   * @param echar the character to find
   * @return the resulting position of echar if found, otherwise -1
   */
  private static int indexOf(char[] chars, int pos, int end, char echar) {
    while (pos < end) {
      if (chars[pos] == echar) {
        return pos;
      }
      pos++;
    }
    return -1;
  }

  private static final Pattern BS_NL_PAT = Pattern.compile("\\\\" + "\n");

  // Keep this in sync with the auxiliary binary's scanning output format.
  private static final ImmutableMap<Character, Kind> KIND_MAP = ImmutableMap.of(
      '"', Kind.QUOTE,
      '<', Kind.ANGLE,
      'q', Kind.NEXT_QUOTE,
      'a', Kind.NEXT_ANGLE);

  /**
   * Processes the output generated by an auxiliary include-scanning binary. Closes the stream upon
   * completion.
   *
   * <p>If a source file has the following include statements:
   * <pre>
   *   #include &lt;string&gt;
   *   #include "directory/header.h"
   * </pre>
   *
   * <p>Then the output file has the following contents:
   * <pre>
   *   "directory/header.h
   *   &lt;string
   * </pre>
   * <p>Each line of the output is translated into an Inclusion object.
   */
  public static List<Inclusion> processIncludes(Object streamName, InputStream is)
      throws IOException {
    List<Inclusion> inclusions = new ArrayList<>();
    InputStreamReader reader = new InputStreamReader(is, ISO_8859_1);
    try {
      for (String line : CharStreams.readLines(reader)) {
        char qchar = line.charAt(0);
        String name = line.substring(1);
        Inclusion.Kind kind = KIND_MAP.get(qchar);
        if (kind == null) {
          throw new IOException("Illegal inclusion kind '" + qchar + "'");
        }
        inclusions.add(new Inclusion(name, kind));
      }
    } catch (IOException e) {
      throw new IOException("Error reading include file " + streamName + ": " + e.getMessage());
    } finally {
      reader.close();
    }
    return inclusions;
  }

  @VisibleForTesting
  Inclusion extractInclusion(String line) {
    return extractInclusion(line.toCharArray(), 0, line.length());
  }

  /**
   * Extracts a new, unresolved an Inclusion from a line of source.
   *
   * @param chars the char array containing the line chars to parse
   * @param lineBegin the position of the first character in the line
   * @param lineEnd the position of the character after the last
   * @return the inclusion object if possible, null if none
   */
  private Inclusion extractInclusion(char[] chars, int lineBegin, int lineEnd) {
    // expect WS#WS(include|include_next)WS("name"|<name>|junk)
    int pos = expectIncludeKeyword(chars, lineBegin, lineEnd);
    if (pos == -1 || pos == lineEnd) {
      return null;
    }
    boolean isNext = false;
    int npos = expect(chars, pos, lineEnd, "_next");
    if (npos >= 0) {
      isNext = true;
      pos = npos;
    }
    if ((pos = skipWhitespace(chars, pos, lineEnd)) == lineEnd) {
      return null;
    }
    if (chars[pos] == '"' || chars[pos] == '<') {
      char qchar = chars[pos++];
      int spos = pos;
      pos = indexOf(chars, pos + 1, lineEnd, qchar == '<' ? '>' : '"');
      if (pos < 0) {
        return null;
      }
      if (chars[spos] == '/') {
        return null;  // disallow absolute paths
      }
      String name = new String(chars, spos, pos - spos);
      if (name.contains("\n")) {  // strip any \+NL pairs within name
        name = BS_NL_PAT.matcher(name).replaceAll("");
      }
      if (isNext) {
        return new Inclusion(name, qchar == '"' ? Kind.NEXT_QUOTE : Kind.NEXT_ANGLE);
      } else {
        return new Inclusion(name, qchar == '"' ? Kind.QUOTE : Kind.ANGLE);
      }
    } else {
      return createOtherInclusion(new String(chars, pos, lineEnd - pos));
    }
  }

  /**
   * Extracts all inclusions from characters of a file.
   *
   * @param chars the file contents to parse & extract inclusions from
   * @return a new set of inclusions, normalized to the cache
   */
  @VisibleForTesting
  List<Inclusion> extractInclusions(char[] chars) {
    List<Inclusion> inclusions = new ArrayList<>();
    int lineBegin = 0;  // the first char of each line
    int end = chars.length;  // the file end
    while (lineBegin < end) {
      int lineEnd = lineBegin;   // the char after the last non-\n in each line
      // skip to the next \n or after end of buffer, ignoring continuations
      while (lineEnd < end) {
        if (chars[lineEnd] == '\n') {
          break;
        } else if (chars[lineEnd] == '\\') {
          lineEnd++;
          if (chars[lineEnd] == '\n') {
            lineEnd++;
          }
        } else {
          lineEnd++;
        }
      }

      // TODO(bazel-team) handle multiline block comments /* */ for the cases:
      //   /* blah blah blah
      //    lalala  */ #include "foo.h"
      // and:
      //   /* blah
      //   #include "foo.h"
      //   */

      // extract the inclusion, and save only the kind we care about.
      Inclusion inclusion = extractInclusion(chars, lineBegin, lineEnd);
      if (inclusion != null) {
        if (isValidInclusionKind(inclusion.kind)) {
          inclusions.add(inclusion);
        } else {
          //System.err.println("Funky include " + inclusion + " in " + file);
        }
      }
      lineBegin = lineEnd + 1;  // next line starts after the previous line
    }
    return inclusions;
  }

  /**
   * Extracts all inclusions from a given source file.
   *
   * @param file the file to parse & extract inclusions from
   * @param greppedFile if non-null, this file has the already-grepped include lines of file.
   * @param actionExecutionContext Services in the scope of the action, like the stream to which
   *     scanning messages are printed
   * @return a new set of inclusions, normalized to the cache
   */
  public Collection<Inclusion> extractInclusions(Artifact file, @Nullable Path greppedFile,
      ActionExecutionContext actionExecutionContext)
          throws IOException, InterruptedException {
    Collection<Inclusion> inclusions;
    if (greppedFile != null) {
      inclusions = processIncludes(greppedFile, greppedFile.getInputStream());
    } else {
      RemoteParseData remoteParseData = remoteExtractor.get() == null
          ? null
          : remoteExtractor.get().shouldParseRemotely(file.getPath());
      if (remoteParseData != null && remoteParseData.shouldParseRemotely()) {
        inclusions =
            remoteExtractor.get().extractInclusions(file, actionExecutionContext,
                remoteParseData);
      } else {
        inclusions = extractInclusions(FileSystemUtils.readContentAsLatin1(file.getPath()));
      }
    }
    if (hints != null) {
      inclusions.addAll(hints.getHintedInclusions(file));
    }
    return ImmutableList.copyOf(inclusions);
  }

  /**
   * Parses include keyword in the provided char array and returns position
   * immediately after include keyword or -1 if keyword was not found. Can be
   * overridden by subclasses.
   */
  protected int expectIncludeKeyword(char[] chars, int position, int end) {
    int pos = expect(chars, skipWhitespace(chars, position, end), end, "#");
    if (pos > 0) {
      int npos = skipWhitespace(chars, pos, end);
      if ((pos = expect(chars, npos, end, "include")) > 0) {
        return pos;
      } else if ((pos = expect(chars, npos, end, "import")) > 0) {
        if (expect(chars, pos, end, "_") == -1) {  // Needed to avoid #import_next.
          return pos;
        }
      }
    }
    return -1;
  }

  /**
   * Returns true if we interested in the given inclusion kind. Can be
   * overridden by the subclass.
   */
  protected boolean isValidInclusionKind(Kind kind) {
    return kind != Kind.OTHER;
  }

  /**
   * Returns inclusion object for non-standard inclusion cases or null if
   * inclusion should be ignored.
   */
  protected Inclusion createOtherInclusion(String inclusionContent) {
    return new Inclusion(inclusionContent, Kind.OTHER);
  }
}
