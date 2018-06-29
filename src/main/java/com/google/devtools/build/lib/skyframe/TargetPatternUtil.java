package com.google.devtools.build.lib.skyframe;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.TargetParsingException;
import com.google.devtools.build.lib.pkgcache.FilteringPolicy;
import com.google.devtools.build.skyframe.SkyFunction.Environment;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.ValueOrException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import javax.annotation.Nullable;

public class TargetPatternUtil {

  @Nullable
  public static ImmutableList<Label> expandTargetPatterns(
      Environment env, List<String> targetPatterns, FilteringPolicy filteringPolicy)
      throws InvalidTargetPatternException, InterruptedException {

    // First parse the patterns, and throw any errors immediately.
    List<TargetPatternValue.TargetPatternKey> patternKeys = new ArrayList<>();
    for (TargetPatternValue.TargetPatternSkyKeyOrException keyOrException :
        TargetPatternValue.keys(targetPatterns, filteringPolicy, "")) {

      try {
        patternKeys.add(keyOrException.getSkyKey());
      } catch (TargetParsingException e) {
        throw new InvalidTargetPatternException(keyOrException.getOriginalPattern(), e);
      }
    }

    // Then, resolve the patterns.
    Map<SkyKey, ValueOrException<TargetParsingException>> resolvedPatterns =
        env.getValuesOrThrow(patternKeys, TargetParsingException.class);
    boolean valuesMissing = env.valuesMissing();
    ImmutableList.Builder<Label> labels = valuesMissing ? null : new ImmutableList.Builder<>();

    for (TargetPatternValue.TargetPatternKey pattern : patternKeys) {
      TargetPatternValue value;
      try {
        value = (TargetPatternValue) resolvedPatterns.get(pattern).get();
        if (!valuesMissing && value != null) {
          labels.addAll(value.getTargets().getTargets());
        }
      } catch (TargetParsingException e) {
        throw new InvalidTargetPatternException(pattern.getPattern(), e);
      }
    }

    if (valuesMissing) {
      return null;
    }

    return labels.build();
  }

  /**
   * Exception used when an error occurs in {@link #expandTargetPatterns(Environment, List,
   * FilteringPolicy)}.
   */
  static final class InvalidTargetPatternException extends Exception {
    private String invalidPattern;
    private TargetParsingException tpe;

    public InvalidTargetPatternException(String invalidPattern, TargetParsingException tpe) {
      super(tpe);
      this.invalidPattern = invalidPattern;
      this.tpe = tpe;
    }

    public String getInvalidPattern() {
      return invalidPattern;
    }

    public TargetParsingException getTpe() {
      return tpe;
    }
  }
}
