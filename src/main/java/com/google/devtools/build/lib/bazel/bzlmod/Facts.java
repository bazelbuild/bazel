package com.google.devtools.build.lib.bazel.bzlmod;

import com.google.auto.value.AutoValue;
import java.util.TreeMap;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.NoneType;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkFloat;
import net.starlark.java.eval.StarlarkInt;
import net.starlark.java.eval.StarlarkList;
import net.starlark.java.eval.Tuple;

@AutoValue
public abstract class Facts {
  public static final Facts EMPTY = new AutoValue_Facts(Starlark.NONE);

  public abstract Object value();

  public static Facts validateAndCreate(Object value) throws EvalException {
    return new AutoValue_Facts(validateAndNormalize(value));
  }

  // This limit only exists to prevent pathological uses of facts, which are meant to be
  // human-readable and friendly to VCS merges.
  private static final int MAX_FACTS_DEPTH = 5;

  private static Object validateAndNormalize(Object facts) throws EvalException {
    return validateAndNormalize(facts, MAX_FACTS_DEPTH);
  }

  private static Object validateAndNormalize(Object facts, int remainingDepth)
      throws EvalException {
    if (remainingDepth == 0) {
      throw Starlark.errorf("Facts cannot be nested more than %s levels deep", MAX_FACTS_DEPTH);
    }
    // Only permit types that can be serialized to JSON and ensure that they contain no information
    // not captured by equality by sorting dicts.
    return switch (facts) {
      case String s -> s;
      case NoneType n -> n;
      case Boolean b -> b;
      case StarlarkFloat f -> f;
      case StarlarkInt i -> i;
      case StarlarkList<?> list -> {
        Object[] normalizedList = new Object[list.size()];
        for (int i = 0; i < list.size(); i++) {
          normalizedList[i] = validateAndNormalize(list.get(i), remainingDepth - 1);
        }
        yield StarlarkList.immutableOf(normalizedList);
      }
      case Tuple tuple -> {
        // Turn a tuple into a list since JSON does not have a tuple type.
        Object[] normalizedList = new Object[tuple.size()];
        for (int i = 0; i < tuple.size(); i++) {
          normalizedList[i] = validateAndNormalize(tuple.get(i), remainingDepth - 1);
        }
        yield StarlarkList.immutableOf(normalizedList);
      }
      case Dict<?, ?> dict -> {
        var builder = new TreeMap<String, Object>();
        for (var entry : dict.entrySet()) {
          if (!(entry.getKey() instanceof String string)) {
            throw Starlark.errorf(
                "Facts keys must be strings, got '%s' (%s)",
                Starlark.repr(entry), Starlark.type(entry.getKey()));
          }
          builder.put(string, validateAndNormalize(entry.getValue(), remainingDepth - 1));
        }
        yield Dict.immutableCopyOf(builder);
      }
      default ->
          throw Starlark.errorf(
              "'%s' (%s) is not supported in facts", Starlark.repr(facts), Starlark.type(facts));
    };
  }
}
