package net.starlark.java.eval;

import org.junit.Test;

import java.lang.reflect.Array;
import java.util.Arrays;
import java.util.stream.IntStream;

import static com.google.common.truth.Truth.assertThat;

public class TupleTest {
  @Test
  public void testToArrayWithObjectArrayParam() throws Exception {
    for (int i = 0; i != 10; ++i) {
      Tuple tuple = Tuple.of(IntStream.range(0, 10).mapToObj(Integer::toString).toArray());
      for (int arrayLength : new int[] { 0, tuple.size() / 2, tuple.size(), tuple.size() * 2 }) {
        for (Class<?> arrayElementClass : new Class[] { Object.class, String.class }) {
          Object[] input = (Object[]) Array.newInstance(arrayElementClass, arrayLength);
          Arrays.fill(input, "x");

          Object[] output = tuple.toArray(input);
          assertThat(input.getClass()).isEqualTo(output.getClass());
          if (input.length < tuple.size()) {
            // assert input is unchanged
            for (Object o : input) {
              assertThat(o).isEqualTo("x");
            }

            Object[] expected = IntStream.range(0, tuple.size()).mapToObj(Integer::toString).toArray();
            assertThat(output).isEqualTo(expected);
          } else {
            assertThat(output).isSameInstanceAs(input);
            for (int j = 0; j != output.length; ++j) {
              if (j < tuple.size()) {
                assertThat(output[j]).isEqualTo(Integer.toString(j));
              } else {
                assertThat(output[j]).isNull();
              }
            }
          }
        }
      }
    }
  }
}
