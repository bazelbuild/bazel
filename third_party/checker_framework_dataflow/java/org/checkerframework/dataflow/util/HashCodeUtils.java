package org.checkerframework.dataflow.util;

/**
 * Utility class to implement the {@code hashCode} method.
 *
 * @author Stefan Heule
 *
 */
public class HashCodeUtils {

    /** Odd prime number. */
    private static int prime = 31;

    /** Seed. */
    private static int seed = 17;

    /** Add a boolean value to a given hash. */
    public static int hash(int hash, boolean item) {
        return hash * prime + (item ? 1 : 0);
    }

    /** Add a char value to a given hash. */
    public static int hash(int hash, char item) {
        return hash * prime + item;
    }

    /** Add an int value to a given hash. */
    public static int hash(int hash, int item) {
        return hash * prime + item;
    }

    /** Add a long value to a given hash. */
    public static int hash(int hash, long item) {
        return hash * prime + (int) (item ^ (item >>> 32));
    }

    /** Add a float value to a given hash. */
    public static int hash(int hash, float item) {
        return hash * prime + Float.floatToIntBits(item);
    }

    /** Add a double value to a given hash. */
    public static int hash(int hash, double item) {
        long l = Double.doubleToLongBits(item);
        return seed * prime + (int) (l ^ (l >>> 32));
    }

    /** Add an object to a given hash. */
    public static int hash(int hash, Object item) {
        if (item == null) {
            return hash * prime;
        }
        return hash * prime + item.hashCode();
    }

    /** Hash a boolean value. */
    public static int hash(boolean item) {
        return (item ? 1 : 0);
    }

    /** Hash a char value. */
    public static int hash(char item) {
        return item;
    }

    /** Hash an int value. */
    public static int hash(int item) {
        return item;
    }

    /** Hash a long value. */
    public static int hash(long item) {
        return (int) (item ^ (item >>> 32));
    }

    /** Hash a float value. */
    public static int hash(float item) {
        return Float.floatToIntBits(item);
    }

    /** Hash a double value. */
    public static int hash(double item) {
        long l = Double.doubleToLongBits(item);
        return (int) (l ^ (l >>> 32));
    }

    /** Hash an object. */
    public static int hash(Object item) {
        if (item == null) {
            return 0;
        }
        return item.hashCode();
    }

    /** Hash multiple objects. */
    public static int hash(Object... items) {
        int result = seed;
        for (Object item : items) {
            result = result * prime + (item == null ? 0 : item.hashCode());
        }
        return result;
    }
}
