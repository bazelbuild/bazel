/*
 * ProGuard -- shrinking, optimization, obfuscation, and preverification
 *             of Java bytecode.
 *
 * Copyright (c) 2002-2019 Guardsquare NV
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the Free
 * Software Foundation; either version 2 of the License, or (at your option)
 * any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
 * more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 */
package proguard.util;

import java.lang.reflect.Array;
import java.util.Arrays;

/**
 * This class contains utility methods operating on arrays.
 */
public class ArrayUtil
{
    /**
     * Returns whether the elements of the two given arrays are the same.
     * @param array1 the first array.
     * @param array2 the second array.
     * @param size   the size of the arrays to be checked.
     * @return whether the elements are the same.
     */
    public static boolean equal(boolean[] array1, boolean[] array2, int size)
    {
        for (int index = 0; index < size; index++)
        {
            if (array1[index] != array2[index])
            {
                return false;
            }
        }

        return true;
    }


    /**
     * Returns whether the elements of the two given arrays are the same.
     * @param array1 the first array.
     * @param array2 the second array.
     * @param size   the size of the arrays to be checked.
     * @return whether the elements are the same.
     */
    public static boolean equal(byte[] array1, byte[] array2, int size)
    {
        for (int index = 0; index < size; index++)
        {
            if (array1[index] != array2[index])
            {
                return false;
            }
        }

        return true;
    }


    /**
     * Returns whether the elements of the two given arrays are the same.
     * @param array1 the first array.
     * @param array2 the second array.
     * @param size   the size of the arrays to be checked.
     * @return whether the elements are the same.
     */
    public static boolean equal(char[] array1, char[] array2, int size)
    {
        for (int index = 0; index < size; index++)
        {
            if (array1[index] != array2[index])
            {
                return false;
            }
        }

        return true;
    }


    /**
     * Returns whether the elements of the two given arrays are the same.
     * @param array1 the first array.
     * @param array2 the second array.
     * @param size   the size of the arrays to be checked.
     * @return whether the elements are the same.
     */
    public static boolean equal(short[] array1, short[] array2, int size)
    {
        for (int index = 0; index < size; index++)
        {
            if (array1[index] != array2[index])
            {
                return false;
            }
        }

        return true;
    }


    /**
     * Returns whether the elements of the two given arrays are the same.
     * @param array1 the first array.
     * @param array2 the second array.
     * @param size   the size of the arrays to be checked.
     * @return whether the elements are the same.
     */
    public static boolean equal(int[] array1, int[] array2, int size)
    {
        for (int index = 0; index < size; index++)
        {
            if (array1[index] != array2[index])
            {
                return false;
            }
        }

        return true;
    }


    /**
     * Returns whether the elements of the two given arrays are the same.
     * @param array1 the first array.
     * @param array2 the second array.
     * @param size   the size of the arrays to be checked.
     * @return whether the elements are the same.
     */
    public static boolean equal(long[] array1, long[] array2, int size)
    {
        for (int index = 0; index < size; index++)
        {
            if (array1[index] != array2[index])
            {
                return false;
            }
        }

        return true;
    }


    /**
     * Returns whether the elements of the two given arrays are the same.
     * @param array1 the first array.
     * @param array2 the second array.
     * @param size   the size of the arrays to be checked.
     * @return whether the elements are the same.
     */
    public static boolean equal(float[] array1, float[] array2, int size)
    {
        for (int index = 0; index < size; index++)
        {
            if (array1[index] != array2[index])
            {
                return false;
            }
        }

        return true;
    }


    /**
     * Returns whether the elements of the two given arrays are the same.
     * @param array1 the first array.
     * @param array2 the second array.
     * @param size   the size of the arrays to be checked.
     * @return whether the elements are the same.
     */
    public static boolean equal(double[] array1, double[] array2, int size)
    {
        for (int index = 0; index < size; index++)
        {
            if (array1[index] != array2[index])
            {
                return false;
            }
        }

        return true;
    }


    /**
     * Returns whether the elements of the two given arrays are the same.
     * @param array1 the first array.
     * @param array2 the second array.
     * @param size   the size of the arrays to be checked.
     * @return whether the elements are the same.
     */
    public static boolean equal(Object[] array1, Object[] array2, int size)
    {
        for (int index = 0; index < size; index++)
        {
            if (!array1[index].equals(array2[index]))
            {
                return false;
            }
        }

        return true;
    }


    /**
     * Returns whether the elements of the two given arrays are the same, or
     * both arrays are null.
     * @param array1 the first array.
     * @param array2 the second array.
     * @return whether the elements are the same.
     */
    public static boolean equalOrNull(Object[] array1, Object[] array2)
    {
        return array1 == null ? array2 == null :
            equalOrNull(array1, array2, array1.length);
    }


    /**
     * Returns whether the elements of the two given arrays are the same, or
     * both arrays are null.
     * @param array1 the first array.
     * @param array2 the second array.
     * @param size   the size of the arrays to be checked.
     * @return whether the elements are the same.
     */
    public static boolean equalOrNull(Object[] array1, Object[] array2, int size)
    {
        return array1 == null ? array2 == null :
            array2 != null &&
            equal(array1, array2, size);
    }


    /**
     * Returns a hash code for the elements of the given array.
     * @param array the array.
     * @param size  the size of the array to be taken into account.
     * @return a hash code.
     */
    public static int hashCode(boolean[] array, int size)
    {
        int hashCode = 0;

        for (int index = 0; index < size; index++)
        {
            // Rotate the hash code.
            hashCode = (hashCode << 1) | (hashCode >>> 31);

            // XOR the element.
            if (array[index])
            {
                hashCode ^= 1;
            }
        }

        return hashCode;
    }


    /**
     * Returns a hash code for the elements of the given array.
     * @param array the array.
     * @param size  the size of the array to be taken into account.
     * @return a hash code.
     */
    public static int hashCode(byte[] array, int size)
    {
        int hashCode = 0;

        for (int index = 0; index < size; index++)
        {
            // Rotate the hash code.
            hashCode = (hashCode << 1) | (hashCode >>> 31);

            // XOR the element.
            hashCode ^= array[index] & 0xff;
        }

        return hashCode;
    }


    /**
     * Returns a hash code for the elements of the given array.
     * @param array the array.
     * @param size  the size of the array to be taken into account.
     * @return a hash code.
     */
    public static int hashCode(char[] array, int size)
    {
        int hashCode = 0;

        for (int index = 0; index < size; index++)
        {
            // Rotate the hash code.
            hashCode = (hashCode << 1) | (hashCode >>> 31);

            // XOR the element.
            hashCode ^= array[index];
        }

        return hashCode;
    }


    /**
     * Returns a hash code for the elements of the given array.
     * @param array the array.
     * @param size  the size of the array to be taken into account.
     * @return a hash code.
     */
    public static int hashCode(short[] array, int size)
    {
        int hashCode = 0;

        for (int index = 0; index < size; index++)
        {
            // Rotate the hash code.
            hashCode = (hashCode << 1) | (hashCode >>> 31);

            // XOR the element.
            hashCode ^= array[index] & 0xffff;
        }

        return hashCode;
    }


    /**
     * Returns a hash code for the elements of the given array.
     * @param array the array.
     * @param size  the size of the array to be taken into account.
     * @return a hash code.
     */
    public static int hashCode(int[] array, int size)
    {
        int hashCode = 0;

        for (int index = 0; index < size; index++)
        {
            // Rotate the hash code.
            hashCode = (hashCode << 1) | (hashCode >>> 31);

            // XOR the element.
            hashCode ^= array[index];
        }

        return hashCode;
    }


    /**
     * Returns a hash code for the elements of the given array.
     * @param array the array.
     * @param size  the size of the array to be taken into account.
     * @return a hash code.
     */
    public static int hashCode(long[] array, int size)
    {
        int hashCode = 0;

        for (int index = 0; index < size; index++)
        {
            // Rotate the hash code.
            hashCode = (hashCode << 1) | (hashCode >>> 31);

            // XOR the element.
            long longBits = array[index];

            hashCode ^= longBits | (longBits >>> 32);
        }

        return hashCode;
    }


    /**
     * Returns a hash code for the elements of the given array.
     * @param array the array.
     * @param size  the size of the array to be taken into account.
     * @return a hash code.
     */
    public static int hashCode(float[] array, int size)
    {
        int hashCode = 0;

        for (int index = 0; index < size; index++)
        {
            // Rotate the hash code.
            hashCode = (hashCode << 1) | (hashCode >>> 31);

            // XOR the element.
            hashCode ^= Float.floatToRawIntBits(array[index]);
        }

        return hashCode;
    }


    /**
     * Returns a hash code for the elements of the given array.
     * @param array the array.
     * @param size  the size of the array to be taken into account.
     * @return a hash code.
     */
    public static int hashCode(double[] array, int size)
    {
        int hashCode = 0;

        for (int index = 0; index < size; index++)
        {
            // Rotate the hash code.
            hashCode = (hashCode << 1) | (hashCode >>> 31);

            // XOR the element.
            long longBits = Double.doubleToRawLongBits(array[index]);

            hashCode ^= longBits | (longBits >>> 32);
        }

        return hashCode;
    }


    /**
     * Returns a hash code for the elements of the given array.
     * @param array the array.
     * @param size  the size of the array to be taken into account.
     * @return a hash code.
     */
    public static int hashCode(Object[] array, int size)
    {
        int hashCode = 0;

        for (int index = 0; index < size; index++)
        {
            // Rotate the hash code.
            hashCode = (hashCode << 1) | (hashCode >>> 31);

            // XOR the element.
            hashCode ^= array[index].hashCode();
        }

        return hashCode;
    }


    /**
     * Returns a hash code for the elements of the given array, or 0 if it is
     * null.
     * @param array the array.
     * @return a hash code.
     */
    public static int hashCodeOrNull(Object[] array)
    {
        return array == null ? 0 : hashCode(array, array.length);
    }


    /**
     * Returns a hash code for the elements of the given array, or 0 if it is
     * null.
     * @param array the array.
     * @param size  the size of the array to be taken into account.
     * @return a hash code.
     */
    public static int hashCodeOrNull(Object[] array, int size)
    {
        return array == null ? 0 : hashCode(array, size);
    }


    /**
     * Compares the elements of the two given arrays.
     * @param array1 the first array.
     * @param size1  the size of the first array.
     * @param array2 the second array.
     * @param size2  the size of the second array.
     * @return 0 if all elements are the same,
     *         -1 if the first different element in the first array is smaller
     *         than the corresponding element in the second array,
     *         or 1 if it is larger.
     */
    public static int compare(boolean[] array1, int size1,
                              boolean[] array2, int size2)
    {
        int minSize = Math.min(size1, size2);

        for (int index = 0; index < minSize; index++)
        {
            int comparison = Boolean.compare(array1[index], array2[index]);
            if (comparison != 0)
            {
                return comparison;
            }
        }

        return Integer.compare(size1,  size2);
    }


    /**
     * Compares the elements of the two given arrays.
     * @param array1 the first array.
     * @param size1  the size of the first array.
     * @param array2 the second array.
     * @param size2  the size of the second array.
     * @return 0 if all elements are the same,
     *         -1 if the first different element in the first array is smaller
     *         than the corresponding element in the second array,
     *         or 1 if it is larger.
     */
    public static int compare(byte[] array1, int size1,
                              byte[] array2, int size2)
    {
        int minSize = Math.min(size1, size2);

        for (int index = 0; index < minSize; index++)
        {
            int comparison = Byte.compare(array1[index], array2[index]);
            if (comparison != 0)
            {
                return comparison;
            }
        }

        return Integer.compare(size1,  size2);
    }


    /**
     * Compares the elements of the two given arrays.
     * @param array1 the first array.
     * @param size1  the size of the first array.
     * @param array2 the second array.
     * @param size2  the size of the second array.
     * @return 0 if all elements are the same,
     *          -1 if the first different element in the first array is smaller
     *          than the corresponding element in the second array,
     *          or 1 if it is larger.
     */
    public static int compare(char[] array1, int size1,
                              char[] array2, int size2)
    {
        int minSize = Math.min(size1, size2);

        for (int index = 0; index < minSize; index++)
        {
            int comparison = Character.compare(array1[index], array2[index]);
            if (comparison != 0)
            {
                return comparison;
            }
        }

        return Integer.compare(size1,  size2);
    }


    /**
     * Compares the elements of the two given arrays.
     * @param array1 the first array.
     * @param size1  the size of the first array.
     * @param array2 the second array.
     * @param size2  the size of the second array.
     * @return 0 if all elements are the same,
     *         -1 if the first different element in the first array is smaller
     *         than the corresponding element in the second array,
     *         or 1 if it is larger.
     */
    public static int compare(short[] array1, int size1,
                              short[] array2, int size2)
    {
        int minSize = Math.min(size1, size2);

        for (int index = 0; index < minSize; index++)
        {
            int comparison = Short.compare(array1[index], array2[index]);
            if (comparison != 0)
            {
                return comparison;
            }
        }

        return Integer.compare(size1,  size2);
    }


    /**
     * Compares the elements of the two given arrays.
     * @param array1 the first array.
     * @param size1  the size of the first array.
     * @param array2 the second array.
     * @param size2  the size of the second array.
     * @return 0 if all elements are the same,
     *         -1 if the first different element in the first array is smaller
     *         than the corresponding element in the second array,
     *         or 1 if it is larger.
     */
    public static int compare(int[] array1, int size1,
                              int[] array2, int size2)
    {
        int minSize = Math.min(size1, size2);

        for (int index = 0; index < minSize; index++)
        {
            int comparison = Integer.compare(array1[index], array2[index]);
            if (comparison != 0)
            {
                return comparison;
            }
        }

        return Integer.compare(size1,  size2);
    }


    /**
     * Compares the elements of the two given arrays.
     * @param array1 the first array.
     * @param size1  the size of the first array.
     * @param array2 the second array.
     * @param size2  the size of the second array.
     * @return 0 if all elements are the same,
     *         -1 if the first different element in the first array is smaller
     *         than the corresponding element in the second array,
     *         or 1 if it is larger.
     */
    public static int compare(long[] array1, int size1,
                              long[] array2, int size2)
    {
        int minSize = Math.min(size1, size2);

        for (int index = 0; index < minSize; index++)
        {
            int comparison = Long.compare(array1[index], array2[index]);
            if (comparison != 0)
            {
                return comparison;
            }
        }

        return Integer.compare(size1,  size2);
    }


    /**
     * Compares the elements of the two given arrays.
     * @param array1 the first array.
     * @param size1  the size of the first array.
     * @param array2 the second array.
     * @param size2  the size of the second array.
     * @return 0 if all elements are the same,
     *         -1 if the first different element in the first array is smaller
     *         than the corresponding element in the second array,
     *         or 1 if it is larger.
     */
    public static int compare(float[] array1, int size1,
                              float[] array2, int size2)
    {
        int minSize = Math.min(size1, size2);

        for (int index = 0; index < minSize; index++)
        {
            int comparison = Float.compare(array1[index], array2[index]);
            if (comparison != 0)
            {
                return comparison;
            }
        }

        return Integer.compare(size1,  size2);
    }


    /**
     * Compares the elements of the two given arrays.
     * @param array1 the first array.
     * @param size1  the size of the first array.
     * @param array2 the second array.
     * @param size2  the size of the second array.
     * @return 0 if all elements are the same,
     *         -1 if the first different element in the first array is smaller
     *         than the corresponding element in the second array,
     *         or 1 if it is larger.
     */
    public static int compare(double[] array1, int size1,
                              double[] array2, int size2)
    {
        int minSize = Math.min(size1, size2);

        for (int index = 0; index < minSize; index++)
        {
            int comparison = Double.compare(array1[index], array2[index]);
            if (comparison != 0)
            {
                return comparison;
            }
        }

        return Integer.compare(size1,  size2);
    }


    /**
     * Compares the elements of the two given arrays.
     * @param array1 the first array.
     * @param size1  the size of the first array.
     * @param array2 the second array.
     * @param size2  the size of the second array.
     * @return 0 if all elements are the same,
     *         -1 if the first different element in the first array is smaller
     *         than the corresponding element in the second array,
     *         or 1 if it is larger.
     */
    public static int compare(Comparable[] array1, int size1,
                              Comparable[] array2, int size2)
    {
        int minSize = Math.min(size1, size2);

        for (int index = 0; index < minSize; index++)
        {
            int comparison = ObjectUtil.compare(array1[index], array2[index]);
            if (comparison != 0)
            {
                return comparison;
            }
        }

        return Integer.compare(size1,  size2);
    }


    /**
     * Returns a shallow copy of the given array, or null if the input is null.
     * @param array the array.
     * @return      a shallow copy of the original array, or null if the array is null.
     */
    public static <T> T[] cloneOrNull(T[] array)
    {
        return array != null ? array.clone() : null;
    }


    /**
     * Ensures the given array has a given size.
     * @param array the array.
     * @param size  the target size of the array.
     * @return      the original array, or a copy if it had to be extended.
     */
    public static boolean[] extendArray(boolean[] array, int size)
    {
        // Reuse the existing array if possible.
        if (array.length >= size)
        {
            return array;
        }

        // Otherwise create and initialize a new array.
        boolean[] newArray = new boolean[size];

        System.arraycopy(array, 0,
                         newArray, 0,
                         array.length);

        return newArray;
    }


    /**
     * Ensures the given array has a given size.
     * @param array        the array.
     * @param size         the target size of the array.
     * @param initialValue the initial value of the elements.
     * @return             the original array, or a copy if it had to be
     *                     extended.
     */
    public static boolean[] ensureArraySize(boolean[] array,
                                            int       size,
                                            boolean   initialValue)
    {
        // Is the existing array large enough?
        if (array.length >= size)
        {
            // Reinitialize the existing array.
            Arrays.fill(array, 0, size, initialValue);
        }
        else
        {
            // Otherwise create and initialize a new array.
            array = new boolean[size];

            if (initialValue)
            {
                Arrays.fill(array, 0, size, initialValue);
            }
        }

        return array;
    }


    /**
     * Adds the given element to the given array.
     * The array is extended if necessary.
     * @param array   the array.
     * @param size    the original size of the array.
     * @param element the element to be added.
     * @return        the original array, or a copy if it had to be extended.
     */
    public static byte[] add(byte[] array, int size, byte element)
    {
        array = extendArray(array, size + 1);

        array[size] = element;

        return array;
    }


    /**
     * Inserts the given element in the given array.
     * The array is extended if necessary.
     * @param array   the array.
     * @param size    the original size of the array.
     * @param index   the index at which the element is to be added.
     * @param element the element to be added.
     * @return        the original array, or a copy if it had to be extended.
     */
    public static byte[] insert(byte[] array, int size, int index, byte element)
    {
        array = extendArray(array, size + 1);

        // Move the last part.
        System.arraycopy(array, index,
                         array, index + 1,
                         size - index);

        array[index] = element;

        return array;
    }


    /**
     * Removes the specified element from the given array.
     * @param array the array.
     * @param size  the original size of the array.
     * @param index the index of the element to be removed.
     */
    public static void remove(byte[] array, int size, int index)
    {
        System.arraycopy(array, index + 1,
                         array, index,
                         array.length - index - 1);

        array[size - 1] = 0;
    }


    /**
     * Ensures the given array has a given size.
     * @param array the array.
     * @param size  the target size of the array.
     * @return      the original array, or a copy if it had to be extended.
     */
    public static byte[] extendArray(byte[] array, int size)
    {
        // Reuse the existing array if possible.
        if (array.length >= size)
        {
            return array;
        }

        // Otherwise create and initialize a new array.
        byte[] newArray = new byte[size];

        System.arraycopy(array, 0,
                         newArray, 0,
                         array.length);

        return newArray;
    }


    /**
     * Ensures the given array has a given size.
     * @param array        the array.
     * @param size         the target size of the array.
     * @param initialValue the initial value of the elements.
     * @return             the original array, or a copy if it had to be
     *                     extended.
     */
    public static byte[] ensureArraySize(byte[] array,
                                         int    size,
                                         byte   initialValue)
    {
        // Is the existing array large enough?
        if (array.length >= size)
        {
            // Reinitialize the existing array.
            Arrays.fill(array, 0, size, initialValue);
        }
        else
        {
            // Otherwise create and initialize a new array.
            array = new byte[size];

            if (initialValue != 0)
            {
                Arrays.fill(array, 0, size, initialValue);
            }
        }

        return array;
    }


    /**
     * Adds the given element to the given array.
     * The array is extended if necessary.
     * @param array   the array.
     * @param size    the original size of the array.
     * @param element the element to be added.
     * @return        the original array, or a copy if it had to be extended.
     */
    public static char[] add(char[] array, int size, char element)
    {
        array = extendArray(array, size + 1);

        array[size] = element;

        return array;
    }


    /**
     * Inserts the given element in the given array.
     * The array is extended if necessary.
     * @param array   the array.
     * @param size    the original size of the array.
     * @param index   the index at which the element is to be added.
     * @param element the element to be added.
     * @return        the original array, or a copy if it had to be extended.
     */
    public static char[] insert(char[] array, int size, int index, char element)
    {
        array = extendArray(array, size + 1);

        // Move the last part.
        System.arraycopy(array, index,
                         array, index + 1,
                         size - index);

        array[index] = element;

        return array;
    }


    /**
     * Removes the specified element from the given array.
     * @param array the array.
     * @param size  the original size of the array.
     * @param index the index of the element to be removed.
     */
    public static void remove(char[] array, int size, int index)
    {
        System.arraycopy(array, index + 1,
                         array, index,
                         array.length - index - 1);

        array[size - 1] = 0;
    }


    /**
     * Ensures the given array has a given size.
     * @param array the array.
     * @param size  the target size of the array.
     * @return      the original array, or a copy if it had to be extended.
     */
    public static char[] extendArray(char[] array, int size)
    {
        // Reuse the existing array if possible.
        if (array.length >= size)
        {
            return array;
        }

        // Otherwise create and initialize a new array.
        char[] newArray = new char[size];

        System.arraycopy(array, 0,
                         newArray, 0,
                         array.length);

        return newArray;
    }


    /**
     * Ensures the given array has a given size.
     * @param array        the array.
     * @param size         the target size of the array.
     * @param initialValue the initial value of the elements.
     * @return             the original array, or a copy if it had to be
     *                     extended.
     */
    public static char[] ensureArraySize(char[] array,
                                         int    size,
                                         char   initialValue)
    {
        // Is the existing array large enough?
        if (array.length >= size)
        {
            // Reinitialize the existing array.
            Arrays.fill(array, 0, size, initialValue);
        }
        else
        {
            // Otherwise create and initialize a new array.
            array = new char[size];

            if (initialValue != 0)
            {
                Arrays.fill(array, 0, size, initialValue);
            }
        }

        return array;
    }


    /**
     * Adds the given element to the given array.
     * The array is extended if necessary.
     * @param array   the array.
     * @param size    the original size of the array.
     * @param element the element to be added.
     * @return        the original array, or a copy if it had to be extended.
     */
    public static short[] add(short[] array, int size, short element)
    {
        array = extendArray(array, size + 1);

        array[size] = element;

        return array;
    }


    /**
     * Inserts the given element in the given array.
     * The array is extended if necessary.
     * @param array   the array.
     * @param size    the original size of the array.
     * @param index   the index at which the element is to be added.
     * @param element the element to be added.
     * @return        the original array, or a copy if it had to be extended.
     */
    public static short[] insert(short[] array, int size, int index, short element)
    {
        array = extendArray(array, size + 1);

        // Move the last part.
        System.arraycopy(array, index,
                         array, index + 1,
                         size - index);

        array[index] = element;

        return array;
    }


    /**
     * Removes the specified element from the given array.
     * @param array the array.
     * @param size  the original size of the array.
     * @param index the index of the element to be removed.
     */
    public static void remove(short[] array, int size, int index)
    {
        System.arraycopy(array, index + 1,
                         array, index,
                         array.length - index - 1);

        array[size - 1] = 0;
    }


    /**
     * Ensures the given array has a given size.
     * @param array the array.
     * @param size  the target size of the array.
     * @return      the original array, or a copy if it had to be extended.
     */
    public static short[] extendArray(short[] array, int size)
    {
        // Reuse the existing array if possible.
        if (array.length >= size)
        {
            return array;
        }

        // Otherwise create and initialize a new array.
        short[] newArray = new short[size];

        System.arraycopy(array, 0,
                         newArray, 0,
                         array.length);

        return newArray;
    }


    /**
     * Ensures the given array has a given size.
     * @param array        the array.
     * @param size         the target size of the array.
     * @param initialValue the initial value of the elements.
     * @return             the original array, or a copy if it had to be
     *                     extended.
     */
    public static short[] ensureArraySize(short[] array,
                                          int     size,
                                          short   initialValue)
    {
        // Is the existing array large enough?
        if (array.length >= size)
        {
            // Reinitialize the existing array.
            Arrays.fill(array, 0, size, initialValue);
        }
        else
        {
            // Otherwise create and initialize a new array.
            array = new short[size];

            if (initialValue != 0)
            {
                Arrays.fill(array, 0, size, initialValue);
            }
        }

        return array;
    }


    /**
     * Adds the given element to the given array.
     * The array is extended if necessary.
     * @param array   the array.
     * @param size    the original size of the array.
     * @param element the element to be added.
     * @return        the original array, or a copy if it had to be extended.
     */
    public static int[] add(int[] array, int size, int element)
    {
        array = extendArray(array, size + 1);

        array[size] = element;

        return array;
    }


    /**
     * Inserts the given element in the given array.
     * The array is extended if necessary.
     * @param array   the array.
     * @param size    the original size of the array.
     * @param index   the index at which the element is to be added.
     * @param element the element to be added.
     * @return        the original array, or a copy if it had to be extended.
     */
    public static int[] insert(int[] array, int size, int index, int element)
    {
        array = extendArray(array, size + 1);

        // Move the last part.
        System.arraycopy(array, index,
                         array, index + 1,
                         size - index);

        array[index] = element;

        return array;
    }


    /**
     * Removes the specified element from the given array.
     * @param array the array.
     * @param size  the original size of the array.
     * @param index the index of the element to be removed.
     */
    public static void remove(int[] array, int size, int index)
    {
        System.arraycopy(array, index + 1,
                         array, index,
                         array.length - index - 1);

        array[size - 1] = 0;
    }


    /**
     * Ensures the given array has a given size.
     * @param array the array.
     * @param size  the target size of the array.
     * @return      the original array, or a copy if it had to be extended.
     */
    public static int[] extendArray(int[] array, int size)
    {
        // Reuse the existing array if possible.
        if (array.length >= size)
        {
            return array;
        }

        // Otherwise create and initialize a new array.
        int[] newArray = new int[size];

        System.arraycopy(array, 0,
                         newArray, 0,
                         array.length);

        return newArray;
    }


    /**
     * Ensures the given array has a given size.
     * @param array        the array.
     * @param size         the target size of the array.
     * @param initialValue the initial value of the elements.
     * @return             the original array, or a copy if it had to be
     *                     extended.
     */
    public static int[] ensureArraySize(int[] array,
                                        int   size,
                                        int   initialValue)
    {
        // Is the existing array large enough?
        if (array.length >= size)
        {
            // Reinitialize the existing array.
            Arrays.fill(array, 0, size, initialValue);
        }
        else
        {
            // Otherwise create and initialize a new array.
            array = new int[size];

            if (initialValue != 0)
            {
                Arrays.fill(array, 0, size, initialValue);
            }
        }

        return array;
    }


    /**
     * Adds the given element to the given array.
     * The array is extended if necessary.
     * @param array   the array.
     * @param size    the original size of the array.
     * @param element the element to be added.
     * @return        the original array, or a copy if it had to be extended.
     */
    public static long[] add(long[] array, int size, long element)
    {
        array = extendArray(array, size + 1);

        array[size] = element;

        return array;
    }


    /**
     * Inserts the given element in the given array.
     * The array is extended if necessary.
     * @param array   the array.
     * @param size    the original size of the array.
     * @param index   the index at which the element is to be added.
     * @param element the element to be added.
     * @return        the original array, or a copy if it had to be extended.
     */
    public static long[] insert(long[] array, int size, int index, long element)
    {
        array = extendArray(array, size + 1);

        // Move the last part.
        System.arraycopy(array, index,
                         array, index + 1,
                         size - index);

        array[index] = element;

        return array;
    }


    /**
     * Removes the specified element from the given array.
     * @param array the array.
     * @param size  the original size of the array.
     * @param index the index of the element to be removed.
     */
    public static void remove(long[] array, int size, int index)
    {
        System.arraycopy(array, index + 1,
                         array, index,
                         array.length - index - 1);

        array[size - 1] = 0;
    }


    /**
     * Ensures the given array has a given size.
     * @param array the array.
     * @param size  the target size of the array.
     * @return      the original array, or a copy if it had to be extended.
     */
    public static long[] extendArray(long[] array, int size)
    {
        // Reuse the existing array if possible.
        if (array.length >= size)
        {
            return array;
        }

        // Otherwise create and initialize a new array.
        long[] newArray = new long[size];

        System.arraycopy(array, 0,
                         newArray, 0,
                         array.length);

        return newArray;
    }


    /**
     * Ensures the given array has a given size.
     * @param array        the array.
     * @param size         the target size of the array.
     * @param initialValue the initial value of the elements.
     * @return             the original array, or a copy if it had to be
     *                     extended.
     */
    public static long[] ensureArraySize(long[] array,
                                         int    size,
                                         long   initialValue)
    {
        // Is the existing array large enough?
        if (array.length >= size)
        {
            // Reinitialize the existing array.
            Arrays.fill(array, 0, size, initialValue);
        }
        else
        {
            // Otherwise create and initialize a new array.
            array = new long[size];

            if (initialValue != 0L)
            {
                Arrays.fill(array, 0, size, initialValue);
            }
        }

        return array;
    }


    /**
     * Adds the given element to the given array.
     * The array is extended if necessary.
     * @param array   the array.
     * @param size    the original size of the array.
     * @param element the element to be added.
     * @return        the original array, or a copy if it had to be extended.
     */
    public static float[] add(float[] array, int size, float element)
    {
        array = extendArray(array, size + 1);

        array[size] = element;

        return array;
    }


    /**
     * Inserts the given element in the given array.
     * The array is extended if necessary.
     * @param array   the array.
     * @param size    the original size of the array.
     * @param index   the index at which the element is to be added.
     * @param element the element to be added.
     * @return        the original array, or a copy if it had to be extended.
     */
    public static float[] insert(float[] array, int size, int index, float element)
    {
        array = extendArray(array, size + 1);

        // Move the last part.
        System.arraycopy(array, index,
                         array, index + 1,
                         size - index);

        array[index] = element;

        return array;
    }


    /**
     * Removes the specified element from the given array.
     * @param array the array.
     * @param size  the original size of the array.
     * @param index the index of the element to be removed.
     */
    public static void remove(float[] array, int size, int index)
    {
        System.arraycopy(array, index + 1,
                         array, index,
                         array.length - index - 1);

        array[size - 1] = 0;
    }


    /**
     * Ensures the given array has a given size.
     * @param array the array.
     * @param size  the target size of the array.
     * @return      the original array, or a copy if it had to be extended.
     */
    public static float[] extendArray(float[] array, int size)
    {
        // Reuse the existing array if possible.
        if (array.length >= size)
        {
            return array;
        }

        // Otherwise create and initialize a new array.
        float[] newArray = new float[size];

        System.arraycopy(array, 0,
                         newArray, 0,
                         array.length);

        return newArray;
    }


    /**
     * Ensures the given array has a given size.
     * @param array        the array.
     * @param size         the target size of the array.
     * @param initialValue the initial value of the elements.
     * @return             the original array, or a copy if it had to be
     *                     extended.
     */
    public static float[] ensureArraySize(float[] array,
                                         int    size,
                                         float   initialValue)
    {
        // Is the existing array large enough?
        if (array.length >= size)
        {
            // Reinitialize the existing array.
            Arrays.fill(array, 0, size, initialValue);
        }
        else
        {
            // Otherwise create and initialize a new array.
            array = new float[size];

            if (initialValue != 0)
            {
                Arrays.fill(array, 0, size, initialValue);
            }
        }

        return array;
    }


    /**
     * Adds the given element to the given array.
     * The array is extended if necessary.
     * @param array   the array.
     * @param size    the original size of the array.
     * @param element the element to be added.
     * @return        the original array, or a copy if it had to be extended.
     */
    public static double[] add(double[] array, int size, double element)
    {
        array = extendArray(array, size + 1);

        array[size] = element;

        return array;
    }


    /**
     * Inserts the given element in the given array.
     * The array is extended if necessary.
     * @param array   the array.
     * @param size    the original size of the array.
     * @param index   the index at which the element is to be added.
     * @param element the element to be added.
     * @return        the original array, or a copy if it had to be extended.
     */
    public static double[] insert(double[] array, int size, int index, double element)
    {
        array = extendArray(array, size + 1);

        // Move the last part.
        System.arraycopy(array, index,
                         array, index + 1,
                         size - index);

        array[index] = element;

        return array;
    }


    /**
     * Removes the specified element from the given array.
     * @param array the array.
     * @param size  the original size of the array.
     * @param index the index of the element to be removed.
     */
    public static void remove(double[] array, int size, int index)
    {
        System.arraycopy(array, index + 1,
                         array, index,
                         array.length - index - 1);

        array[size - 1] = 0;
    }


    /**
     * Ensures the given array has a given size.
     * @param array the array.
     * @param size  the target size of the array.
     * @return      the original array, or a copy if it had to be extended.
     */
    public static double[] extendArray(double[] array, int size)
    {
        // Reuse the existing array if possible.
        if (array.length >= size)
        {
            return array;
        }

        // Otherwise create and initialize a new array.
        double[] newArray = new double[size];

        System.arraycopy(array, 0,
                         newArray, 0,
                         array.length);

        return newArray;
    }


    /**
     * Ensures the given array has a given size.
     * @param array        the array.
     * @param size         the target size of the array.
     * @param initialValue the initial value of the elements.
     * @return             the original array, or a copy if it had to be
     *                     extended.
     */
    public static double[] ensureArraySize(double[] array,
                                         int    size,
                                         double   initialValue)
    {
        // Is the existing array large enough?
        if (array.length >= size)
        {
            // Reinitialize the existing array.
            Arrays.fill(array, 0, size, initialValue);
        }
        else
        {
            // Otherwise create and initialize a new array.
            array = new double[size];

            if (initialValue != 0)
            {
                Arrays.fill(array, 0, size, initialValue);
            }
        }

        return array;
    }


    /**
     * Adds the given element to the given array.
     * The array is extended if necessary.
     * @param array   the array.
     * @param size    the original size of the array.
     * @param element the element to be added.
     * @return        the original array, or a copy if it had to be extended.
     */
    public static <T> T[] add(T[] array, int size, T element)
    {
        array = extendArray(array, size + 1);

        array[size] = element;

        return array;
    }


    /**
     * Inserts the given element in the given array.
     * The array is extended if necessary.
     * @param array   the array.
     * @param size    the original size of the array.
     * @param index   the index at which the element is to be added.
     * @param element the element to be added.
     * @return        the original array, or a copy if it had to be extended.
     */
    public static <T> T[] insert(T[] array, int size, int index, T element)
    {
        array = extendArray(array, size + 1);

        // Move the last part.
        System.arraycopy(array, index,
                         array, index + 1,
                         size - index);

        array[index] = element;

        return array;
    }


    /**
     * Removes the specified element from the given array.
     * @param array the array.
     * @param size  the original size of the array.
     * @param index the index of the element to be removed.
     */
    public static void remove(Object[] array, int size, int index)
    {
        System.arraycopy(array, index + 1,
                         array, index,
                         array.length - index - 1);

        array[size - 1] = null;
    }


    /**
     * Ensures the given array has a given size.
     * @param array the array.
     * @param size  the target size of the array.
     * @return      the original array, or a copy if it had to be extended.
     */
    public static <T> T[] extendArray(T[] array, int size)
    {
        // Reuse the existing array if possible.
        if (array.length >= size)
        {
            return array;
        }

        // Otherwise create and initialize a new array.
        T[] newArray = (T[])Array.newInstance(array.getClass().getComponentType(), size);

        System.arraycopy(array, 0,
                         newArray, 0,
                         array.length);

        return newArray;
    }


    /**
     * Ensures the given array has a given size.
     * @param array        the array.
     * @param size         the target size of the array.
     * @param initialValue the initial value of the elements.
     * @return             the original array, or a copy if it had to be
     *                     extended.
     */
    public static <T> T[] ensureArraySize(T[] array,
                                          int size,
                                          T   initialValue)
    {
        // Is the existing array large enough?
        if (array.length >= size)
        {
            // Reinitialize the existing array.
            Arrays.fill(array, 0, size, initialValue);
        }
        else
        {
            // Otherwise create and initialize a new array.
            array = (T[])Array.newInstance(array.getClass().getComponentType(), size);

            if (initialValue != null)
            {
                Arrays.fill(array, 0, size, initialValue);
            }
        }

        return array;
    }
}
