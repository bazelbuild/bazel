/*
 * plist - An open source library to parse and generate property lists
 * Copyright (C) 2014 Daniel Dreibrodt
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

package com.dd.plist;

import java.io.IOException;
import java.util.Arrays;

/**
 * Represents an Array.
 *
 * @author Daniel Dreibrodt
 */
public class NSArray extends NSObject {

    private NSObject[] array;

    /**
     * Creates an empty array of the given length.
     *
     * @param length The number of elements this array will be able to hold.
     */
    public NSArray(int length) {
        array = new NSObject[length];
    }

    /**
     * Creates a array from an existing one
     *
     * @param a The array which should be wrapped by the NSArray
     */
    public NSArray(NSObject... a) {
        array = a;
    }

    /**
     * Returns the object stored at the given index.
     * Equivalent to <code>getArray()[i]</code>.
     *
     * @param i The index of the object.
     * @return The object at the given index.
     */
    public NSObject objectAtIndex(int i) {
        return array[i];
    }

    /**
     * Remove the i-th element from the array.
     * The array will be resized.
     *
     * @param i The index of the object
     */
    public void remove(int i) {
        if ((i >= array.length) || (i < 0))
            throw new ArrayIndexOutOfBoundsException("invalid index:" + i + ";the array length is " + array.length);
        NSObject[] newArray = new NSObject[array.length - 1];
        System.arraycopy(array, 0, newArray, 0, i);
        System.arraycopy(array, i + 1, newArray, i, array.length - i - 1);
        array = newArray;
    }

    /**
     * Stores an object at the specified index.
     * If there was another object stored at that index it will be replaced.
     * Equivalent to <code>getArray()[key] = value</code>.
     *
     * @param key   The index where to store the object.
     * @param value The object.
     */
    public void setValue(int key, Object value) {
        array[key] = NSObject.wrap(value);
    }

    /**
     * Returns the array of NSObjects represented by this NSArray.
     * Any changes to the values of this array will also affect the NSArray.
     *
     * @return The actual array represented by this NSArray.
     */
    public NSObject[] getArray() {
        return array;
    }

    /**
     * Returns the size of the array.
     *
     * @return The number of elements that this array can store.
     */
    public int count() {
        return array.length;
    }

    /**
     * Checks whether an object is present in the array or whether it is equal
     * to any of the objects in the array.
     *
     * @param obj The object to look for.
     * @return <code>true</code>, when the object could be found. <code>false</code> otherwise.
     * @see Object#equals(java.lang.Object)
     */
    public boolean containsObject(Object obj) {
        NSObject nso = NSObject.wrap(obj);
        for (NSObject elem : array) {
            if(elem == null) {
                if(obj == null)
                    return true;
                continue;
            }
            if (elem.equals(nso)) {
                return true;
            }
        }
        return false;
    }

    /**
     * Searches for an object in the array. If it is found its index will be
     * returned. This method also returns an index if the object is not the same
     * as the one stored in the array but has equal contents.
     *
     * @param obj The object to look for.
     * @return The index of the object, if it was found. -1 otherwise.
     * @see Object#equals(java.lang.Object)
     * @see #indexOfIdenticalObject(Object)
     */
    public int indexOfObject(Object obj) {
        NSObject nso = NSObject.wrap(obj);
        for (int i = 0; i < array.length; i++) {
            if (array[i].equals(nso)) {
                return i;
            }
        }
        return -1;
    }

    /**
     * Searches for an object in the array. If it is found its index will be
     * returned. This method only returns the index of an object that is
     * <b>identical</b> to the given one. Thus objects that might contain the
     * same value as the given one will not be considered.
     *
     * @param obj The object to look for.
     * @return The index of the object, if it was found. -1 otherwise.
     * @see #indexOfObject(Object)
     */
    public int indexOfIdenticalObject(Object obj) {
        NSObject nso = NSObject.wrap(obj);
        for (int i = 0; i < array.length; i++) {
            if (array[i] == nso) {
                return i;
            }
        }
        return -1;
    }

    /**
     * Returns the last object contained in this array.
     * Equivalent to <code>getArray()[getArray().length-1]</code>.
     *
     * @return The value of the highest index in the array.
     */
    public NSObject lastObject() {
        return array[array.length - 1];
    }

    /**
     * Returns a new array containing only the values stored at the given
     * indices. The values are sorted by their index.
     *
     * @param indexes The indices of the objects.
     * @return The new array containing the objects stored at the given indices.
     */
    public NSObject[] objectsAtIndexes(int... indexes) {
        NSObject[] result = new NSObject[indexes.length];
        Arrays.sort(indexes);
        for (int i = 0; i < indexes.length; i++)
            result[i] = array[indexes[i]];
        return result;
    }

    @Override
    public boolean equals(Object obj) {
        if(obj == null)
            return false;
        if(obj.getClass().equals(NSArray.class)) {
            return Arrays.equals(((NSArray) obj).getArray(), this.array);
        } else {
            NSObject nso = NSObject.wrap(obj);
            if(nso.getClass().equals(NSArray.class)) {
                return Arrays.equals(((NSArray) nso).getArray(), this.array);
            }
        }
        return false;
    }

    @Override
    public int hashCode() {
        int hash = 7;
        hash = 89 * hash + Arrays.deepHashCode(this.array);
        return hash;
    }

    @Override
    void toXML(StringBuilder xml, int level) {
        indent(xml, level);
        xml.append("<array>");
        xml.append(NSObject.NEWLINE);
        for (NSObject o : array) {
            o.toXML(xml, level + 1);
            xml.append(NSObject.NEWLINE);
        }
        indent(xml, level);
        xml.append("</array>");
    }

    @Override
    void assignIDs(BinaryPropertyListWriter out) {
        super.assignIDs(out);
        for (NSObject obj : array) {
            obj.assignIDs(out);
        }
    }

    @Override
    void toBinary(BinaryPropertyListWriter out) throws IOException {
        out.writeIntHeader(0xA, array.length);
        for (NSObject obj : array) {
            out.writeID(out.getID(obj));
        }
    }

    /**
     * Generates a valid ASCII property list which has this NSArray as its
     * root object. The generated property list complies with the format as
     * described in <a href="https://developer.apple.com/library/mac/#documentation/Cocoa/Conceptual/PropertyLists/OldStylePlists/OldStylePLists.html">
     * Property List Programming Guide - Old-Style ASCII Property Lists</a>.
     *
     * @return ASCII representation of this object.
     */
    public String toASCIIPropertyList() {
        StringBuilder ascii = new StringBuilder();
        toASCII(ascii, 0);
        ascii.append(NEWLINE);
        return ascii.toString();
    }

    /**
     * Generates a valid ASCII property list in GnuStep format which has this
     * NSArray as its root object. The generated property list complies with
     * the format as described in <a href="http://www.gnustep.org/resources/documentation/Developer/Base/Reference/NSPropertyList.html">
     * GnuStep - NSPropertyListSerialization class documentation
     * </a>
     *
     * @return GnuStep ASCII representation of this object.
     */
    public String toGnuStepASCIIPropertyList() {
        StringBuilder ascii = new StringBuilder();
        toASCIIGnuStep(ascii, 0);
        ascii.append(NEWLINE);
        return ascii.toString();
    }

    @Override
    protected void toASCII(StringBuilder ascii, int level) {
        indent(ascii, level);
        ascii.append(ASCIIPropertyListParser.ARRAY_BEGIN_TOKEN);
        int indexOfLastNewLine = ascii.lastIndexOf(NEWLINE);
        for (int i = 0; i < array.length; i++) {
            Class<?> objClass = array[i].getClass();
            if ((objClass.equals(NSDictionary.class) || objClass.equals(NSArray.class) || objClass.equals(NSData.class))
                    && indexOfLastNewLine != ascii.length()) {
                ascii.append(NEWLINE);
                indexOfLastNewLine = ascii.length();
                array[i].toASCII(ascii, level + 1);
            } else {
                if (i != 0)
                    ascii.append(" ");
                array[i].toASCII(ascii, 0);
            }

            if (i != array.length - 1)
                ascii.append(ASCIIPropertyListParser.ARRAY_ITEM_DELIMITER_TOKEN);

            if (ascii.length() - indexOfLastNewLine > ASCII_LINE_LENGTH) {
                ascii.append(NEWLINE);
                indexOfLastNewLine = ascii.length();
            }
        }
        ascii.append(ASCIIPropertyListParser.ARRAY_END_TOKEN);
    }

    @Override
    protected void toASCIIGnuStep(StringBuilder ascii, int level) {
        indent(ascii, level);
        ascii.append(ASCIIPropertyListParser.ARRAY_BEGIN_TOKEN);
        int indexOfLastNewLine = ascii.lastIndexOf(NEWLINE);
        for (int i = 0; i < array.length; i++) {
            Class<?> objClass = array[i].getClass();
            if ((objClass.equals(NSDictionary.class) || objClass.equals(NSArray.class) || objClass.equals(NSData.class))
                    && indexOfLastNewLine != ascii.length()) {
                ascii.append(NEWLINE);
                indexOfLastNewLine = ascii.length();
                array[i].toASCIIGnuStep(ascii, level + 1);
            } else {
                if (i != 0)
                    ascii.append(" ");
                array[i].toASCIIGnuStep(ascii, 0);
            }

            if (i != array.length - 1)
                ascii.append(ASCIIPropertyListParser.ARRAY_ITEM_DELIMITER_TOKEN);

            if (ascii.length() - indexOfLastNewLine > ASCII_LINE_LENGTH) {
                ascii.append(NEWLINE);
                indexOfLastNewLine = ascii.length();
            }
        }
        ascii.append(ASCIIPropertyListParser.ARRAY_END_TOKEN);
    }
}
