/*
 * plist - An open source library to parse and generate property lists
 * Copyright (C) 2011-2014 Daniel Dreibrodt
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

import java.io.*;
import java.math.BigInteger;

/**
 * Parses property lists that are in Apple's binary format.
 * Use this class when you are sure about the format of the property list.
 * Otherwise use the PropertyListParser class.
 *
 * Parsing is done by calling the static <code>parse</code> methods.
 *
 * @author Daniel Dreibrodt
 */
public class BinaryPropertyListParser {

    /**
     * Major version of the property list format
     */
    @SuppressWarnings("FieldCanBeLocal") //Useful when the features of different format versions are implemented
    private int majorVersion;

    /**
     * Minor version of the property list format
     */
    @SuppressWarnings("FieldCanBeLocal") //Useful when the features of different format versions are implemented
    private int minorVersion;

    /**
     * property list in bytes
     */
    private byte[] bytes;

    /**
     * Length of an object reference in bytes
     */
    private int objectRefSize;

    /**
     * The table holding the information at which offset each object is found
     */
    private int[] offsetTable;

    /**
     * Protected constructor so that instantiation is fully controlled by the
     * static parse methods.
     *
     * @see BinaryPropertyListParser#parse(byte[])
     */
    protected BinaryPropertyListParser() {
        /** empty **/
    }

    /**
     * Parses a binary property list from a byte array.
     *
     * @param data The binary property list's data.
     * @return The root object of the property list. This is usually a NSDictionary but can also be a NSArray.
     * @throws PropertyListFormatException When the property list's format could not be parsed.
     * @throws java.io.UnsupportedEncodingException When a NSString object could not be decoded.
     */
    public static NSObject parse(byte[] data) throws PropertyListFormatException, UnsupportedEncodingException {
        BinaryPropertyListParser parser = new BinaryPropertyListParser();
        return parser.doParse(data);
    }

    /**
     * Parses a binary property list from a byte array.
     *
     * @param data The binary property list's data.
     * @return The root object of the property list. This is usually a NSDictionary but can also be a NSArray.
     * @throws PropertyListFormatException When the property list's format could not be parsed.
     * @throws java.io.UnsupportedEncodingException When a NSString object could not be decoded.
     */
    private NSObject doParse(byte[] data) throws PropertyListFormatException, UnsupportedEncodingException {
        bytes = data;
        String magic = new String(copyOfRange(bytes, 0, 8));
        if (!magic.startsWith("bplist")) {
            throw new IllegalArgumentException("The given data is no binary property list. Wrong magic bytes: " + magic);
        }

        majorVersion = magic.charAt(6) - 0x30; //ASCII number
        minorVersion = magic.charAt(7) - 0x30; //ASCII number

        // 0.0 - OS X Tiger and earlier
        // 0.1 - Leopard
        // 0.? - Snow Leopard
        // 1.5 - Lion
        // 2.0 - Snow Lion

        if (majorVersion > 0) {
            throw new IllegalArgumentException("Unsupported binary property list format: v" + majorVersion + "." + minorVersion + ". " +
                    "Version 1.0 and later are not yet supported.");
            //Version 1.0+ is not even supported by OS X's own parser
        }

        /*
         * Handle trailer, last 32 bytes of the file
         */
        byte[] trailer = copyOfRange(bytes, bytes.length - 32, bytes.length);
        //6 null bytes (index 0 to 5)

        int offsetSize = (int) parseUnsignedInt(trailer, 6, 7);
        objectRefSize = (int) parseUnsignedInt(trailer, 7, 8);
        int numObjects = (int) parseUnsignedInt(trailer, 8, 16);
        int topObject = (int) parseUnsignedInt(trailer, 16, 24);
        int offsetTableOffset = (int) parseUnsignedInt(trailer, 24, 32);

        /*
         * Handle offset table
         */
        offsetTable = new int[numObjects];

        for (int i = 0; i < numObjects; i++) {
            offsetTable[i] = (int) parseUnsignedInt(bytes, offsetTableOffset + i * offsetSize, offsetTableOffset + (i + 1) * offsetSize);
        }

        return parseObject(topObject);
    }

    /**
     * Parses a binary property list from an input stream.
     *
     * @param is The input stream that points to the property list's data.
     * @return The root object of the property list. This is usually a NSDictionary but can also be a NSArray.
     * @throws PropertyListFormatException When the property list's format could not be parsed.
     * @throws java.io.IOException When a NSString object could not be decoded or an InputStream IO error occurs.
     */
    public static NSObject parse(InputStream is) throws IOException, PropertyListFormatException {
        byte[] buf = PropertyListParser.readAll(is);
        return parse(buf);
    }

    /**
     * Parses a binary property list file.
     *
     * @param f The binary property list file
     * @return The root object of the property list. This is usually a NSDictionary but can also be a NSArray.
     * @throws PropertyListFormatException When the property list's format could not be parsed.
     * @throws java.io.UnsupportedEncodingException When a NSString object could not be decoded or a file IO error occurs.
     */
    public static NSObject parse(File f) throws IOException, PropertyListFormatException {
        return parse(new FileInputStream(f));
    }

    /**
     * Parses an object inside the currently parsed binary property list.
     * For the format specification check
     * <a href="http://www.opensource.apple.com/source/CF/CF-855.17/CFBinaryPList.c">
     * Apple's binary property list parser implementation</a>.
     *
     * @param obj The object ID.
     * @return The parsed object.
     * @throws PropertyListFormatException When the property list's format could not be parsed.
     * @throws java.io.UnsupportedEncodingException When a NSString object could not be decoded.
     */
    private NSObject parseObject(int obj) throws PropertyListFormatException, UnsupportedEncodingException {
        int offset = offsetTable[obj];
        byte type = bytes[offset];
        int objType = (type & 0xF0) >> 4; //First  4 bits
        int objInfo = (type & 0x0F);      //Second 4 bits
        switch (objType) {
            case 0x0: {
                //Simple
                switch (objInfo) {
                    case 0x0: {
                        //null object (v1.0 and later)
                        return null;
                    }
                    case 0x8: {
                        //false
                        return new NSNumber(false);
                    }
                    case 0x9: {
                        //true
                        return new NSNumber(true);
                    }
                    case 0xC: {
                        //URL with no base URL (v1.0 and later)
                        //TODO Implement binary URL parsing (not yet even implemented in Core Foundation as of revision 855.17)
                        throw new UnsupportedOperationException("The given binary property list contains a URL object. Parsing of this object type is not yet implemented.");
                    }
                    case 0xD: {
                        //URL with base URL (v1.0 and later)
                        //TODO Implement binary URL parsing (not yet even implemented in Core Foundation as of revision 855.17)
                        throw new UnsupportedOperationException("The given binary property list contains a URL object. Parsing of this object type is not yet implemented.");
                    }
                    case 0xE: {
                        //16-byte UUID (v1.0 and later)
                        //TODO Implement binary UUID parsing (not yet even implemented in Core Foundation as of revision 855.17)
                        throw new UnsupportedOperationException("The given binary property list contains a UUID object. Parsing of this object type is not yet implemented.");
                    }
                    default: {
                        throw new PropertyListFormatException("The given binary property list contains an object of unknown type (" + objType + ")");
                    }
                }
            }
            case 0x1: {
                //integer
                int length = (int) Math.pow(2, objInfo);
                return new NSNumber(bytes, offset + 1, offset + 1 + length, NSNumber.INTEGER);
            }
            case 0x2: {
                //real
                int length = (int) Math.pow(2, objInfo);
                return new NSNumber(bytes, offset + 1, offset + 1 + length, NSNumber.REAL);
            }
            case 0x3: {
                //Date
                if (objInfo != 0x3) {
                    throw new PropertyListFormatException("The given binary property list contains a date object of an unknown type ("+objInfo+")");
                }
                return new NSDate(bytes, offset + 1, offset + 9);
            }
            case 0x4: {
                //Data
                int[] lengthAndOffset = readLengthAndOffset(objInfo, offset);
                int length = lengthAndOffset[0];
                int dataOffset = lengthAndOffset[1];
                return new NSData(copyOfRange(bytes, offset + dataOffset, offset + dataOffset + length));
            }
            case 0x5: {
                //ASCII string
                int[] lengthAndOffset = readLengthAndOffset(objInfo, offset);
                int length = lengthAndOffset[0];  //Each character is 1 byte
                int strOffset = lengthAndOffset[1];
                return new NSString(bytes, offset + strOffset, offset + strOffset + length, "ASCII");
            }
            case 0x6: {
                //UTF-16-BE string
                int[] lengthAndOffset = readLengthAndOffset(objInfo, offset);
                int characters = lengthAndOffset[0];
                int strOffset = lengthAndOffset[1];
                //UTF-16 characters can have variable length, but the Core Foundation reference implementation
                //assumes 2 byte characters, thus only covering the Basic Multilingual Plane
                int length = characters * 2;
                return new NSString(bytes, offset + strOffset, offset + strOffset + length, "UTF-16BE");
            }
            case 0x7: {
                //UTF-8 string (v1.0 and later)
                int[] lengthAndOffset = readLengthAndOffset(objInfo, offset);
                int strOffset = lengthAndOffset[1];
                int characters = lengthAndOffset[0];
                //UTF-8 characters can have variable length, so we need to calculate the byte length dynamically
                //by reading the UTF-8 characters one by one
                int length = calculateUtf8StringLength(bytes, offset + strOffset, characters);
                return new NSString(bytes, offset + strOffset, offset + strOffset + length, "UTF-8");
            }
            case 0x8: {
                //UID (v1.0 and later)
                int length = objInfo + 1;
                return new UID(String.valueOf(obj), copyOfRange(bytes, offset + 1, offset + 1 + length));
            }
            case 0xA: {
                //Array
                int[] lengthAndOffset = readLengthAndOffset(objInfo, offset);
                int length = lengthAndOffset[0];
                int arrayOffset = lengthAndOffset[1];

                NSArray array = new NSArray(length);
                for (int i = 0; i < length; i++) {
                    int objRef = (int) parseUnsignedInt(bytes, offset + arrayOffset + i * objectRefSize, offset + arrayOffset + (i + 1) * objectRefSize);
                    array.setValue(i, parseObject(objRef));
                }
                return array;
            }
            case 0xB: {
                //Ordered set (v1.0 and later)
                int[] lengthAndOffset = readLengthAndOffset(objInfo, offset);
                int length = lengthAndOffset[0];
                int contentOffset = lengthAndOffset[1];

                NSSet set = new NSSet(true);
                for (int i = 0; i < length; i++) {
                    int objRef = (int) parseUnsignedInt(bytes, offset + contentOffset + i * objectRefSize, offset + contentOffset + (i + 1) * objectRefSize);
                    set.addObject(parseObject(objRef));
                }
                return set;
            }
            case 0xC: {
                //Set (v1.0 and later)
                int[] lengthAndOffset = readLengthAndOffset(objInfo, offset);
                int length = lengthAndOffset[0];
                int contentOffset = lengthAndOffset[1];

                NSSet set = new NSSet();
                for (int i = 0; i < length; i++) {
                    int objRef = (int) parseUnsignedInt(bytes, offset + contentOffset + i * objectRefSize, offset + contentOffset + (i + 1) * objectRefSize);
                    set.addObject(parseObject(objRef));
                }
                return set;
            }
            case 0xD: {
                //Dictionary
                int[] lengthAndOffset = readLengthAndOffset(objInfo, offset);
                int length = lengthAndOffset[0];
                int contentOffset = lengthAndOffset[1];

                NSDictionary dict = new NSDictionary();
                for (int i = 0; i < length; i++) {
                    int keyRef = (int) parseUnsignedInt(bytes, offset + contentOffset + i * objectRefSize, offset + contentOffset + (i + 1) * objectRefSize);
                    int valRef = (int) parseUnsignedInt(bytes, offset + contentOffset + (length * objectRefSize) + i * objectRefSize, offset + contentOffset + (length * objectRefSize) + (i + 1) * objectRefSize);
                    NSObject key = parseObject(keyRef);
                    NSObject val = parseObject(valRef);
                    assert key != null; //Encountering a null object at this point would either be a fundamental error in the parser or an error in the property list
                    dict.put(key.toString(), val);
                }
                return dict;
            }
            default: {
                throw new PropertyListFormatException("The given binary property list contains an object of unknown type (" + objType + ")");
            }
        }
    }

    /**
     * Reads the length for arrays, sets and dictionaries.
     *
     * @param objInfo Object information byte.
     * @param offset  Offset in the byte array at which the object is located.
     * @return An array with the length two. First entry is the length, second entry the offset at which the content starts.
     */
    private int[] readLengthAndOffset(int objInfo, int offset) {
        int lengthValue = objInfo;
        int offsetValue = 1;
        if (objInfo == 0xF) {
            int int_type = bytes[offset + 1];
            int intType = (int_type & 0xF0) >> 4;
            if (intType != 0x1) {
                System.err.println("BinaryPropertyListParser: Length integer has an unexpected type" + intType + ". Attempting to parse anyway...");
            }
            int intInfo = int_type & 0x0F;
            int intLength = (int) Math.pow(2, intInfo);
            offsetValue = 2 + intLength;
            if (intLength < 3) {
                lengthValue = (int) parseUnsignedInt(bytes, offset + 2, offset + 2 + intLength);
            } else {
                lengthValue = new BigInteger(copyOfRange(bytes, offset + 2, offset + 2 + intLength)).intValue();
            }
        }
        return new int[]{lengthValue, offsetValue};
    }

    private int calculateUtf8StringLength(byte[] bytes, int offset, int numCharacters) {
        int length = 0;
        for(int i = 0; i < numCharacters; i++) {
            int tempOffset = offset + length;
            if(bytes.length <= tempOffset) {
                //WARNING: Invalid UTF-8 string, fall back to length = number of characters
                return numCharacters;
            }
            if(bytes[tempOffset] < 0x80) {
                length++;
            }
            if(bytes[tempOffset] < 0xC2) {
                //Invalid value (marks continuation byte), fall back to length = number of characters
                return numCharacters;
            }
            else if(bytes[tempOffset] < 0xE0) {
                if((bytes[tempOffset + 1] & 0xC0) != 0x80) {
                    //Invalid continuation byte, fall back to length = number of characters
                    return numCharacters;
                }
                length += 2;
            }
            else if(bytes[tempOffset] < 0xF0) {
                if((bytes[tempOffset + 1] & 0xC0) != 0x80
                        || (bytes[tempOffset + 2] & 0xC0) != 0x80) {
                    //Invalid continuation byte, fall back to length = number of characters
                    return numCharacters;
                }
                length += 3;
            }
            else if(bytes[tempOffset] < 0xF5) {
                if((bytes[tempOffset + 1] & 0xC0) != 0x80
                        || (bytes[tempOffset + 2] & 0xC0) != 0x80
                        || (bytes[tempOffset + 3] & 0xC0) != 0x80) {
                    //Invalid continuation byte, fall back to length = number of characters
                    return numCharacters;
                }
                length += 4;
            }
        }
        return length;
    }

    /**
     * Parses an unsigned integers from a byte array.
     *
     * @param bytes The byte array containing the unsigned integer.
     * @return The unsigned integer represented by the given bytes.
     */
    @SuppressWarnings("unused")
    public static long parseUnsignedInt(byte[] bytes) {
        return parseUnsignedInt(bytes, 0, bytes.length);
    }

    /**
     * Parses an unsigned integer from a byte array.
     *
     * @param bytes The byte array containing the unsigned integer.
     * @param startIndex Beginning of the unsigned int in the byte array.
     * @param endIndex End of the unsigned int in the byte array.
     * @return The unsigned integer represented by the given bytes.
     */
    public static long parseUnsignedInt(byte[] bytes, int startIndex, int endIndex) {
        long l = 0;
        for (int i = startIndex; i < endIndex; i++) {
            l <<= 8;
            l |= bytes[i] & 0xFF;
        }
        l &= 0xFFFFFFFFL;
        return l;
    }

    /**
     * Parses a long from a (big-endian) byte array.
     *
     * @param bytes The bytes representing the long integer.
     * @return The long integer represented by the given bytes.
     */
    @SuppressWarnings("unused")
    public static long parseLong(byte[] bytes) {
        return parseLong(bytes, 0, bytes.length);
    }

    /**
     * Parses a long from a (big-endian) byte array.
     *
     * @param bytes The bytes representing the long integer.
     * @param startIndex Beginning of the long in the byte array.
     * @param endIndex End of the long in the byte array.
     * @return The long integer represented by the given bytes.
     */
    public static long parseLong(byte[] bytes, int startIndex, int endIndex) {
        long l = 0;
        for (int i = startIndex; i < endIndex; i++) {
            l <<= 8;
            l |= bytes[i] & 0xFF;
        }
        return l;
    }

    /**
     * Parses a double from a (big-endian) byte array.
     *
     * @param bytes The bytes representing the double.
     * @return The double represented by the given bytes.
     */
    @SuppressWarnings("unused")
    public static double parseDouble(byte[] bytes) {
        return parseDouble(bytes, 0, bytes.length);
    }

    /**
     * Parses a double from a (big-endian) byte array.
     *
     * @param bytes The bytes representing the double.
     * @param startIndex Beginning of the double in the byte array.
     * @param endIndex End of the double in the byte array.
     * @return The double represented by the given bytes.
     */
    public static double parseDouble(byte[] bytes, int startIndex, int endIndex) {
        if (endIndex - startIndex == 8) {
            return Double.longBitsToDouble(parseLong(bytes, startIndex, endIndex));
        } else if (endIndex - startIndex == 4) {
            return Float.intBitsToFloat((int)parseLong(bytes, startIndex, endIndex));
        } else {
            throw new IllegalArgumentException("endIndex ("+endIndex+") - startIndex ("+startIndex+") != 4 or 8");
        }
    }

    /**
     * Copies a part of a byte array into a new array.
     *
     * @param src        The source array.
     * @param startIndex The index from which to start copying.
     * @param endIndex   The index until which to copy.
     * @return The copied array.
     */
    public static byte[] copyOfRange(byte[] src, int startIndex, int endIndex) {
        int length = endIndex - startIndex;
        if (length < 0) {
            throw new IllegalArgumentException("startIndex (" + startIndex + ")" + " > endIndex (" + endIndex + ")");
        }
        byte[] dest = new byte[length];
        System.arraycopy(src, startIndex, dest, 0, length);
        return dest;
    }
}

