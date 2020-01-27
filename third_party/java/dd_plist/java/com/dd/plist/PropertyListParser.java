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

import org.xml.sax.SAXException;

import javax.xml.parsers.ParserConfigurationException;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.text.ParseException;

/**
 * This class provides methods to parse property lists. It can handle files,
 * input streams and byte arrays. All known property list formats are supported.
 *
 * This class also provides methods to save and convert property lists.
 *
 * @author Daniel Dreibrodt
 */
public class PropertyListParser {

    private static final int TYPE_XML = 0;
    private static final int TYPE_BINARY = 1;
    private static final int TYPE_ASCII = 2;
    private static final int TYPE_ERROR_BLANK = 10;
    private static final int TYPE_ERROR_UNKNOWN = 11;

    private static final int READ_BUFFER_LENGTH = 2048;

    /**
     * Prevent instantiation.
     */
    protected PropertyListParser() {
        /** empty **/
    }

    /**
     * Determines the type of a property list by means of the first bytes of its data
     * @param dataBeginning The very first bytes of data of the property list (minus any whitespace) as a string
     * @return The type of the property list
     */
    private static int determineType(String dataBeginning) {
        dataBeginning = dataBeginning.trim();
        if(dataBeginning.length() == 0) {
            return TYPE_ERROR_BLANK;
        }
        if(dataBeginning.startsWith("bplist")) {
            return TYPE_BINARY;
        }
        if(dataBeginning.startsWith("(") || dataBeginning.startsWith("{") || dataBeginning.startsWith("/")) {
            return TYPE_ASCII;
        }
        if(dataBeginning.startsWith("<")) {
            return TYPE_XML;
        }
        return TYPE_ERROR_UNKNOWN;
    }

    /**
     * Determines the type of a property list by means of the first bytes of its data
     * @param bytes The very first bytes of data of the property list (minus any whitespace)
     * @return The type of the property list
     */
    private static int determineType(byte[] bytes) {
        //Skip any possible whitespace at the beginning of the file
        int offset = 0;
        if(bytes.length >= 3 && (bytes[0] & 0xFF) == 0xEF && (bytes[1] & 0xFF) == 0xBB && (bytes[2] & 0xFF) == 0xBF) {
            //Skip Unicode byte order mark (BOM)
            offset += 3;
        }
        while(offset < bytes.length && (bytes[offset] == ' ' || bytes[offset] == '\t' || bytes[offset] == '\r' || bytes[offset] == '\n' || bytes[offset] == '\f')) {
            offset++;
        }
        return determineType(new String(bytes, offset, Math.min(8, bytes.length - offset)));
    }

    /**
     * Determines the type of a property list by means of the first bytes of its data
     * @param is An input stream pointing to the beginning of the property list data.
     *           If the stream supports marking it will be reset to the beginning of the property
     *           list data after the type has been determined.
     * @return The type of the property list
     */
    private static int determineType(InputStream is) throws IOException {
        //Skip any possible whitespace at the beginning of the file
        byte[] magicBytes = new byte[8];
        int b;
        long index = -1;
        boolean bom = false;
        do {
            if(is.markSupported())
                is.mark(16);
            b = is.read();
            index++;
            //Check if we are reading the Unicode byte order mark (BOM) and skip it
            bom = index < 3 && ((index == 0 && b == 0xEF) || (bom && ((index == 1 && b == 0xBB) || (index == 2 && b == 0xBF))));
        }
        while(b != -1 && b == ' ' || b == '\t' || b == '\r' || b == '\n' || b == '\f' || bom);
        magicBytes[0] = (byte)b;
        int read = is.read(magicBytes, 1, 7);
        int type = determineType(new String(magicBytes, 0, read));
        if(is.markSupported())
            is.reset();
        return type;
    }

    /**
     * Reads all bytes from an InputStream and stores them in an array, up to
     * a maximum count.
     *
     * @param in  The InputStream pointing to the data that should be stored in the array.
     * @return An array containing all bytes that were read from the input stream.
     * @throws java.io.IOException When an IO error while reading from the input stream.
     */
    protected static byte[] readAll(InputStream in) throws IOException {
        ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
        byte[] buf = new byte[READ_BUFFER_LENGTH];
        int read;
        while ((read = in.read(buf, 0, READ_BUFFER_LENGTH)) != -1) {
            outputStream.write(buf, 0, read);
        }
        return outputStream.toByteArray();
    }

    /**
     * Parses a property list from a file.
     *
     * @param filePath Path to the property list file.
     * @return The root object in the property list. This is usually a NSDictionary but can also be a NSArray.
     * @throws javax.xml.parsers.ParserConfigurationException If a document builder for parsing a XML property list
     *                                                        could not be created. This should not occur.
     * @throws java.io.IOException If any IO error occurs while reading the file.
     * @throws org.xml.sax.SAXException If any parse error occurs.
     * @throws com.dd.plist.PropertyListFormatException If the given property list has an invalid format.
     * @throws java.text.ParseException If a date string could not be parsed.
     */
    public static NSObject parse(String filePath) throws ParserConfigurationException, ParseException, SAXException, PropertyListFormatException, IOException {
        return parse(new File(filePath));
    }

    /**
     * Parses a property list from a file.
     *
     * @param f The property list file.
     * @return The root object in the property list. This is usually a NSDictionary but can also be a NSArray.
     * @throws javax.xml.parsers.ParserConfigurationException If a document builder for parsing a XML property list
     *                                                        could not be created. This should not occur.
     * @throws java.io.IOException If any IO error occurs while reading the file.
     * @throws org.xml.sax.SAXException If any parse error occurs.
     * @throws com.dd.plist.PropertyListFormatException If the given property list has an invalid format.
     * @throws java.text.ParseException If a date string could not be parsed.
     */
    public static NSObject parse(File f) throws IOException, PropertyListFormatException, ParseException, ParserConfigurationException, SAXException {
        FileInputStream fis = new FileInputStream(f);
        int type = determineType(fis);
        fis.close();
        switch(type) {
            case TYPE_BINARY:
                return BinaryPropertyListParser.parse(f);
            case TYPE_XML:
                return XMLPropertyListParser.parse(f);
            case TYPE_ASCII:
                return ASCIIPropertyListParser.parse(f);
            default:
                throw new PropertyListFormatException("The given file is not a property list of a supported format.");
        }
    }

    /**
     * Parses a property list from a byte array.
     *
     * @param bytes The property list data as a byte array.
     * @return The root object in the property list. This is usually a NSDictionary but can also be a NSArray.
     * @throws javax.xml.parsers.ParserConfigurationException If a document builder for parsing a XML property list
     *                                                        could not be created. This should not occur.
     * @throws java.io.IOException If any IO error occurs while reading the byte array.
     * @throws org.xml.sax.SAXException If any parse error occurs.
     * @throws com.dd.plist.PropertyListFormatException If the given property list has an invalid format.
     * @throws java.text.ParseException If a date string could not be parsed.
     */
    public static NSObject parse(byte[] bytes) throws IOException, PropertyListFormatException, ParseException, ParserConfigurationException, SAXException {
        switch(determineType(bytes)) {
            case TYPE_BINARY:
                return BinaryPropertyListParser.parse(bytes);
            case TYPE_XML:
                return XMLPropertyListParser.parse(bytes);
            case TYPE_ASCII:
                return ASCIIPropertyListParser.parse(bytes);
            default:
                throw new PropertyListFormatException("The given data is not a property list of a supported format.");
        }
    }

    /**
     * Parses a property list from an InputStream.
     *
     * @param is The InputStream delivering the property list data.
     * @return The root object of the property list. This is usually a NSDictionary but can also be a NSArray.
     * @throws javax.xml.parsers.ParserConfigurationException If a document builder for parsing a XML property list
     *                                                        could not be created. This should not occur.
     * @throws java.io.IOException If any IO error occurs while reading the input stream.
     * @throws org.xml.sax.SAXException If any parse error occurs.
     * @throws com.dd.plist.PropertyListFormatException If the given property list has an invalid format.
     * @throws java.text.ParseException If a date string could not be parsed.
     */
    public static NSObject parse(InputStream is) throws IOException, PropertyListFormatException, ParseException, ParserConfigurationException, SAXException {
        return parse(readAll(is));
    }

    /**
     * Saves a property list with the given object as root into a XML file.
     *
     * @param root The root object.
     * @param out  The output file.
     * @throws IOException When an error occurs during the writing process.
     */
    public static void saveAsXML(NSObject root, File out) throws IOException {
        File parent = out.getParentFile();
        if (!parent.exists())
            if(!parent.mkdirs())
                throw new IOException("The output directory does not exist and could not be created.");
        FileOutputStream fous = new FileOutputStream(out);
        saveAsXML(root, fous);
        fous.close();
    }

    /**
     * Saves a property list with the given object as root in XML format into an output stream.
     *
     * @param root The root object.
     * @param out  The output stream.
     * @throws IOException When an error occurs during the writing process.
     */
    public static void saveAsXML(NSObject root, OutputStream out) throws IOException {
        OutputStreamWriter w = new OutputStreamWriter(out, "UTF-8");
        w.write(root.toXMLPropertyList());
        w.close();
    }

    /**
     * Converts a given property list file into the OS X and iOS XML format.
     *
     * @param in  The source file.
     * @param out The target file.
     *
     * @throws javax.xml.parsers.ParserConfigurationException If a document builder for parsing a XML property list
     *                                                        could not be created. This should not occur.
     * @throws java.io.IOException If any IO error occurs while reading the input file or writing the output file.
     * @throws org.xml.sax.SAXException If any parse error occurs.
     * @throws com.dd.plist.PropertyListFormatException If the given property list has an invalid format.
     * @throws java.text.ParseException If a date string could not be parsed.
     */
    public static void convertToXml(File in, File out) throws ParserConfigurationException, ParseException, SAXException, PropertyListFormatException, IOException {
        NSObject root = parse(in);
        saveAsXML(root, out);
    }

    /**
     * Saves a property list with the given object as root into a binary file.
     *
     * @param root The root object.
     * @param out  The output file.
     * @throws IOException When an error occurs during the writing process.
     */
    public static void saveAsBinary(NSObject root, File out) throws IOException {
        File parent = out.getParentFile();
        if (!parent.exists())
            if(!parent.mkdirs())
                throw new IOException("The output directory does not exist and could not be created.");
        BinaryPropertyListWriter.write(out, root);
    }

    /**
     * Saves a property list with the given object as root in binary format into an output stream.
     *
     * @param root The root object.
     * @param out  The output stream.
     * @throws IOException When an error occurs during the writing process.
     */
    public static void saveAsBinary(NSObject root, OutputStream out) throws IOException {
        BinaryPropertyListWriter.write(out, root);
    }

    /**
     * Converts a given property list file into the OS X and iOS binary format.
     *
     * @param in  The source file.
     * @param out The target file.
     * @throws javax.xml.parsers.ParserConfigurationException If a document builder for parsing a XML property list
     *                                                        could not be created. This should not occur.
     * @throws java.io.IOException If any IO error occurs while reading the input file or writing the output file.
     * @throws org.xml.sax.SAXException If any parse error occurs.
     * @throws com.dd.plist.PropertyListFormatException If the given property list has an invalid format.
     * @throws java.text.ParseException If a date string could not be parsed.
     */
    public static void convertToBinary(File in, File out) throws IOException, ParserConfigurationException, ParseException, SAXException, PropertyListFormatException {
        NSObject root = parse(in);
        saveAsBinary(root, out);
    }

    /**
     * Saves a property list with the given object as root into a ASCII file.
     *
     * @param root The root object.
     * @param out  The output file.
     * @throws IOException When an error occurs during the writing process.
     */
    public static void saveAsASCII(NSDictionary root, File out) throws IOException {
        File parent = out.getParentFile();
        if (!parent.exists())
            if(!parent.mkdirs())
                throw new IOException("The output directory does not exist and could not be created.");
        OutputStreamWriter w = new OutputStreamWriter(new FileOutputStream(out), "ASCII");
        w.write(root.toASCIIPropertyList());
        w.close();
    }

    /**
     * Saves a property list with the given object as root into a ASCII file.
     *
     * @param root The root object.
     * @param out  The output file.
     * @throws IOException When an error occurs during the writing process.
     */
    public static void saveAsASCII(NSArray root, File out) throws IOException {
        OutputStreamWriter w = new OutputStreamWriter(new FileOutputStream(out), "ASCII");
        w.write(root.toASCIIPropertyList());
        w.close();
    }

    /**
     * Converts a given property list file into ASCII format.
     *
     * @param in  The source file.
     * @param out The target file.
     * @throws javax.xml.parsers.ParserConfigurationException If a document builder for parsing a XML property list
     *                                                        could not be created. This should not occur.
     * @throws java.io.IOException If any IO error occurs while reading the input file or writing the output file.
     * @throws org.xml.sax.SAXException If any parse error occurs.
     * @throws com.dd.plist.PropertyListFormatException If the given property list has an invalid format.
     * @throws java.text.ParseException If a date string could not be parsed.
     */
    public static void convertToASCII(File in, File out) throws ParserConfigurationException, ParseException, SAXException, PropertyListFormatException, IOException {
        NSObject root = parse(in);
        if(root instanceof NSDictionary) {
            saveAsASCII((NSDictionary) root, out);
        }
        else if(root instanceof NSArray) {
            saveAsASCII((NSArray) root, out);
        }
        else {
            throw new PropertyListFormatException("The root of the given input property list "
                    + "is neither a Dictionary nor an Array!");
        }
    }

    /**
     * Saves a property list with the given object as root into a ASCII file.
     *
     * @param root The root object.
     * @param out  The output file.
     * @throws IOException When an error occurs during the writing process.
     */
    public static void saveAsGnuStepASCII(NSDictionary root, File out) throws IOException {
        File parent = out.getParentFile();
        if (!parent.exists())
            if(!parent.mkdirs())
                throw new IOException("The output directory does not exist and could not be created.");
        OutputStreamWriter w = new OutputStreamWriter(new FileOutputStream(out), "ASCII");
        w.write(root.toGnuStepASCIIPropertyList());
        w.close();
    }

    /**
     * Saves a property list with the given object as root into a ASCII file.
     *
     * @param root The root object.
     * @param out  The output file.
     * @throws IOException When an error occurs during the writing process.
     */
    public static void saveAsGnuStepASCII(NSArray root, File out) throws IOException {
        File parent = out.getParentFile();
        if (!parent.exists())
            if(!parent.mkdirs())
                throw new IOException("The output directory does not exist and could not be created.");
        OutputStreamWriter w = new OutputStreamWriter(new FileOutputStream(out), "ASCII");
        w.write(root.toGnuStepASCIIPropertyList());
        w.close();
    }

    /**
     * Converts a given property list file into ASCII format.
     *
     * @param in  The source file.
     * @param out The target file.
     * @throws javax.xml.parsers.ParserConfigurationException If a document builder for parsing a XML property list
     *                                                        could not be created. This should not occur.
     * @throws java.io.IOException If any IO error occurs while reading the input file or writing the output file.
     * @throws org.xml.sax.SAXException If any parse error occurs.
     * @throws com.dd.plist.PropertyListFormatException If the given property list has an invalid format.
     * @throws java.text.ParseException If a date string could not be parsed.
     */
    public static void convertToGnuStepASCII(File in, File out) throws ParserConfigurationException, ParseException, SAXException, PropertyListFormatException, IOException {
        NSObject root = parse(in);
        if(root instanceof NSDictionary) {
            saveAsGnuStepASCII((NSDictionary) root, out);
        }
        else if(root instanceof NSArray) {
            saveAsGnuStepASCII((NSArray) root, out);
        }
        else {
            throw new PropertyListFormatException("The root of the given input property list "
                    + "is neither a Dictionary nor an Array!");
        }
    }
}
