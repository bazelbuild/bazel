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

import proguard.Configuration;

import java.io.*;

/**
 * Utility code for creating PrintWriters for printing mappings etc.
 * @author Johan Leys
 */
public class PrintWriterUtil
{
    /**
     * Returns a print writer for the given file, or the standard output if
     * the file name is empty.
     */
    public static PrintWriter createPrintWriterOut(File outputFile)
            throws FileNotFoundException, UnsupportedEncodingException
    {
        return createPrintWriterOut(outputFile, false);
    }

   /**
     * Returns a print writer for the given file, or the standard output if
     * the file name is empty.
     */
    public static PrintWriter createPrintWriterOut(File outputFile, boolean append)
            throws FileNotFoundException, UnsupportedEncodingException
    {

        return createPrintWriter(outputFile, new PrintWriter(System.out, true), append);
    }


    /**
     * Returns a print writer for the given file, or the standard output if
     * the file name is empty.
     */
    public static PrintWriter createPrintWriterErr(File outputFile)
            throws FileNotFoundException, UnsupportedEncodingException
    {
        return createPrintWriter(outputFile, new PrintWriter(System.err, true));
    }


    /**
     * Returns a print writer for the given file, or the standard output if
     * the file name is empty.
     */
    public static PrintWriter createPrintWriter(File outputFile, PrintWriter console)
            throws FileNotFoundException, UnsupportedEncodingException
    {
        return createPrintWriter(outputFile, console, false);
    }


    /**
     * Returns a print writer for the given file, or the standard output if
     * the file name is empty.
     */
    public static PrintWriter createPrintWriter(File outputFile,
                                                PrintWriter console,
                                                boolean append)
    throws FileNotFoundException, UnsupportedEncodingException
    {
        return outputFile == Configuration.STD_OUT ?
            console :
            new PrintWriter(
            new BufferedWriter(
            new OutputStreamWriter(
            new FileOutputStream(outputFile, append), "UTF-8")));
    }

    /**
     * Closes the given print writer, or flushes it if is the standard output.
     */
    public static void closePrintWriter(File file, PrintWriter printWriter)
    {
        if (file == Configuration.STD_OUT)
        {
            printWriter.flush();
        }
        else
        {
            printWriter.close();
        }
    }


    /**
     * Returns the canonical file name for the given file, or "standard output"
     * if the file name is empty.
     */
    public static String fileName(File file)
    {
        if (file == Configuration.STD_OUT)
        {
            return "standard output";
        }
        else
        {
            try
            {
                return file.getCanonicalPath();
            }
            catch (IOException ex)
            {
                return file.getPath();
            }
        }
    }


    // Hide constructor for util class.
    private PrintWriterUtil() {}
}
