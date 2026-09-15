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
package proguard.io;

import proguard.classfile.ClassConstants;
import proguard.util.ExtensionMatcher;

import java.io.*;

/**
 * This DataEntryReader writes the ZIP entries and files that it reads to a
 * given DataEntryWriter.
 *
 * @author Eric Lafortune
 */
public class DataEntryCopier implements DataEntryReader
{
    private static final int BUFFER_SIZE = 1024;

    private final DataEntryWriter dataEntryWriter;
    private final byte[]          buffer = new byte[BUFFER_SIZE];


    /**
     * Creates a new DataEntryCopier.
     */
    public DataEntryCopier(DataEntryWriter dataEntryWriter)
    {
        this.dataEntryWriter = dataEntryWriter;
    }


    // Implementations for DataEntryReader.

    public void read(DataEntry dataEntry) throws IOException
    {
        try
        {
            if (dataEntry.isDirectory())
            {
                dataEntryWriter.createDirectory(dataEntry);
            }
            else
            {
                // Get the output entry corresponding to this input entry.
                OutputStream outputStream = dataEntryWriter.createOutputStream(dataEntry);
                if (outputStream != null)
                {
                    try
                    {
                        InputStream inputStream = dataEntry.getInputStream();

                        try
                        {
                            // Copy the data from the input entry to the output entry.
                            copyData(inputStream, outputStream);

                            // Flush the output stream, just to be sure.
                            outputStream.flush();
                        }
                        finally
                        {
                            // Close the input stream.
                            dataEntry.closeInputStream();
                        }
                    }
                    finally
                    {
                        // Close the output stream.
                        outputStream.close();
                    }
                }
            }
        }
        catch (IOException ex)
        {
            System.err.println("Warning: can't write resource [" + dataEntry.getName() + "] (" + ex.getMessage() + ")");
        }
        catch (Exception ex)
        {
            throw (IOException)new IOException("Can't write resource ["+dataEntry.getName()+"] ("+ex.getMessage()+")").initCause(ex);
        }
    }


    /**
     * Copies all data that it can read from the given input stream to the
     * given output stream. The caller of this method will open and
     * afterwards flush and close the input stream and the output stream.
     * The implementation of this method needs to make sure that any wrapping
     * output streams are flushed before returning.
     */
    protected void copyData(InputStream  inputStream,
                            OutputStream outputStream)
    throws IOException
    {
        while (true)
        {
            int count = inputStream.read(buffer);
            if (count < 0)
            {
                break;
            }
            outputStream.write(buffer, 0, count);
        }
    }


    /**
     * A main method for testing file/archive/directory copying.
     */
    public static void main(String[] args)
    {
        try
        {
            String input  = args[0];
            String output = args[1];

            boolean outputIsApk  = output.endsWith(".apk") ||
                                   output.endsWith(".ap_");
            boolean outputIsJar  = output.endsWith(".jar");
            boolean outputIsAar  = output.endsWith(".aar");
            boolean outputIsWar  = output.endsWith(".war");
            boolean outputIsEar  = output.endsWith(".ear");
            boolean outputIsJmod = output.endsWith(".jmod");
            boolean outputIsZip  = output.endsWith(".zip");

            DataEntryWriter writer = new DirectoryWriter(new File(output),
                                                         outputIsApk  ||
                                                         outputIsJar  ||
                                                         outputIsAar  ||
                                                         outputIsWar  ||
                                                         outputIsEar  ||
                                                         outputIsJmod ||
                                                         outputIsZip);

            // Zip up any zips, if necessary.
            DataEntryWriter zipWriter = new JarWriter(writer);
            if (outputIsZip)
            {
                // Always zip.
                writer = zipWriter;
            }
            else
            {
                // Only zip up zips.
                writer = new FilteredDataEntryWriter(new DataEntryParentFilter(
                                                     new DataEntryNameFilter(
                                                     new ExtensionMatcher(".zip"))),
                                                     zipWriter,
                                                     writer);
            }

            // Zip up any jmods, if necessary.
            DataEntryWriter jmodWriter = new JarWriter(ClassConstants.JMOD_HEADER, writer);
            if (outputIsJmod)
            {
                // Always zip.
                writer = jmodWriter;
            }
            else
            {
                // Only zip up jmods.
                writer = new FilteredDataEntryWriter(new DataEntryParentFilter(
                                                     new DataEntryNameFilter(
                                                     new ExtensionMatcher(".jmod"))),
                                                     jmodWriter,
                                                     writer);
            }

            // Zip up any wars, if necessary.
            DataEntryWriter warWriter = new JarWriter(writer);
            if (outputIsWar)
            {
                // Always zip.
                writer = warWriter;
            }
            else
            {
                // Only zip up wars.
                writer = new FilteredDataEntryWriter(new DataEntryParentFilter(
                                                     new DataEntryNameFilter(
                                                     new ExtensionMatcher(".war"))),
                                                     warWriter,
                                                     writer);
            }

            // Zip up any aars, if necessary.
            DataEntryWriter aarWriter = new JarWriter(writer);
            if (outputIsWar)
            {
                // Always zip.
                writer = aarWriter;
            }
            else
            {
                // Only zip up aars.
                writer = new FilteredDataEntryWriter(new DataEntryParentFilter(
                                                     new DataEntryNameFilter(
                                                     new ExtensionMatcher(".aar"))),
                                                     aarWriter,
                                                     writer);
            }

            // Zip up any jars, if necessary.
            DataEntryWriter jarWriter = new JarWriter(writer);
            if (outputIsJar)
            {
                // Always zip.
                writer = jarWriter;
            }
            else
            {
                // Only zip up jars.
                writer = new FilteredDataEntryWriter(new DataEntryParentFilter(
                                                     new DataEntryNameFilter(
                                                     new ExtensionMatcher(".jar"))),
                                                     jarWriter,
                                                     writer);
            }

            // Zip up any apks, if necessary.
            DataEntryWriter apkWriter = new JarWriter(writer);
            if (outputIsApk)
            {
                // Always zip.
                writer = apkWriter;
            }
            else
            {
                // Only zip up apks.
                writer = new FilteredDataEntryWriter(new DataEntryParentFilter(
                                                     new DataEntryNameFilter(
                                                     new ExtensionMatcher(".apk"))),
                                                     apkWriter,
                                                     writer);
            }


            // Create the copying DataEntryReader.
            DataEntryReader reader = new DataEntryCopier(writer);

            boolean inputIsApk  = input.endsWith(".apk") ||
                                  input.endsWith(".ap_");
            boolean inputIsJar  = input.endsWith(".jar");
            boolean inputIsAar  = input.endsWith(".aar");
            boolean inputIsWar  = input.endsWith(".war");
            boolean inputIsEar  = input.endsWith(".ear");
            boolean inputIsJmod = input.endsWith(".jmod");
            boolean inputIsZip  = input.endsWith(".zip");

            // Unzip any apks, if necessary.
            DataEntryReader apkReader = new JarReader(reader);
            if (inputIsApk)
            {
                // Always unzip.
                reader = apkReader;
            }
            else
            {
                // Only unzip apk entries.
                reader = new FilteredDataEntryReader(new DataEntryNameFilter(
                                                     new ExtensionMatcher(".apk")),
                                                     apkReader,
                                                     reader);

                // Unzip any jars, if necessary.
                DataEntryReader jarReader = new JarReader(reader);
                if (inputIsJar)
                {
                    // Always unzip.
                    reader = jarReader;
                }
                else
                {
                    // Only unzip jar entries.
                    reader = new FilteredDataEntryReader(new DataEntryNameFilter(
                                                         new ExtensionMatcher(".jar")),
                                                         jarReader,
                                                         reader);

                    // Unzip any aars, if necessary.
                    DataEntryReader aarReader = new JarReader(reader);
                    if (inputIsAar)
                    {
                        // Always unzip.
                        reader = aarReader;
                    }
                    else
                    {
                        // Only unzip aar entries.
                        reader = new FilteredDataEntryReader(new DataEntryNameFilter(
                                                             new ExtensionMatcher(".aar")),
                                                             aarReader,
                                                             reader);

                        // Unzip any wars, if necessary.
                        DataEntryReader warReader = new JarReader(reader);
                        if (inputIsWar)
                        {
                            // Always unzip.
                            reader = warReader;
                        }
                        else
                        {
                            // Only unzip war entries.
                            reader = new FilteredDataEntryReader(new DataEntryNameFilter(
                                                                 new ExtensionMatcher(".war")),
                                                                 warReader,
                                                                 reader);

                            // Unzip any ears, if necessary.
                            DataEntryReader earReader = new JarReader(reader);
                            if (inputIsEar)
                            {
                                // Always unzip.
                                reader = earReader;
                            }
                            else
                            {
                                // Only unzip ear entries.
                                reader = new FilteredDataEntryReader(new DataEntryNameFilter(
                                                                     new ExtensionMatcher(".ear")),
                                                                     earReader,
                                                                     reader);

                                // Unzip any jmods, if necessary.
                                DataEntryReader jmodReader = new JarReader(reader, true);
                                if (inputIsJmod)
                                {
                                    // Always unzip.
                                    reader = jmodReader;
                                }
                                else
                                {
                                    // Only unzip jmod entries.
                                    reader = new FilteredDataEntryReader(new DataEntryNameFilter(
                                                                         new ExtensionMatcher(".jmod")),
                                                                         jmodReader,
                                                                         reader);

                                    // Unzip any zips, if necessary.
                                    DataEntryReader zipReader = new JarReader(reader);
                                    if (inputIsZip)
                                    {
                                        // Always unzip.
                                        reader = zipReader;
                                    }
                                    else
                                    {
                                        // Only unzip zip entries.
                                        reader = new FilteredDataEntryReader(new DataEntryNameFilter(
                                                                             new ExtensionMatcher(".zip")),
                                                                             zipReader,
                                                                             reader);
                                    }
                                }
                            }
                        }
                    }
                }
            }

            DirectoryPump directoryReader = new DirectoryPump(new File(input));

            directoryReader.pumpDataEntries(reader);

            writer.close();
        }
        catch (Exception ex)
        {
            ex.printStackTrace();
        }
    }
}
