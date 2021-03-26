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
package proguard.obfuscate;

import java.io.*;


/**
 * This class can parse mapping files and invoke a processor for each of the
 * mapping entries.
 *
 * @author Eric Lafortune
 */
public class MappingReader
{
    private final File mappingFile;


    public MappingReader(File mappingFile)
    {
        this.mappingFile = mappingFile;
    }


    /**
     * Reads the mapping file, presenting all of the encountered mapping entries
     * to the given processor.
     */
    public void pump(MappingProcessor mappingProcessor) throws IOException
    {
        LineNumberReader reader =
            new LineNumberReader(
            new BufferedReader(
            new InputStreamReader(
            new FileInputStream(mappingFile), "UTF-8")));
        try
        {
            String className = null;

            // Read the subsequent class mappings and class member mappings.
            while (true)
            {
                String line = reader.readLine();

                if (line == null)
                {
                    break;
                }

                line = line.trim();

                // Is it a non-comment line?
                if (!line.startsWith("#"))
                {
                    // Is it a class mapping or a class member mapping?
                    if (line.endsWith(":"))
                    {
                        // Process the class mapping and remember the class's
                        // old name.
                        className = processClassMapping(line, mappingProcessor);
                    }
                    else if (className != null)
                    {
                        // Process the class member mapping, in the context of
                        // the current old class name.
                        processClassMemberMapping(className, line, mappingProcessor);
                    }
                }
            }
        }
        catch (IOException ex)
        {
            throw new IOException("Can't process mapping file (" + ex.getMessage() + ")");
        }
        finally
        {
            try
            {
                reader.close();
            }
            catch (IOException ex)
            {
                // This shouldn't happen.
            }
        }
    }


    /**
     * Parses the given line with a class mapping and processes the
     * results with the given mapping processor. Returns the old class name,
     * or null if any subsequent class member lines can be ignored.
     */
    private String processClassMapping(String           line,
                                       MappingProcessor mappingProcessor)
    {
        // See if we can parse "___ -> ___:", containing the original
        // class name and the new class name.

        int arrowIndex = line.indexOf("->");
        if (arrowIndex < 0)
        {
            return null;
        }

        int colonIndex = line.indexOf(':', arrowIndex + 2);
        if (colonIndex < 0)
        {
            return null;
        }

        // Extract the elements.
        String className    = line.substring(0, arrowIndex).trim();
        String newClassName = line.substring(arrowIndex + 2, colonIndex).trim();

        // Process this class name mapping.
        boolean interested = mappingProcessor.processClassMapping(className, newClassName);

        return interested ? className : null;
    }


    /**
     * Parses the given line with a class member mapping and processes the
     * results with the given mapping processor.
     */
    private void processClassMemberMapping(String           className,
                                           String           line,
                                           MappingProcessor mappingProcessor)
    {
        // See if we can parse one of
        //     ___ ___ -> ___
        //     ___:___:___ ___(___) -> ___
        //     ___:___:___ ___(___):___ -> ___
        //     ___:___:___ ___(___):___:___ -> ___
        // containing the optional line numbers, the return type, the original
        // field/method name, optional arguments, the optional original line
        // numbers, and the new field/method name. The original field/method
        // name may contain an original class name "___.___".

        int colonIndex1    =                           line.indexOf(':');
        int colonIndex2    = colonIndex1    < 0 ? -1 : line.indexOf(':', colonIndex1    + 1);
        int spaceIndex     =                           line.indexOf(' ', colonIndex2    + 2);
        int argumentIndex1 =                           line.indexOf('(', spaceIndex     + 1);
        int argumentIndex2 = argumentIndex1 < 0 ? -1 : line.indexOf(')', argumentIndex1 + 1);
        int colonIndex3    = argumentIndex2 < 0 ? -1 : line.indexOf(':', argumentIndex2 + 1);
        int colonIndex4    = colonIndex3    < 0 ? -1 : line.indexOf(':', colonIndex3    + 1);
        int arrowIndex     =                           line.indexOf("->", (colonIndex4    >= 0 ? colonIndex4    :
                                                                           colonIndex3    >= 0 ? colonIndex3    :
                                                                           argumentIndex2 >= 0 ? argumentIndex2 :
                                                                                                 spaceIndex) + 1);

        if (spaceIndex < 0 ||
            arrowIndex < 0)
        {
            return;
        }

        // Extract the elements.
        String type    = line.substring(colonIndex2 + 1, spaceIndex).trim();
        String name    = line.substring(spaceIndex + 1, argumentIndex1 >= 0 ? argumentIndex1 : arrowIndex).trim();
        String newName = line.substring(arrowIndex + 2).trim();

        // Does the method name contain an explicit original class name?
        String newClassName = className;
        int dotIndex = name.lastIndexOf('.');
        if (dotIndex >= 0)
        {
            className = name.substring(0, dotIndex);
            name      = name.substring(dotIndex + 1);
        }

        // Process this class member mapping.
        if (type.length()    > 0 &&
            name.length()    > 0 &&
            newName.length() > 0)
        {
            // Is it a field or a method?
            if (argumentIndex2 < 0)
            {
                mappingProcessor.processFieldMapping(className,
                                                     type,
                                                     name,
                                                     newClassName,
                                                     newName);
            }
            else
            {
                int firstLineNumber = 0;
                int lastLineNumber  = 0;
                int newFirstLineNumber = 0;
                int newLastLineNumber  = 0;

                if (colonIndex2 >= 0)
                {
                    firstLineNumber = newFirstLineNumber = Integer.parseInt(line.substring(0, colonIndex1).trim());
                    lastLineNumber  = newLastLineNumber  = Integer.parseInt(line.substring(colonIndex1 + 1, colonIndex2).trim());
                }

                if (colonIndex3 >= 0)
                {
                    firstLineNumber = Integer.parseInt(line.substring(colonIndex3 + 1, colonIndex4 > 0 ? colonIndex4 : arrowIndex).trim());
                    lastLineNumber  = colonIndex4 < 0 ? firstLineNumber :
                                      Integer.parseInt(line.substring(colonIndex4 + 1, arrowIndex).trim());
                }

                String arguments = line.substring(argumentIndex1 + 1, argumentIndex2).trim();

                mappingProcessor.processMethodMapping(className,
                                                      firstLineNumber,
                                                      lastLineNumber,
                                                      type,
                                                      name,
                                                      arguments,
                                                      newClassName,
                                                      newFirstLineNumber,
                                                      newLastLineNumber,
                                                      newName);
            }
        }
    }
}
